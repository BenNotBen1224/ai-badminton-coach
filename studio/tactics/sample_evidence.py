import argparse
import hashlib
import json
import math
import pathlib
import re
import subprocess
import tempfile

import pymupdf


ROOT = pathlib.Path(__file__).resolve().parents[3]
EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
COLORS = {"near": (1, 0.25, 0.4), "far": (0.1, 1, 0.8)}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=pathlib.Path)
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    probe = json.loads(subprocess.check_output([
        str(ROOT / "tools" / "ffprobe.exe"), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,start_time",
        "-of", "json", str(args.source),
    ]))["streams"][0]
    numerator, denominator = map(int, probe["avg_frame_rate"].split("/"))
    fps = numerator / denominator
    if probe["r_frame_rate"] != probe["avg_frame_rate"] or float(probe["start_time"]) != 0:
        raise ValueError("This sampler requires a zero-start constant-rate source; do not guess a VFR mapping.")
    index = json.loads(args.index.read_text(encoding="utf-8"))
    caches = {}
    samples = []
    for requested in args.times:
        segments = [s for s in index["segments"] if s["reel_start_s"] <= requested < s["reel_end_s"]]
        if len(segments) != 1:
            raise ValueError(f"{requested}: expected one cache segment, found {len(segments)}")
        segment = segments[0]
        path = pathlib.Path(segment["pose_path"])
        if path not in caches:
            caches[path] = json.loads(path.read_text(encoding="utf-8"))
        document = caches[path]
        if pathlib.Path(document["source"]["original_file"]).resolve() != args.source.resolve():
            raise ValueError("Pose cache and requested video are different sources.")
        if (document["source"]["width"], document["source"]["height"]) != (probe["width"], probe["height"]):
            raise ValueError("Pose and source coordinate systems differ.")
        frame = min(document["frames"], key=lambda f: abs(segment["reel_start_s"] + f["timestamp_seconds"] - requested))
        time = segment["reel_start_s"] + frame["timestamp_seconds"]
        samples.append({"requested_s": requested, "pose_time_s": time, "source_frame": round(time * fps),
                        "pose_frame": frame["frame_index"], "cache": str(path), "players": frame["players"]})
    samples.sort(key=lambda s: s["source_frame"])
    if len({s["source_frame"] for s in samples}) != len(samples):
        raise ValueError("Requested samples resolve to duplicate source frames.")
    args.out.mkdir(parents=True, exist_ok=True)
    selected = "+".join(f"eq(n\\,{s['source_frame']})" for s in samples)
    with tempfile.TemporaryDirectory() as temporary:
        temporary = pathlib.Path(temporary)
        result = subprocess.run([
            str(ROOT / "tools" / "ffmpeg.exe"), "-hide_banner", "-loglevel", "info",
            "-threads", "1", "-i", str(args.source), "-an", "-vf", f"select={selected},showinfo",
            "-frames:v", str(len(samples)), "-fps_mode", "vfr", "-q:v", "2", "-threads:v", "1",
            str(temporary / "%03d.jpg"),
        ], check=True, capture_output=True, text=True)
        times = [float(t) for t in re.findall(r"\bn:\s*\d+.*?\bpts_time:([0-9.]+)", result.stderr)]
        if len(times) != len(samples):
            raise ValueError(f"Expected {len(samples)} decoded timestamps, got {len(times)}.")
        for number, (sample, decoded_time) in enumerate(zip(samples, times), 1):
            if abs(decoded_time - sample["pose_time_s"]) > 0.5 / fps + 0.001:
                raise ValueError("Decoded picture and pose timestamps do not align.")
            image = temporary / f"{number:03d}.jpg"
            original = args.out / f"source-{sample['source_frame']:06d}.jpg"
            original.write_bytes(image.read_bytes())
            sample["decoded_time_s"] = decoded_time
            sample["source_image"] = str(original)
            sample["ignored_unassigned_tracks"] = [p["track_id"] for p in sample["players"] if p.get("role") not in COLORS]
            sample["warnings"] = []
            with pymupdf.open() as pdf:
                page = pdf.new_page(width=probe["width"], height=probe["height"])
                page.insert_image(page.rect, filename=str(image))
                for role, color in COLORS.items():
                    players = [p for p in sample["players"] if p.get("role") == role]
                    if len(players) != 1:
                        sample["warnings"].append(f"{role}: expected one assigned track, found {len(players)}; not drawn")
                        continue
                    player = players[0]
                    joints = player["joints"]
                    valid = {i for i, (x, y, confidence) in enumerate(joints)
                             if confidence >= 0.65 and all(map(math.isfinite, (x, y, confidence)))
                             and 0 <= x < probe["width"] and 0 <= y < probe["height"]}
                    for a, b in EDGES:
                        if a in valid and b in valid:
                            page.draw_line(joints[a][:2], joints[b][:2], color=color, width=4, stroke_opacity=0.85)
                    for i in valid & set(range(5, 17)):
                        page.draw_circle(joints[i][:2], 5, color=(0, 0, 0), fill=color, width=1)
                    x, y = player["bbox_xyxy"][:2]
                    page.insert_text((max(5, x), max(20, y - 10)), f"{role} / track {player['track_id']}",
                                     fontsize=20, color=color)
                page.draw_rect((12, 1026, 1680, 1075), color=None, fill=(0, 0, 0), fill_opacity=0.8)
                page.insert_text((24, 1057), f"Source {decoded_time:.3f}s | frame {sample['source_frame']} | real cached pose | scores <0.65 hidden",
                                 fontsize=23, color=(1, 1, 1))
                output = args.out / f"pose-{sample['source_frame']:06d}.png"
                page.get_pixmap(matrix=pymupdf.Matrix(0.75, 0.75)).save(str(output))
                sample["overlay_image"] = str(output)
                sample["overlay_sha256"] = digest(output)
    report = {
        "source": str(args.source), "source_sha256": digest(args.source), "probe": probe,
        "cache_hashes": {str(path): digest(path) for path in caches},
        "limits": [
            "Pose scores are model scores, not calibrated probabilities.",
            "These images verify picture/pose synchronization; they do not prove racket-shuttle contact.",
            "Airborne ankle projections must not be presented as precise floor positions.",
        ],
        "samples": samples,
    }
    (args.out / "evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(samples), "warnings": [w for s in samples for w in s["warnings"]],
                      "output": str(args.out / "evidence.json")}, indent=2))


if __name__ == "__main__":
    main()
