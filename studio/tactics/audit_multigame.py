"""Audit delivered editorial media without importing or modifying coach."""

import argparse
from array import array
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FFMPEG = ROOT / "tools" / "ffmpeg.exe"
FFPROBE = ROOT / "tools" / "ffprobe.exe"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(*arguments):
    return subprocess.run(arguments, check=True, capture_output=True, timeout=180)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit(folder, destination, expected_handoff=None, expected_analysis=None):
    destination.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "narrated_media", ROOT / "obj2" / "tools" / "audit_narrated_media.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if expected_handoff:
        require(digest(folder / "HANDOFF.json") == expected_handoff, "Handoff changed")
    handoff = load(folder / "HANDOFF.json")
    if "videos" in handoff:
        media_entries = handoff["videos"]
    elif "media" in handoff:
        media_entries = handoff["media"]
    else:
        raise ValueError("Handoff has no recognized media list")
    require(set(media_entries) == {"mens", "womens"}, "Expected both match examples")
    analysis_path = folder / "analysis.json"
    if expected_analysis:
        require(digest(analysis_path) == expected_analysis, "Analysis changed")
    claims = load(analysis_path)["claims"] if analysis_path.exists() else []
    drafts = {item["id"]: item for item in load(HERE / "multigame-scripts.json")["videos"]}
    results = {}
    for name, video_info in media_entries.items():
        video = ROOT / video_info["path"]
        require(digest(video) == video_info["sha256"], f"{name}: video hash mismatch")
        schedule_path = folder / name / "schedule-pre-encode.json"
        map_path = folder / name / "frame-map.csv"
        if "hashes" in handoff:
            require(digest(schedule_path) == handoff["hashes"][f"{name}_schedule"],
                    f"{name}: schedule hash mismatch")
            require(digest(map_path) == handoff["hashes"][f"{name}_frame_map"],
                    f"{name}: frame map hash mismatch")
        schedule = load(schedule_path)
        media = module.audit(video, FFMPEG, schedule["output_duration_s"])
        require(media["technical_media_pass"], f"{name}: media decode failed")
        require(media["decoded_frames"] == schedule["output_frames"],
                f"{name}: decoded frame count differs")
        streams = json.loads(run(
            str(FFPROBE), "-v", "error", "-show_streams", "-of", "json", str(video)
        ).stdout)["streams"]
        picture = next(stream for stream in streams if stream["codec_type"] == "video")
        sound = next(stream for stream in streams if stream["codec_type"] == "audio")
        require((picture["width"], picture["height"], picture["codec_name"],
                 picture["r_frame_rate"], sound["codec_name"])
                == (1920, 1080, "h264", "30/1", "aac"), f"{name}: stream shape differs")
        rows = read_csv(map_path)
        require(len(rows) == media["decoded_frames"], f"{name}: map length differs")
        source = Path(drafts[name]["source"])
        require(digest(source) == drafts[name]["source_sha256"], f"{name}: source changed")
        suffix = "ms" if name == "mens" else "ws"
        clock_path = source.parents[1] / f"clock-map-{suffix}.csv"
        clock = {int(row["norm_index"]): row for row in read_csv(clock_path)}
        previous = -1
        longest_run = current_run = 0
        for index, row in enumerate(rows):
            source_index = int(row["norm_index"])
            require(int(row["output_index"]) == index, f"{name}: missing output index")
            require(abs(float(row["output_time_s"]) - index / 30) < 0.00006,
                    f"{name}: incorrect output time")
            require(source_index >= previous, f"{name}: repeated source action")
            require(schedule["window_s"][0] - 1 / 30 <= source_index / 30
                    <= schedule["window_s"][1] + 1 / 30, f"{name}: source outside window")
            original = clock[source_index]
            require(row["source_frame_index"] == original["source_frame_index"]
                    and abs(float(row["source_pts_s"]) - float(original["source_pts_s"])) < 0.0001,
                    f"{name}: incorrect capture clock join")
            current_run = current_run + 1 if source_index == previous else 1
            longest_run = max(longest_run, current_run)
            previous = source_index

        measurements = []
        video_claims = [claim for claim in claims if claim["video"] == name]
        measured_claims = [claim for claim in video_claims if "norm_index" in claim]
        if measured_claims:
            pose_index_path = Path(drafts[name]["pose_index"])
            pose_index = load(pose_index_path)
            targets = {claim["norm_index"] for claim in measured_claims}
            pose_frames = {}
            pose_hashes = {}
            for segment in pose_index["segments"]:
                if not any(segment["reel_start_s"] <= index / 30 <= segment["reel_end_s"]
                           for index in targets):
                    continue
                pose_path = Path(segment["pose_path"])
                pose_hashes[str(pose_path)] = digest(pose_path)
                for frame in load(pose_path)["frames"]:
                    index = round(frame["reel_time_seconds"] * 30)
                    if index in targets:
                        require(index not in pose_frames, f"{name}: duplicate evidence frame")
                        pose_frames[index] = frame
            for claim in measured_claims:
                index = claim["norm_index"]
                frame = pose_frames[index]
                players = {p["role"]: p for p in frame["players"] if p.get("role")}
                comparisons = {}
                for role in ("near", "far"):
                    player = players[role]
                    key = f"{role}_hip"
                    if key in claim:
                        hips = [player["joints"][joint] for joint in (11, 12)]
                        require(min(hip[2] for hip in hips) >= 0.5,
                                f"{claim['id']}: uncertain hips")
                        actual = [(hips[0][axis] + hips[1][axis]) / 2 for axis in (0, 1)]
                        require(all(abs(a - b) <= 0.51 for a, b in zip(actual, claim[key])),
                                f"{claim['id']}: hip differs from real pose")
                        comparisons[key] = actual
                    key = f"{role}_box_x"
                    if key in claim:
                        actual = [player["bbox_xyxy"][axis] for axis in (0, 2)]
                        require(all(abs(a - b) <= 0.51 for a, b in zip(actual, claim[key])),
                                f"{claim['id']}: box differs from detector")
                        comparisons[key] = actual
                    key = f"{role}_box_h"
                    if key in claim:
                        actual = player["bbox_xyxy"][3] - player["bbox_xyxy"][1]
                        require(abs(actual - claim[key]) <= 0.51,
                                f"{claim['id']}: box height differs")
                        comparisons[key] = actual
                lo, hi = claim["on_screen_s"]
                displayed = [int(row["output_index"]) / 30 for row in rows
                             if int(row["norm_index"]) == index]
                require(displayed and any(lo - 1 / 30 <= t <= hi + 1 / 30 for t in displayed),
                        f"{claim['id']}: evidence anchor outside stated display span")
                original = clock[index]
                require(int(original["source_frame_index"]) == claim["source_frame_index"]
                        and abs(float(original["source_pts_s"]) - claim["source_pts_s"]) < 0.0001,
                        f"{claim['id']}: evidence capture clock differs")
                measurements.append({"id": claim["id"], "actual": comparisons,
                                     "source_anchor_output_s": displayed})
            all_ids = {claim["id"] for claim in measured_claims}
            for claim in video_claims:
                if claim["kind"] == "conditional":
                    require(set(claim["binds"]) <= all_ids,
                            f"{claim['id']}: unresolved evidence ID")

        decoded = run(
            str(FFMPEG), "-v", "error", "-nostdin", "-threads", "1",
            "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
            "-f", "s16le", "-"
        )
        samples = array("h", decoded.stdout)
        if sys.byteorder != "little":
            samples.byteswap()
        speech = []
        selections = []
        beat_lengths = {beat["id"]: beat["audio_s"] for beat in schedule["beats"]}
        beat_lengths["summary"] = schedule["summary_audio_s"]
        for beat, start in schedule["audio_start_s"].items():
            end = start + beat_lengths[beat]
            require(0 <= start < end <= media["video_duration_s"],
                    f"{name}/{beat}: speech outside video")
            window = samples[round(start * 16000):round(end * 16000)]
            rms = math.sqrt(sum(float(value) ** 2 for value in window) / len(window)) / 32768
            require(rms > 0.005, f"{name}/{beat}: silent speech window")
            speech.append({"beat": beat, "start_s": start, "end_s": end,
                           "rms_db": 20 * math.log10(rms)})
            candidates = [i for i, row in enumerate(rows) if row["beat_id"] == beat]
            require(candidates, f"{name}/{beat}: missing visual beat")
            require(candidates[0] / 30 - 0.04 <= start
                    and end <= (candidates[-1] + 1) / 30 + 0.04,
                    f"{name}/{beat}: speech does not fit its visual beat")
            held = [i for i in candidates if rows[i]["hold"] == "1"]
            selections.append(held[len(held) // 2] if held else candidates[len(candidates) // 2])

        expression = "+".join(f"eq(n\\,{index})" for index in selections)
        sheet = destination / f"{name}-decoded.jpg"
        run(str(FFMPEG), "-v", "error", "-y", "-nostdin", "-threads", "1",
            "-i", str(video), "-vf",
            f"select={expression},scale=900:506,tile=2x3", "-frames:v", "1",
            "-threads:v", "1", "-q:v", "2", str(sheet))
        poster = destination / f"{name}-poster.jpg"
        run(str(FFMPEG), "-v", "error", "-y", "-nostdin", "-threads", "1",
            "-i", str(video), "-vf", f"select=eq(n\\,{selections[2]})",
            "-frames:v", "1", "-threads:v", "1", "-q:v", "2", str(poster))
        results[name] = {
            "media": media, "schedule_sha256": digest(schedule_path),
            "frame_map_sha256": digest(map_path), "source_sha256": digest(source),
            "clock_sha256": digest(clock_path), "clock_rows_checked": len(rows),
            "monotonic_source_order": True, "speech": speech,
            "longest_same_source_frame_s_including_summary": longest_run / 30,
            "visual_samples": [{"output_frame": i, "normalized_frame": int(rows[i]["norm_index"]),
                                "beat": rows[i]["beat_id"]} for i in selections],
            "decoded_sheet": str(sheet), "poster": str(poster),
            "author_schedule_hashes_provided": "hashes" in handoff,
            "measurements_recomputed": measurements,
            "pose_sha256": pose_hashes if measured_claims else {},
        }
    result = {
        "handoff_sha256": digest(folder / "HANDOFF.json"), "videos": results,
        "analysis_sha256": digest(analysis_path) if analysis_path.exists() else None,
        "limits": [
            "Technical audit, not a coaching-quality or automatic-detection acceptance.",
            "Monotonic order is checked against the declared frame map; image alignment needs visual review.",
            "Non-silent decoded speech does not establish intelligibility or naturalness.",
        ],
    }
    output = destination / "audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(output), "videos": {
        name: {"frames": item["media"]["decoded_frames"], "speech_windows": len(item["speech"])}
        for name, item in results.items()
    }}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--expected-handoff")
    parser.add_argument("--expected-analysis")
    arguments = parser.parse_args()
    audit(arguments.folder, arguments.destination,
          arguments.expected_handoff, arguments.expected_analysis)
