"""Gate publication on the reviewed coaching script and the actual encoded media."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import wave

import cv2
import numpy as np

from audit_concise import ROOT, HERE, FFMPEG, FFPROBE, load, digest, require, run, pictures, trimmed_error


SPEC = HERE / "coaching-content-spec.json"


def check_script(script_path, spec_path=SPEC):
    script = load(script_path)
    spec = load(spec_path)
    actual = script["beats"]
    expected = spec["beats"]
    require(len(actual) == len(expected), "The reviewed revision has three coaching beats, no filler summary")
    for beat, approved in zip(actual, expected):
        require(beat["id"] == approved["id"], "Coaching beat order changed")
        require(beat["range"] == approved["range"], f"Wrong source passage: {beat['id']}")
        require(beat["text"] == approved["text"], f"Unreviewed narration: {beat['id']}")
        for field in ("intent_quote", "technique_quote"):
            require(approved[field] in beat["text"], f"Missing reviewed {field}: {beat['id']}")
    return {
        "script_sha256": digest(script_path),
        "reviewed_spec_sha256": digest(spec_path),
        "beats": [beat["id"] for beat in actual],
        "meaning": "Exact text matches parent-reviewed intent and technique; this is not automated semantic scoring.",
    }


def audit(folder, expected_hash, destination, spec_path=SPEC):
    content = check_script(folder / "script.json", spec_path)
    destination.mkdir(parents=True, exist_ok=True)
    video = folder / "editorial.mp4"
    require(digest(video) == expected_hash, "Delivered video hash differs")
    schedule = load(folder / "planned-schedule.json")
    mapping = load(folder / "frame-map.json")
    analysis = load(folder / "analysis.json")
    voices = load(folder / "voice.json")
    handoff = load(folder / "handoff.json")
    provenance = load(folder / "run.json")
    for item in handoff["files"]:
        path = Path(item["path"]).resolve()
        require(path.is_relative_to(folder.resolve()), "Manifest escapes output folder")
        require(digest(path) == item["sha256"], f"Changed artifact: {path.name}")
    duration = schedule["duration_s"]
    count = schedule["frame_count"]
    require(30 <= duration <= 35, "The coaching revision is outside the 30-35 second target")
    require(schedule["fps"] == 30 and abs(count / 30 - duration) < 1e-8, "Invalid frame target")
    require(digest(folder / "planned-schedule.json")
            == provenance["planned_schedule_sha256_before_encode"], "Pre-encode plan changed")
    require(len(mapping["frames"]) == len(schedule["frames"]) == count, "Frame-map length differs")
    prior = -1
    for index, (plan, row) in enumerate(zip(schedule["frames"], mapping["frames"])):
        require(row["output_frame"] == index, "Missing frame")
        require(all(row[key] == value for key, value in plan.items()), "Plan/map mismatch")
        require(row["source_position"] > prior and row["kind"] == "play", "Replay or freeze")
        prior = row["source_position"]
    require([beat["text"] for beat in schedule["beats"]]
            == [beat["text"] for beat in load(spec_path)["beats"]], "Encoded schedule changed the narration")
    evidence = {row["id"]: row for row in analysis["evidence"] + analysis["context_evidence"]}
    refs = [ref for beat in load(spec_path)["beats"] for ref in beat["evidence_ids"]]
    require(all(ref in evidence for ref in refs), "Unresolved coaching evidence")
    source = Path(analysis["source"]["path"])
    require(digest(source) == analysis["source"]["sha256"], "Source footage changed")

    module_spec = importlib.util.spec_from_file_location(
        "narrated_media", ROOT / "obj2" / "tools" / "audit_narrated_media.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    media = module.audit(video, FFMPEG, duration)
    require(media["technical_media_pass"] and media["decoded_frames"] == count, "Media decode failed")
    streams = json.loads(run(str(FFPROBE), "-v", "error", "-show_streams",
                             "-of", "json", str(video)))["streams"]
    stream = next(row for row in streams if row["codec_type"] == "video")
    require((stream["width"], stream["height"], stream["r_frame_rate"])
            == (1920, 1440, "30/1"), "Unexpected picture format")
    audio = np.frombuffer(run(
        str(FFMPEG), "-v", "error", "-nostdin", "-threads", "1", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "24000", "-f", "f32le", "-"
    ), dtype="<f4")
    speech = []
    selected = set()
    for beat in schedule["beats"]:
        voice = voices[beat["id"]]
        require(voice["text"] == beat["text"], "TTS input differs from reviewed narration")
        require(digest(Path(voice["path"])) == beat["voice_sha256"], "Voice WAV changed")
        with wave.open(voice["path"], "rb") as handle:
            require((handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
                    == (1, 2, 24000), "Unexpected voice format")
            reference = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        start, end = beat["speech_span_s"]
        lo, hi = beat["display_span_s"]
        require(lo <= start < end <= hi, "Voice falls outside its scene")
        require(abs(end - start - len(reference) / 24000) < 1e-6, "Voice duration differs")
        offset = round(start * 24000)
        rendered = audio[offset:offset + len(reference)]
        require(len(rendered) == len(reference), "Voice truncated")
        correlation = float(np.corrcoef(reference, rendered)[0, 1])
        require(correlation > 0.99, f"Wrong encoded voice or timing: {beat['id']}")
        speech.append({"beat": beat["id"], "correlation": correlation, "span_s": [start, end]})
        first, last = beat["first_frame"], beat["end_frame_exclusive"] - 1
        selected.update(round(first + fraction * (last - first)) for fraction in (0, 0.2, 0.5, 0.8, 1))

    needed = set()
    for index in selected:
        pair = mapping["frames"][index]["picture_frames"]
        needed.update(pair)
        needed.update(value + 60 for value in pair)
    originals = pictures(source, needed)
    exports = pictures(video, selected, crop=True)
    pixel_checks, tiles = [], []
    for index in sorted(selected):
        row = mapping["frames"][index]
        left, right = row["picture_frames"]
        weight = row["picture_weight"]
        expected = originals[left].astype(float) * (1 - weight) + originals[right] * weight
        wrong = originals[left + 60].astype(float) * (1 - weight) + originals[right + 60] * weight
        error = trimmed_error(exports[index], expected)
        changed = np.abs(expected - wrong).mean(axis=2) > 12
        require(int(changed.sum()) > 100, "Insufficient motion to test source identity")
        correct_changed = float(np.abs(exports[index] - expected).mean(axis=2)[changed].mean())
        wrong_changed = float(np.abs(exports[index] - wrong).mean(axis=2)[changed].mean())
        require(error < 8 and correct_changed * 2 < wrong_changed,
                f"Source picture identity failed at frame {index}")
        pixel_checks.append({"frame": index, "trimmed_mae": error,
                             "moving_region_correct_mae": correct_changed,
                             "moving_region_wrong_mae": wrong_changed})
        tile = exports[index].copy()
        cv2.putText(tile, f"{index / 30:.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    require(cv2.imwrite(str(destination / "contact.jpg"),
                        np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])),
            "Contact sheet not written")
    for beat in schedule["beats"]:
        midpoint = sum(beat["display_span_s"]) / 2
        run(str(FFMPEG), "-v", "error", "-nostdin", "-y", "-threads", "1",
            "-ss", str(midpoint), "-i", str(video), "-frames:v", "1", "-threads:v", "1",
            "-q:v", "2", str(destination / f"{beat['id']}.jpg"))
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "content_review": content,
        "media": media,
        "manifest_files_checked": len(handoff["files"]),
        "mapped_frames_checked": count,
        "evidence_refs_checked": len(refs),
        "speech": speech,
        "pixel_checks": pixel_checks,
        "pass": True,
        "limits": "Parent-reviewed coaching advice, not known player intent or independently measured technique. "
                  "No claim of expert/user acceptance, human voice review, or automatic engine acceptance.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-script", type=Path)
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--expected-hash")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.check_script:
        print(json.dumps(check_script(args.check_script, args.spec), ensure_ascii=False, indent=2))
    else:
        if not (args.folder and args.expected_hash and args.output):
            parser.error("--folder, --expected-hash and --output are required for media audit")
        lock = ROOT / "obj2" / "work" / "cpu.lock"
        owner = f"parent-coaching-content-audit {os.getpid()}"
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(owner)
        try:
            result = audit(args.folder, args.expected_hash, args.output, args.spec)
            (args.output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2))
        finally:
            if lock.read_text(encoding="utf-8") == owner:
                lock.unlink()
