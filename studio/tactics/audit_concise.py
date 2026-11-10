"""Check the concise export without running the coach or the renderer."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import wave

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FOLDER = ROOT / "badminton-coach" / "outputs" / "tactics-original-editorial-20261105"
CONCISE = FOLDER / "concise-30-35s"
DEST = HERE / "concise-audit"
FFMPEG = ROOT / "tools" / "ffmpeg.exe"
FFPROBE = ROOT / "tools" / "ffprobe.exe"
EXPECTED = "188e45f391973f257c4916106fb77351ca04d1361b2ea5340167f860cd5b35de"
ORIGINAL = "e933259865dec966f3001ac7fdda65107826605b97b71a6e1f78863a871a1ad3"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, timeout=240).stdout


def pictures(path, indices, crop=False):
    indices = sorted(set(indices))
    select = "+".join(f"eq(n\\,{index})" for index in indices)
    transform = "crop=1920:1080:0:120," if crop else ""
    raw = run(
        str(FFMPEG), "-v", "error", "-nostdin", "-threads", "1", "-i", str(path),
        "-an", "-vf", f"select={select},{transform}scale=480:270:flags=area",
        "-frames:v", str(len(indices)), "-fps_mode", "passthrough",
        "-threads:v", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "-",
    )
    require(len(raw) == len(indices) * 480 * 270 * 3, "Selected picture count differs")
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 270, 480, 3)
    return dict(zip(indices, frames))


def trimmed_error(left, right):
    delta = np.abs(left.astype(np.float32) - right).mean(axis=2).ravel()
    count = int(len(delta) * 0.9)
    return float(np.partition(delta, count)[:count].mean())


def audit():
    DEST.mkdir(parents=True, exist_ok=True)
    video = CONCISE / "editorial.mp4"
    require(digest(video) == EXPECTED, "Concise video changed")
    require(digest(FOLDER / "editorial.mp4") == ORIGINAL, "Original video changed")
    handoff = load(CONCISE / "handoff.json")
    for item in handoff["files"]:
        path = Path(item["path"]).resolve()
        require(path.is_relative_to(CONCISE.resolve()), "Manifest path outside concise folder")
        require(path.stat().st_size == item["bytes"], f"Size differs: {path.name}")
        require(digest(path) == item["sha256"], f"Hash differs: {path.name}")
    schedule = load(CONCISE / "planned-schedule.json")
    mapping = load(CONCISE / "frame-map.json")
    analysis = load(CONCISE / "analysis.json")
    provenance = load(CONCISE / "run.json")
    require(digest(CONCISE / "planned-schedule.json")
            == provenance["planned_schedule_sha256_before_encode"], "Plan changed")
    require(schedule["frame_count"] == 912 and schedule["fps"] == 30, "Wrong frame target")
    require(schedule["duration_s"] == 30.4, "Wrong duration target")
    require(schedule["picture_box"] == [0, 120, 1920, 1200], "Unexpected picture rectangle")
    require(len(mapping["frames"]) == len(schedule["frames"]) == 912, "Map length differs")
    previous = -1
    for index, (plan, actual) in enumerate(zip(schedule["frames"], mapping["frames"])):
        require(actual["output_frame"] == index, "Missing output frame")
        require(all(actual[key] == value for key, value in plan.items()), "Map differs from plan")
        require(actual["source_position"] > previous, "Hold or backwards source movement")
        require(actual["kind"] == "play", "Unexpected held picture")
        previous = actual["source_position"]
    require([beat["range"] for beat in schedule["beats"]]
            == [[57.8, 64.8], [126.6, 134.0], [272.8, 280.0], [280.0, 285.0]],
            "Source passages changed")
    require(all(beat["hold_frames"] == 0 for beat in schedule["beats"]), "Unexpected hold")
    evidence = {row["id"]: row for row in analysis["evidence"] + analysis["context_evidence"]}
    refs = [ref for beat in analysis["beats"] for ref in beat["evidence_ids"]]
    require(len(refs) == 15 and all(ref in evidence for ref in refs), "Unresolved evidence")
    for beat in analysis["beats"]:
        require(all(ref in evidence for ref in beat["context_evidence_ids"]), "Missing context")
    source = Path(analysis["source"]["path"])
    require(digest(source) == analysis["source"]["sha256"], "Source changed")

    spec = importlib.util.spec_from_file_location(
        "media_audit", ROOT / "obj2" / "tools" / "audit_narrated_media.py"
    )
    media_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(media_module)
    media = media_module.audit(video, FFMPEG, 30.4)
    require(media["technical_media_pass"] and media["decoded_frames"] == 912, "Decode failed")
    streams = json.loads(run(str(FFPROBE), "-v", "error", "-show_streams",
                             "-of", "json", str(video)))["streams"]
    image_stream = next(row for row in streams if row["codec_type"] == "video")
    require((image_stream["width"], image_stream["height"], image_stream["r_frame_rate"])
            == (1920, 1440, "30/1"), "Unexpected video format")

    voices = load(CONCISE / "voice.json")
    decoded = np.frombuffer(run(
        str(FFMPEG), "-v", "error", "-nostdin", "-threads", "1", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "24000", "-f", "f32le", "-"
    ), dtype="<f4")
    speech = []
    samples = set()
    for beat in schedule["beats"]:
        voice = voices[beat["id"]]
        require(digest(Path(voice["path"])) == beat["voice_sha256"], "Voice changed")
        with wave.open(voice["path"], "rb") as handle:
            require((handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
                    == (1, 2, 24000), "Unexpected voice format")
            reference = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        start, end = beat["speech_span_s"]
        lo, hi = beat["display_span_s"]
        require(lo <= start < end <= hi, "Narration outside displayed beat")
        require(abs((end - start) - len(reference) / 24000) < 1e-6, "Voice duration differs")
        offset = round(start * 24000)
        rendered = decoded[offset:offset + len(reference)]
        require(len(rendered) == len(reference), "Truncated narration")
        correlation = float(np.corrcoef(reference, rendered)[0, 1])
        require(correlation > 0.99, f"Wrong speech or timing: {beat['id']}")
        speech.append({"beat": beat["id"], "span_s": [start, end], "correlation": correlation})
        first, last = beat["first_frame"], beat["end_frame_exclusive"] - 1
        samples.update([first, last])
        samples.update(round(first + fraction * (last - first)) for fraction in (0.2, 0.5, 0.8))

    source_indices = set()
    for index in samples:
        row = mapping["frames"][index]
        source_indices.update(row["picture_frames"])
        source_indices.update(value + 60 for value in row["picture_frames"])
    originals = pictures(source, source_indices)
    exports = pictures(video, samples, crop=True)
    errors = []
    tiles = []
    for index in sorted(samples):
        row = mapping["frames"][index]
        left, right = row["picture_frames"]
        weight = row["picture_weight"]
        expected = originals[left].astype(float) * (1 - weight) + originals[right] * weight
        wrong = originals[left + 60].astype(float) * (1 - weight) + originals[right + 60] * weight
        correct_error = trimmed_error(exports[index], expected)
        wrong_error = trimmed_error(exports[index], wrong)
        require(correct_error < 8, f"Wrong source pixels at export frame {index}")
        changed = np.abs(expected - wrong).mean(axis=2) > 12
        changed_pixels = int(changed.sum())
        require(changed_pixels > 100, f"Insufficient visible motion for control: {index}")
        changed_correct = float(np.abs(exports[index] - expected).mean(axis=2)[changed].mean())
        changed_wrong = float(np.abs(exports[index] - wrong).mean(axis=2)[changed].mean())
        errors.append({"output_frame": index, "trimmed_mae": correct_error,
                       "wrong_source_plus_1s_mae": wrong_error,
                       "changed_pixels": changed_pixels,
                       "changed_region_correct_mae": changed_correct,
                       "changed_region_wrong_mae": changed_wrong})
        tile = exports[index].copy()
        cv2.putText(tile, f"out {index / 30:.2f}s / source {row['source_position'] / 60:.2f}s",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    (DEST / "pixel-samples.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    # Static court pixels can hide a displaced player in a whole-picture mean.
    require(all(row["changed_region_correct_mae"] * 2 < row["changed_region_wrong_mae"]
                for row in errors), "Wrong-source control failed in moving regions")
    sheet = np.vstack([np.hstack(tiles[index:index + 4]) for index in range(0, len(tiles), 4)])
    require(cv2.imwrite(str(DEST / "source-bound-contact.jpg"), sheet), "Contact sheet not saved")
    run(str(FFMPEG), "-v", "error", "-nostdin", "-y", "-threads", "1",
        "-ss", "2", "-i", str(video), "-frames:v", "1", "-threads:v", "1",
        "-q:v", "2", str(DEST / "poster.jpg"))
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "audit_script_sha256": digest(Path(__file__)),
        "media": media,
        "original_unchanged_sha256": ORIGINAL,
        "manifest_files_checked": len(handoff["files"]),
        "schedule_sha256": digest(CONCISE / "planned-schedule.json"),
        "source_sha256": analysis["source"]["sha256"],
        "mapped_frames_checked": 912,
        "evidence_references_resolved": len(refs),
        "speech": speech,
        "source_pixels": errors,
        "pass": True,
        "limits": "Technical/source/voice binding only; not independent coaching acceptance, "
                  "human listening, exact contact detection or the strict WHY/NEXT display rubric.",
    }


if __name__ == "__main__":
    lock = ROOT / "obj2" / "work" / "cpu.lock"
    owner = f"parent-concise-audit {os.getpid()}"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(owner)
    try:
        result = audit()
        (DEST / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        if lock.read_text(encoding="utf-8") == owner:
            lock.unlink()
