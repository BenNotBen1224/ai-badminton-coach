"""Recheck the actual encoded short against its saved source-frame bindings."""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

from .app.backend import PROJECT, sha256_file
from .analyze import analyze
from .export import _tools
from .play_state import exchange_mask
from .schema import Role
from .short_review import CHANGED_SOURCES, FPS, MAX_FREEZE_S, SAFE, _dump, render_frame, select_points
from .coaching_points import PRIORITIES
from .coaching_corpus.corpus_learn import HEDGE_ZH
from .tracks import load_tracks


def verify(folder, source):
    folder, source = Path(folder), Path(source)
    schedule = json.loads((folder / "schedule.json").read_text("utf-8"))
    mapping = json.loads((folder / "frame-map.json").read_text("utf-8"))
    doc = json.loads((folder / "pose.json").read_text("utf-8"))
    points_doc = json.loads((folder / "points.json").read_text("utf-8"))
    video = folder / "tiktok.mp4"
    if sha256_file(source) != points_doc["source_sha256"]:
        raise AssertionError("This is not the bound source video")
    if sha256_file(PRIORITIES) != points_doc["priorities_sha256"]:
        raise AssertionError("The teaching priorities differ from the bound ranking")
    tracks = load_tracks(doc)
    report = analyze(doc)
    regenerated = select_points(doc, report)
    active = exchange_mask(tracks, report.events)
    for saved, fresh in zip(schedule["points"], regenerated):
        for key in ("source_frame", "metric", "value", "n", "confidence", "keypoints", "freeze_crop", "bbox",
                    "source_start_frame", "source_end_frame", "cycle", "cue", "priority", "balance_arm"):
            if saved.get(key) != fresh.get(key):
                raise AssertionError(f"Evidence does not reproduce: {key}")
        if saved.get("cycle"):
            cycle = saved["cycle"]
            matching = [event for event in report.events if event.data.get("knee_stride")
                        and event.player.value == saved["player"]
                        and event.data["plant_frame"] == cycle["plant_frame"]
                        and event.data["bottom_frame"] == cycle["bottom_frame"]
                        and event.data["end_frame"] == cycle["return_frame"]]
            if not matching:
                raise AssertionError("No matching measured lunge cycle supports this excerpt")
            if (cycle["start_frame"] - saved["source_start_frame"] > round(tracks.fps * 0.5)
                    or saved["source_end_frame"] > cycle["next_move_end_frame"]):
                raise AssertionError("Excerpt pads outside its measured action context")
        elif not active[saved["source_start_frame"]:saved["source_end_frame"] + 1].all():
            raise AssertionError("Excerpt enters an unsupported waiting period")
        for source_frame, state in zip(mapping["source_frame_per_output_frame"], mapping["states"]):
            if state["point"] != schedule["points"].index(saved):
                continue
            if not saved["shot"][0] <= source_frame <= saved["shot"][1]:
                raise AssertionError("Point spans a camera cut")
            if state["phase"] in ("freeze", "hook") and source_frame != saved["source_frame"]:
                raise AssertionError("Freeze pose is not bound to its exact source frame")
    if len(schedule["points"]) != 3 or len(regenerated) != 3:
        raise AssertionError("Exactly three evidence-backed points required")
    frames = mapping["source_frame_per_output_frame"]
    runs = []
    start = 0
    for index in range(1, len(frames) + 1):
        if index == len(frames) or frames[index] != frames[start]:
            runs.append((index - start) / FPS)
            start = index
    if max(runs) > MAX_FREEZE_S:
        raise AssertionError("Actual constant-source hold exceeds freeze cap")
    measured_holds = sum(duration for duration in runs if duration >= 0.5)
    if measured_holds / schedule["planned_duration_s"] > 0.35:
        raise AssertionError("Actual held-source time exceeds paused-time cap")
    font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    fonts = tuple(ImageFont.truetype(str(font_dir / "msyh.ttc"), size) for size in (58, 50, 44, 30))
    encoded = cv2.VideoCapture(str(video))
    original = cv2.VideoCapture(str(source))
    checks = []
    samples = [0]
    for point in schedule["points"]:
        samples += [round((point["freeze_start_s"] - 0.2) * FPS),
                    round((point["freeze_start_s"] + 0.8) * FPS),
                    round((point["freeze_end_s"] + 0.2) * FPS)]
    for output_frame in samples:
        source_frame = frames[output_frame]
        state = mapping["states"][output_frame]
        point = schedule["points"][state["point"]]
        original.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        encoded.set(cv2.CAP_PROP_POS_FRAMES, output_frame)
        ok_source, raw = original.read()
        ok_output, actual = encoded.read()
        if not ok_source or not ok_output:
            raise AssertionError(f"Decode failure at output frame {output_frame}")
        expected, boxes = render_frame(raw, tracks, point, {**state, "source_frame": source_frame},
                                       output_frame / FPS, schedule["captions"], fonts,
                                       schedule.get("credit", {}).get("endcard",
                                                                      "Badminton Sphere · CC BY 3.0\nWikimedia Commons · 剪辑与标注"))
        error = float(np.mean(np.abs(expected.astype(np.float32) - actual.astype(np.float32))))
        if error > 6:
            raise AssertionError(f"Source/pose/caption alignment mismatch at {output_frame}: MAE {error}")
        if not all(SAFE[0] <= b[0] <= b[2] <= SAFE[2] and SAFE[1] <= b[1] <= b[3] <= SAFE[3] for b in boxes):
            raise AssertionError("Burned text crosses the safe area")
        checks.append({"output_frame": output_frame, "source_frame": source_frame,
                       "phase": state["phase"], "pixel_mae": error})
        if state["phase"] == "freeze":
            cv2.imwrite(str(folder / f'encoded-point-{state["point"] + 1}.jpg'), actual)
    original.release()
    encoded.release()
    cover = cv2.imread(str(folder / "cover.jpg"))
    if cover is None or cover.shape[:2] != (1920, 1080):
        raise AssertionError("Cover is missing or not 1080x1920")
    ffmpeg, ffprobe = _tools()
    streams = json.loads(subprocess.check_output([
        str(ffprobe), "-v", "error", "-show_streams", "-of", "json", str(video)]))["streams"]
    video_stream = next(stream for stream in streams if stream["codec_type"] == "video")
    audio_stream = next(stream for stream in streams if stream["codec_type"] == "audio")
    if (video_stream["codec_name"] != "h264" or audio_stream["codec_name"] != "aac"
            or video_stream["r_frame_rate"] != "30/1"
            or [video_stream["width"], video_stream["height"]] != [1080, 1920]):
        raise AssertionError("Encoded media does not match the portrait H.264/AAC/30fps contract")
    atoms = []
    with video.open("rb") as handle:
        while header := handle.read(8):
            if len(header) != 8:
                raise AssertionError("Truncated MP4 atom header")
            size, kind = struct.unpack(">I4s", header)
            header_size = 8
            if size == 1:
                size = struct.unpack(">Q", handle.read(8))[0]
                header_size = 16
            atoms.append(kind.decode("ascii"))
            if size == 0:
                break
            if size < header_size:
                raise AssertionError("Invalid MP4 atom size")
            handle.seek(size - header_size, 1)
    if "moov" not in atoms or "mdat" not in atoms or atoms.index("moov") > atoms.index("mdat"):
        raise AssertionError("MP4 is not faststart")
    rate = 16000
    raw_audio = subprocess.check_output([
        str(ffmpeg), "-v", "error", "-i", str(video), "-map", "0:a:0", "-ac", "1",
        "-ar", str(rate), "-f", "f32le", "pipe:1"])
    audio = np.frombuffer(raw_audio, dtype="<f4")
    speech_checks = []
    for caption in schedule["captions"]:
        if HEDGE_ZH.search(caption["text"]):
            raise AssertionError("Narration contains a hedge instead of an actionable coaching cue")
        samples = audio[round(caption["start_s"] * rate):round(caption["end_s"] * rate)]
        if not samples.size or np.max(np.abs(samples)) < 0.01:
            raise AssertionError("Caption has no corresponding audible speech")
        point = schedule["points"][caption["point"]]
        if not (point["output_start_s"] <= caption["speech_start_s"] <= caption["start_s"]
                < caption["end_s"] <= caption["speech_end_s"] <= point["output_end_s"]):
            raise AssertionError("Caption spills beyond measured speech or its evidence window")
        speech_checks.append({"text": caption["text"], "start_s": caption["start_s"],
                              "end_s": caption["end_s"], "decoded_rms": float(np.sqrt(np.mean(samples ** 2)))})
    result = {
        "video": str(video.resolve()), "sha256": sha256_file(video),
        "source_sha256": sha256_file(source), "evidence_reproduced": True,
        "encoded_alignment": checks, "speech_windows": speech_checks,
        "max_actual_source_hold_s": max(runs), "actual_held_source_time_s": measured_holds,
        "actual_held_source_share": measured_holds / schedule["planned_duration_s"],
        "hold_definition": "Consecutive equal source-frame runs lasting at least 0.5s, including slow-motion boundary repeats.",
        "cover_shape": [1080, 1920], "no_cut_crossing": True,
        "video_codec": "h264", "audio_codec": "aac", "fps": 30, "faststart": True,
        "limits": ["Speech content/naturalness still needs listening; RMS is not semantic verification.",
                   "Model confidence is not advice correctness. Single-camera geometry is a projection."],
    }
    _dump(folder / "alignment-verification.json", result)
    verification = json.loads((folder / "verification.json").read_text("utf-8"))
    verification["encoded_alignment_verified"] = True
    verification["faststart"] = True
    verification["video_codec"] = "h264"
    verification["audio_codec"] = "aac"
    verification["actual_held_source_time_s"] = measured_holds
    verification["actual_moving_source_time_s"] = schedule["planned_duration_s"] - measured_holds
    verification["phase_totals"] = {"moving_s": schedule["moving_s"], "paused_s": schedule["paused_s"]}
    for key in ("moving_s", "paused_s", "paused_share"):
        verification.pop(key, None)
    verification["measured_source_holds"] = {
        "held_s": measured_holds, "moving_s": schedule["planned_duration_s"] - measured_holds,
        "share": measured_holds / schedule["planned_duration_s"],
        "definition": result["hold_definition"],
    }
    _dump(folder / "verification.json", verification)
    (folder / "narration.txt").write_text("\n\n".join(
        f'{row["start_s"]:.3f}–{row["end_s"]:.3f}s · 源帧 {schedule["points"][row["point"]]["source_frame"]}\n{row["text"]}'
        for row in schedule["captions"]), "utf-8")
    handoff_path = folder / "handoff.json"
    handoff = json.loads(handoff_path.read_text("utf-8")) if handoff_path.is_file() else {}
    tests_path, audit_path = folder / "tests.json", folder / "independent-media-audit.log"
    if tests_path.is_file():
        tests = json.loads(tests_path.read_text("utf-8"))
        if tests["failures"] or tests["errors"]:
            raise AssertionError("Recorded applicable test suite did not pass")
        handoff["tests"] = tests
        handoff["test_pass_gate"] = {
            "scope": tests.get("scope", "Recorded coach test modules"),
            "passed": True,
            "chat_and_plan_included": all(name in tests["modules"] for name in
                                          ("coach.tests.test_chat", "coach.tests.test_plan")),
        }
    full_path = folder / "full-suite-with-legacy-chat.json"
    if full_path.is_file():
        full = json.loads(full_path.read_text("utf-8"))
        handoff["broader_suite"] = {
            "run": full["run"], "passed": full["passed"],
            "skipped": len(full["skipped"]), "failed": len(full["failures"]),
            "errors": len(full["errors"]),
            "all_passed": not full["failures"] and not full["errors"],
            "failure_details": str(full_path.resolve()),
            "classification": str((folder / "chat-test-classification.json").resolve()),
        }
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text("utf-8-sig"))
        if not audit["technical_media_pass"] or audit["sha256"] != result["sha256"]:
            raise AssertionError("Independent media audit is failed or refers to another video")
        handoff["independent_media_audit"] = audit
    handoff.update({
        "status": "verified-render", "output": str(video.resolve()),
        "planned_duration_s": schedule["planned_duration_s"],
        "phase_totals": verification["phase_totals"],
        "measured_source_holds": verification["measured_source_holds"],
        "source_sha256": result["source_sha256"],
        "points": schedule["points"], "narration": schedule["captions"],
        "changed_source_files": {
            str(name): {"path": str((PROJECT / name).resolve()), "sha256": sha256_file(PROJECT / name)}
            for name in CHANGED_SOURCES},
        "artifacts": {p.name: {"path": str(p.resolve()), "sha256": sha256_file(p)}
                      for p in folder.iterdir() if p.is_file() and p.name != "handoff.json"},
    })
    for key in ("moving_s", "paused_s"):
        handoff.pop(key, None)
    _dump(handoff_path, handoff)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.folder, args.source)))


if __name__ == "__main__":
    main()
