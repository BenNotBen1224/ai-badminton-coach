"""Decode an exported video and its narration without running the coach."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


def run(ffmpeg, video, *options):
    return subprocess.run(
        [str(ffmpeg), "-hide_banner", "-nostdin", "-nostats", "-xerror",
         "-i", str(video),
         *options],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )


def last_number(text, key):
    values = re.findall(rf"^{re.escape(key)}=(\d+)$", text, re.MULTILINE)
    return int(values[-1]) if values else None


def audit(video, ffmpeg, expected_duration):
    with video.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    inspection = run(ffmpeg, video, "-map", "0:v:0", "-map", "0:a:0",
                     "-c", "copy", "-t", "0", "-f", "null", "-")
    checks = {
        "video_stream_present": bool(
            re.search(r"^\s*Stream #0:.*\bVideo:", inspection.stderr, re.MULTILINE)
        ),
        "audio_stream_present": bool(
            re.search(r"^\s*Stream #0:.*\bAudio:", inspection.stderr, re.MULTILINE)
        ),
        "stream_mapping_succeeded": inspection.returncode == 0,
    }
    result = {
        "video": str(video.resolve()),
        "bytes": video.stat().st_size,
        "sha256": digest,
        "checks": checks,
        "limits": (
            "Technical media checks only. Passing does not prove natural Chinese "
            "speech, narration-to-action alignment, correct coaching, source-frame "
            "alignment, or complete application acceptance."
        ),
    }
    if not all(checks.values()):
        result["inspection_error"] = inspection.stderr.strip()
        result["technical_media_pass"] = False
        return result

    decoded_video = run(
        ffmpeg, video, "-map", "0:v:0", "-an", "-sn", "-dn",
        "-progress", "pipe:1", "-f", "null", "-",
    )
    decoded_audio = run(
        ffmpeg, video, "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-af", "astats=metadata=0:reset=0", "-progress", "pipe:1",
        "-f", "null", "-",
    )
    frames = last_number(decoded_video.stdout, "frame")
    video_us = last_number(decoded_video.stdout, "out_time_us")
    audio_us = last_number(decoded_audio.stdout, "out_time_us")
    peaks = re.findall(r"Peak level dB:\s*(-?inf|[-\d.]+)", decoded_audio.stderr)
    peak = float(peaks[-1]) if peaks else None
    video_s = video_us / 1_000_000 if video_us is not None else None
    audio_s = audio_us / 1_000_000 if audio_us is not None else None
    checks.update({
        "video_decodes": decoded_video.returncode == 0 and bool(frames),
        "audio_decodes": decoded_audio.returncode == 0 and bool(audio_us),
        "audio_is_not_silent": peak is not None and math.isfinite(peak),
    })
    if expected_duration is not None:
        checks["video_matches_expected_duration"] = (
            video_s is not None and abs(video_s - expected_duration) <= 0.1
        )
    result.update({
        "decoded_frames": frames,
        "video_duration_s": video_s,
        "audio_duration_s": audio_s,
        "audio_video_duration_difference_s": (
            audio_s - video_s
            if audio_s is not None and video_s is not None else None
        ),
        "audio_peak_db": peak if peak is not None and math.isfinite(peak) else None,
        "technical_media_pass": all(checks.values()),
    })
    for name, process in (("video", decoded_video), ("audio", decoded_audio)):
        if process.returncode != 0:
            result[f"{name}_decode_error"] = process.stderr.strip()
    return result


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("duration must be positive and finite")
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--expected-duration", type=positive_seconds)
    args = parser.parse_args()
    for path in (args.video, args.ffmpeg):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    result = audit(args.video, args.ffmpeg, args.expected_duration)
    print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    return 0 if result["technical_media_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
