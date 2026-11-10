"""Prove the technical reel checker rejects missing white text and muted narration."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from render_reels import tool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    parser.add_argument("--font")
    args = parser.parse_args()
    ffmpeg, ffprobe = tool(args.ffmpeg, "ffmpeg"), tool(args.ffprobe, "ffprobe")
    args.out.mkdir(parents=True, exist_ok=False)
    cases = {
        "missing-white-caption": (
            ["-vf", "drawbox=x=0:y=1206:w=1080:h=90:color=0x07161d:t=fill:enable='between(t,5.2,6.999)'"],
            "Encoded caption is missing or changed",
        ),
        "muted-voice": (
            ["-af", "volume=0:enable='between(t,5.28,6.5575)'"],
            "Encoded voice does not match planned sound",
        ),
    }
    results = []
    for name, (filters, expected_error) in cases.items():
        target = args.out / name
        target.mkdir()
        for file in ("expected-frames.npz", "render-inputs.json", "mix.wav"):
            shutil.copyfile(args.render / file, target / file)
        command = [ffmpeg, "-v", "error", "-i", str(args.render / "reel.mp4"), *filters,
                   "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "18",
                   "-c:a", "aac", "-b:a", "160k", str(target / "reel.mp4")]
        encoded = subprocess.run(command, capture_output=True)
        if encoded.returncode:
            raise RuntimeError(encoded.stderr.decode("utf-8", "replace"))
        verify = [sys.executable, str(Path(__file__).with_name("render_reels.py")),
                  "--edition", "technical", "--verify-only", "--source", str(args.source),
                  "--inputs", str(args.inputs), "--out", str(target),
                  "--ffmpeg", ffmpeg, "--ffprobe", ffprobe]
        if args.font:
            verify.extend(["--font", args.font])
        checked = subprocess.run(verify, capture_output=True, text=True, encoding="utf-8")
        if checked.returncode == 0 or expected_error not in checked.stderr:
            raise RuntimeError(f"{name} did not fail the intended check:\n{checked.stdout}\n{checked.stderr}")
        results.append({"case": name, "rejected": True, "expected_error": expected_error,
                        "returncode": checked.returncode})
    (args.out / "negative-controls.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
