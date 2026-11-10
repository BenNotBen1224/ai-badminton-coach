from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from badminton_coach.__main__ import render_videos, require_ffmpeg


OUTPUT_NAMES = ("annotated.mp4", "side-by-side.mp4")


def rerender(run_dir: Path, video_path: Path) -> None:
    keypoints_file = run_dir / "keypoints.json"
    if not keypoints_file.is_file():
        raise FileNotFoundError(f"No keypoints.json found in {run_dir}.")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    ffmpeg, ffprobe = require_ffmpeg()
    staging_dir = Path(tempfile.mkdtemp(prefix=".render-", dir=run_dir))
    try:
        render_videos(keypoints_file, video_path.resolve(), staging_dir, ffmpeg, ffprobe)
        try:
            for name in OUTPUT_NAMES:
                os.replace(staging_dir / name, run_dir / name)
        except PermissionError as error:
            raise RuntimeError(
                "Could not replace the old videos. "
                "Close any player that has them open, then run this again."
            ) from error
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    print(f"Re-rendered {' and '.join(OUTPUT_NAMES)} in {run_dir}. keypoints.json was not changed.")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="badminton-coach-render",
        description=(
            "Redraw annotated.mp4 and side-by-side.mp4 from a finished run's keypoints.json. "
            "Detection is not run again."
        ),
    )
    parser.add_argument("run_folder", type=Path, help="finished run folder containing keypoints.json")
    parser.add_argument("video", type=Path, help="the original video that run was made from")
    args = parser.parse_args()
    try:
        rerender(args.run_folder.resolve(), args.video)
    except KeyboardInterrupt:
        print("\nRender cancelled. The existing videos were left as they were.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        details = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) else str(error)
        print(f"Render failed: {details}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
