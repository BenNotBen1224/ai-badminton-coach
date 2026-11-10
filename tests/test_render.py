import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from badminton_coach.__main__ import render_videos, require_ffmpeg
from badminton_coach.render import rerender


WIDTH, HEIGHT, FRAME_COUNT = 160, 96, 6
JOINT_Y = 48

try:
    FFMPEG, FFPROBE = require_ffmpeg()
except FileNotFoundError:
    FFMPEG = FFPROBE = None


def joint_x(frame_index):
    return 20 + 20 * frame_index


def make_clip(path, frame_count):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (WIDTH, HEIGHT))
    for index in range(frame_count):
        writer.write(np.full((HEIGHT, WIDTH, 3), 40 + 20 * index, dtype=np.uint8))
    writer.release()


def make_keypoints(frame_count):
    frames = []
    for index in range(frame_count):
        x = joint_x(index)
        joints = [None] * 17
        joints[5] = [float(x), float(JOINT_Y), 0.9]
        frames.append({
            "frame_index": index,
            "timestamp_seconds": index / 30,
            "players": [{
                "track_id": 1,
                "bbox_xyxy": [x - 12.0, 20.0, x + 12.0, 80.0],
                "box_confidence": 0.9,
                "joints": joints,
            }],
        })
    return {
        "source": {
            "file": "clip.mp4", "width": WIDTH, "height": HEIGHT,
            "fps": {"numerator": 30, "denominator": 1}, "frame_count": frame_count,
        },
        "frames": frames,
    }


def read_frames(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def white_centroid(frame):
    ys, xs = np.nonzero(frame.min(axis=2) > 200)
    return (xs.mean(), ys.mean()) if len(xs) else None


@unittest.skipIf(FFMPEG is None, "FFmpeg is not installed")
class RenderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.clip = self.directory / "clip.mp4"
        self.keypoints = self.directory / "keypoints.json"

    def write_inputs(self, video_frames, json_frames):
        make_clip(self.clip, video_frames)
        self.keypoints.write_text(json.dumps(make_keypoints(json_frames)), encoding="utf-8")

    def test_skeleton_lands_on_its_joint_in_every_frame(self):
        self.write_inputs(FRAME_COUNT, FRAME_COUNT)
        render_videos(self.keypoints, self.clip, self.directory, FFMPEG, FFPROBE)

        annotated = read_frames(self.directory / "annotated.mp4")
        side_by_side = read_frames(self.directory / "side-by-side.mp4")
        self.assertEqual(FRAME_COUNT, len(annotated))
        self.assertEqual((HEIGHT, WIDTH, 3), annotated[0].shape)
        self.assertEqual((HEIGHT, WIDTH * 2, 3), side_by_side[0].shape)
        for index, frame in enumerate(annotated):
            x, y = white_centroid(frame)
            self.assertAlmostEqual(joint_x(index), x, delta=3, msg=f"frame {index}")
            self.assertAlmostEqual(JOINT_Y, y, delta=3, msg=f"frame {index}")
        for index, frame in enumerate(side_by_side):
            self.assertIsNone(white_centroid(frame[:, :WIDTH]), f"original half, frame {index}")

    def test_a_video_with_a_different_frame_count_is_refused(self):
        self.write_inputs(FRAME_COUNT - 1, FRAME_COUNT)
        with self.assertRaisesRegex(ValueError, "does not match keypoints.json"):
            render_videos(self.keypoints, self.clip, self.directory, FFMPEG, FFPROBE)
        self.assertFalse((self.directory / "annotated.mp4").exists())

    def test_incomplete_keypoints_are_refused(self):
        self.write_inputs(FRAME_COUNT, FRAME_COUNT)
        data = make_keypoints(FRAME_COUNT)
        del data["frames"][3]
        self.keypoints.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            render_videos(self.keypoints, self.clip, self.directory, FFMPEG, FFPROBE)

    def test_rerender_replaces_the_videos_and_leaves_the_json_alone(self):
        self.write_inputs(FRAME_COUNT, FRAME_COUNT)
        run_dir = self.directory / "run"
        run_dir.mkdir()
        run_keypoints = run_dir / "keypoints.json"
        run_keypoints.write_bytes(self.keypoints.read_bytes())
        for name in ("annotated.mp4", "side-by-side.mp4"):
            (run_dir / name).write_bytes(b"old thin skeleton")

        rerender(run_dir, self.clip)

        self.assertEqual(self.keypoints.read_bytes(), run_keypoints.read_bytes())
        self.assertEqual(FRAME_COUNT, len(read_frames(run_dir / "annotated.mp4")))
        self.assertEqual(FRAME_COUNT, len(read_frames(run_dir / "side-by-side.mp4")))
        self.assertEqual(
            ["annotated.mp4", "keypoints.json", "side-by-side.mp4"],
            sorted(path.name for path in run_dir.iterdir()),
        )

    def test_a_failed_rerender_keeps_the_existing_videos(self):
        self.write_inputs(FRAME_COUNT - 1, FRAME_COUNT)
        run_dir = self.directory / "run"
        run_dir.mkdir()
        (run_dir / "keypoints.json").write_bytes(self.keypoints.read_bytes())
        (run_dir / "annotated.mp4").write_bytes(b"keep me")

        with self.assertRaises(ValueError):
            rerender(run_dir, self.clip)

        self.assertEqual(b"keep me", (run_dir / "annotated.mp4").read_bytes())
        self.assertEqual(
            ["annotated.mp4", "keypoints.json"],
            sorted(path.name for path in run_dir.iterdir()),
        )


if __name__ == "__main__":
    unittest.main()
