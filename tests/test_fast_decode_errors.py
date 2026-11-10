import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fast"))
import pipeline  # noqa: E402


class BrokenCapture:
    """Opens and reports a 32-frame video, then fails to read any frame."""

    def __init__(self, path):
        pass

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_COUNT: 32, cv2.CAP_PROP_FRAME_WIDTH: 64,
                cv2.CAP_PROP_FRAME_HEIGHT: 48, cv2.CAP_PROP_FPS: 30.0}.get(prop, 0)

    def set(self, prop, value):
        return True

    def read(self):
        raise MemoryError("no memory for a frame")

    def release(self):
        pass


class IdleSegment:
    error = None

    def __init__(self, *args):
        pass

    def start(self):
        pass

    def join(self):
        pass


class DecodeErrors(unittest.TestCase):
    def test_run_reports_the_decoders_own_error(self):
        with patch.object(pipeline.cv2, "VideoCapture", BrokenCapture), \
                patch.object(pipeline, "Models", lambda *a, **k: None), \
                patch.object(pipeline, "Court", lambda *a: None), \
                patch.object(pipeline, "Segment", IdleSegment):
            with self.assertRaisesRegex(MemoryError, "^no memory for a frame$"):
                pipeline.run("clip.mp4", "pose.json", court_lines=[[0, 0]] * 4)

    def test_follower_reader_reports_its_error_instead_of_waiting(self):
        raised = []

        def read_one():
            try:
                pipeline.Reader("clip.mp4", 5).get()
            except Exception as e:  # noqa: BLE001 - the assertion below inspects it
                raised.append(e)

        with patch.object(pipeline.cv2, "VideoCapture", BrokenCapture):
            reader = threading.Thread(target=read_one, daemon=True)
            reader.start()
            reader.join(5)
        self.assertFalse(reader.is_alive(), "get() is still waiting for a frame")
        self.assertEqual([type(e) for e in raised], [MemoryError])


if __name__ == "__main__":
    unittest.main()
