"""One-time graph conversion/compilation only; no video or saved pose JSON is read."""
import json
import os
import time
import detect
import pose
from ovcache import BUNDLE_NAME, BundleWriter

from pipeline import Models, WEIGHTS
from pose import DEFAULT_POSE, DEFAULT_SMALL_POSE


if __name__ == "__main__":
    start = time.perf_counter()
    writer = BundleWriter()
    pose.compile_model = detect.compile_model = writer.compile
    models = Models(DEFAULT_POSE, WEIGHTS / "yolo26s-448x768.onnx",
                    WEIGHTS / "yolo26n-384x640.onnx", streams=8, pose_small_onnx=DEFAULT_SMALL_POSE)
    models.one
    writer.save(WEIGHTS.parent / "cache" / BUNDLE_NAME)
    print(json.dumps({"prepared_primary": str(DEFAULT_POSE), "prepared_small": str(DEFAULT_SMALL_POSE),
                      "streams": 8, "followers_prepared": True,
                      "bundle": BUNDLE_NAME, "compiled_graphs": len(writer.entries),
                      "setup_seconds": time.perf_counter() - start}), flush=True)
    os._exit(0)
