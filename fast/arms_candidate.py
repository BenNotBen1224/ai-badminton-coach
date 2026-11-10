"""Screen the arm-focused student using a separate prepared cache, preserving production."""
import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

sys.modules.setdefault("openvino.tools.ovc", None)
import openvino as ov
import pipeline
import pose
from ovcache import BUNDLE_NAME, BundleWriter
from tail_candidate import route_tail

MODEL = pipeline.WEIGHTS / "rtmpose-t-ft4-arms.onnx"
ASSETS = pipeline.HERE / "runs" / "step4-arms-assets"


def prepare():
    writer = BundleWriter()
    existing = pipeline.HERE / "cache" / BUNDLE_NAME
    with zipfile.ZipFile(existing) as z:
        writer.entries = json.loads(z.read("manifest.json"))
        writer.data = {name: z.read(name) for name in writer.entries}
    original = pose.compile_model
    pose.compile_model = writer.compile
    core = ov.Core()
    for hint, streams in (("THROUGHPUT", 8), ("LATENCY", None)):
        pose.Pose(MODEL, core=core, hint=hint, streams=streams, dynamic=True)
    pose.compile_model = original
    writer.save(ASSETS / "cache" / BUNDLE_NAME)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--tail", type=float, default=0)
    a = ap.parse_args()
    if a.prepare:
        prepare()
    else:
        if not a.video or not a.out:
            ap.error("video and out are required without --prepare")
        pipeline.HERE = ASSETS
        if a.tail:
            pipeline.Segment.route_poses = lambda self, *args: route_tail(self, *args, threshold=a.tail)
        print(json.dumps(pipeline.run(a.video, a.out, pose_small=MODEL)), flush=True)
    os._exit(0)
