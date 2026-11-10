"""Time model loading in a fresh process: python loadbench.py [cache|nocache]"""
import sys
import time

t0 = time.perf_counter()
import openvino as ov  # noqa: E402

from pipeline import HERE, WEIGHTS  # noqa: E402
from pose import Pose  # noqa: E402
from detect import Detector  # noqa: E402

t_import = time.perf_counter() - t0
core = ov.Core()
if sys.argv[1:] != ["nocache"]:
    core.set_property({"CACHE_DIR": str(HERE / "cache")})
t = time.perf_counter()
Pose(WEIGHTS / "rtmpose-s.onnx", core=core, streams=6, dynamic=True)
t_pose = time.perf_counter() - t
t = time.perf_counter()
Detector(WEIGHTS / "yolo26s-448x768.onnx", core=core, streams=6)
t_s = time.perf_counter() - t
t = time.perf_counter()
Detector(WEIGHTS / "yolo26n-384x640.onnx", core=core, streams=6)
t_n = time.perf_counter() - t
print(f"import {t_import:.2f}  pose {t_pose:.2f}  det-s {t_s:.2f}  det-n {t_n:.2f}")
