"""Micro-benchmarks for the sub-10 s budget (clip 01, this CPU)."""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "sample-videos" / "01-singles-rally.mp4"
WEIGHTS = ROOT / "comparison" / "weights"


def bench_decode():
    out = {}
    for threads in (0, 8, 16):
        cap = cv2.VideoCapture(str(VIDEO), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, threads])
        t = time.perf_counter()
        n = 0
        while cap.read()[0]:
            n += 1
        out[f"cv2_read_threads{threads}"] = [n, round(time.perf_counter() - t, 3)]
    return out


def bench_ort(model, shape, batches=(1, 2, 4), threads=(4, 8, 16)):
    import onnxruntime as ort
    out = {}
    for th in threads:
        so = ort.SessionOptions()
        so.intra_op_num_threads = th
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        s = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        for b in batches:
            x = np.random.rand(b, *shape).astype(np.float32)
            try:
                s.run(None, {name: x})
            except Exception as e:  # static batch
                out[f"ort_t{th}_b{b}"] = f"fail {type(e).__name__}"
                continue
            t = time.perf_counter()
            for _ in range(20):
                s.run(None, {name: x})
            out[f"ort_t{th}_b{b}_ms_per_crop"] = round((time.perf_counter() - t) / 20 / b * 1000, 2)
    return out


def bench_ov(model, shape, batches=(1, 2, 4)):
    import openvino as ov
    core = ov.Core()
    out = {}
    for b in batches:
        m = core.read_model(str(model))
        m.reshape([b, *shape])
        for hint in ("LATENCY", "THROUGHPUT"):
            c = core.compile_model(m, "CPU", {"PERFORMANCE_HINT": hint})
            x = np.random.rand(b, *shape).astype(np.float32)
            q = ov.AsyncInferQueue(c)
            n = 40
            q.start_async({0: x})
            q.wait_all()
            t = time.perf_counter()
            for _ in range(n):
                q.start_async({0: x})
            q.wait_all()
            out[f"ov_{hint}_b{b}_ms_per_crop"] = round((time.perf_counter() - t) / n / b * 1000, 2)
    return out


if __name__ == "__main__":
    what = sys.argv[1:] or ["decode", "rtmpose"]
    res = {"cpu_count": os.cpu_count()}
    if "decode" in what:
        res["decode"] = bench_decode()
    if "rtmpose" in what:
        res["rtmpose_m_ort"] = bench_ort(WEIGHTS / "rtmpose.onnx", (3, 256, 192))
        res["rtmpose_m_ov"] = bench_ov(WEIGHTS / "rtmpose.onnx", (3, 256, 192))
    print(json.dumps(res, indent=1))
