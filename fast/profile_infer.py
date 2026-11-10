"""Untimed per-request tensor/lock/wait census; never acceptance evidence."""
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.modules.setdefault("openvino.tools.ovc", None)
import pipeline
import ovcache

records = []
lock = threading.Lock()


def measured(request, x):
    a = time.perf_counter()
    with ovcache._tensors:
        b = time.perf_counter()
        t = request.get_input_tensor(0)
        if tuple(t.shape) != x.shape:
            t.shape = x.shape
        t.data[:] = x
        request.start_async()
    c = time.perf_counter()
    request.wait()
    d = time.perf_counter()
    with ovcache._tensors:
        e = time.perf_counter()
        out = [request.get_output_tensor(i).data.copy() for i in range(len(request.model_outputs))]
    f = time.perf_counter()
    with lock:
        records.append([b-a, c-b, d-c, e-d, f-e])
    return out


if __name__ == "__main__":
    pipeline.infer = measured
    internal = pipeline.run(sys.argv[1], sys.argv[2] + ".poses.json")
    total = [sum(r[i] for r in records) for i in range(5)]
    Path(sys.argv[2]).write_text(json.dumps({"note": __doc__, "requests": len(records),
        "summed_request_seconds": dict(zip(("input_lock", "set_start", "wait", "output_lock", "output_copy"), total)),
        "internal_not_timing": internal}, indent=1))
    print(total, flush=True)
    os._exit(0)
