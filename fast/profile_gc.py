"""Untimed diagnostic run: collection pauses only, not a performance result."""
import gc
import json
import os
import sys
import time
from pathlib import Path

import pipeline


if __name__ == "__main__":
    started = {}
    pauses = []

    def collection(phase, info):
        gen = info["generation"]
        if phase == "start":
            started[gen] = time.perf_counter_ns()
        else:
            pauses.append({"generation": gen, "ns": time.perf_counter_ns() - started.pop(gen),
                           "collected": info["collected"], "uncollectable": info["uncollectable"]})

    gc.callbacks.append(collection)
    stats = pipeline.run(sys.argv[1], sys.argv[2])
    gc.callbacks.remove(collection)
    Path(sys.argv[3]).write_text(json.dumps({
        "note": "Instrumented run; never include as a benchmark sample.",
        "gc_total_s": sum(p["ns"] for p in pauses) / 1e9,
        "gc_longest_s": max((p["ns"] for p in pauses), default=0) / 1e9,
        "collections": pauses, "internal": stats,
    }, indent=1), "utf-8")
    os._exit(0)
