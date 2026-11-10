"""Untimed startup diagnostic: per-model compile/load spans and slow-thread dumps."""
import argparse
import contextlib
import faulthandler
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path


def child(style, output, video=None):
    sys.modules.setdefault("openvino.tools.ovc", None)
    import pipeline
    import pose
    import detect
    import ovcache

    spans = []
    gate = threading.Lock()
    original = ovcache.compile_model
    original_core = pipeline.ov.Core
    core_spans = []
    bundle_lock = threading.Lock()
    bundle_data = []

    def measured_core():
        start = time.perf_counter()
        core = original_core()
        core_spans.append(time.perf_counter() - start)
        return core

    def measured(core, onnx, key, build, config):
        queued = time.perf_counter()
        with gate if style == "serial" else contextlib.nullcontext():
            started = time.perf_counter()
            export_dir = Path(__file__).parent / "runs" / "step2-export-models"
            name = hashlib.sha256(repr((str(onnx), key, config)).encode()).hexdigest()[:16]
            exported = export_dir / f"{Path(onnx).stem}-{name}.blob"
            if style == "ir":
                st = Path(onnx).stat()
                name = hashlib.sha1(repr((key, st.st_size, st.st_mtime_ns)).encode()).hexdigest()[:12]
                xml = Path(core.get_property("CACHE_DIR")) / f"{Path(onnx).stem}-{name}.xml"
                result = core.compile_model(str(xml), "CPU", {**config, "CACHE_DIR": ""})
            elif style in ("import", "bundle"):
                if style == "bundle":
                    with bundle_lock:
                        if not bundle_data:
                            bundle_data.append(export_dir.with_suffix(".zip").read_bytes())
                    with zipfile.ZipFile(io.BytesIO(bundle_data[0])) as z:
                        data = z.read(exported.name)
                else:
                    data = exported.read_bytes()
                result = core.import_model(data, "CPU", config)
            else:
                result = (core.compile_model(build(core.read_model(str(onnx))), "CPU",
                                             {**config, "CACHE_DIR": ""}) if style == "export"
                          else original(core, onnx, key, build, config))
                if style == "export":
                    export_dir.mkdir(exist_ok=True)
                    stream = result.export_model()
                    exported.write_bytes(stream if isinstance(stream, bytes) else stream.getvalue())
            spans.append({"model": Path(onnx).name, "config": config,
                          "queue_s": started - queued, "compile_s": time.perf_counter() - started})
        return result

    pose.compile_model = detect.compile_model = measured
    pipeline.ov.Core = measured_core
    timings = {}
    original_models = pipeline.Models

    class TimedModels(original_models):
        def __init__(self, *args, **kwargs):
            start = time.perf_counter()
            super().__init__(*args, **kwargs)
            timings["models_init_s"] = time.perf_counter() - start
            faulthandler.cancel_dump_traceback_later()

    pipeline.Models = TimedModels
    faulthandler.dump_traceback_later(1.5, repeat=True)
    start = time.perf_counter()
    if video:
        timings["pipeline"] = pipeline.run(video, str(Path(output).with_suffix(".poses.json")))
    else:
        models = pipeline.Models(pose.DEFAULT_POSE, pipeline.WEIGHTS / "yolo26s-448x768.onnx",
                                 pipeline.WEIGHTS / "yolo26n-384x640.onnx", streams=8,
                                 pose_small_onnx=pose.DEFAULT_SMALL_POSE)
        pipeline._ALIVE.append(models)
        if style == "export":
            models.one
            directory = Path(__file__).parent / "runs" / "step2-export-models"
            with zipfile.ZipFile(directory.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED,
                                 compresslevel=1) as z:
                for p in directory.glob("*.blob"):
                    z.write(p, p.name)
    elapsed = time.perf_counter() - start
    faulthandler.cancel_dump_traceback_later()
    Path(output).write_text(json.dumps({"style": style, "load_s": elapsed,
                                       "core_s": core_spans, "spans": spans, "timings": timings}, indent=1), "utf-8")
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--child", choices=["parallel", "serial", "export", "import", "bundle", "ir"])
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--styles", nargs="+", choices=["parallel", "serial", "export", "import", "bundle", "ir"],
                    default=["parallel", "serial"])
    ap.add_argument("--sample-load", action="store_true")
    a = ap.parse_args()
    if a.child:
        child(a.child, a.out, a.video)
    else:
        a.out.mkdir(parents=True, exist_ok=False)
        rows = []
        for i in range(a.runs):
            for style in (a.styles if i % 2 == 0 else list(reversed(a.styles))):
                path = a.out / f"{style}-{i}.json"
                load = None
                if a.sample_load:
                    from acceptance import load_snapshot
                    load = load_snapshot()
                with path.with_suffix(".stderr.txt").open("wb") as err, \
                        path.with_suffix(".stdout.txt").open("wb") as stdout:
                    command = [sys.executable, __file__, str(path), "--child", style]
                    if a.video:
                        command += ["--video", str(a.video)]
                    p = subprocess.run(command, stdout=stdout, stderr=err, timeout=30)
                if p.returncode:
                    raise RuntimeError(f"diagnostic child failed: {path}")
                rows.append(json.loads(path.read_text("utf-8")))
                rows[-1]["load_before"] = load
                path.write_text(json.dumps(rows[-1], indent=1), "utf-8")
                print(style, i, rows[-1]["load_s"], flush=True)
        (a.out / "summary.json").write_text(json.dumps({
            "note": "Instrumented startup diagnostics; not acceptance timings.",
            "runs": rows,
        }, indent=1), "utf-8")
