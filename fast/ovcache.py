"""Import prepared compiled graphs, or compile through a saved OpenVINO IR copy.

The prepared bundle opens once per process; its source/config/runtime identities and
per-graph hashes are checked before import. The IR cache remains available without a bundle.
"""
import hashlib
import io
import json
import os
import platform
import threading
import zipfile
from pathlib import Path

import openvino as ov

_lock = threading.Lock()
_tensors = threading.Lock()
_bundles = {}
_source_hashes = {}
BUNDLE_NAME = "prepared-models.zip"


def _sha(data):
    return hashlib.sha256(data).hexdigest().upper()


def _identity(core, onnx, key, config):
    path = Path(onnx).resolve()
    st = path.stat()
    source_key = (str(path), st.st_size, st.st_mtime_ns)
    if source_key not in _source_hashes:
        _source_hashes[source_key] = _sha(path.read_bytes())
    identity = {"model": path.name, "source_sha256": _source_hashes[source_key],
                "preprocessing": key, "config": config, "openvino": ov.__version__,
                "cpu": core.get_property("CPU", "FULL_DEVICE_NAME"),
                "architecture": platform.machine(), "logical_cpus": os.cpu_count()}
    return _sha(json.dumps(identity, sort_keys=True).encode()), identity


class BundleWriter:
    """Setup-only compiler; exports fresh graphs, not graphs imported from OpenVINO's cache."""

    def __init__(self):
        self.entries = {}
        self.data = {}

    def compile(self, core, onnx, key, build, config):
        with _lock:
            digest, identity = _identity(core, onnx, key, config)
        compiled = core.compile_model(build(core.read_model(str(onnx))), "CPU", {**config, "CACHE_DIR": ""})
        stream = compiled.export_model()
        data = stream if isinstance(stream, bytes) else stream.getvalue()
        with _lock:
            self.entries[digest] = {**identity, "blob_sha256": _sha(data)}
            self.data[digest] = data
        return compiled

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.zip")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            z.writestr("manifest.json", json.dumps(self.entries, sort_keys=True))
            for digest, data in self.data.items():
                z.writestr(digest, data)
        os.replace(tmp, path)


def _import_bundle(core, path, onnx, key, config):
    with _lock:
        if path not in _bundles:
            data = path.read_bytes()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                manifest = json.loads(z.read("manifest.json"))
            _bundles[path] = (data, manifest)
        data, manifest = _bundles[path]
        digest, identity = _identity(core, onnx, key, config)
        if digest not in manifest:
            raise RuntimeError(f"Prepared cache does not match {identity['model']}, preprocessing, config or runtime. "
                               "Run fast\\prepare_cache.py again; custom configurations need their own prepared cache.")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        blob = z.read(digest)
    if _sha(blob) != manifest[digest]["blob_sha256"]:
        raise RuntimeError(f"Prepared graph hash mismatch: {identity['model']}")
    return core.import_model(blob, "CPU", config)


def infer(request, x):
    """`request.infer({0: x})` as a list of output arrays, minus a rare whole-process hang.

    OpenVINO 2026.0 can deadlock when two threads touch tensor memory at once: one holds
    OpenVINO's lock and waits for the GIL while the other holds the GIL and waits for that
    lock. So tensor copies take turns; the inference itself still runs in parallel.
    """
    with _tensors:
        t = request.get_input_tensor(0)
        if tuple(t.shape) != x.shape:
            t.shape = x.shape
        t.data[:] = x
        request.start_async()
    request.wait()
    with _tensors:
        return [request.get_output_tensor(i).data.copy() for i in range(len(request.model_outputs))]


def compile_model(core, onnx, key, build, config):
    """`build(model)` adds preprocessing to the read ONNX graph; `key` holds everything it depends on."""
    cache = core.get_property("CACHE_DIR")
    if not cache:
        return core.compile_model(build(core.read_model(str(onnx))), "CPU", config)
    bundle = Path(cache) / BUNDLE_NAME
    if bundle.is_file():
        return _import_bundle(core, bundle, onnx, key, config)
    st = Path(onnx).stat()
    digest = hashlib.sha1(repr((key, st.st_size, st.st_mtime_ns)).encode()).hexdigest()[:12]
    xml = Path(cache) / f"{Path(onnx).stem}-{digest}.xml"
    with _lock:
        if not xml.exists():
            tmp = xml.with_suffix(".tmp.xml")
            ov.save_model(build(core.read_model(str(onnx))), str(tmp), compress_to_fp16=False)
            os.replace(tmp.with_suffix(".bin"), xml.with_suffix(".bin"))
            os.replace(tmp, xml)
    return core.compile_model(str(xml), "CPU", config)
