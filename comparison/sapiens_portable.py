"""Convert the official fixed-shape TorchScript graph without changing weights."""
from pathlib import Path
import sys
import time

import numpy as np
import torch
import openvino as ov

sys.path.insert(0, str(Path(__file__).parent))
from run import WEIGHTS, save


def main():
    started = time.perf_counter()
    torch.set_num_threads(1)
    path = WEIGHTS / "sapiens_0.3b_goliath_torchscript.pt2"
    model = torch.jit.load(str(path), map_location="cpu").float().eval()
    target = WEIGHTS / "sapiens.xml"
    if not target.exists():
        print("Converting unchanged Sapiens TorchScript graph", flush=True)
        converted = ov.convert_model(model, input=[1, 3, 1024, 768])
        ov.save_model(converted, str(target), compress_to_fp16=False)
    compiled = ov.Core().compile_model(str(target), "CPU", {
        "INFERENCE_NUM_THREADS": 4, "NUM_STREAMS": 1, "INFERENCE_PRECISION_HINT": "f32",
    })
    conversion_seconds = time.perf_counter() - started
    random = np.random.default_rng(42).standard_normal((1, 3, 1024, 768)).astype(np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        reference = model(torch.from_numpy(random)).numpy()
    print("Reference forward complete", flush=True)
    output = compiled(random)[compiled.output(0)]
    if output.shape != (1, 308, 256, 192):
        raise ValueError(f"Unexpected native output {output.shape}")
    difference = np.abs(output - reference)
    np.testing.assert_allclose(output, reference, atol=.003, rtol=.003)
    save(WEIGHTS / "sapiens-port.json", {
        "backend": "OpenVINO CPU", "precision": "FP32 weights and inference",
        "native_shape": list(output.shape), "input_shape": list(random.shape),
        "checkpoint": str(path.name), "openvino_version": ov.__version__,
        "conversion_and_compile_seconds": round(conversion_seconds, 3),
        "verification_seconds": round(time.perf_counter() - started, 3),
        "max_absolute_heatmap_error": float(difference.max()),
        "mean_absolute_heatmap_error": float(difference.mean()),
        "allclose_atol": .003, "allclose_rtol": .003,
        "verification": "Same fixed random RGB-normalized-scale tensor, full 308-channel heatmap comparison against original FP32 TorchScript on CPU",
    })
    print("Verified faithful Sapiens OpenVINO CPU conversion", flush=True)


if __name__ == "__main__":
    main()
