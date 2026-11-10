import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "models" / "badminton-pipeline-repro" / "scripts" / "tracknet_runtime"))
from utils.general import HEIGHT, WIDTH, get_model  # noqa: E402

clip = HERE / "runs" / f"{sys.argv[1]}.mp4"
starts = [int(s) for s in sys.argv[2:]] or [30, 90, 150, 210]
ckpt = torch.load(HERE.parents[1] / "models" / "TrackNetV3" / "ckpts" / "TrackNet_best.pt", map_location="cpu", weights_only=False)
seq_len, bg_mode = ckpt["param_dict"]["seq_len"], ckpt["param_dict"]["bg_mode"]
print("seq_len", seq_len, "bg_mode", bg_mode, "epoch", ckpt.get("epoch"), "keys", list(ckpt["param_dict"].keys()))
model = get_model("TrackNet", seq_len, bg_mode)
model.load_state_dict(ckpt["model"])
model.eval()

cap = cv2.VideoCapture(str(clip))
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
h, w = frames[0].shape[:2]
median = np.median(np.stack(frames[::2]), 0).astype(np.uint8)[..., ::-1]
median_chw = np.moveaxis(cv2.resize(median, (WIDTH, HEIGHT)), -1, 0)
for start in starts:
    seq = [np.moveaxis(cv2.resize(frames[start + k][..., ::-1], (WIDTH, HEIGHT)), -1, 0) for k in range(seq_len)]
    x = np.concatenate([median_chw] + seq, 0) if bg_mode == "concat" else np.concatenate(seq, 0)
    with torch.no_grad():
        y = model(torch.from_numpy(x[None] / 255.0).float())[0].numpy()
    for k in range(seq_len):
        m = y[k]
        iy, ix = np.unravel_index(m.argmax(), m.shape)
        print(f"frame {start + k:4d} max {m.max():.3f} at ({ix * w / WIDTH:.0f},{iy * h / HEIGHT:.0f}) >0.15 px {(m > 0.15).sum()} >0.5 px {(m > 0.5).sum()}")
    k = seq_len // 2
    heat = cv2.resize((np.clip(y[k], 0, 1) * 255).astype(np.uint8), (w, h))
    overlay = frames[start + k].copy()
    overlay[..., 2] = np.maximum(overlay[..., 2], heat)
    cv2.imwrite(str(HERE / "runs" / f"diag-{sys.argv[1]}-{start + k}.jpg"), overlay)

