"""Fine-tune an RTMPose student on teacher-labelled badminton crops (knowledge distillation).

The student starts from the released --init weights (converted from their ONNX file),
sees crops made the same way the pipeline makes them plus random jitter, and learns
the teachers' joints with RTMPose's own SimCC KL loss. Each epoch is checked on a
held-out clip against the teachers, next to the untouched s and m models.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx2torch import convert

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fast"))
from pose import INPUT_H, INPUT_W, MEAN, STD  # noqa: E402

FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
SPLIT, SIGMA, BETA = 2.0, (4.9, 5.66), 10.0
PATCH_H = 384
BOX_PX = PATCH_H / (1.5 * 1.25)  # unpadded box height in patch pixels (height-limited boxes)
WEIGHTS = Path(__file__).resolve().parents[1] / "fast" / "weights"


def load_onnx(path):
    m = onnx.load(str(path))
    for n in m.graph.node:  # Clip(x, min, "") -> Clip(x, min): onnx2torch can't read an empty input
        while n.input and n.input[-1] == "":
            n.input.pop()
    return convert(m)


def labels(d):
    """Teacher mean position, confidence (halpe26 scale) and a 0..1 trust weight per joint."""
    p = d["preds"]
    xy = p[..., :2].mean(1)
    conf = p[:, :2, :, 2].mean(1)
    spread = np.linalg.norm(p[..., :2] - xy[:, None], axis=-1).max(1) / BOX_PX
    w = np.clip((conf - 0.3) / 0.4, 0, 1) * np.clip(1 - (spread - 0.02) / 0.04, 0, 1)
    return xy.astype(np.float32), conf.astype(np.float32), w.astype(np.float32)


def crop(patch, xy, rng=None):
    """Student crop (and joints in crop pixels) from a patch; random jitter when rng is given."""
    cx, cy, s, flip = PATCH_H * 0.375, PATCH_H / 2, 1.0, False
    if rng is not None and rng.random() > 0.25:
        s = rng.uniform(0.8, 1.25)
        cx += rng.uniform(-0.08, 0.08) * INPUT_W
        cy += rng.uniform(-0.08, 0.08) * INPUT_H
    if rng is not None:
        flip = rng.random() < 0.5
    m = np.array([[1 / s, 0, INPUT_W / 2 - cx / s], [0, 1 / s, INPUT_H / 2 - cy / s]], np.float32)
    img = cv2.warpAffine(patch, m, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
    k = xy @ m[:, :2].T + m[:, 2]
    if rng is not None:
        if rng.random() < 0.3:  # look like a smaller, softer person
            f = rng.uniform(0.4, 0.8)
            small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        a, b = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    if flip:
        img = img[:, ::-1]
        k = k[FLIP]
        k[:, 0] = INPUT_W - 1 - k[:, 0]
    return np.ascontiguousarray(img), k, flip


def to_input(imgs):
    x = (np.stack(imgs)[..., ::-1].astype(np.float32) - np.float32(MEAN)) / np.float32(STD)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


def simcc_targets(k):
    """Gaussian SimCC targets (mmpose SimCCLabel, normalize=False)."""
    bx = torch.arange(int(INPUT_W * SPLIT), dtype=torch.float32)
    by = torch.arange(int(INPUT_H * SPLIT), dtype=torch.float32)
    k = torch.from_numpy(np.stack(k)) * SPLIT
    tx = torch.exp(-((bx - k[..., :1]) ** 2) / (2 * SIGMA[0] ** 2))
    ty = torch.exp(-((by - k[..., 1:2]) ** 2) / (2 * SIGMA[1] ** 2))
    return tx, ty


def kl(pred, target, w):
    """mmpose KLDiscretLoss(beta=10, label_softmax=True, use_target_weight=True)."""
    loss = F.kl_div(F.log_softmax(pred * BETA, -1), F.softmax(target * BETA, -1), reduction="none").mean(-1)
    return (loss * w).sum() / w.sum().clamp(min=1)


def decode(sx, sy):
    x = sx.argmax(-1) / SPLIT
    y = sy.argmax(-1) / SPLIT
    return np.stack([x, y], -1), 0.5 * (sx.max(-1) + sy.max(-1))


def evaluate(run, data):
    """Error vs teachers on exact (unjittered) crops: PCK@0.05 / @0.1 of box height, by role."""
    imgs, ks, _ = zip(*(crop(p, xy) for p, xy in zip(data["patches"], data["xy"])))
    preds = []
    for i in range(0, len(imgs), 64):
        sx, sy = run(to_input(imgs[i:i + 64]))
        preds.append(decode(sx, sy)[0])
    pred = np.concatenate(preds)
    err = np.linalg.norm(pred - np.stack(ks), axis=-1) / BOX_PX  # crop px == patch px at s=1
    ok = data["w"] > 0.5
    out = {}
    for role in ("near", "far"):
        sel = ok & (data["role"] == role)[:, None]
        out[role] = {"pck05": round(float((err[sel] < 0.05).mean()), 4),
                     "pck10": round(float((err[sel] < 0.1).mean()), 4),
                     "arms_pck05": round(float((err[:, 7:11][sel[:, 7:11]] < 0.05).mean()), 4)}
    return out


def load(paths):
    parts = [np.load(p) for p in paths]
    d = {k: np.concatenate([p[k] for p in parts]) for k in ("patches", "preds", "role", "box_h")}
    d["xy"], d["conf"], d["w"] = labels(d)
    return d


def ort_runner(path):
    sess = ort.InferenceSession(str(path))
    return lambda x: [np.asarray(o) for o in sess.run(None, {"input": x.numpy()})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output .onnx")
    ap.add_argument("--init", default=str(WEIGHTS / "rtmpose-s.onnx"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5, help="1e-4 blows the pretrained weights up on step 1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms-weight", type=float, default=1.0, help="loss multiplier for elbows and wrists")
    args = ap.parse_args()
    if not np.isfinite(args.arms_weight) or args.arms_weight <= 0:
        ap.error("--arms-weight must be finite and positive")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train, val = load(args.train), load(args.val)
    print(f"train {len(train['xy'])} val {len(val['xy'])}, trusted joints {float((train['w'] > 0.5).mean()):.3f}")
    log = {"args": vars(args), "reference": {}, "epochs": []}
    for name in ("rtmpose-s.onnx", "rtmpose-m.onnx"):
        log["reference"][name] = evaluate(ort_runner(WEIGHTS / name), val)
        print(name, log["reference"][name], flush=True)

    net = load_onnx(args.init)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(train["xy"]) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    def run_eval(x):
        net.eval()
        with torch.no_grad():
            return [o.numpy() for o in net(x)]

    for epoch in range(args.epochs):
        t0, total = time.perf_counter(), 0.0
        net.train()
        order = rng.permutation(len(train["xy"]))
        for b in range(len(order) // args.batch):
            idx = order[b * args.batch:(b + 1) * args.batch]
            imgs, ks, ws = [], [], []
            for i in idx:
                img, k, flip = crop(train["patches"][i], train["xy"][i], rng)
                inside = (k[:, 0] >= 0) & (k[:, 0] < INPUT_W) & (k[:, 1] >= 0) & (k[:, 1] < INPUT_H)
                imgs.append(img)
                ks.append(k)
                ws.append((train["w"][i][FLIP] if flip else train["w"][i]) * inside)
            tx, ty = simcc_targets(ks)
            sx, sy = net(to_input(imgs))
            w = torch.from_numpy(np.stack(ws))
            w[:, 7:11] *= args.arms_weight
            loss = kl(sx, tx, w) + kl(sy, ty, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss)
        res = evaluate(run_eval, val)
        log["epochs"].append({"epoch": epoch, "loss": round(total / (b + 1), 5), "val": res,
                              "secs": round(time.perf_counter() - t0, 1)})
        print(json.dumps(log["epochs"][-1]), flush=True)

    net.eval()
    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W)
    torch.onnx.export(net, dummy, args.out, input_names=["input"], output_names=["simcc_x", "simcc_y"],
                      dynamic_axes={"input": {0: "batch"}, "simcc_x": {0: "batch"}, "simcc_y": {0: "batch"}},
                      opset_version=17, dynamo=False)
    log["exported"] = evaluate(ort_runner(args.out), val)
    print("exported", log["exported"])
    Path(args.out).with_suffix(".json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
