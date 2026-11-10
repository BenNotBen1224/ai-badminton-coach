"""Fit a small crop-risk tree using only the existing teacher-labelled train/validation windows."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fast"))
from pose import Pose, DEFAULT_POSE, DEFAULT_SMALL_POSE
from ovcache import infer
from acceptance import sha256


def features(k):
    conf = k[:, :, 2]
    return np.concatenate([conf, np.sort(conf, axis=1)[:, :3],
                           conf.mean(1)[:, None], k[:, :, :2].reshape(len(k), -1) / 256], axis=1)


def predict(tree, x):
    if "value" in tree:
        return np.full(len(x), tree["value"])
    left = x[:, tree["feature"]] < tree["threshold"]
    out = np.empty(len(x))
    out[left] = predict(tree["left"], x[left])
    out[~left] = predict(tree["right"], x[~left])
    return out


def fit(x, y, depth=3):
    leaf = {"value": float(y.mean()), "samples": len(y)}
    if depth == 0 or len(y) < 80:
        return leaf
    best = None
    baseline = float(((y - y.mean()) ** 2).sum())
    for j in range(x.shape[1]):
        for threshold in np.unique(np.quantile(x[:, j], np.linspace(.1, .9, 17))):
            left = x[:, j] < threshold
            if min(left.sum(), (~left).sum()) < 40:
                continue
            error = float(((y[left] - y[left].mean()) ** 2).sum() +
                          ((y[~left] - y[~left].mean()) ** 2).sum())
            if best is None or error < best[0]:
                best = (error, j, float(threshold), left)
    if best is None or best[0] >= baseline:
        return leaf
    _, feature, threshold, left = best
    return {"feature": feature, "threshold": threshold,
            "left": fit(x[left], y[left], depth - 1), "right": fit(x[~left], y[~left], depth - 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("distillation", ROOT / "train" / "train.py")
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    tiny, strong = [Pose(p, streams=1, threads=8, dynamic=True) for p in (DEFAULT_SMALL_POSE, DEFAULT_POSE)]
    requests = [p.compiled.create_infer_request() for p in (tiny, strong)]
    sets = {}
    paths = {"train": [ROOT / "train" / n for n in ("lab-ms-0.npz", "lab-ws-0.npz")],
             "validation": [ROOT / "train" / "lab-ms-102.npz"]}
    for name, sources in paths.items():
        data = training.load(sources)
        crops = [training.crop(p, xy) for p, xy in zip(data["patches"], data["xy"])]
        images = np.stack([c[0] for c in crops])
        target = np.stack([c[1] for c in crops])
        predictions = []
        for req in requests:
            parts = []
            for start in range(0, len(images), 32):
                sx, sy = infer(req, images[start:start + 32])
                xy, conf = training.decode(sx, sy)
                parts.append(np.concatenate([xy, conf[:, :, None]], axis=2))
            predictions.append(np.concatenate(parts))
        trusted = data["w"] > .5
        hits = [(np.linalg.norm(p[:, :, :2] - target, axis=2) < .05 * training.BOX_PX) & trusted
                for p in predictions]
        gain = hits[1].sum(1) - hits[0].sum(1)
        eligible = (predictions[0][:, :, 2].mean(1) >= 1.05) & (predictions[0][:, :, 2].min(1) < .65)
        sets[name] = (features(predictions[0]), gain, eligible)
        np.savez_compressed(a.out / f"{name}.npz", tiny=predictions[0], strong=predictions[1],
                            target=target, trusted=trusted, gain=gain, eligible=eligible)
    x, y, eligible = sets["train"]
    tree = fit(x[eligible], y[eligible])
    reports = {}
    for name, (x, y, eligible) in sets.items():
        chosen = eligible & (predict(tree, x) > 0)
        reports[name] = {"samples": len(x), "eligible_tail": int(eligible.sum()),
                         "all_tail_direct_gain": int(y[eligible].sum()), "selected": int(chosen.sum()),
                         "selected_direct_gain": int(y[chosen].sum()),
                         "selected_better": int((y[chosen] > 0).sum()),
                         "selected_worse": int((y[chosen] < 0).sum())}
    doc = {"tree": tree, "selection": "mean>=1.05 and min<0.65 and tree(features)>0",
           "feature_order": "17 joint confidences;3 sorted lowest confidences;mean confidence;34 xy/256",
           "training_only": True, "reports": reports,
           "source_sha256": {str(p): sha256(p) for sources in paths.values() for p in sources},
           "pose_sha256": {str(p): sha256(p) for p in (DEFAULT_SMALL_POSE, DEFAULT_POSE)},
           "script_sha256": sha256(__file__)}
    (a.out / "selector.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc), flush=True)


if __name__ == "__main__":
    main()
