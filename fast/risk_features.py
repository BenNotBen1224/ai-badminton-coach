"""Small learned risk tree evaluated in the same crop coordinates as training."""
import numpy as np


def features(k):
    conf = k[:, :, 2]
    return np.concatenate([conf, np.sort(conf, axis=1)[:, :3],
                           conf.mean(1)[:, None], k[:, :, :2].reshape(len(k), -1) / 256], axis=1)


def crop_features(k, center, scale):
    crop = k.copy()
    crop[:, :2] = ((crop[:, :2] - center) / scale + .5) * [192, 256]
    return features(crop[None])[0]


def risk_value(tree, x):
    while "value" not in tree:
        tree = tree["left"] if x[tree["feature"]] < tree["threshold"] else tree["right"]
    return tree["value"]
