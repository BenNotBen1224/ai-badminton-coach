"""Check the court finder: same corners across a clip, clear rejection when there is no court.

usage: python courtcheck.py OUT.jpg VIDEO [VIDEO ...]      (comparison/.venv)
Prints per video: fit range, worst corner spread in pixels over 8 sampled frames.
"""
import json
import sys

import cv2
import numpy as np

from court import draw, find_court


def frames(video, k=8):
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in np.linspace(0, n - 1, k).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            yield int(i), fr
    cap.release()


def main(out, videos):
    tiles = []
    for v in videos:
        quads, fits, first = [], [], None
        for i, fr in frames(v):
            q, info = find_court(fr)
            fits.append(info.get("fit", 0.0))
            if q is not None:
                quads.append(np.float32(q) * [fr.shape[1], fr.shape[0]])
            if first is None:
                first = draw(fr, q) if q is not None else fr
        spread = float(np.max(np.ptp(np.float32(quads), axis=0))) if len(quads) > 1 else None
        print(json.dumps({"video": str(v).split("\\")[-1], "found": len(quads), "fit_min": min(fits),
                          "fit_max": max(fits), "corner_spread_px": spread and round(spread, 1)}))
        tiles.append(cv2.resize(first, (640, 360)))
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
