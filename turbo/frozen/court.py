"""Find the badminton court lines in a fixed camera view filmed from behind one baseline.

Returns the outer doubles quad (far-left, far-right, near-right, near-left), normalized to 0..1.
Search: thin white lines -> line segments -> try pairs of "across" and "along" lines as known court
lines -> project the whole standard court and keep the fit that lands most of it on white pixels.
"""
import itertools
import sys

import cv2
import numpy as np

W = 960  # working width; court lines stay 1-3 px thick here
COURT_W, COURT_L = 6.1, 13.4
ACROSS = [0.0, 0.76, 4.72, 8.68, 12.64, 13.4]  # back, doubles long service, short service (each half)
ALONG = [0.0, 0.46, 5.64, 6.1]  # doubles and singles sidelines
ACROSS_PAIRS = [(0.0, 13.4), (0.76, 12.64), (0.0, 12.64), (0.76, 13.4), (0.0, 8.68), (4.72, 13.4)]
ALONG_PAIRS = [(0.0, 6.1), (0.46, 5.64)]


def line_mask(gray, tau=4, step=18):
    """Pixels brighter than both neighbours tau px away, across or along: thin bright lines."""
    g = gray.astype(np.int16)
    hor = (g - np.roll(g, tau, 1) > step) & (g - np.roll(g, -tau, 1) > step)
    ver = (g - np.roll(g, tau, 0) > step) & (g - np.roll(g, -tau, 0) > step)
    return ((hor | ver) & (g > 150)).astype(np.uint8)


def model_points(spacing=0.1):
    pts = []
    for y in ACROSS:
        pts += [(x, y) for x in np.linspace(0, COURT_W, round(COURT_W / spacing) + 1)]
    for x in ALONG:
        pts += [(x, y) for y in np.linspace(0, COURT_L, round(COURT_L / spacing) + 1)]
    for y in np.linspace(0, COURT_L, round(COURT_L / spacing) + 1):  # centre line, absent around the net
        if y <= 4.72 or y >= 8.68:
            pts.append((COURT_W / 2, y))
    return np.float32(pts)


FINE, COARSE = model_points(0.1), model_points(0.4)
UNIT = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])


def rect_to_unit(x0, x1, y0, y1):
    return np.float32([[1 / (x1 - x0), 0, -x0 / (x1 - x0)], [0, 1 / (y1 - y0), -y0 / (y1 - y0)], [0, 0, 1]])


# every (across pair, along pair) guess as a map from court metres to the unit square
GUESSES = np.float32([rect_to_unit(x0, x1, y0, y1)
                      for (y0, y1), (x0, x1) in itertools.product(ACROSS_PAIRS, ALONG_PAIRS)])


def as_line(seg):
    x0, y0, x1, y1 = map(float, seg)
    a, b = y1 - y0, x0 - x1
    n = np.hypot(a, b)
    return np.array([a / n, b / n, -(a * x0 + b * y0) / n]), n


def merge(lines, key, tol):
    """Group nearly identical lines by key(line); each group keeps its longest member and total length."""
    groups = []
    for ln, length in sorted(lines, key=lambda t: -t[1]):
        k = key(ln)
        for gr in groups:
            if np.all(np.abs(gr[2] - k) < tol):
                gr[1] += length
                break
        else:
            groups.append([ln, length, k])
    return [(g[0], g[1]) for g in sorted(groups, key=lambda g: -g[1])]


def cross(l1, l2):
    p = np.cross(l1, l2)
    return p[:2] / p[2] if abs(p[2]) > 1e-9 else None


def candidates(mask, w, h):
    segs = cv2.HoughLinesP(mask * 255, 1, np.pi / 360, threshold=50, minLineLength=w // 16, maxLineGap=6)
    across, along = [], []
    for s in [] if segs is None else segs.reshape(-1, 4):
        ln, length = as_line(s)
        angle = np.degrees(np.arctan2(abs(s[3] - s[1]), abs(s[2] - s[0])))
        if angle < 8:
            across.append((ln, length))
        elif angle > 20:
            along.append((ln, length))
    y_at = lambda ln: np.array([-(ln[0] * w / 2 + ln[2]) / ln[1]])
    x_at = lambda ln: np.array([-(ln[1] * y + ln[2]) / ln[0] for y in (h * 0.3, h * 0.9)])
    return merge(across, y_at, 4)[:12], merge(along, x_at, 6)[:10]


def score(hs, pts, hit, w, h):
    """Fraction of the court model that lands on line pixels, for a batch of homographies."""
    p = hs[:, :, :2] @ pts.T + hs[:, :, 2:]
    z = p[:, 2]
    ok = z > 1e-6
    z = np.where(ok, z, 1)
    x = (p[:, 0] / z).round().astype(np.int32)
    y = (p[:, 1] / z).round().astype(np.int32)
    inside = ok & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return (hit[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)] & inside).mean(1)


def refine(quad, mask, band=3):
    """Re-fit each outer edge to the line pixels near it, then intersect the edges."""
    ys, xs = np.nonzero(mask)
    pix = np.c_[xs, ys].astype(np.float32)
    edges = []
    for a, b in zip(quad, np.roll(quad, -1, 0)):
        ln, _ = as_line([*a, *b])
        d = np.abs(pix @ ln[:2] + ln[2])
        t = (pix - a) @ (b - a) / np.dot(b - a, b - a)
        near = pix[(d < band) & (t > 0.02) & (t < 0.98)]
        if len(near) < 20:
            edges.append(ln)
            continue
        vx, vy, x0, y0 = cv2.fitLine(near, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        edges.append(as_line([x0, y0, x0 + vx, y0 + vy])[0])
    return np.float32([cross(edges[i - 1], edges[i]) for i in range(4)])


def find_court(frame):
    h0, w0 = frame.shape[:2]
    h = round(W * h0 / w0)
    gray = cv2.cvtColor(cv2.resize(frame, (W, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    mask = line_mask(gray)
    hit = cv2.dilate(mask, np.ones((3, 3), np.uint8)).astype(bool)
    across, along = candidates(mask, W, h)
    if len(across) < 2 or len(along) < 2:
        return None, {"across": len(across), "along": len(along), "hypotheses": 0}
    hom = np.cross(np.float64([a for a, _ in across])[:, None], np.float64([b for b, _ in along])[None])
    with np.errstate(divide="ignore", invalid="ignore"):
        meet = np.where(np.abs(hom[..., 2:]) > 1e-9, hom[..., :2] / hom[..., 2:], np.nan).astype(np.float32)
    quads = []
    for t, b in itertools.combinations(range(len(across)), 2):
        for l, r in itertools.combinations(range(len(along)), 2):
            c = meet[[t, t, b, b], [l, r, r, l]]
            if np.isnan(c).any():
                continue
            if c[0, 1] > c[3, 1]:  # top line must be the far one
                c = c[[3, 2, 1, 0]]
            if c[0, 0] > c[1, 0]:
                c = c[[1, 0, 3, 2]]
            far, near = c[1, 0] - c[0, 0], c[2, 0] - c[3, 0]
            if 0 < far <= near * 1.05 and cv2.contourArea(c) > 0.02 * W * h and cv2.isContourConvex(c):
                quads.append(cv2.getPerspectiveTransform(UNIT, c))
    info = {"across": len(across), "along": len(along), "hypotheses": len(quads) * len(GUESSES)}
    if not quads:
        return None, info
    hyps = (np.float32(quads)[:, None] @ GUESSES[None]).reshape(-1, 3, 3)
    coarse = np.concatenate([score(hyps[i:i + 4096], COARSE, hit, W, h) for i in range(0, len(hyps), 4096)])
    top = np.argsort(-coarse)[:32]
    fine = score(hyps[top], FINE, hit, W, h)
    best = hyps[top[int(fine.argmax())]]
    corners = np.float32([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]])
    quad = refine(cv2.perspectiveTransform(corners[None], best)[0], mask)
    fit = float(score(cv2.getPerspectiveTransform(corners, quad)[None], FINE, hit, W, h)[0])
    norm = (quad / [W, h]).round(4).tolist()
    return norm, {"fit": round(fit, 3), "fit_before_refine": round(float(fine.max()), 3), **info}


MIN_FIT = 0.85  # real courts score 0.95-0.99; no court or a side-on handheld view scores 0-0.65


def first_frame(video):
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {video}")
    return frame


def court_from_video(video, tries=(0.0, 0.25, 0.5, 0.75)):
    """Court lines from the first frame that shows them (intro cards and cutaways are skipped)."""
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    best = (None, {"fit": 0.0})
    for t in tries:
        if t:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * n))
        ok, frame = cap.read()
        if not ok:
            continue
        quad, info = find_court(frame)
        if quad is not None and info["fit"] > best[1]["fit"]:
            best = (quad, {**info, "frame": int(t * n)})
        if best[1]["fit"] >= MIN_FIT:
            break
    cap.release()
    if best[1]["fit"] < MIN_FIT:
        raise RuntimeError(f"no court found in {video} (best fit {best[1]['fit']:.2f}); "
                           "the camera must be fixed behind a baseline, or pass --court x,y,... (8 numbers)")
    return best


def draw(frame, quad):
    h, w = frame.shape[:2]
    hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]]),
                                     np.float32(np.float32(quad) * [w, h]))
    out = frame.copy()
    segs = [((0, y), (COURT_W, y)) for y in ACROSS] + [((x, 0), (x, COURT_L)) for x in ALONG]
    segs += [((COURT_W / 2, 0), (COURT_W / 2, 4.72)), ((COURT_W / 2, 8.68), (COURT_W / 2, COURT_L))]
    for a, b in segs:
        p = cv2.perspectiveTransform(np.float32([[a, b]]), hm)[0].round().astype(int)
        cv2.line(out, tuple(p[0]), tuple(p[1]), (0, 0, 255), 2, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    import json
    import time
    for v in sys.argv[1:]:
        t = time.perf_counter()
        fr = first_frame(v)
        t1 = time.perf_counter()
        quad, info = find_court(fr)
        print(json.dumps({"video": v, "quad": quad, **info, "read_s": round(t1 - t, 3),
                          "find_s": round(time.perf_counter() - t1, 3)}))
