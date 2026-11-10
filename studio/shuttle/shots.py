import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ENGINE = HERE / "engine"
RUNS = HERE / "runs"
SPANS = {"space": (56.8, 65.8), "time": (125.6, 135.0), "options": (271.8, 286.0)}
SEGMENT = {"space": 2, "time": 5, "options": 10}
SCALE = 1.5
FPS = 30
COURT_M = np.float32([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]])


def track(name):
    rows = list(csv.DictReader((RUNS / f"{name}_ball.csv").open(encoding="utf-8")))
    xy = np.full((len(rows), 2), np.nan)
    for row in rows:
        if int(row["Visibility"]) and (float(row["X"]), float(row["Y"])) != (0.0, 0.0):
            xy[int(row["Frame"])] = float(row["X"]) * SCALE, float(row["Y"]) * SCALE
    keep = np.zeros(len(xy), bool)
    for i in np.flatnonzero(~np.isnan(xy[:, 0])):
        for j in range(max(0, i - 3), min(len(xy), i + 4)):
            if j != i and not np.isnan(xy[j, 0]) and np.linalg.norm(xy[i] - xy[j]) <= 120 * abs(i - j):
                keep[i] = True
                break
    xy[~keep] = np.nan
    pieces, current, last = [], [], None
    for i in np.flatnonzero(~np.isnan(xy[:, 0])):
        if last is not None and (i - last > 3 or np.linalg.norm(xy[i] - xy[last]) > 120 * (i - last)):
            pieces.append(current)
            current = []
        current.append(i)
        last = i
    pieces.append(current)
    for n, piece in enumerate(pieces):
        if len(piece) >= 6:
            continue
        linked = False
        for m in (n - 1, n + 1):
            if 0 <= m < len(pieces) and len(pieces[m]) >= 6:
                a, b = (pieces[m][-1], piece[0]) if m < n else (piece[-1], pieces[m][0])
                linked |= abs(b - a) <= 6 and np.linalg.norm(xy[b] - xy[a]) <= 120 * abs(b - a)
        if not linked:
            xy[piece] = np.nan
    return xy


def poses(name, count):
    segment = json.loads((ENGINE / "pose_cache" / f"segment-{SEGMENT[name]:02d}.json").read_text(encoding="utf-8"))
    times = np.array([f["reel_time_seconds"] for f in segment["frames"]])
    start = SPANS[name][0]
    out = []
    for j in range(count):
        k = int(np.argmin(np.abs(times - (start + j / FPS))))
        frame = {}
        if abs(times[k] - (start + j / FPS)) <= 1 / FPS:
            for p in segment["frames"][k]["players"]:
                if p.get("role") in ("near", "far"):
                    frame[p["role"]] = p
        out.append(frame)
    quad = np.float32(segment["court_lines_normalized"]) * np.float32([1920, 1080])
    return out, cv2.getPerspectiveTransform(quad, COURT_M)


def joint_array(player):
    return np.array([j if j is not None else [0.0, 0.0, 0.0] for j in player["joints"]], dtype=float)


def feet(player, homography):
    joints = joint_array(player)
    ankles = joints[[15, 16]]
    good = ankles[ankles[:, 2] >= 0.5]
    x1, y1, x2, y2 = player["bbox_xyxy"]
    point = good[:, :2].mean(0) if len(good) else np.array([(x1 + x2) / 2, y2])
    return cv2.perspectiveTransform(point.reshape(1, 1, 2).astype(np.float32), homography)[0, 0]


def reach(player, point):
    x1, y1, x2, y2 = player["bbox_xyxy"]
    w, h = x2 - x1, y2 - y1
    bx1, bx2, by1, by2 = x1 - 0.6 * w, x2 + 0.6 * w, y1 - 0.7 * h, y2
    dx = max(bx1 - point[0], 0, point[0] - bx2)
    dy = max(by1 - point[1], 0, point[1] - by2)
    return float(np.hypot(dx, dy) / h)


def side_point(xy, i, step):
    j = i + step
    while 0 <= j < len(xy) and abs(j - i) <= 5:
        if not np.isnan(xy[j, 0]):
            return j
        j += step
    return None


def closeness(player, point):
    joints = joint_array(player)
    x1, y1, x2, y2 = player["bbox_xyxy"]
    shoulders = joints[[5, 6]]
    centre = shoulders[:, :2].mean(0) if (shoulders[:, 2] >= 0.3).all() else np.array([(x1 + x2) / 2, y1 + 0.25 * (y2 - y1)])
    return float(np.linalg.norm(point - centre) / (y2 - y1))


def contacts(xy, frames, radius=0.9):
    visible = np.flatnonzero(~np.isnan(xy[:, 0]))
    found = []
    for role in ("near", "far"):
        d = np.full(len(xy), np.nan)
        for i in visible:
            if role in frames[i]:
                d[i] = closeness(frames[i][role], xy[i])
        for i in np.flatnonzero(d <= radius):
            window = d[max(0, i - 6):i + 7]
            if d[i] <= np.nanmin(window):
                found.append({"frame": int(i), "role": role, "closeness": float(d[i])})
    found.sort(key=lambda c: c["frame"])
    shots = []
    for c in found:
        if shots and (shots[-1]["role"] == c["role"] or c["frame"] - shots[-1]["frame"] < 8):
            if c["closeness"] < shots[-1]["closeness"] and shots[-1]["role"] == c["role"]:
                shots[-1] = c
            continue
        shots.append(c)
    return shots


def height(player, y):
    joints = joint_array(player)
    nose, hips = joints[0], joints[[11, 12]]
    shoulder = joints[[5, 6]]
    if nose[2] < 0.3 or (hips[:, 2] < 0.3).all():
        return None, None
    hip_y = hips[hips[:, 2] >= 0.3, 1].mean()
    rel = (hip_y - y) / max(hip_y - nose[1], 1)
    label = "overhead" if rel >= 1.15 else "low" if rel <= 0.35 else "mid"
    return label, float(rel)


def zone(role, y_m):
    depth = y_m if role == "far" else 13.4 - y_m
    return "rear" if depth < 2.6 else "front" if depth > 4.6 else "mid"


def analyse(name):
    xy = track(name)
    frames, homography = poses(name, len(xy))
    start = SPANS[name][0]
    shots = contacts(xy, frames)
    for n, c in enumerate(shots):
        i = c["frame"]
        player = frames[i][c["role"]]
        c["time_s"] = round(start + i / FPS, 3)
        c["xy_px"] = [round(float(v), 1) for v in xy[i]]
        c["hitter_m"] = [round(float(v), 2) for v in feet(player, homography)]
        c["zone"] = zone(c["role"], c["hitter_m"][1])
        c["contact_height"], c["height_ratio"] = height(player, xy[i][1])
        nxt = shots[n + 1]["frame"] if n + 1 < len(shots) else min(len(xy), i + 60)
        seg = xy[i:nxt]
        vis = ~np.isnan(seg[:, 0])
        c["flight_s"] = round((nxt - i) / FPS, 3) if n + 1 < len(shots) else None
        c["apex_y_px"] = round(float(np.nanmin(seg[:, 1])), 1) if vis.any() else None
        c["visible_frac"] = round(float(vis.mean()), 2)
        early = [k for k in range(i + 1, min(nxt, i + 6)) if not np.isnan(xy[k, 0])]
        c["exit_speed_px_s"] = round(float(np.linalg.norm(xy[early[-1]] - xy[i]) / (early[-1] - i) * FPS), 0) if early else None
        c["dy_out_px_s"] = round(float((xy[early[-1], 1] - xy[i, 1]) / (early[-1] - i) * FPS), 0) if early else None
    for n, c in enumerate(shots):
        if n + 1 < len(shots):
            c["receiver_m"] = shots[n + 1]["hitter_m"]
            c["receiver_zone"] = shots[n + 1]["zone"]
    clicks = [h["time_s"] for h in json.loads((ENGINE / "hits.json").read_text(encoding="utf-8"))["hits"]
              if "audio_click" in h["sources"] and SPANS[name][0] <= h["time_s"] <= SPANS[name][1]]
    for c in shots:
        near = min(clicks, key=lambda t: abs(t - c["time_s"])) if clicks else None
        c["audio_click_offset_s"] = round(near - c["time_s"], 3) if near is not None else None
    return xy, frames, shots


def sheet(name, xy, frames, shots):
    cap = cv2.VideoCapture(str(RUNS / f"{name}.mp4"))
    images = []
    for c in shots:
        i = c["frame"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        img = cv2.resize(img, (1920, 1080))
        for k in range(max(0, i - 12), min(len(xy), i + 13)):
            if not np.isnan(xy[k, 0]):
                colour = (0, 255, 255) if k < i else (255, 0, 255) if k > i else (0, 0, 255)
                cv2.circle(img, (int(xy[k, 0]), int(xy[k, 1])), 6 if k == i else 4, colour, -1)
        x1, y1, x2, y2 = (int(v) for v in frames[i][c["role"]]["bbox_xyxy"])
        cx, cy = int(xy[i, 0]), int((y1 + y2) / 2 + min(xy[i, 1], y1)) // 2
        h = max(int((y2 - y1) * 2.4), 320)
        crop = img[max(0, cy - h // 2):cy + h // 2, max(0, cx - h // 2):cx + h // 2]
        crop = cv2.resize(crop, (360, 360))
        label = f"{c['time_s']:.2f} {c['role']} {c['contact_height']} {c['zone']} fl={c['flight_s']}"
        cv2.rectangle(crop, (0, 0), (360, 22), (0, 0, 0), -1)
        cv2.putText(crop, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        images.append(crop)
    while len(images) % 5:
        images.append(np.zeros((360, 360, 3), np.uint8))
    rows = [np.hstack(images[k:k + 5]) for k in range(0, len(images), 5)]
    cv2.imwrite(str(RUNS / f"shots-{name}.jpg"), np.vstack(rows))


def main():
    for name in sys.argv[1:] or list(SPANS):
        xy, frames, shots = analyse(name)
        (RUNS / f"shots-{name}.json").write_text(json.dumps({"span": SPANS[name], "detected_frames": int((~np.isnan(xy[:, 0])).sum()),
                                                             "frames": len(xy), "shots": shots}, indent=1), encoding="utf-8")
        sheet(name, xy, frames, shots)
        print(name, "visible", int((~np.isnan(xy[:, 0])).sum()), "/", len(xy))
        for c in shots:
            print(f"  {c['time_s']:7.2f} {c['role']:4} {str(c['contact_height']):8} {c['zone']:5} m={c['hitter_m']} fl={c['flight_s']} "
                  f"apex={c['apex_y_px']} v={c['exit_speed_px_s']} dy={c['dy_out_px_s']} rz={c.get('receiver_zone')} click={c['audio_click_offset_s']}")


if __name__ == "__main__":
    main()

