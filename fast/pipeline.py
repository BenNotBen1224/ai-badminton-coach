"""Fast two-player pose pipeline: segment-parallel, pose-guided tracking.

A few threads decode; S workers each own a contiguous segment of frames. Inside a
segment, people are found by a detector at the start and every K frames, and
followed in between with boxes derived from their previous skeleton. Segments
are stitched by box overlap at the boundaries. The two people who move the most
are labeled as the near and far players. The court lines are found automatically
(court.py) while the models load; the camera must be fixed behind a baseline.

The pose model is RTMPose-s fine-tuned on badminton crops labelled by larger teacher
models (../train: label.py, train.py; log in ft1.json). Stock rtmpose-s.onnx still works.

usage: python pipeline.py VIDEO OUT.json [--segments 8] [--every 16] [--court x0,y0,...,x3,y3]
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

if __name__ == "__main__":
    # openvino imports its model converter (0.35 s) on startup; our models are already converted
    sys.modules.setdefault("openvino.tools.ovc", None)

import cv2
import numpy as np
import openvino as ov

from court import COURT_L, COURT_W, court_from_video
from detect import Court, Detector
from ovcache import infer
from pose import DEFAULT_POSE, DEFAULT_SMALL_CONFIDENCE, DEFAULT_SMALL_MIN_JOINT_CONFIDENCE, DEFAULT_SMALL_POSE, INPUT_H, INPUT_W, Pose, crop_transform, decode

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights"
_ALIVE = []


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


TORSO = [5, 6, 11, 12]
EXPANSION = 1.25  # pose-derived box size relative to the skeleton's extent


def pose_to_box(k, expansion=None, min_conf=0.3, mode="extent"):
    """Box around a skeleton. 'torso' re-centres it horizontally on shoulders+hips, so a stray
    limb point (e.g. a neighbour's arm) cannot drag the next crop off the player."""
    expansion = expansion or EXPANSION
    good = k[:, 2] >= min_conf
    pts = k[good, :2] if good.sum() >= 4 else k[:, :2]
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    half = (hi - lo) / 2
    torso = k[TORSO][good[TORSO], 0]
    if mode != "extent" and len(torso) >= 2:
        tx = float(torso.mean())
        if mode == "sym":
            half[0] = max(tx - lo[0], hi[0] - tx)
        c[0] = tx
    return np.concatenate([c - half * expansion, c + half * expansion]).tolist()


def torso_center(k, min_conf=0.3):
    pts = k[TORSO][k[TORSO, 2] >= min_conf, :2]
    return pts.mean(0) if len(pts) >= 2 else None


CUT_LEVEL, CUT_FRAC = 30, 0.022  # an edit cut moves >2.2% of pixels by >30 levels at once; live play peaks at 1.6%


def cut_sig(image):
    return image[::16, ::16, 1].astype(np.int16)


def is_cut(a, b):
    return a is not None and float((np.abs(a - b) > CUT_LEVEL).mean()) > CUT_FRAC


def thumb(crop):
    """8x8-pixel block means of a pose crop, for change detection."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (INPUT_W // 8, INPUT_H // 8), interpolation=cv2.INTER_AREA).astype(np.int16)


class Models:
    """All models, compiled side by side (loading from the cache is mostly single-threaded)."""

    def __init__(self, pose_onnx, start_det_onnx, det_onnx, streams, pose_small_onnx=None):
        core = ov.Core()
        core.set_property({"CACHE_DIR": str(HERE / "cache")})
        self.core, self.pose_onnx, self.det_onnx = core, pose_onnx, det_onnx
        self.pose_small_onnx, self._one = pose_small_onnx, None
        jobs = {"pose": lambda: Pose(pose_onnx, core=core, streams=streams, dynamic=True),
                "start_det": lambda: Detector(start_det_onnx, core=core, streams=streams),
                "det": lambda: Detector(det_onnx, core=core, streams=streams)}
        if pose_small_onnx:
            jobs["pose_small"] = lambda: Pose(pose_small_onnx, core=core, streams=streams, dynamic=True)
        ex = ThreadPoolExecutor(len(jobs))
        futures = {k: ex.submit(f) for k, f in jobs.items()}
        ex.shutdown(wait=False)
        m = {k: f.result() for k, f in futures.items()}
        self.pose, self.pose_small = m["pose"], m.get("pose_small")
        self.start_det, self.det = m["start_det"], m["det"]

    @property
    def one(self):
        """Load latency-mode models only when a segment boundary needs a follower."""
        if self._one is None:
            jobs = {"pose": lambda: Pose(self.pose_onnx, core=self.core, hint="LATENCY", dynamic=True),
                    "det": lambda: Detector(self.det_onnx, core=self.core, hint="LATENCY")}
            if self.pose_small_onnx:
                jobs["pose_small"] = lambda: Pose(self.pose_small_onnx, core=self.core, hint="LATENCY", dynamic=True)
            with ThreadPoolExecutor(len(jobs)) as ex:
                futures = {k: ex.submit(f) for k, f in jobs.items()}
                m = {k: f.result() for k, f in futures.items()}
            self._one = SimpleNamespace(pose=m["pose"], pose_small=m.get("pose_small"),
                                        start_det=self.start_det, det=m["det"])
        return self._one


class Reader:
    """Frames on demand from `start`, for players carried past their segment's end.

    A helper thread decodes a few frames ahead so decoding overlaps the follower's pose work.
    """

    def __init__(self, video, start, ahead=16):
        self.q, self.done = queue.Queue(maxsize=ahead), threading.Event()
        self.error = None
        threading.Thread(target=self._fill, args=(video, start), daemon=True).start()

    def _fill(self, video, start):
        try:
            self._read(video, start)
        except Exception as e:  # noqa: BLE001 - re-raised by get()
            self.error = e
            self.q.put((None, None))

    def _read(self, video, start):
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        i = start
        while not self.done.is_set():
            ok, frame = cap.read()
            self.q.put((i, frame if ok else None))
            if not ok:
                return
            i += 1

    def get(self):
        i, frame = self.q.get()
        if frame is None:
            if self.error is not None:
                raise self.error
            raise RuntimeError(f"could not read frame {i}")
        return i, frame

    def close(self):
        self.done.set()
        while not self.q.empty():  # unblock a pending put so the thread sees `done`
            self.q.get_nowait()


class Segment(threading.Thread):
    """Tracks people through frames [first, last).

    With `follow`, it only carries the given `tracks` forward (no new people) and lets a
    track go as soon as `follow(idx)` - the regular segment's rows - already has that person.
    """

    def __init__(self, models, court, first, last, every, frames, reuse=8, max_hold=16, box="torso",
                 predict=False, refine=0.0, small_h=0.0, official_n=0, small_confidence=0.0,
                 tracks=None, follow=None):
        super().__init__(daemon=True)
        self.m, self.court, self.first, self.last, self.every = models, court, first, last, every
        self.reuse, self.max_hold, self.box, self.predict = reuse, max_hold, box, predict
        self.refine, self.small_h, self.official_n = refine, small_h, official_n
        self.small_confidence = small_confidence
        self.ignore = []  # detector boxes of officials: never tracked again
        self.official_rows = set()  # (frame, track) rows that showed someone was an official
        self.init_tracks, self.follow, self.met, self.final_tracks = tracks or {}, follow, [], {}
        self.cuts, self.first_sig, self.last_sig = [], None, None  # edit cuts: nobody is tracked across one
        self.frames = frames  # queue of (index, image)
        self.out = {}  # frame_index -> list of (local_track, box, det_conf, joints[17,3])
        self.pose_req = models.pose.compiled.create_infer_request()
        self.pose_small_req = models.pose_small.compiled.create_infer_request() if models.pose_small else None
        self.start_req = models.start_det.compiled.create_infer_request()
        self.det_req = models.det.compiled.create_infer_request()
        self.error = None
        self.calls = {"start": 0, "det": 0, "pose": 0, "pose_small": 0, "reused": 0, "refined": 0,
                      "officials": 0, "fallback": 0, "tail_fallback": 0,
                      "pose_requests": 0, "pose_small_requests": 0}
        self.secs = {"wait": 0.0, "det": 0.0, "pose": 0.0}

    def small(self, box):
        return self.pose_small_req is not None and box[3] - box[1] < self.small_h

    def detect(self, image, start):
        det, req = (self.m.start_det, self.start_req) if start else (self.m.det, self.det_req)
        self.calls["start" if start else "det"] += 1
        t = time.perf_counter()
        raw = det(image, self.court.roi, request=req, min_conf=0.25)
        self.secs["det"] += time.perf_counter() - t
        found = sorted(((b, s) for b, s in raw if self.court.contains_feet(b)
                        and all(iou(b, g) <= 0.5 for g in self.ignore)), key=lambda d: -d[1])
        kept = []
        for b, s in found:  # NMS-free models still emit the odd low-score twin box
            if all(iou(b, k) <= 0.6 for k, _ in kept):
                kept.append((b, s))
        return kept

    def poses(self, image, boxes):
        crops, meta = [], []
        for b in boxes:
            m, c, s = crop_transform(b)
            crops.append(cv2.warpAffine(image, m, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR))
            meta.append((c, s))
        flags = [False] * len(boxes) if self.small_confidence else [self.small(b) for b in boxes]
        return self.pose_crops(crops, meta, flags)

    def route_poses(self, crops, meta, boxes, tracks):
        if not self.small_confidence:
            return self.pose_crops(crops, meta, [self.small(b) for b in boxes])
        flags = [not t.get("large_left", 0) for t in tracks]
        kps = self.pose_crops(crops, meta, flags)
        mean_weak = [flag and float(k[:, 2].mean()) < self.small_confidence
                     for flag, k in zip(flags, kps)]
        redo = [i for i, (flag, k) in enumerate(zip(flags, kps))
                if flag and (mean_weak[i] or float(k[:, 2].min()) < DEFAULT_SMALL_MIN_JOINT_CONFIDENCE)]
        for flag, t in zip(flags, tracks):
            if not flag:
                t["large_left"] -= 1
        if redo:
            refined = self.pose_crops([crops[i] for i in redo], [meta[i] for i in redo], [False] * len(redo))
            self.calls["fallback"] += len(redo)
            self.calls["tail_fallback"] += sum(not mean_weak[i] for i in redo)
            for i, k in zip(redo, refined):
                kps[i] = k
                if mean_weak[i]:
                    # Globally weak crops stay on s; isolated weak joints retry only this frame.
                    tracks[i]["large_left"] = 8
        return kps

    def pose_crops(self, crops, meta, small):
        t = time.perf_counter()
        out = [None] * len(crops)
        for flag, req in ((False, self.pose_req), (True, self.pose_small_req)):
            idx = [i for i, s in enumerate(small) if s == flag]
            if not idx:
                continue
            res = infer(req, np.stack([crops[i] for i in idx]))
            self.calls["pose_small_requests" if flag else "pose_requests"] += 1
            sx, sy = res[0], res[1]
            for j, i in enumerate(idx):
                out[i] = decode(sx[j], sy[j], *meta[i])
        self.secs["pose"] += time.perf_counter() - t
        self.calls["pose"] += len(crops)
        self.calls["pose_small"] += sum(small)
        return out

    def backfill(self, tid, box, conf, recent):
        """A person first found now was probably there a few frames earlier: trace them back."""
        for idx, image in reversed(recent):
            if any(iou(box, row[1]) > 0.5 for row in self.out[idx]) or not self.court.contains_feet(box):
                return
            k = self.poses(image, [box])[0]
            if float(np.median(k[:, 2])) < 0.3:
                return
            self.out[idx].append((tid, list(box), conf, k))
            box = pose_to_box(k, mode=self.box)

    def run(self):
        try:
            self._run()
        except Exception as e:  # surfaced by the main thread
            self.error = e
        self.finished = time.perf_counter()

    def is_official(self, t):
        """Stood still for the last official_n frames, feet well outside the court lines."""
        trail = t.get("trail", [])[-self.official_n:]
        if not self.official_n or len(trail) < self.official_n:
            return False
        a = np.float32(trail)
        wander = float((a[:, :2].max(0) - a[:, :2].min(0)).max())
        return wander < STILL * float(np.median(a[:, 2])) and float(np.median(a[:, 3])) > OUTSIDE_M

    def remember(self, t, box, k, idx):
        s = k[[5, 6]]
        s = s[s[:, 2] >= 0.25, :2]
        x, y = self.court.feet_metres(box)
        out = float(np.hypot(max(0.0, -x, x - COURT_W), max(0.0, -y, y - COURT_L)))
        p = s.mean(0) if len(s) else ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        t.setdefault("trail", []).append((p[0], p[1], max(box[3] - box[1], 1), out, idx))
        del t["trail"][:-self.official_n or None]

    def _run(self):
        tracks = dict(self.init_tracks)  # id -> {"box", "conf", "weak", "ref" (thumb of last posed crop), "kp", "age"}
        next_id = max(tracks, default=-1) + 1
        recent = []  # frames since the last detection, for backfill
        for idx in range(self.first, self.last):
            t = time.perf_counter()
            i, image = self.frames.get()
            self.secs["wait"] += time.perf_counter() - t
            if idx == self.first:
                self.started = time.perf_counter()
            assert i == idx
            local = idx - self.first
            sig = cut_sig(image)
            if is_cut(self.last_sig, sig):
                self.cuts.append(idx)
                if self.follow is not None:
                    break
                tracks, recent = {}, []
            self.first_sig = sig if self.first_sig is None else self.first_sig
            self.last_sig = sig
            lost = any(t["weak"] for t in tracks.values())
            if local % self.every == 0 or lost or not tracks:
                for tid in [tid for tid, t in tracks.items() if self.is_official(t)]:
                    t = tracks.pop(tid)
                    self.ignore.append(t.get("det_box"))
                    self.official_rows.update((int(e[4]), tid) for e in t["trail"])
                    self.calls["officials"] += 1
                self.ignore = [g for g in self.ignore if g is not None]
                dets = self.detect(image, start=(local == 0 or not tracks) and self.follow is None)
                pairs = sorted(((iou(t["box"], b), tid, j) for tid, t in tracks.items()
                                for j, (b, _) in enumerate(dets)), reverse=True)
                used, matched = set(), set()
                for o, tid, j in pairs:
                    if o < 0.3:
                        break
                    if tid in matched or j in used:
                        continue
                    matched.add(tid)
                    used.add(j)
                    t = tracks[tid]
                    t["box"], t["det_box"], t["conf"], t["weak"] = dets[j][0], dets[j][0], dets[j][1], False
                for tid, t in tracks.items():
                    if tid not in matched and t["weak"]:
                        t["drop"] = True
                for j, (b, s) in enumerate(dets):
                    if self.follow is None and j not in used and all(iou(b, t["box"]) <= 0.5 for t in tracks.values()):
                        tracks[next_id] = {"box": b, "det_box": b, "conf": s, "weak": False, "ref": None}
                        if local:
                            self.backfill(next_id, b, s, recent)
                        next_id += 1
                tracks = {k: v for k, v in tracks.items() if not v.get("drop")}
                recent = []
            recent.append((idx, image))
            self.out[idx] = []
            if self.follow is not None:  # the regular segment has this person now: hand them over
                others = self.follow(idx)
                for tid in list(tracks):
                    o, otid = max(((iou(tracks[tid]["box"], r[1]), r[0]) for r in others), default=(0, None))
                    if o >= 0.3:
                        self.met.append((tid, otid, idx))
                        del tracks[tid]
                if not tracks:
                    break
            if not tracks:
                continue
            ids = sorted(tracks)
            used = {tid: list(tracks[tid]["box"]) for tid in ids}
            todo, crops, meta = [], [], []
            for tid in ids:
                t = tracks[tid]
                t["age"] = t.get("age", 0) + 1
                # same pixels where we last ran pose -> the model would say the same thing
                if self.reuse and t["ref"] is not None and t["age"] <= self.max_hold:
                    region = cv2.warpAffine(image, t["ref_m"], (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
                    if np.abs(thumb(region) - t["ref"]).max() <= self.reuse:
                        self.calls["reused"] += 1
                        continue
                m, c, s = crop_transform(t["box"])
                crop = cv2.warpAffine(image, m, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
                todo.append(tid)
                crops.append(crop)
                meta.append((c, s))
                t["ref"], t["ref_m"], t["age"] = thumb(crop), m, 0
            if todo:
                kps = self.route_poses(crops, meta, [used[tid] for tid in todo], [tracks[tid] for tid in todo])
                if self.refine:  # crop was off the person -> look again where the skeleton says they are
                    redo = [i for i, (tid, k) in enumerate(zip(todo, kps))
                            if iou(used[tid], pose_to_box(k, mode=self.box)) < self.refine]
                    if redo:
                        self.calls["refined"] += len(redo)
                        again = self.poses(image, [pose_to_box(kps[i], mode=self.box) for i in redo])
                        for i, k in zip(redo, again):
                            kps[i] = k
                for tid, k in zip(todo, kps):
                    t = tracks[tid]
                    t["kp"] = k
                    t["weak"] = float(np.median(k[:, 2])) < 0.3
                    t["box"] = pose_to_box(k, mode=self.box)
                    if self.predict:  # the next crop goes where the torso is heading
                        tc = torso_center(k)
                        if tc is not None and t.get("tc_idx") == idx - 1:
                            v = (tc - t["tc"]).tolist()
                            t["box"] = [t["box"][0] + v[0], t["box"][1] + v[1], t["box"][2] + v[0], t["box"][3] + v[1]]
                        t["tc"], t["tc_idx"] = tc, idx if tc is not None else None
            self.out[idx] = [(tid, used[tid], tracks[tid]["conf"], tracks[tid]["kp"])
                             for tid in ids] + self.out[idx]
            if self.official_n:
                for tid in ids:
                    self.remember(tracks[tid], used[tid], tracks[tid]["kp"], idx)
            # two tracks collapsing onto one person: keep the older one
            for a in ids:
                for b in ids:
                    if a < b and a in tracks and b in tracks and iou(tracks[a]["box"], tracks[b]["box"]) > 0.6:
                        del tracks[b]
        self.final_tracks = tracks


def stitch(segments):
    """Map (segment, local id) -> global id by box overlap across each boundary."""
    gid, next_gid = {}, 0
    for si, seg in enumerate(segments):
        if si and is_cut(segments[si - 1].last_sig, seg.first_sig):
            seg.cuts.insert(0, seg.first)
        prev_rows = segments[si - 1].out.get(seg.first - 1, []) if si and seg.first not in seg.cuts else []
        taken = set()
        for tid, box, _, _ in seg.out.get(seg.first, []):
            best = max(((iou(box, pb), ptid) for ptid, pb, _, _ in prev_rows
                        if gid[(si - 1, ptid)] not in taken), default=(0, None))
            if best[0] >= 0.3:
                gid[(si, tid)] = gid[(si - 1, best[1])]
                taken.add(gid[(si, tid)])
        for rows in seg.out.values():
            for tid, *_ in rows:
                if (si, tid) not in gid:
                    gid[(si, tid)] = next_gid
                    next_gid += 1
    return gid


def joint_record(k, threshold=0.25):
    return [[round(float(x), 2), round(float(y), 2), round(float(c), 4)] if c >= threshold else None
            for x, y, c in k]


def player_record(track_id, box, conf, k, official=False):
    return {"track_id": track_id, "bbox_xyxy": [round(v, 2) for v in box],
            "box_confidence": round(conf, 4), "joints": joint_record(k), **({"official": True} if official else {})}


def handoff(workers, gid, frames, players, models, court, video, n, seg_of, every, *opts):
    """Carry players across segment boundaries the way one long tracker would.

    A segment cold-starts with a detector, which can miss a small far player standing at
    the back. A player still tracked when the previous segment ends but absent from the
    next one is followed on by pose alone until the regular segments have them again.
    """
    followers = []
    for si in range(len(workers) - 1):
        wk, nxt = workers[si], workers[si + 1]
        if nxt.first in nxt.cuts:
            continue
        carried = {gid[(si + 1, r[0])] for r in nxt.out.get(nxt.first, [])}
        orphans = {tid: t for tid, t in wk.final_tracks.items()
                   if gid.get((si, tid)) in players and gid[(si, tid)] not in carried}
        if orphans:
            f = Segment(models.one, court, nxt.first, n, every, Reader(video, nxt.first), *opts,
                        tracks=orphans, follow=lambda idx: workers[seg_of[idx]].out[idx])
            f.origin = si
            followers.append(f)
    for f in followers:
        f.start()
    for f in followers:
        f.join()
        f.frames.close()
        if f.error:
            raise f.error
    first_seen = {}
    for fr in frames:
        for p in fr["players"]:
            first_seen.setdefault(p["track_id"], fr["frame_index"])
    remap = {}  # a track the regular segment found later is the followed player: same id
    for f in followers:
        for tid, otid, idx in f.met:
            other = gid[(int(seg_of[idx]), otid)]
            if f.first < first_seen[other] and first_seen[other] >= idx - 2 * every:
                remap[other] = gid[(f.origin, tid)]
        for idx, rows in f.out.items():
            frames[idx]["players"] += [{**player_record(gid[(f.origin, tid)], b, c, k, (idx, tid) in f.official_rows),
                                        "followed": True} for tid, b, c, k in rows]
    for fr in frames:
        mine = {}
        for p in fr["players"]:
            while p["track_id"] in remap:
                p["track_id"] = remap[p["track_id"]]
            q = mine.get(p["track_id"])
            if q is None or q.get("followed"):  # the regular segment's row wins a tie
                mine[p["track_id"]] = p
        fr["players"] = list(mine.values())
        for p in fr["players"]:
            p.pop("followed", None)
    return followers


def relink(frames, every, cuts=()):
    """A track that dies and is reborn moments later where it was is one person.

    When players overlap on screen, the hidden player's skeleton can slide onto the other
    one and be merged away; the next detection then finds them again under a new id.
    Never across an edit cut: there, being in the same place says nothing about who it is.
    """
    rows = {}
    for f in frames:
        for p in f["players"]:
            rows.setdefault(p["track_id"], []).append((f["frame_index"], p["bbox_xyxy"]))
    start = {t: r[0][0] for t, r in rows.items()}
    end = {t: r[-1][0] for t, r in rows.items()}
    remap, taken = {}, set()
    for new in sorted(rows, key=start.get):
        s, box = start[new], rows[new][0][1]
        best, link = 0.3, None
        for old in rows:
            if old == new or old in taken or not (start[old] < s and s - 2 * every <= end[old] <= s + 2):
                continue
            if any(end[old] < c <= s for c in cuts):
                continue
            o = max(iou(box, b) for i, b in rows[old] if i >= end[old] - every)
            if o >= best:
                best, link = o, old
        if link is not None:
            taken.add(link)
            remap[new] = remap.get(link, link)
    for f in frames:  # where old and new overlap, the newer track is the one still on its person
        keep = {}
        for p in f["players"]:
            gid = remap.get(p["track_id"], p["track_id"])
            if gid not in keep or start[p["track_id"]] > start[keep[gid]["track_id"]]:
                keep[gid] = p
        f["players"] = [p for p in f["players"] if keep[remap.get(p["track_id"], p["track_id"])] is p]
        for p in f["players"]:
            p["track_id"] = remap.get(p["track_id"], p["track_id"])
    return {int(k): int(v) for k, v in remap.items()}


STILL = 0.1  # an official's shoulders move less than this many body heights within 2 s...
OUTSIDE_M = 0.5  # ...and they stand at least this far outside the court lines
OFFICIAL_S = 1.0  # while tracking, 1 s of that is enough to stop following (and posing) them
SMOOTH_S = 0.5  # feet are medians over +-0.5 s, so a jump does not change a player's half


def windows(a, half):
    return np.lib.stride_tricks.sliding_window_view(np.pad(a, ((half, half), (0, 0)), mode="edge"),
                                                    2 * half + 1, axis=0)


def stillness(a, n):
    """How far points wander within the n frames before, or the n frames after, each row
    (whichever is calmer). Too little data on both sides counts as moving."""
    p = np.pad(np.float32(a), ((n, n), (0, 0)), constant_values=np.nan)
    w = np.lib.stride_tricks.sliding_window_view(p, n + 1, axis=0)
    r = (np.nanmax(w, axis=2) - np.nanmin(w, axis=2)).max(axis=1)
    r[np.isnan(w[:, 0]).sum(axis=1) > n // 2] = np.inf
    return np.minimum(r[:len(a)], r[n:n + len(a)])


def label_players(frames, court, fps):
    """Players run around in the court; officials and line judges sit still outside it.

    Per frame and per half, a person standing still outside the lines is not a player; of
    the rest, the track with the most travel (foot path in body heights) is that half's
    player. The roles follow the person, so a track that slides from a player onto an
    official loses its role there and the player's new track takes it. Stillness is read
    from the shoulders: an official's feet are often hidden behind boards or a chair.
    """
    smooth, still_n = max(1, round(SMOOTH_S * fps)), max(2, round(2 * fps))
    rows = {}
    for f in frames:
        for p in f["players"]:
            rows.setdefault(p["track_id"], []).append(p)
    travel, state = {}, {}
    for t, ps in rows.items():
        b = np.float32([p["bbox_xyxy"] for p in ps])
        height = np.maximum(b[:, 3] - b[:, 1], 1)
        foot = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], 1)
        travel[t] = float((np.linalg.norm(np.diff(foot, axis=0), axis=1) / height[1:]).sum())
        med = np.median(windows(cv2.perspectiveTransform(foot[None], court.to_court)[0], smooth), axis=2)
        out = np.hypot(np.maximum.reduce([0 * med[:, 0], -med[:, 0], med[:, 0] - COURT_W]),
                       np.maximum.reduce([0 * med[:, 1], -med[:, 1], med[:, 1] - COURT_L]))
        shoulders = np.float32([np.mean([j[:2] for j in s], 0) if s else ((x0 + x1) / 2, (y0 + y1) / 2)
                                for (x0, y0, x1, y1), s in
                                zip(b, ([j for j in (p["joints"][5], p["joints"][6]) if j] for p in ps))])
        still = stillness(shoulders, still_n) < STILL * float(np.median(height))
        seen = np.array([bool(p.get("official")) for p in ps])  # the tracker saw them stand still outside
        for p, far, official in zip(ps, med[:, 1] < COURT_L / 2, still & (out > OUTSIDE_M) | seen):
            state[id(p)] = None if official else "far" if far else "near"
    held = {}
    for f in frames:
        best = {}
        for p in f["players"]:
            p["role"] = None
            side = state[id(p)]
            if side and (side not in best or travel[p["track_id"]] > travel[best[side]["track_id"]]):
                best[side] = p
        for side, p in best.items():
            p["role"] = side
            held[p["track_id"]] = travel[p["track_id"]]
    return {t: round(v, 1) for t, v in sorted(held.items(), key=lambda kv: -kv[1])}


def run(video, out, segments=8, every=16, court_lines=None,
        pose_onnx=DEFAULT_POSE, start_det=WEIGHTS / "yolo26s-448x768.onnx",
        det=WEIGHTS / "yolo26n-384x640.onnx", reuse=8, max_hold=16, box="torso", predict=False,
        refine=0.0, pose_small=DEFAULT_SMALL_POSE, small_frac=1 / 6, decoders=2, official_s=OFFICIAL_S,
        small_confidence=DEFAULT_SMALL_CONFIDENCE):
    if not np.isfinite(small_confidence) or small_confidence < 0:
        raise ValueError("small_confidence must be finite and non-negative")
    if small_confidence and pose_small is None:
        raise ValueError("confidence routing needs a small pose model")
    t0, c0 = time.perf_counter(), time.process_time()
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    bounds = np.linspace(0, n, segments + 1).round().astype(int)
    queues = [queue.Queue() for _ in range(segments)]
    seg_of = np.searchsorted(bounds[1:], np.arange(n), side="right")
    decoded, decode_errors = {}, []

    def decode(a, b):  # starts before the models load, so decoding overlaps compilation
        try:
            read(a, b)
        except Exception as e:  # noqa: BLE001 - re-raised on the main thread
            decode_errors.append(e)

    def read(a, b):
        cap = cv2.VideoCapture(str(video))
        if a:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(a))
        i = a
        while i < b:
            ok, frame = cap.read()
            if not ok:
                break
            queues[seg_of[i]].put((i, frame))
            i += 1
        decoded[a] = (i - a, time.perf_counter() - t0)

    # one H.264 stream decodes no faster than ~3 threads allow; with several, the later
    # segments get their first frames early instead of after everything before them
    groups = np.array_split(np.arange(segments), decoders)
    spans = [(int(bounds[g[0]]), int(bounds[g[-1] + 1])) for g in groups if len(g)]
    readers = [threading.Thread(target=decode, args=span, daemon=True) for span in spans]
    for r in readers:
        r.start()
    located = {}

    def locate():
        try:
            located["court"] = court_from_video(video)
        except Exception as e:  # noqa: BLE001 - re-raised on the main thread
            located["error"] = e

    finder = threading.Thread(target=locate, daemon=True)
    if court_lines is None:
        finder.start()
    models = Models(pose_onnx, start_det, det, streams=segments, pose_small_onnx=pose_small)
    t_load = time.perf_counter() - t0
    if court_lines is None:
        finder.join()
        if "error" in located:
            raise located["error"]
        court_lines, court_info = located["court"]
    else:
        court_info = {"given": True}
    t_court = time.perf_counter() - t0
    court = Court(court_lines, w, h)
    opts = (reuse, max_hold, box, predict, refine, small_frac * h, round(official_s * fps), small_confidence)
    workers = [Segment(models, court, a, b, every, q, *opts)
               for a, b, q in zip(bounds[:-1], bounds[1:], queues)]
    for wk in workers:
        wk.start()
    for r in readers:
        r.join()
    if decode_errors:
        raise decode_errors[0]
    i, t_decoded = sum(c for c, _ in decoded.values()), max(t for _, t in decoded.values())
    for wk in workers:
        wk.join()
        if wk.error:
            raise wk.error
    if i != n:
        raise RuntimeError(f"decoded {i} of {n} frames")
    gid = stitch(workers)
    frames = []
    for si, wk in enumerate(workers):
        for idx in range(wk.first, wk.last):
            frames.append({"frame_index": idx, "timestamp_seconds": round(idx / fps, 4),
                           "players": [player_record(gid[(si, tid)], box, conf, k, (idx, tid) in wk.official_rows)
                                       for tid, box, conf, k in wk.out[idx]]})
    t_follow = time.perf_counter()
    followers = handoff(workers, gid, frames, label_players(frames, court, fps), models, court, video, n, seg_of,
                        every, *opts)
    t_follow = time.perf_counter() - t_follow
    cuts = sorted(int(c) for wk in workers for c in wk.cuts)
    relinked = relink(frames, every, cuts)
    player_tracks = label_players(frames, court, fps)
    calls = {k: sum(w.calls[k] for w in [*workers, *followers]) for k in workers[0].calls}
    header = {"schema_version": 3, "source": {"file": Path(video).name, "width": w, "height": h,
                                              "fps": fps, "frame_count": n},
              "model": {"pose": Path(pose_onnx).name, "pose_small": pose_small and Path(pose_small).name,
                        "small_below_frame_frac": small_frac if pose_small and not small_confidence else None,
                        "small_min_mean_confidence": small_confidence or None,
                        "small_min_joint_confidence": DEFAULT_SMALL_MIN_JOINT_CONFIDENCE if small_confidence else None,
                        "large_hold_inferences": 8 if small_confidence else None,
                        "start_detector": Path(start_det).name,
                        "detector": Path(det).name, "segments": segments, "detect_every": every,
                        "reuse_threshold": reuse, "max_hold": max_hold, "box": box, "predict": predict,
                        "refine": refine, "official_s": official_s,
                        "joint_threshold": 0.25},
              "court_lines_normalized": court_lines, "court_fit": court_info,
              "play_area_normalized": (np.float32(court.poly) / [w, h]).round(4).tolist(),
              "cuts": cuts, "player_tracks_travel": player_tracks}
    header["processing_counts"] = {
        "primary_fresh_pose_crops": calls["pose"] - calls["pose_small"],
        "small_fresh_pose_crops": calls["pose_small"],
        "primary_inference_requests": calls["pose_requests"],
        "small_inference_requests": calls["pose_small_requests"],
        "reused_pose_rows": calls["reused"], "small_to_primary_fallbacks": calls["fallback"],
        "weak_joint_only_fallbacks": calls["tail_fallback"],
    }
    Path(out).write_text(json.dumps({**header, "frames": frames}), "utf-8")
    _ALIVE.append(models)  # freeing compiled models joins OpenVINO's thread pools (~0.25 s); the CLI exits first
    total = time.perf_counter() - t0
    return {"total_s": round(total, 2), "load_s": round(t_load, 2), "court_s": round(t_court, 2),
            "decode_done_s": round(t_decoded, 2),
            "follow_s": round(t_follow, 2), "relinked": relinked, "cuts": cuts,
            "followers": [[f.origin, int(f.first), max((int(i) for i, r in f.out.items() if r), default=None),
                           [[int(v) for v in m] for m in f.met]] for f in followers],
            "crops": sum(len(f["players"]) for f in frames), "frames": len(frames),
            "workers": [[round(wk.started - t0, 2), round(wk.finished - t0, 2)] for wk in workers],
            "cpu_s": round(time.process_time() - c0, 1),
            "secs": {k: [round(wk.secs[k], 2) for wk in workers] for k in workers[0].secs},
            "calls": calls}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--every", type=int, default=16, help="detector cadence; skeletons carry people in between")
    ap.add_argument("--reuse", type=float, default=8, help="max thumbnail change to reuse a pose; 0 = off")
    ap.add_argument("--max-hold", type=int, default=16)
    ap.add_argument("--box", choices=["extent", "torso", "sym"], default="torso")
    ap.add_argument("--pose", default=str(DEFAULT_POSE))
    ap.add_argument("--det", default=str(WEIGHTS / "yolo26n-384x640.onnx"), help="periodic re-detection model")
    ap.add_argument("--predict", action="store_true", help="shift each crop by the torso's last motion")
    ap.add_argument("--refine", type=float, default=0.0,
                    help="rerun pose on the skeleton's own box when it overlaps the crop box less than this")
    ap.add_argument("--expand", type=float, default=EXPANSION, help="pose-derived box expansion")
    ap.add_argument("--pose-small", default=str(DEFAULT_SMALL_POSE), help="tiny model for confidence routing")
    ap.add_argument("--small-frac", type=float, default=1 / 6, help="box height / frame height")
    ap.add_argument("--small-confidence", type=float, default=DEFAULT_SMALL_CONFIDENCE,
                    help="low mean confidence uses --pose for 8 inferences; a weak joint retries only this frame")
    ap.add_argument("--single-model", action="store_true", help="disable tiny-model routing and use --pose alone")
    ap.add_argument("--decoders", type=int, default=2, help="parallel video decoders")
    ap.add_argument("--court", default=None,
                    help="court line corners far-left,far-right,near-right,near-left as 8 numbers in 0..1; "
                         "default: find them")
    ap.add_argument("--official-s", type=float, default=OFFICIAL_S,
                    help="stop tracking someone standing still this long outside the lines; 0 = off")
    a = ap.parse_args()
    if a.single_model:
        a.pose_small, a.small_confidence = None, 0.0
    EXPANSION = a.expand
    lines = None
    if a.court:
        v = [float(x) for x in a.court.split(",")]
        if len(v) != 8:
            ap.error("--court needs 8 numbers")
        lines = [v[i:i + 2] for i in range(0, 8, 2)]
    print(json.dumps(run(a.video, a.out, a.segments, a.every, court_lines=lines, pose_onnx=a.pose, det=a.det,
                         reuse=a.reuse,
                         max_hold=a.max_hold, box=a.box, predict=a.predict, refine=a.refine,
                         pose_small=a.pose_small, small_frac=a.small_frac, decoders=a.decoders,
                         official_s=a.official_s, small_confidence=a.small_confidence)), flush=True)
    os._exit(0)  # results are written; skip the ~0.4 s interpreter/OpenVINO teardown
