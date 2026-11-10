"""Warm resident pose service: preview, refined, and window tiers.

Preview samples reference frames, poses those crops, and interpolates the rest.
Nothing here is cached by video content. Settings are global, not per clip.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from . import decode, sampler
except ImportError:
    import decode
    import sampler

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "frozen"
CACHE = ROOT / "cache"

# One operating point for every clip. Change it only by measuring, then log the change.
EVERY_REF = 4
EXTRA_BIG = 2
DETECT_EVERY = 30
MAX_GAP = 48
DECODE_THREADS = 2
POSE_THREADS = None
POSE_STREAMS = 4
MAX_TRACKS = 4
GOP_PARALLEL = 2
POSE_SLOTS = 4
FOLLOW = "pose"
FLOW_BATCH = 4


def _boot():
    sys.modules.setdefault("openvino.tools.ovc", None)
    if str(FROZEN) not in sys.path:
        sys.path.insert(0, str(FROZEN))


def _frozen():
    _boot()
    import court
    import detect
    import ovcache
    import pipeline
    import pose
    return court, detect, ovcache, pipeline, pose


class TurboService:
    def __init__(self, models_dir, decode_method="serial"):
        self.models_dir = Path(models_dir)
        if decode_method not in ("serial", "gop"):
            raise ValueError(decode_method)
        self.decode_method = decode_method
        self._core = None
        self._pose = None
        self._det = None
        self._dense = None

    def warm_up(self, video_path=None):
        """Compile every model and run one dummy crop. Not part of the timed call."""
        self._preview_models()
        self._refined_models()
        self._touch()
        return {"warm": True, "video": None if video_path is None else Path(video_path).name}

    def analyze(self, video_path, out_json_path=None, tier="preview", windows=None, base_doc=None):
        if tier == "preview":
            return self._preview(video_path, out_json_path)
        if tier == "refined":
            return self._refined(video_path, out_json_path)
        if tier == "window":
            return self._window(video_path, out_json_path, windows, base_doc)
        raise ValueError(tier)

    def _core_or_new(self):
        if self._core is None:
            _boot()
            import openvino as ov
            core = ov.Core()
            CACHE.mkdir(exist_ok=True)
            core.set_property({"CACHE_DIR": str(CACHE)})
            self._core = core
        return self._core

    def _onnx(self, name):
        local = self.models_dir / name
        return local if local.is_file() else ROOT.parent / "fast" / "weights" / name

    def _preview_models(self):
        if self._pose is not None:
            return self._pose, self._det
        _, detect, _, _, pose = _frozen()
        core = self._core_or_new()
        self._pose = pose.Pose(
            self._onnx("rtmpose-t-ft3.onnx"), core=core, streams=POSE_STREAMS,
            threads=POSE_THREADS, dynamic=False)
        self._det = detect.Detector(
            self._onnx("yolo26n-384x640.onnx"), hint="LATENCY", core=core, threads=2)
        return self._pose, self._det

    def _refined_models(self):
        if self._dense is not None:
            return self._dense
        _, _, _, pipeline, _ = _frozen()
        self._dense = pipeline.Models(
            self._onnx("rtmpose-s-ft1.onnx"),
            self._onnx("yolo26s-448x768.onnx"),
            self._onnx("yolo26n-384x640.onnx"),
            streams=8,
            pose_small_onnx=self._onnx("rtmpose-t-ft3.onnx"))
        return self._dense

    def _touch(self):
        _, _, ovcache, _, pose = _frozen()
        pose_m, det = self._preview_models()
        req = pose_m.compiled.create_infer_request()
        ovcache.infer(req, np.zeros((1, pose.INPUT_H, pose.INPUT_W, 3), np.uint8))
        det(np.zeros((64, 64, 3), np.uint8), (0, 0, 64, 64))

    def _preview(self, video, out_path):
        court_mod, detect, ovcache, pipeline, pose = _frozen()
        pose_m, det = self._preview_models()
        t0 = time.perf_counter()
        box, stream, packets = decode.demux(video)
        n = int(stream.frames) or len(packets)
        fps = float(stream.average_rate)
        width, height = stream.codec_context.width, stream.codec_context.height
        ranges = decode.gop_ranges(packets)
        keep = decode.make_keep(EVERY_REF, decode.big_packet_pts(packets, ranges, EXTRA_BIG))
        box.close()

        located = {}
        trace = []
        # First real failure first. It is recorded before `stop` is set, so a
        # decode.Cancelled that follows can never become errors[0].
        errors = []
        stop = threading.Event()

        def fail(exc):
            errors.append(exc)
            stop.set()

        def mark(name):
            trace.append((name, round(time.perf_counter() - t0, 4)))

        def find_court():
            if os.environ.get("TURBO_CENSUS") == "1":
                located["thread"] = threading.get_native_id()
            t = time.perf_counter()
            try:
                located["court"] = court_mod.court_from_video(video)
            except Exception as exc:
                located["error"] = exc
                fail(exc)
            located["s"] = time.perf_counter() - t
            mark("court_done")

        finder = threading.Thread(target=find_court, daemon=True)
        finder.start()
        gop_rows = [None] * len(ranges)
        stats = [None] * len(ranges)
        census_rows = []
        decoded_n = 0
        # Unbounded: a pose worker waits on court, so a full queue would stall decode forever.
        kept_q = [queue.Queue() for _ in ranges]
        decode_box = {}
        # Pose slots are not decode contexts. Take one only after a frame is waiting,
        # so an idle GOP cannot hold a slot the decoded frames need.
        pose_slots = threading.Semaphore(POSE_SLOTS)

        def pose_worker(g):
            saw_end = False
            holding = False
            try:
                finder.join()
                if stop.is_set():
                    return
                if "error" in located:
                    raise located["error"]
                lines, _info = located["court"]
                court = detect.Court(lines, width, height)
                tracker = Tracker(
                    court, det, pose, ovcache, det_reqs[g], pose_reqs[g], DETECT_EVERY,
                    follow=FOLLOW)
                while True:
                    item = kept_q[g].get()
                    if stop.is_set():
                        saw_end = item is None
                        return
                    if item is None:
                        saw_end = True
                        tracker.finish()
                        break
                    if not holding:
                        mark(f"pose_ready_{g}")
                        pose_slots.acquire()
                        holding = True
                        # Another worker may have failed while this one waited for the slot.
                        if stop.is_set():
                            return
                        mark(f"pose_start_{g}")
                    tracker.step(item)
                gop_rows[g] = tracker.out
                stats[g] = (tracker.det_s, tracker.pose_s, tracker.crops, tracker.retrieve_s)
                census_rows.append({
                    "g": g,
                    "thread": threading.get_native_id(),
                    "steps": tracker.step_n,
                    "kept": len(tracker.out),
                    "crops": tracker.crops,
                    "detect_n": tracker.detect_n,
                    "detect_s": round(tracker.det_s, 4),
                    "pose_calls": tracker.pose_calls,
                    "pose_s": round(tracker.pose_s, 4),
                    "retrieve_s": round(tracker.retrieve_s, 4),
                    "flow_n": tracker.flow_n,
                    "flow_s": round(tracker.flow_s, 4),
                })
                if holding:
                    mark(f"pose_done_{g}")
            except Exception as exc:
                fail(exc)
            finally:
                if not saw_end:
                    while kept_q[g].get() is not None:
                        pass
                if holding:
                    pose_slots.release()

        def run_decode():
            nonlocal decoded_n
            if os.environ.get("TURBO_CENSUS") == "1":
                decode_box["thread"] = threading.get_native_id()
            t_decode = time.perf_counter()
            try:
                if self.decode_method == "gop":
                    _emitted, decoded_n, _groups = decode.decode_gops(
                        video, keep, max_parallel=GOP_PARALLEL, threads=1, skip="NONREF",
                        loop_filter=True, on_kept=lambda g, frame: kept_q[g].put(frame),
                        on_gop=lambda g, _frames: (mark(f"decode_done_{g}"), kept_q[g].put(None)),
                        stop=stop.is_set)
                else:
                    _emitted, decoded_n = decode.decode_serial(
                        video, keep, skip="NONREF", loop_filter=True, threads=DECODE_THREADS,
                        on_kept=lambda g, frame: kept_q[g].put(frame),
                        on_gop=lambda g, _frames: (mark(f"decode_done_{g}"), kept_q[g].put(None)),
                        stop=stop.is_set)
            except Exception as exc:
                fail(exc)
            finally:
                decode_box["s"] = time.perf_counter() - t_decode
                for q in kept_q:
                    q.put(None)

        started = [finder]
        try:
            n_req = 4 if FOLLOW == "flow" else 2
            pose_reqs = [
                [pose_m.compiled.create_infer_request() for _ in range(n_req)] for _ in ranges]
            det_reqs = [det.compiled.create_infer_request() for _ in ranges]
            posers = [threading.Thread(target=pose_worker, args=(g,), daemon=True) for g in range(len(ranges))]
            for poser in posers:
                poser.start()
                started.append(poser)
            worker = threading.Thread(target=run_decode, daemon=True)
            worker.start()
        except BaseException as exc:
            # Startup broke part way. The decoder starts last, so it never ran and cannot
            # end the pose queues: stop, end them here, and wait for what did start.
            fail(exc)
            for q in kept_q:
                q.put(None)
            for thread in started:
                thread.join()
            raise
        worker.join()
        for poser in posers:
            poser.join()
        if errors:
            raise errors[0]
        retrieved = sum(len(rows or {}) for rows in gop_rows)
        decode_s = decode_box.get("s", 0.0)
        finder.join()
        if "error" in located:
            raise located["error"]
        lines, info = located["court"]
        court = detect.Court(lines, width, height)
        for i, rows in enumerate(gop_rows):
            if rows is None:
                gop_rows[i] = {}
        gid = sampler.stitch_ids(gop_rows)
        measured = sampler.apply_ids(gop_rows, gid)
        t_interp = time.perf_counter()
        frames = sampler.fill_frames(measured, n, fps, MAX_GAP)
        travel = pipeline.label_players(frames, court, fps)
        interp_s = time.perf_counter() - t_interp
        mark("tail_done")
        ready = [s for s in stats if s]
        det_s = sum(s[0] for s in ready)
        pose_s = sum(s[1] for s in ready)
        crops = sum(s[2] for s in ready)
        retrieve_s = sum(s[3] for s in ready)
        doc = {
            "schema_version": 3,
            "tier": "preview",
            "sampler": {
                "every_ref": EVERY_REF,
                "extra_big": EXTRA_BIG,
                "detect_every": DETECT_EVERY,
                "max_gap": MAX_GAP,
                "decode": self.decode_method,
                "skip": "NONREF",
                "skip_loop_filter": True,
                "pose": "rtmpose-t-ft3.onnx",
                "detector": "yolo26n-384x640.onnx",
            },
            "source": {
                "file": Path(video).name, "width": width, "height": height,
                "fps": fps, "frame_count": n,
            },
            "model": {
                "pose": "rtmpose-t-ft3.onnx",
                "detector": "yolo26n-384x640.onnx",
                "detect_every": DETECT_EVERY,
                "joint_threshold": sampler.JOINT_MIN,
            },
            "court_lines_normalized": lines,
            "court_fit": info,
            "play_area_normalized": (np.float32(court.poly) / [width, height]).round(4).tolist(),
            "cuts": [],
            "player_tracks_travel": travel,
            "frames": frames,
        }
        text, write_s = _dump(doc, out_path)
        doc = json.loads(text)
        total = time.perf_counter() - t0
        timings = _times(
            "preview", located.get("s", 0.0), decode_s, det_s, pose_s, interp_s, write_s,
            crops, decoded_n, retrieved, total)
        timings["retrieve"] = round(retrieve_s, 4)
        timings["stages_overlap"] = True
        timings["trace"] = trace
        if os.environ.get("TURBO_CENSUS") == "1":
            timings["actors"] = census_rows
            timings["census_threads"] = {
                "main": threading.get_native_id(),
                "court": located.get("thread"),
                "decode": decode_box.get("thread"),
            }
        return doc, timings

    def _refined(self, video, out_path):
        import tempfile
        _, _, _, pipeline, _ = _frozen()
        models = self._refined_models()
        CACHE.mkdir(exist_ok=True)
        handle = tempfile.NamedTemporaryFile(suffix=".json", dir=CACHE, delete=False)
        target = Path(handle.name)
        handle.close()
        t0 = time.perf_counter()
        try:
            info = pipeline.run(str(video), str(target), models=models)
            doc = json.loads(target.read_text(encoding="utf-8"))
        finally:
            target.unlink(missing_ok=True)
        doc["tier"] = "refined"
        _text, write_s = _dump(doc, out_path)
        total = time.perf_counter() - t0
        timings = _times(
            "refined", info.get("court_s", 0.0), info.get("decode_done_s", 0.0),
            sum(info.get("secs", {}).get("det", []) or [0.0]),
            sum(info.get("secs", {}).get("pose", []) or [0.0]),
            0.0, write_s, info.get("crops", 0), info.get("frames", 0), info.get("frames", 0), total)
        timings["frozen"] = info
        return doc, timings

    def _window(self, video, out_path, windows, base_doc):
        if not windows or base_doc is None:
            raise ValueError("window tier needs windows and base_doc")
        _, detect, ovcache, pipeline, pose = _frozen()
        # Dense pose, not the preview model. The window replaces guesses with measured rows.
        refined = self._refined_models()
        pose_m, det = refined.pose, refined.det
        doc = json.loads(json.dumps(base_doc, default=_jsonable))
        width, height = doc["source"]["width"], doc["source"]["height"]
        fps = doc["source"]["fps"]
        court = detect.Court(doc["court_lines_normalized"], width, height)
        t0 = time.perf_counter()
        per = []
        pose_req = pose_m.compiled.create_infer_request()
        det_req = det.compiled.create_infer_request()
        # Same inclusive end as coach.schema.Window. decode_span still wants an exclusive stop.
        inclusive = []
        for start, end in windows:
            start, end = int(start), int(end)
            if end < start:
                raise ValueError("window end is inclusive and must be >= start")
            inclusive.append((start, end))
        covered = {fi for start, end in inclusive for fi in range(start, end + 1)}
        reserved = {
            int(p["track_id"])
            for frame in doc["frames"]
            for p in frame.get("players") or []
        }
        for start, end in inclusive:
            t = time.perf_counter()
            frames, decoded_n = decode.decode_span(video, start, end + 1)
            rows, det_s, pose_s, crops, retrieve_s = _track(
                frames, court, det, pose, ovcache, det_req, pose_req, detect_every=8)
            # Local ids restart at zero. Bind them before label_players groups by track_id.
            bound = _bind_window_ids(doc["frames"], rows, start, end, reserved)
            for fi, people in rows.items():
                if fi in covered and start <= fi <= end:
                    doc["frames"][fi]["players"] = [
                        sampler.player_row(bound[tid], box, conf, joints, fi, fps, "measured")
                        for tid, box, conf, joints in people
                    ]
            per.append({
                "start": start, "end": end,
                "s": round(time.perf_counter() - t, 4),
                "crops": crops, "frames_decoded": decoded_n,
                "retrieve": round(retrieve_s, 4),
                "detect": round(det_s, 4), "pose": round(pose_s, 4),
            })
        saved = json.loads(json.dumps(doc))
        pipeline.label_players(doc["frames"], court, fps)
        for i, frame in enumerate(saved["frames"]):
            if i not in covered:
                doc["frames"][i] = frame
        for key in ("source", "court_lines_normalized", "court_fit", "play_area_normalized", "cuts"):
            if key in saved:
                doc[key] = saved[key]
        # Window is an operation. Keep the base tier. Omitted tier means refined to load_tracks.
        if doc.get("tier") not in ("preview", "refined", None):
            raise ValueError(f"base tier must be preview or refined, got {doc.get('tier')!r}")
        doc["windows_refined"] = list(saved.get("windows_refined") or []) + per
        text, write_s = _dump(doc, out_path)
        doc = json.loads(text)
        timings = _times(
            "window", 0.0, sum(w["s"] for w in per), 0.0, 0.0, 0.0, write_s,
            sum(w["crops"] for w in per), sum(w["frames_decoded"] for w in per),
            sum(w["crops"] for w in per), time.perf_counter() - t0)
        timings["windows"] = per
        timings["retrieve"] = round(sum(w.get("retrieve", 0.0) for w in per), 4)
        return doc, timings


def _bind_window_ids(frames, rows, start, end, reserved):
    """Map one window's local track ids onto the document.

    Evidence is box overlap with the rows about to be replaced, and with the
    frame on each side of the window. Below 0.3 IoU is not a match. A new id
    does not collide with an id already in the document or allocated earlier
    in this call.
    """
    local = {}
    for fi, people in rows.items():
        if fi < start or fi > end:
            continue
        for tid, box, *_rest in people:
            local.setdefault(tid, []).append((fi, list(box)))
    scores = {}

    def consider(tid, box, players):
        for person in players:
            overlap = sampler._iou(box, person["bbox_xyxy"])
            key = (tid, int(person["track_id"]))
            if overlap > scores.get(key, 0.0):
                scores[key] = overlap

    for tid, samples in local.items():
        for fi, box in samples:
            if 0 <= fi < len(frames):
                consider(tid, box, frames[fi].get("players") or [])
            if fi == start and start > 0:
                consider(tid, box, frames[start - 1].get("players") or [])
            if fi == end and end + 1 < len(frames):
                consider(tid, box, frames[end + 1].get("players") or [])
    mapping = {}
    taken = set()
    pairs = sorted(
        ((score, tid, gid) for (tid, gid), score in scores.items() if score >= 0.3),
        reverse=True)
    for _score, tid, gid in pairs:
        if tid in mapping or gid in taken:
            continue
        mapping[tid] = gid
        taken.add(gid)
    nxt = (max(reserved) + 1) if reserved else 0
    for tid in sorted(local):
        if tid in mapping:
            continue
        while nxt in reserved:
            nxt += 1
        mapping[tid] = nxt
        reserved.add(nxt)
        nxt += 1
    return mapping


class Tracker:
    """One GOP's boxes.

    pose follow waits for this sample's skeleton before the next crop. flow follow
    moves the box from the Y plane, so several frames can be posed together.
    """

    def __init__(self, court, det, pose, ovcache, det_req, pose_reqs, detect_every, follow="pose"):
        if follow not in ("pose", "flow"):
            raise ValueError(follow)
        self.court = court
        self.det = det
        self.pose = pose
        self.ovcache = ovcache
        self.det_req = det_req
        self.pose_reqs = pose_reqs if isinstance(pose_reqs, list) else [pose_reqs]
        self.detect_every = detect_every
        self.follow = follow
        self.tracks, self.nxt = {}, 0
        self.last_det = None
        self.prev_y = None
        self.pending = []
        self.out = {}
        self.det_s = self.pose_s = self.retrieve_s = self.flow_s = 0.0
        self.crops = self.flow_n = self.detect_n = self.step_n = self.pose_calls = 0

    def step(self, yuv):
        from pipeline import iou, pose_to_box
        detecting = self.last_det is None or yuv.fi - self.last_det >= self.detect_every or not self.tracks
        self.step_n += 1
        if self.follow == "flow" and not detecting and self.prev_y is not None and self.tracks:
            t_flow = time.perf_counter()
            _flow_shift(self.prev_y, yuv.y, self.tracks, yuv.w, yuv.h)
            self.flow_s += time.perf_counter() - t_flow
            self.flow_n += 1
        self.prev_y = yuv.y
        if detecting:
            self.detect_n += 1
            t = time.perf_counter()
            dets, took = _detect(yuv, self.court, self.det, self.det_req)
            self.retrieve_s += took
            pairs = sorted(
                ((iou(t["box"], b), tid, j)
                 for tid, t in self.tracks.items() for j, (b, _) in enumerate(dets)),
                reverse=True)
            used, matched = set(), set()
            for overlap, tid, j in pairs:
                if overlap < 0.3 or tid in matched or j in used:
                    continue
                matched.add(tid)
                used.add(j)
                self.tracks[tid]["box"] = dets[j][0]
                self.tracks[tid]["conf"] = dets[j][1]
                self.tracks[tid]["miss"] = 0
            for tid in list(self.tracks):
                if tid not in matched:
                    self.tracks[tid]["miss"] = self.tracks[tid].get("miss", 0) + 1
                    if self.tracks[tid]["miss"] >= 2:
                        del self.tracks[tid]
            for j, (box, score) in enumerate(dets):
                if j in used or len(self.tracks) >= MAX_TRACKS:
                    continue
                if all(iou(box, t["box"]) <= 0.5 for t in self.tracks.values()):
                    self.tracks[self.nxt] = {"box": box, "conf": score, "miss": 0}
                    self.nxt += 1
            self.last_det = yuv.fi
            self.det_s += time.perf_counter() - t
        if not self.tracks:
            self.out[yuv.fi] = []
            return
        ids = sorted(self.tracks)
        jobs = []
        boxes, confs = {}, {}
        for tid in ids:
            boxes[tid] = list(self.tracks[tid]["box"])
            confs[tid] = self.tracks[tid]["conf"]
            crop, center, scale, took = _warp(yuv, boxes[tid], self.pose)
            self.retrieve_s += took
            jobs.append((crop, center, scale))
        if self.follow == "pose":
            self._apply(yuv.fi, ids, boxes, confs, jobs, pose_to_box)
            return
        self.pending.append({
            "fi": yuv.fi, "ids": ids, "boxes": boxes, "confs": confs, "jobs": jobs,
        })
        if len(self.pending) >= FLOW_BATCH:
            self._flush(pose_to_box)

    def finish(self):
        from pipeline import pose_to_box
        self._flush(pose_to_box)

    def _apply(self, fi, ids, boxes, confs, jobs, pose_to_box):
        self.pose_calls += 1
        t = time.perf_counter()
        kps = _infer_crops(self.pose_reqs, jobs, self.pose, self.ovcache)
        self.pose_s += time.perf_counter() - t
        self.crops += len(ids)
        self.out[fi] = self._rows(ids, boxes, confs, kps, pose_to_box)

    def _flush(self, pose_to_box):
        if not self.pending:
            return
        self.pose_calls += 1
        jobs = []
        spans = []
        for item in self.pending:
            spans.append((len(jobs), item))
            jobs.extend(item["jobs"])
        t = time.perf_counter()
        kps = _infer_crops(self.pose_reqs, jobs, self.pose, self.ovcache)
        self.pose_s += time.perf_counter() - t
        self.crops += len(jobs)
        for start, item in spans:
            n = len(item["ids"])
            self.out[item["fi"]] = self._rows(
                item["ids"], item["boxes"], item["confs"], kps[start:start + n], pose_to_box)
        self.pending.clear()

    def _rows(self, ids, boxes, confs, kps, pose_to_box):
        rows = []
        for tid, k in zip(ids, kps):
            rows.append((tid, boxes[tid], confs[tid], k))
            if tid not in self.tracks:
                continue
            if float(np.median(k[:, 2])) < 0.3:
                self.tracks[tid]["miss"] = self.tracks[tid].get("miss", 0) + 1
            else:
                self.tracks[tid]["box"] = pose_to_box(k, mode="torso")
                self.tracks[tid]["miss"] = 0
        for tid in [tid for tid, tr in self.tracks.items() if tr.get("miss", 0) >= 2]:
            del self.tracks[tid]
        return rows


def _infer_crops(reqs, jobs, pose, ovcache):
    """One native request per crop. A failed request raises after its wave joins."""
    out = [None] * len(jobs)
    errors = {}

    def run(i, req):
        try:
            crop, center, scale = jobs[i]
            res = ovcache.infer(req, crop[None])
            out[i] = pose.decode(res[0][0], res[1][0], center, scale)
        except Exception as exc:
            errors[i] = exc

    pending = list(range(len(jobs)))
    while pending:
        wave = pending[:len(reqs)]
        pending = pending[len(reqs):]
        if len(wave) == 1:
            run(wave[0], reqs[0])
        else:
            threads = []
            try:
                for n, i in enumerate(wave):
                    thread = threading.Thread(target=run, args=(i, reqs[n]))
                    thread.start()
                    threads.append(thread)
            finally:
                for thread in threads:
                    thread.join()
        if errors:
            raise errors[min(errors)]
    return out


def _track(frames, court, det, pose, ovcache, det_req, pose_req, detect_every):
    tracker = Tracker(court, det, pose, ovcache, det_req, pose_req, detect_every)
    for yuv in frames:
        tracker.step(yuv)
    tracker.finish()
    return tracker.out, tracker.det_s, tracker.pose_s, tracker.crops, tracker.retrieve_s


def _detect(yuv, court, det, req):
    from pipeline import iou
    x0, y0, x1, y1 = court.roi
    t = time.perf_counter()
    bgr, (ox, oy) = yuv.region_bgr(x0, y0, x1, y1)
    took = time.perf_counter() - t
    raw = det(bgr, (0, 0, bgr.shape[1], bgr.shape[0]), request=req, min_conf=0.25)
    found = []
    for box, score in raw:
        full = [box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy]
        if court.contains_feet(full):
            found.append((full, score))
    found.sort(key=lambda item: -item[1])
    kept = []
    for box, score in found:
        if all(iou(box, other) <= 0.6 for other, _ in kept):
            kept.append((box, score))
    return kept, took


def _flow_shift(prev_y, y, tracks, w, h):
    """Move each box by the median Y-plane flow. A wild jump is ignored."""
    pts, owner = [], []
    for tid, tr in tracks.items():
        x0, y0, x1, y1 = [float(v) for v in tr["box"]]
        if x1 - x0 < 12 or y1 - y0 < 12:
            continue
        for fx in (0.3, 0.5, 0.7):
            for fy in (0.3, 0.5, 0.7):
                x = x0 + (x1 - x0) * fx
                yy = y0 + (y1 - y0) * fy
                if 0 <= x < w and 0 <= yy < h:
                    pts.append((x, yy))
                    owner.append(tid)
    if len(pts) < 4:
        return
    arr = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    nxt, st, _err = cv2.calcOpticalFlowPyrLK(
        prev_y, y, arr, None, winSize=(21, 21), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03))
    by = {}
    for i, tid in enumerate(owner):
        if st is None or int(st.reshape(-1)[i]) == 0:
            continue
        by.setdefault(tid, []).append((
            float(nxt[i, 0, 0] - arr[i, 0, 0]),
            float(nxt[i, 0, 1] - arr[i, 0, 1])))
    for tid, shifts in by.items():
        if len(shifts) < 4 or tid not in tracks:
            continue
        dx = float(np.median([s[0] for s in shifts]))
        dy = float(np.median([s[1] for s in shifts]))
        x0, y0, x1, y1 = [float(v) for v in tracks[tid]["box"]]
        bw, bh = x1 - x0, y1 - y0
        if abs(dx) > 0.5 * bw or abs(dy) > 0.5 * bh:
            continue
        x0 = min(max(0.0, x0 + dx), max(0.0, w - bw))
        y0 = min(max(0.0, y0 + dy), max(0.0, h - bh))
        tracks[tid]["box"] = [x0, y0, x0 + bw, y0 + bh]


def _warp(yuv, box, pose):
    matrix, center, scale = pose.crop_transform(box)
    half = scale / 2.0
    t = time.perf_counter()
    bgr, (ox, oy) = yuv.region_bgr(
        float(center[0] - half[0]) - 2, float(center[1] - half[1]) - 2,
        float(center[0] + half[0]) + 2, float(center[1] + half[1]) + 2)
    took = time.perf_counter() - t
    moved = decode.shift_warp(matrix, ox, oy)
    crop = cv2.warpAffine(bgr, moved, (pose.INPUT_W, pose.INPUT_H), flags=cv2.INTER_LINEAR)
    return crop, center, scale, took


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj))


def _dump(doc, out_path):
    t = time.perf_counter()
    text = json.dumps(doc, default=_jsonable, separators=(",", ":"))
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
    return text, time.perf_counter() - t


def _times(tier, court, decode_s, det_s, pose_s, interp_s, write_s, crops, decoded, retrieved, total):
    return {
        "tier": tier,
        "court": round(float(court), 4),
        "decode": round(float(decode_s), 4),
        "retrieve": 0.0,
        "detect": round(float(det_s), 4),
        "pose": round(float(pose_s), 4),
        "interpolate": round(float(interp_s), 4),
        "write": round(float(write_s), 4),
        "crops": int(crops),
        "frames_decoded": int(decoded),
        "frames_retrieved": int(retrieved),
        "total_s": round(float(total), 4),
    }


class PoseRunner:
    IS_REPLAY = False

    def __init__(self, models_dir):
        self.svc = TurboService(models_dir)

    def warm_up(self, video_path):
        return self.svc.warm_up(video_path)

    def run(self, video_path, work_dir):
        out = Path(work_dir) / "pose.json"
        doc, timings = self.svc.analyze(video_path, out_json_path=out, tier="preview")
        return {"doc": doc, "timings": timings}
