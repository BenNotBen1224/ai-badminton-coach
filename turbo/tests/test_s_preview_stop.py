"""A preview stops its own work after the first real failure, and cleans up a partial start.

Fake PyAV and fake models drive the real `_preview` threads and the real decode loops on
a 4-GOP x 3-packet clip. Hooks pin each race in one order. No video, no models, and no
files written. Run from turbo:

    python -m unittest tests.test_s_preview_stop -v
"""
import concurrent.futures
import queue
import sys
import threading
import traceback
import types
import unittest
from fractions import Fraction
from pathlib import Path

import decode
import service

WAIT = 5.0
HANG = 20.0
W, H = 4, 2
GOPS, PER = 4, 3
MODES = ("serial", "gop")
LINES = [[0.1, 0.2, 0.9, 0.8]]
INFO = {"fit": "fake"}
SLOT_KEYS = ("slot_acquired_free", "slot_waits", "slot_acquired_after_wait", "slot_released")
# Player 0's x0 in each preview frame: y + u/1e3 + v/1e6 of that frame's first pixel.
X0 = [0.1, 16.12, 32.13, 48.15, 64.16, 80.18, 96.2, 112.21, 128.23, 144.24, 160.0, 176.02]


class CourtError(RuntimeError):
    pass


class DecodeError(RuntimeError):
    pass


class PoseError(RuntimeError):
    pass


class StartError(RuntimeError):
    pass


class RequestError(RuntimeError):
    pass


def plane_bytes(pts, offset, size):
    return bytes((pts * 16 + offset + i) % 256 for i in range(size))


def frame_sig(pts):
    """Decoding packet `pts` returns its own planes at frame index `pts`."""
    return [pts, pts, pts % PER == 0, plane_bytes(pts, 0, W * H).hex(),
            plane_bytes(pts, 100, W * H // 4).hex(), plane_bytes(pts, 200, W * H // 4).hex()]


class Plane(bytearray):
    def __init__(self, data, height, line_size):
        super().__init__(data)
        self.height = height
        self.line_size = line_size


class Frame:
    def __init__(self, pts):
        self.pts = pts
        self.width, self.height = W, H
        self.planes = [
            Plane(plane_bytes(pts, 0, W * H), H, W),
            Plane(plane_bytes(pts, 100, W * H // 4), H // 2, W // 2),
            Plane(plane_bytes(pts, 200, W * H // 4), H // 2, W // 2),
        ]
        self.color_range = 1
        self.colorspace = 1


class Packet:
    def __init__(self, pts):
        self.pts = pts
        self.is_keyframe = pts % PER == 0
        self.size = 100 + pts


class Stream:
    start_time = 0
    time_base = Fraction(1, 30)
    average_rate = Fraction(30)
    frames = GOPS * PER
    codec_context = types.SimpleNamespace(width=W, height=H, name="fake", extradata=None)


class Box:
    def __init__(self, world):
        self.world = world

    def close(self):
        self.world.count("box_closed")
        with self.world.lock:
            self.world.inflight_at_close.append(len(self.world.calls) - self.world.completed)


class Codec:
    def __init__(self, world):
        self.world = world

    def decode(self, packet):
        w = self.world
        with w.lock:
            call = len(w.calls) + 1
            w.calls.append((call, None if packet is None else packet.pts))
        w.on_call(w, call, packet)
        with w.lock:
            w.completed += 1
        return [] if packet is None else [Frame(packet.pts)]


class World:
    """One scenario: the fake clip, counters, and the hooks that pin one race order."""

    def __init__(self, on_call=None, on_context=None, on_court=None, on_step=None, on_stop=None):
        self.lock = threading.Lock()
        self.on_call = on_call or (lambda w, call, packet: None)
        self.on_context = on_context or (lambda w: None)
        self.on_court = on_court or (lambda w: (LINES, INFO))
        self.on_step = on_step or (lambda w, g, item: None)
        self.on_stop = on_stop or (lambda w: None)
        self.packets = [Packet(i) for i in range(GOPS * PER)]
        self.calls = []
        self.completed = 0
        self.n = {}
        self.queues = []
        self.started = []
        self.joins = []
        self.caller = self.finder = self.decoder = self.poser0 = None
        self.calls_at_court_error = None
        self.court_entered = threading.Event()
        self.court_release = threading.Event()
        self.decoder_seen = threading.Event()
        self.poser0_seen = threading.Event()
        self.flag = threading.Event()
        self.flag2 = threading.Event()
        self.slot_wait = threading.Event()
        self.stop_seen = threading.Event()
        self.inflight_at_close = []
        self.timeouts = []

    def count(self, key, by=1):
        with self.lock:
            self.n[key] = self.n.get(key, 0) + by

    def wait(self, what, label):
        if isinstance(what, threading.Thread):
            what.join(WAIT)
            ok = not what.is_alive()
        elif isinstance(what, threading.Barrier):
            try:
                what.wait(WAIT)
                ok = True
            except threading.BrokenBarrierError:
                ok = False
        else:
            ok = what.wait(WAIT)
        if not ok:
            with self.lock:
                self.timeouts.append(label)
        return ok

    def demux(self, path):
        self.count("box_opened")
        if self.caller is not None and threading.current_thread() is not self.caller:
            self.decoder = threading.current_thread()
            self.decoder_seen.set()
        return Box(self), Stream(), list(self.packets)

    def context(self, stream, threads, skip, loop_filter):
        self.count("contexts")
        self.on_context(self)
        return Codec(self)

    def court_from_video(self, video):
        self.finder = threading.current_thread()
        self.court_entered.set()
        try:
            return self.on_court(self)
        except Exception:
            with self.lock:
                self.calls_at_court_error = len(self.calls)
            raise


def tracker_class(world):
    class FakeTracker:
        def __init__(self, court, det, pose, ovcache, det_req, pose_reqs, detect_every, follow="pose"):
            self.g = det_req.index
            self.out = {}
            self.det_s = self.pose_s = self.retrieve_s = self.flow_s = 0.0
            self.crops = self.flow_n = self.detect_n = self.step_n = self.pose_calls = 0
            world.count("trackers")

        def step(self, item):
            world.count(f"steps_g{self.g}")
            world.on_step(world, self.g, item)
            self.step_n += 1
            self.detect_n = 1
            self.pose_calls += 1
            self.crops += 1
            # The row is built from decoded pixels, so the document checks decode output.
            v = float(item.y[0, 0]) + float(item.u[0, 0]) / 1e3 + float(item.v[0, 0]) / 1e6
            box = [v, 10.0, v + 20.0, 40.0]
            self.out[item.fi] = [(0, box, 0.9, [[v, 20.0 + j, 0.9] for j in range(17)])]

        def finish(self):
            world.count("finishes")

    return FakeTracker


class Req:
    def __init__(self, kind, index):
        self.kind, self.index = kind, index


class Compiled:
    def __init__(self, kind, world, fail_at=None, err=None):
        self.kind, self.world, self.fail_at, self.err = kind, world, fail_at, err
        self.made = 0

    def create_infer_request(self):
        if self.made == self.fail_at:
            # Court detection is running when the request fails.
            self.world.wait(self.world.court_entered, "court_entered")
            raise self.err
        req = Req(self.kind, self.made)
        self.made += 1
        return req


class Court:
    def __init__(self, lines, width, height):
        self.poly = [[0, 0], [width, 0], [width, height], [0, height]]


def label_players(frames, court, fps):
    ids = sorted({p["track_id"] for f in frames for p in f["players"]})
    return {str(i): sum(p["track_id"] == i for f in frames for p in f["players"]) for i in ids}


def fake_queue(world):
    """`queue` for the service: each pose queue is recorded and counts its get() calls."""

    class Watched(queue.Queue):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.entered = self.returned = 0
            self.getting = threading.Event()
            with world.lock:
                world.queues.append(self)

        def get(self, block=True, timeout=None):
            with world.lock:
                self.entered += 1
            self.getting.set()
            try:
                return super().get(block, timeout)
            finally:
                with world.lock:
                    self.returned += 1

    ns = types.SimpleNamespace(**{k: getattr(queue, k) for k in dir(queue) if not k.startswith("_")})
    ns.Queue = Watched
    return ns


def fake_threading(world, fail_role=None, err=None, before_fail=None):
    """`threading` for the service: named threads, one start() that can fail, a watched
    pose slot, and a stop event that reports to `world`."""

    class RoleThread(threading.Thread):
        def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
            self.role = target.__name__ + "".join(f"({a})" for a in args)
            super().__init__(group, target, f"svc-{self.role}", args, kwargs, daemon=daemon)

        def start(self):
            if self.role == fail_role:
                if before_fail is not None:
                    before_fail(world)
                raise err
            super().start()
            with world.lock:
                world.started.append(self.role)

        def join(self, timeout=None):
            with world.lock:
                world.joins.append([threading.current_thread().name, self.role])
            if self.role == "find_court":
                world.court_release.set()
            super().join(timeout)

    class Slots(threading.Semaphore):
        def acquire(self, blocking=True, timeout=None):
            if super().acquire(blocking=False):
                world.count("slot_acquired_free")
                return True
            world.count("slot_waits")
            world.slot_wait.set()
            got = super().acquire(blocking, timeout)
            if got:
                world.count("slot_acquired_after_wait")
            return got

        def release(self, n=1):
            world.count("slot_released", n)
            super().release(n)

    class Stop(threading.Event):
        def set(self):
            first = not self.is_set()
            super().set()
            world.stop_seen.set()
            if first:
                world.on_stop(world)

    ns = types.SimpleNamespace(
        **{k: getattr(threading, k) for k in dir(threading) if not k.startswith("_")})
    ns.Thread, ns.Semaphore, ns.Event = RoleThread, Slots, Stop
    return ns


def snapshot(world, before, runner):
    frames = sys._current_frames()
    alive = []
    for t in threading.enumerate():
        if t in before or t is runner or not t.is_alive():
            continue
        where = None
        frame = frames.get(t.ident)
        for fs in traceback.extract_stack(frame) if frame is not None else ():
            if Path(fs.filename).name == "service.py":
                where = [fs.name, fs.line]
        alive.append({"thread": t.name, "blocked_at": where})
    with world.lock:
        return {
            "alive": sorted(alive, key=lambda a: a["thread"]),
            "queues": [{"frames": sum(x is not None for x in list(q.queue)),
                        "gets": [q.entered, q.returned]} for q in world.queues],
            "native_calls": len(world.calls),
            "native_calls_completed": world.completed,
            "call_pts": [pts for _, pts in world.calls],
            "calls_at_court_error": world.calls_at_court_error,
            "contexts": world.n.get("contexts", 0),
            "boxes": [world.n.get("box_opened", 0), world.n.get("box_closed", 0)],
            "trackers": world.n.get("trackers", 0),
            "steps": {int(k[7:]): v for k, v in sorted(world.n.items()) if k.startswith("steps_g")},
            "finishes": world.n.get("finishes", 0),
            "slots": {k: world.n.get(k, 0) for k in SLOT_KEYS},
        }


def run_preview(mode, world, fail_role=None, fail_request=None, err=None, before_fail=None,
                slots=None):
    """Run the real `_preview` with fakes. Record what is left the moment it returns."""
    saved = []

    def put(obj, attr, value):
        saved.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, value)

    frozen = (
        types.SimpleNamespace(court_from_video=world.court_from_video),
        types.SimpleNamespace(Court=Court),
        types.SimpleNamespace(),
        types.SimpleNamespace(label_players=label_players),
        types.SimpleNamespace(),
    )
    models = (types.SimpleNamespace(compiled=Compiled("pose", world, fail_request, err)),
              types.SimpleNamespace(compiled=Compiled("det", world)))
    before = set(threading.enumerate())
    box = {}

    def call():
        world.caller = threading.current_thread()
        try:
            box["value"] = svc.analyze("clip.mp4", tier="preview")
        except BaseException as exc:
            box["error"] = exc

    runner = threading.Thread(target=call, name=f"call-{mode}", daemon=True)
    try:
        put(service, "_frozen", lambda: frozen)
        put(service, "Tracker", tracker_class(world))
        put(service, "queue", fake_queue(world))
        put(service, "threading", fake_threading(world, fail_role, err, before_fail))
        if slots is not None:
            put(service, "POSE_SLOTS", slots)
        put(decode, "demux", world.demux)
        put(decode, "_apply_stream", world.context)
        put(decode, "_open_context", world.context)
        svc = service.TurboService(Path("no-models"), decode_method=mode)
        svc._preview_models = lambda: models
        runner.start()
        runner.join(HANG)
        out = {"error": box.get("error"), "hung": runner.is_alive(),
               **snapshot(world, before, runner)}
    finally:
        # The evidence is taken. Release what is left: court detection ends and each
        # pose queue gets an end signal, so a leak fails the test instead of hanging it.
        world.court_release.set()
        for q in list(world.queues):
            q.put(None)
        if runner.ident is not None:
            runner.join(WAIT)
        for t in threading.enumerate():
            if t not in before and t is not runner:
                t.join(WAIT)
        for obj, attr, value in reversed(saved):
            setattr(obj, attr, value)
    out["started"] = list(world.started)
    out["caller_joins"] = sorted(role for joiner, role in world.joins if joiner == runner.name)
    out["timeouts"] = list(world.timeouts)
    out["doc"] = out["counts"] = None
    if "value" in box:
        doc, timings = box["value"]
        out["doc"] = doc
        out["counts"] = {k: timings[k] for k in ("crops", "frames_decoded", "frames_retrieved")}
    return out


def sig(f):
    return [f.fi, f.pts, f.key, f.y.tobytes().hex(), f.u.tobytes().hex(), f.v.tobytes().hex()]


def run_decode(mode, world, max_parallel=2, pool=None, **kwargs):
    names = ("demux", "_apply_stream", "_open_context") + (("ThreadPoolExecutor",) if pool else ())
    saved = [(a, getattr(decode, a)) for a in names]
    decode.demux = world.demux
    decode._apply_stream = world.context
    decode._open_context = world.context
    if pool is not None:
        decode.ThreadPoolExecutor = pool
    kept, gops = [], []
    lock = threading.Lock()

    def on_kept(g, item):
        with lock:
            kept.append([g, item.fi])

    def on_gop(g, frames):
        with lock:
            gops.append([g, [f.fi for f in frames]])

    keep = decode.make_keep(4, decode.big_packet_pts(
        world.packets, decode.gop_ranges(world.packets), 2))
    out = {"error": None, "frames": None, "decoded": None, "groups": None}
    before = set(threading.enumerate())
    try:
        if mode == "serial":
            frames, n = decode.decode_serial(
                "clip.mp4", keep, threads=1, on_gop=on_gop, on_kept=on_kept, **kwargs)
        else:
            frames, n, found = decode.decode_gops(
                "clip.mp4", keep, max_parallel=max_parallel, threads=1,
                on_gop=on_gop, on_kept=on_kept, **kwargs)
            out["groups"] = [[f.fi for f in g] for g in found]
        out["frames"] = [sig(f) for f in frames]
        out["decoded"] = n
    except Exception as exc:
        out["error"] = exc
    finally:
        for attr, value in saved:
            setattr(decode, attr, value)
    out.update({
        "calls": [list(c) for c in world.calls],
        "kept": kept if mode == "serial" else sorted(kept),
        "gops": gops if mode == "serial" else sorted(gops),
        "contexts": world.n.get("contexts", 0),
        "boxes": [world.n.get("box_opened", 0), world.n.get("box_closed", 0)],
        "inflight_at_close": list(world.inflight_at_close),
        "leftover_threads": [t.name for t in threading.enumerate()
                             if t not in before and t.is_alive()],
        "timeouts": list(world.timeouts),
    })
    return out


ALL_GOPS = [[0, [0, 1, 2]], [1, [3, 4, 5]], [2, [6, 7, 8]], [3, [9, 10, 11]]]


class DecodeStop(unittest.TestCase):
    def test_d1_unset_stop_returns_every_frame(self):
        variants = {"no stop": {}, "stop None": {"stop": None}, "stop false": {"stop": lambda: False}}
        for mode in MODES:
            for label, kwargs in variants.items():
                with self.subTest(mode=mode, variant=label):
                    r = run_decode(mode, World(), **kwargs)
                    self.assertIsNone(r["error"])
                    self.assertEqual(r["frames"], [frame_sig(i) for i in range(12)])
                    self.assertEqual(r["frames"][5][3], "5051525354555657")
                    self.assertEqual(r["decoded"], 12)
                    self.assertEqual(r["kept"], [[i // PER, i] for i in range(12)])
                    self.assertEqual(r["gops"], ALL_GOPS)
                    pts = [p for _, p in r["calls"]]
                    if mode == "serial":
                        self.assertEqual(pts, list(range(12)) + [None])
                        self.assertEqual(r["contexts"], 1)
                    else:
                        self.assertEqual(sorted(p for p in pts if p is not None), list(range(12)))
                        self.assertEqual(pts.count(None), 4)
                        self.assertEqual(r["contexts"], 4)
                        self.assertEqual(r["groups"], [g for _, g in ALL_GOPS])
                    self.assertEqual(r["boxes"], [1, 1])
                    self.assertEqual(r["leftover_threads"], [])

    def test_d2_preset_stop_opens_nothing(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                r = run_decode(mode, World(), stop=lambda: True)
                self.assertIs(type(r["error"]), decode.Cancelled)
                self.assertEqual(r["calls"], [])
                self.assertEqual(r["contexts"], 0)
                self.assertEqual(r["boxes"], [0, 0])
                self.assertEqual(r["kept"], [])
                self.assertEqual(r["gops"], [])

    def test_d3_serial_stop_during_a_call_ends_before_the_next_call(self):
        cases = ((5, ALL_GOPS[:1]), (12, ALL_GOPS[:3]))
        for stop_at, want_gops in cases:
            with self.subTest(stop_at=stop_at):
                def on_call(w, call, packet, stop_at=stop_at):
                    if call == stop_at:
                        w.flag.set()

                world = World(on_call=on_call)
                r = run_decode("serial", world, stop=world.flag.is_set)
                self.assertIs(type(r["error"]), decode.Cancelled)
                self.assertIsNone(r["frames"])
                self.assertEqual([c for c, _ in r["calls"]], list(range(1, stop_at + 1)))
                self.assertEqual(r["gops"], want_gops)
                self.assertEqual([fi for _, fi in r["kept"]], list(range(stop_at)))
                self.assertEqual(r["boxes"], [1, 1])

    def test_d4_gop_stop_with_two_calls_in_flight(self):
        barrier = threading.Barrier(2)

        def on_call(w, call, packet):
            if packet is not None and packet.pts in (0, 3):
                w.wait(barrier, "barrier")
                w.flag.set()

        world = World(on_call=on_call)
        r = run_decode("gop", world, max_parallel=2, stop=world.flag.is_set)
        self.assertIs(type(r["error"]), decode.Cancelled)
        self.assertEqual(sorted(p for _, p in r["calls"]), [0, 3])
        self.assertEqual(r["contexts"], 2)
        self.assertEqual(r["gops"], [])
        self.assertEqual(r["kept"], [[0, 0], [1, 3]])
        self.assertEqual(r["boxes"], [1, 1])
        self.assertEqual(r["timeouts"], [])
        self.assertEqual(r["leftover_threads"], [])

    def test_d5_gop_error_stops_later_gops_and_keeps_identity(self):
        boom = DecodeError("decoder broke at pts 4")

        def on_call(w, call, packet):
            if packet is not None and packet.pts == 4:
                raise boom

        r = run_decode("gop", World(on_call=on_call), max_parallel=1)
        self.assertIs(r["error"], boom)
        self.assertEqual(r["calls"], [[1, 0], [2, 1], [3, 2], [4, None], [5, 3], [6, 4]])
        self.assertEqual(r["contexts"], 2)
        self.assertEqual(r["gops"], ALL_GOPS[:1])
        self.assertEqual(r["boxes"], [1, 1])

    def test_d6_serial_error_keeps_identity_and_closes_the_container(self):
        boom = DecodeError("decoder broke at call 5")

        def on_call(w, call, packet):
            if call == 5:
                raise boom

        r = run_decode("serial", World(on_call=on_call))
        self.assertIs(r["error"], boom)
        self.assertEqual([c for c, _ in r["calls"]], [1, 2, 3, 4, 5])
        self.assertEqual(r["boxes"], [1, 1])

    def test_d7_gop_executor_setup_failure_closes_the_container(self):
        # The real executor rejects zero workers after the container is open.
        r = run_decode("gop", World(), max_parallel=0)
        self.assertIs(type(r["error"]), ValueError)
        self.assertIn("max_workers", str(r["error"]))
        self.assertEqual(r["calls"], [])
        self.assertEqual(r["contexts"], 0)
        self.assertEqual(r["boxes"], [1, 1])
        self.assertEqual(r["inflight_at_close"], [0])
        self.assertEqual(r["leftover_threads"], [])

    def test_d8_gop_submit_failure_halts_submitted_gops_then_closes(self):
        boom = RuntimeError("can't start new thread")
        world = World()

        class FailingPool(concurrent.futures.ThreadPoolExecutor):
            submitted = 0

            def submit(self, fn, /, *args, **kwargs):
                self.submitted += 1
                if self.submitted == 2:
                    # GOP 0 is inside its first native call when the second submit fails.
                    world.wait(world.flag, "gop0_in_call")
                    raise boom
                return super().submit(fn, *args, **kwargs)

            def shutdown(self, wait=True, *, cancel_futures=False):
                world.flag2.set()
                return super().shutdown(wait=wait, cancel_futures=cancel_futures)

        def on_call(w, call, packet):
            if call == 1:
                w.flag.set()
                w.wait(w.flag2, "pool_shutdown")

        world.on_call = on_call
        r = run_decode("gop", world, pool=FailingPool)
        self.assertIs(r["error"], boom)
        self.assertEqual(r["calls"], [[1, 0]])
        self.assertEqual(r["contexts"], 1)
        self.assertEqual(r["kept"], [[0, 0]])
        self.assertEqual(r["gops"], [])
        self.assertEqual(r["boxes"], [1, 1])
        self.assertEqual(r["inflight_at_close"], [0])
        self.assertEqual(r["timeouts"], [])
        self.assertEqual(r["leftover_threads"], [])


class PreviewStop(unittest.TestCase):
    def assert_clean(self, r):
        """Every service thread ended, no frame is left queued, every container closed."""
        self.assertFalse(r["hung"])
        self.assertEqual(r["alive"], [])
        self.assertEqual(r["timeouts"], [])
        self.assertEqual([q["frames"] for q in r["queues"]], [0] * len(r["queues"]))
        self.assertEqual(r["boxes"][0], r["boxes"][1])

    def test_s1_success_document_comes_from_every_decoded_frame(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                r = run_preview(mode, World())
                self.assertIsNone(r["error"])
                self.assert_clean(r)
                doc = r["doc"]
                self.assertEqual(r["counts"], {"crops": 12, "frames_decoded": 12, "frames_retrieved": 12})
                self.assertEqual(doc["source"], {"file": "clip.mp4", "width": 4, "height": 2,
                                                 "fps": 30.0, "frame_count": 12})
                self.assertEqual(doc["sampler"]["decode"], mode)
                measured = [p["bbox_xyxy"][0] for f in doc["frames"] for p in f["players"]
                            if "src" not in p]
                self.assertEqual(measured, X0)
                self.assertEqual(r["native_calls"], 13 if mode == "serial" else 16)
                self.assertEqual(r["trackers"], 4)
                self.assertEqual(r["steps"], {0: 3, 1: 3, 2: 3, 3: 3})
                self.assertEqual(r["finishes"], 4)

    def test_s2_court_error_before_decoding_starts_no_native_call(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                err = CourtError("no court found")

                def on_court(w, err=err):
                    raise err

                def on_context(w):
                    w.wait(w.court_entered, "court_entered")
                    w.wait(w.finder, "finder_end")

                r = run_preview(mode, World(on_court=on_court, on_context=on_context))
                self.assertIs(r["error"], err)
                self.assert_clean(r)
                self.assertEqual(r["native_calls"], 0)
                self.assertEqual(r["trackers"], 0)
                self.assertEqual(r["steps"], {})

    def test_s3_court_error_during_decoding_lets_only_in_flight_calls_finish(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                err = CourtError("no court found")
                barrier = threading.Barrier(2)

                def on_call(w, call, packet, barrier=barrier, mode=mode):
                    if mode == "serial" and call != 5:
                        return
                    if mode == "gop":
                        if packet is None or packet.pts not in (0, 3):
                            return
                        w.wait(barrier, "barrier")
                    w.flag.set()
                    w.wait(w.court_entered, "court_entered")
                    w.wait(w.finder, "finder_end")

                def on_court(w, err=err):
                    w.wait(w.flag, "decode_in_flight")
                    raise err

                r = run_preview(mode, World(on_call=on_call, on_court=on_court))
                self.assertIs(r["error"], err)
                self.assert_clean(r)
                in_flight = 5 if mode == "serial" else 2
                self.assertEqual(r["calls_at_court_error"], in_flight)
                self.assertEqual(r["native_calls"], in_flight)
                self.assertEqual(r["native_calls_completed"], in_flight)
                self.assertEqual(r["trackers"], 0)
                if mode == "gop":
                    self.assertEqual(sorted(r["call_pts"]), [0, 3])
                    self.assertEqual(r["contexts"], 2)

    def test_s4_decoder_error_skips_pose_work(self):
        err = DecodeError("decoder broke at call 7")

        def on_call(w, call, packet):
            if call == 7:
                raise err

        def on_court(w):
            w.wait(w.decoder_seen, "decoder_seen")
            w.wait(w.decoder, "decoder_end")
            return LINES, INFO

        r = run_preview("serial", World(on_call=on_call, on_court=on_court))
        self.assertIs(r["error"], err)
        self.assert_clean(r)
        self.assertEqual(r["native_calls"], 7)
        self.assertEqual(r["trackers"], 0)
        self.assertEqual(r["steps"], {})

    def test_s5_pose_error_stops_decode_and_other_workers(self):
        err = PoseError("pose broke on gop 0")

        def on_step(w, g, item):
            if g == 0 and item.fi == 0:
                w.poser0 = threading.current_thread()
                w.poser0_seen.set()
                w.wait(w.flag, "g1_step_started")
                raise err
            if g == 1 and item.fi == 3:
                w.flag.set()
                w.wait(w.poser0_seen, "poser0_seen")
                w.wait(w.poser0, "poser0_end")

        def on_call(w, call, packet):
            if call == 7:
                w.wait(w.poser0_seen, "poser0_seen")
                w.wait(w.poser0, "poser0_end")

        r = run_preview("serial", World(on_call=on_call, on_step=on_step))
        self.assertIs(r["error"], err)
        self.assert_clean(r)
        self.assertEqual(r["native_calls"], 7)
        self.assertEqual(r["steps"], {0: 1, 1: 1})
        self.assertEqual(r["finishes"], 0)

    def test_s6_first_cause_is_raised(self):
        # decoder_then_court: the decoder fails first, the court later.
        # court_then_in_flight_decoder: the court fails first, then the decode call
        # already in flight fails too.
        for order in ("decoder_then_court", "court_then_in_flight_decoder"):
            with self.subTest(order=order):
                cerr = CourtError("no court found")
                derr = DecodeError("decoder broke")
                if order == "decoder_then_court":
                    def on_call(w, call, packet, derr=derr):
                        if call == 2:
                            raise derr

                    def on_court(w, cerr=cerr):
                        w.wait(w.decoder_seen, "decoder_seen")
                        w.wait(w.decoder, "decoder_end")
                        raise cerr
                    first, calls = derr, 2
                else:
                    def on_call(w, call, packet, derr=derr):
                        if call == 5:
                            w.flag.set()
                            w.wait(w.court_entered, "court_entered")
                            w.wait(w.finder, "finder_end")
                            raise derr

                    def on_court(w, cerr=cerr):
                        w.wait(w.flag, "decode_in_flight")
                        raise cerr
                    first, calls = cerr, 5
                r = run_preview("serial", World(on_call=on_call, on_court=on_court))
                self.assertIs(r["error"], first)
                self.assert_clean(r)
                self.assertEqual(r["native_calls"], calls)

    def test_s7_cause_is_recorded_before_stop_is_visible(self):
        # Setting the stop pauses until the decoder thread has ended. If the cause were
        # recorded after the stop, the decoder's Cancelled would be recorded first.
        err = CourtError("no court found")

        def on_stop(w):
            if threading.current_thread() is not w.decoder:
                w.flag2.set()
                w.wait(w.decoder_seen, "decoder_seen")
                w.wait(w.decoder, "decoder_end_after_stop")

        def on_call(w, call, packet):
            if call == 2:
                w.flag.set()
                w.wait(w.flag2, "stop_set")

        def on_court(w):
            w.wait(w.flag, "decode_started")
            raise err

        r = run_preview("serial", World(on_call=on_call, on_court=on_court, on_stop=on_stop))
        self.assertIs(r["error"], err)
        self.assert_clean(r)
        self.assertEqual(r["native_calls"], 2)

    def test_s8_worker_queued_on_the_pose_slot_starts_no_step_after_stop(self):
        # One pose slot. Worker 1 queues for it while worker 0 holds it in a step, then
        # that step fails. Worker 1 gets the slot only after the stop.
        err = PoseError("pose broke on frame 0")

        def on_step(w, g, item):
            if g == 0 and item.fi == 0:
                w.poser0_seen.set()
                w.wait(w.slot_wait, "slot_wait")
                raise err

        def on_call(w, call, packet):
            # Call 4 yields worker 1's first frame: release it only once worker 0 holds
            # the slot, so worker 1 is the one that queues. Hold call 5 until the stop.
            if call == 4:
                w.wait(w.poser0_seen, "poser0_in_step")
            if call == 5:
                w.wait(w.stop_seen, "stop_set")

        r = run_preview("serial", World(on_call=on_call, on_step=on_step), slots=1)
        self.assertIs(r["error"], err)
        self.assert_clean(r)
        self.assertEqual(r["slots"], {"slot_acquired_free": 1, "slot_waits": 1,
                                      "slot_acquired_after_wait": 1, "slot_released": 2})
        self.assertEqual(r["steps"], {0: 1})
        self.assertEqual(r["finishes"], 0)
        self.assertEqual(r["native_calls"], 5)


def court_runs_until_joined(w):
    # Real court detection has no stop input. This one ends once a thread waits for it,
    # so a court thread that nobody joins is still running when `analyze` returns.
    w.wait(w.court_release, "court_runs_until_joined")
    return LINES, INFO


def all_workers_wait_for_frames(w):
    for g, q in enumerate(w.queues):
        w.wait(q.getting, f"worker{g}_waits_for_frames")


class PreviewStartup(unittest.TestCase):
    """A thread or an inference request fails to start part way through `_preview`."""

    def assert_cleaned(self, r, err, started, trackers):
        self.assertIs(r["error"], err)
        self.assertFalse(r["hung"])
        self.assertIsNone(r["doc"])
        self.assertEqual(r["timeouts"], [])
        self.assertEqual(r["alive"], [])
        self.assertEqual(r["started"], started)
        self.assertEqual(r["caller_joins"], sorted(started))
        self.assertEqual([q["frames"] for q in r["queues"]], [0] * len(r["queues"]))
        self.assertEqual(r["native_calls"], 0)
        self.assertEqual(r["contexts"], 0)
        self.assertEqual(r["boxes"], [1, 1])
        self.assertEqual(r["trackers"], trackers)
        self.assertEqual(r["steps"], {})
        self.assertEqual(r["finishes"], 0)

    def test_u1_court_thread_start_failure_joins_nothing(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                err = StartError("can't start new thread")
                r = run_preview(mode, World(), fail_role="find_court", err=err)
                self.assert_cleaned(r, err, started=[], trackers=0)

    def test_u2_request_failure_after_court_start_waits_for_the_court_thread(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                err = RequestError("request failed")
                r = run_preview(mode, World(on_court=court_runs_until_joined),
                                fail_request=3, err=err)
                self.assert_cleaned(r, err, started=["find_court"], trackers=0)

    def test_u3_worker_start_failure_ends_the_started_worker(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                err = StartError("can't start new thread")
                r = run_preview(
                    mode, World(), fail_role="pose_worker(1)", err=err,
                    before_fail=lambda w: w.wait(w.queues[0].getting, "worker0_waits_for_frames"))
                self.assert_cleaned(r, err, started=["find_court", "pose_worker(0)"], trackers=1)
                self.assertEqual(r["queues"][0]["gets"], [1, 1])

    def test_u4_decoder_start_failure_ends_every_worker(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                err = StartError("can't start new thread")
                r = run_preview(mode, World(), fail_role="run_decode", err=err,
                                before_fail=all_workers_wait_for_frames)
                workers = [f"pose_worker({g})" for g in range(GOPS)]
                self.assert_cleaned(r, err, started=["find_court"] + workers, trackers=GOPS)
                self.assertEqual([q["gets"] for q in r["queues"]], [[1, 1]] * GOPS)


if __name__ == "__main__":
    unittest.main()
