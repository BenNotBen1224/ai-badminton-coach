"""Decode a preview subset of frames.

One sequential decoder is the default. It is correct by construction and yields
each GOP as soon as that GOP ends, so pose can run while later GOPs still
decode. Independent GOP contexts are a speed lever only. `align_check` must
show they match the sequential decoder's pixels, including the frame after a
GOP boundary, before anything uses them. A decode-only PTS list is not that check.
"""
from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

AV_SITE = Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages"


def av():
    if str(AV_SITE) not in sys.path:
        sys.path.insert(0, str(AV_SITE))
    import av as _av
    _av.logging.set_level(_av.logging.ERROR)
    return _av


class YuvFrame:
    """Copied YUV planes. The decoder may reuse its own buffers on the next frame."""

    def __init__(self, fi, pts, y, u, v, key, color_range, colorspace):
        self.fi = int(fi)
        self.pts = pts
        self.y = y
        self.u = u
        self.v = v
        self.key = bool(key)
        self.color_range = color_range
        self.colorspace = colorspace
        self.h, self.w = y.shape

    def region_bgr(self, x0, y0, x1, y1):
        x0, y0, x1, y1 = even_box(x0, y0, x1, y1, self.w, self.h)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("empty crop")
        bgr = yuv_region_bgr(
            self.y, self.u, self.v, x0, y0, x1, y1, self.color_range, self.colorspace)
        return bgr, (x0, y0)


def shift_warp(matrix, ox, oy):
    """Move a full-frame warp so it addresses a crop whose top-left is (ox, oy)."""
    moved = matrix.copy()
    moved[0, 2] = matrix[0, 2] + matrix[0, 0] * ox
    moved[1, 2] = matrix[1, 2] + matrix[1, 1] * oy
    return moved


def even_box(x0, y0, x1, y1, w, h):
    x0 = max(0, int(np.floor(x0)) & ~1)
    y0 = max(0, int(np.floor(y0)) & ~1)
    x1 = min(int(w), int(np.ceil(x1) + 1) & ~1)
    y1 = min(int(h), int(np.ceil(y1) + 1) & ~1)
    return x0, y0, x1, y1


def copy_plane(plane, height, width):
    raw = np.frombuffer(plane, np.uint8)
    rows = raw.reshape(plane.height, plane.line_size)
    return rows[:height, :width].copy()


def copy_yuv(frame):
    h, w = frame.height, frame.width
    return (
        copy_plane(frame.planes[0], h, w),
        copy_plane(frame.planes[1], h // 2, w // 2),
        copy_plane(frame.planes[2], h // 2, w // 2),
    )


def _blit(plane, arr):
    view = np.frombuffer(plane, np.uint8).reshape(plane.height, plane.line_size)
    view[: arr.shape[0], : arr.shape[1]] = arr


def yuv_region_bgr(y, u, v, x0, y0, x1, y1, color_range=None, colorspace=None):
    """BGR of one even rectangle via swscale, so it can match `to_ndarray`."""
    Y = y[y0:y1, x0:x1]
    U = u[y0 // 2:y1 // 2, x0 // 2:x1 // 2]
    V = v[y0 // 2:y1 // 2, x0 // 2:x1 // 2]
    h, w = Y.shape
    fr = av().VideoFrame(w, h, "yuv420p")
    if color_range is not None:
        fr.color_range = color_range
    if colorspace is not None:
        fr.colorspace = colorspace
    _blit(fr.planes[0], Y)
    _blit(fr.planes[1], U)
    _blit(fr.planes[2], V)
    return fr.to_ndarray(format="bgr24", threads=1)


def frame_index(stream, pts):
    origin = stream.start_time or 0
    return int(round((pts - origin) * float(stream.time_base) * float(stream.average_rate)))


def demux(path):
    box = av().open(str(path))
    stream = box.streams.video[0]
    packets = [p for p in box.demux(stream) if p.size]
    return box, stream, packets


def gop_ranges(packets):
    starts = [i for i, p in enumerate(packets) if p.is_keyframe]
    if not starts or starts[0] != 0:
        starts = [0, *starts]
    starts = sorted(set(starts))
    return list(zip(starts, starts[1:] + [len(packets)]))


def big_packet_pts(packets, ranges, extra):
    chosen = set()
    if extra <= 0:
        return chosen
    for a, b in ranges:
        top = sorted(packets[a:b], key=lambda p: p.size, reverse=True)[:extra]
        chosen.update(p.pts for p in top)
    return chosen


def make_keep(every_ref, extra_pts):
    """Keep keyframes, every Nth reference frame inside a GOP, and large packets."""

    def keep(ordinal, fi, pts, is_key):
        return is_key or ordinal % every_ref == 0 or pts in extra_pts

    return keep


def _open_context(stream, threads, skip, loop_filter):
    cc = av().CodecContext.create(stream.codec_context.name, "r")
    cc.extradata = stream.codec_context.extradata
    cc.thread_type = "AUTO"
    cc.thread_count = threads
    cc.skip_frame = skip
    if loop_filter:
        cc.options = {"skip_loop_filter": "all"}
    return cc


def _apply_stream(stream, threads, skip, loop_filter):
    cc = stream.codec_context
    cc.thread_type = "AUTO"
    cc.thread_count = threads
    cc.skip_frame = skip
    if loop_filter:
        cc.options = {"skip_loop_filter": "all"}
    return cc


def _as_yuv(frame, stream, key_pts):
    if frame.pts is None:
        return None
    y, u, v = copy_yuv(frame)
    return YuvFrame(
        frame_index(stream, frame.pts), frame.pts, y, u, v,
        frame.pts in key_pts, frame.color_range, frame.colorspace,
    )


def _consume(frames, stream, key_pts, keep, ordinal, kept, on_kept=None):
    last = None
    for frame in frames:
        if frame.pts is None:
            continue
        fi = frame_index(stream, frame.pts)
        is_key = frame.pts in key_pts
        if keep(ordinal, fi, frame.pts, is_key):
            item = _as_yuv(frame, stream, key_pts)
            if item is not None:
                kept.append(item)
                if on_kept is not None:
                    on_kept(item)
        ordinal += 1
        last = frame
    return ordinal, last


def _keep_tail(frame, stream, key_pts, kept, on_kept=None):
    """Copy now. The next decode call may reuse this frame's buffer."""
    if frame is None or (kept and kept[-1].pts == frame.pts):
        return
    tail = _as_yuv(frame, stream, key_pts)
    if tail is not None:
        kept.append(tail)
        if on_kept is not None:
            on_kept(tail)


class Cancelled(Exception):
    """The caller's stop signal ended decoding. Frames decoded so far are not a result."""


def _check(stop):
    if stop is not None and stop():
        raise Cancelled()


def decode_packet_group(cc, packets, stream, key_pts, keep, flush, on_kept=None, stop=None):
    """`stop()` is checked before each native decode call. A call already running finishes."""
    kept, ordinal = [], 0
    last = None
    for packet in packets:
        _check(stop)
        ordinal, got = _consume(
            cc.decode(packet), stream, key_pts, keep, ordinal, kept, on_kept)
        if got is not None:
            last = got
    _keep_tail(last, stream, key_pts, kept, on_kept)
    if flush:
        _check(stop)
        ordinal, got = _consume(
            cc.decode(None), stream, key_pts, keep, ordinal, kept, on_kept)
        _keep_tail(got, stream, key_pts, kept, on_kept)
    return kept, ordinal


def decode_serial(path, keep, skip="NONREF", loop_filter=True, threads=4, on_gop=None, on_kept=None, stop=None):
    """One context, in order. `on_gop(index, frames)` fires as each GOP finishes.

    The shared context is flushed only after the last packet. Flushing between
    GOPs would drop the references the next GOP still needs. Once `stop()` is
    true this raises Cancelled before the next native call instead of returning.
    """
    _check(stop)
    box, stream, packets = demux(path)
    try:
        key_pts = {p.pts for p in packets if p.is_keyframe}
        cc = _apply_stream(stream, threads, skip, loop_filter)
        ranges = gop_ranges(packets)
        emitted, decoded = [], 0
        for g, (a, b) in enumerate(ranges):
            def emit(item, g=g):
                if on_kept is not None:
                    on_kept(g, item)

            kept, n = decode_packet_group(
                cc, packets[a:b], stream, key_pts, keep, flush=(g == len(ranges) - 1),
                on_kept=emit if on_kept is not None else None, stop=stop)
            decoded += n
            emitted.extend(kept)
            if on_gop is not None:
                on_gop(g, kept)
    finally:
        box.close()
    return emitted, decoded


def decode_gops(path, keep, max_parallel=3, threads=2, skip="NONREF", loop_filter=True, on_gop=None, on_kept=None, stop=None):
    """At most `max_parallel` independent GOP contexts. Flush each one: it owns its GOP.

    `on_gop(index, frames)` runs as that GOP finishes, so pose can overlap later GOPs.
    The first GOP failure, or `stop()`, ends the other GOPs before their next native
    call. That failure is raised; a stop alone raises Cancelled. Neither returns frames.
    Do not use this for a timed preview until `align_check` passes on a GOP boundary.
    """
    _check(stop)
    box, stream, packets = demux(path)
    failed = []
    halt = threading.Event()
    try:
        key_pts = {p.pts for p in packets if p.is_keyframe}
        ranges = gop_ranges(packets)
        found = [None] * len(ranges)
        counts = [0] * len(ranges)

        def stopped():
            return halt.is_set() or (stop is not None and stop())

        def work(g):
            try:
                _check(stopped)
                a, b = ranges[g]
                cc = _open_context(stream, threads, skip, loop_filter)

                def emit(item, g=g):
                    if on_kept is not None:
                        on_kept(g, item)

                kept, n = decode_packet_group(
                    cc, packets[a:b], stream, key_pts, keep, flush=True,
                    on_kept=emit if on_kept is not None else None, stop=stopped)
                found[g] = kept
                counts[g] = n
                if on_gop is not None:
                    on_gop(g, kept)
            except Cancelled:
                pass
            except Exception as exc:
                failed.append(exc)
                halt.set()

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            try:
                list(pool.map(work, range(len(ranges))))
            except BaseException:
                # GOPs already submitted stop before their next native call; exit joins them.
                halt.set()
                raise
    finally:
        box.close()
    if failed:
        raise failed[0]
    if any(group is None for group in found):
        raise Cancelled()
    frames = [f for group in found for f in group]
    frames.sort(key=lambda f: (f.fi, f.pts))
    return frames, sum(counts), found


def decode_span(path, start, end, threads=2):
    """DEFAULT frames in [start, end), starting at the preceding keyframe.

    Stopping at the next keyframe after `end` keeps B-frame reorder honest.
    """
    box, stream, packets = demux(path)
    key_pts = {p.pts for p in packets if p.is_keyframe}
    ranges = gop_ranges(packets)
    begin = 0
    stop = len(packets)
    for a, b in ranges:
        fi = frame_index(stream, packets[a].pts)
        if fi <= start:
            begin = a
        elif fi > end:
            stop = a
            break
    cc = _apply_stream(stream, threads, "DEFAULT", False)
    kept, decoded = [], 0
    for packet in packets[begin:stop]:
        for frame in cc.decode(packet):
            if frame.pts is None:
                continue
            decoded += 1
            fi = frame_index(stream, frame.pts)
            if start <= fi < end:
                item = _as_yuv(frame, stream, key_pts)
                if item is not None:
                    kept.append(item)
    for frame in cc.decode(None):
        if frame.pts is None:
            continue
        decoded += 1
        fi = frame_index(stream, frame.pts)
        if start <= fi < end:
            item = _as_yuv(frame, stream, key_pts)
            if item is not None:
                kept.append(item)
    box.close()
    kept.sort(key=lambda f: f.fi)
    return kept, decoded


def stream_gops(path, keep, out, skip="NONREF", loop_filter=True, threads=4):
    """Decode on this thread. Put `(gop_index, frames)` on `out`, then None."""
    try:
        decode_serial(
            path, keep, skip=skip, loop_filter=loop_filter, threads=threads,
            on_gop=lambda g, frames: out.put((g, frames)))
        out.put(None)
    except Exception as exc:
        out.put(exc)


def y_signature(frame):
    plane = frame.planes[0]
    y = np.frombuffer(plane, np.uint8).reshape(plane.height, plane.line_size)
    return y[: frame.height:8, : frame.width:8].copy()


def _signatures(stream, packets, ranges, skip, loop_filter, threads, parallel):
    def pull(cc, group, flush):
        sigs = {}
        for packet in group:
            for frame in cc.decode(packet):
                if frame.pts is not None:
                    sigs[frame_index(stream, frame.pts)] = y_signature(frame)
        if flush:
            for frame in cc.decode(None):
                if frame.pts is not None:
                    sigs[frame_index(stream, frame.pts)] = y_signature(frame)
        return sigs

    if not parallel:
        cc = _apply_stream(stream, threads, skip, loop_filter)
        sigs = {}
        for g, (a, b) in enumerate(ranges):
            sigs.update(pull(cc, packets[a:b], flush=g == len(ranges) - 1))
        return sigs

    parts = [None] * len(ranges)

    def work(g):
        a, b = ranges[g]
        cc = _open_context(stream, threads, skip, loop_filter)
        parts[g] = pull(cc, packets[a:b], flush=True)

    with ThreadPoolExecutor(max_workers=min(3, len(ranges))) as pool:
        list(pool.map(work, range(len(ranges))))
    merged = {}
    for part in parts:
        merged.update(part)
    return merged


def align_check(path, limit_gops=2):
    """Compare GOP-parallel luma with one sequential decoder.

    DEFAULT (loop filter on) checks that splitting at keyframes does not change
    pixels. NONREF + skip_loop_filter is the preview setting; those pixels differ
    from DEFAULT on purpose, so this only requires the two decoders to agree
    with each other.
    """
    import time
    box, stream, packets = demux(path)
    ranges = gop_ranges(packets)[:limit_gops]
    report = {
        "clip": Path(path).name,
        "gops_used": len(ranges),
        "gop_packets": [b - a for a, b in ranges],
    }
    box.close()
    modes = (("DEFAULT", False), ("NONREF", True))
    for skip, loop_filter in modes:
        t = time.perf_counter()
        box, stream, packets = demux(path)
        serial = _signatures(stream, packets, ranges, skip, loop_filter, 1, False)
        box.close()
        box, stream, packets = demux(path)
        key_fi = frame_index(stream, packets[ranges[1][0]].pts) if len(ranges) > 1 else None
        parallel = _signatures(stream, packets, ranges, skip, loop_filter, 2, True)
        box.close()
        only_serial = sorted(set(serial) - set(parallel))
        only_parallel = sorted(set(parallel) - set(serial))
        both = sorted(set(serial) & set(parallel))
        diffs = [int(np.abs(serial[i].astype(np.int16) - parallel[i].astype(np.int16)).max()) for i in both]
        near = [i for i in both if key_fi is not None and abs(i - key_fi) <= 2]
        near_diff = [int(np.abs(serial[i].astype(np.int16) - parallel[i].astype(np.int16)).max()) for i in near]
        report[skip] = {
            "serial": len(serial),
            "parallel": len(parallel),
            "only_serial": only_serial[:8],
            "only_parallel": only_parallel[:8],
            "max_abs": max(diffs) if diffs else None,
            "boundary_fi": key_fi,
            "boundary_frames": near,
            "boundary_max_abs": max(near_diff) if near_diff else None,
            "wall_s": round(time.perf_counter() - t, 3),
            "pass": not only_serial and not only_parallel and bool(diffs) and max(diffs) <= 1 and bool(near_diff),
        }
    report["pass"] = all(report[name]["pass"] for name, _ in modes)
    return report


def conversion_check(path, box=(700, 400, 980, 780)):
    """One frame: a YUV crop against the same rectangle of `to_ndarray`."""
    container = av().open(str(path))
    stream = container.streams.video[0]
    stream.codec_context.skip_frame = "NONREF"
    frame = next(container.decode(stream))
    full = frame.to_ndarray(format="bgr24")
    y, u, v = copy_yuv(frame)
    item = YuvFrame(0, frame.pts, y, u, v, True, frame.color_range, frame.colorspace)
    x0, y0, x1, y1 = even_box(*box, frame.width, frame.height)
    got, origin = item.region_bgr(x0, y0, x1, y1)
    ref = full[origin[1]:origin[1] + got.shape[0], origin[0]:origin[0] + got.shape[1]]
    container.close()
    err = np.abs(got.astype(np.int16) - ref.astype(np.int16))
    return {
        "mean_abs": float(err.mean()),
        "max_abs": int(err.max()),
        "shape": list(got.shape),
        "pass": float(err.mean()) <= 1.5 and int(err.max()) <= 8,
    }
