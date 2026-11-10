"""Render a review MP4 by executing the published timeline.

Slow motion is cv2 DIS. On clip 04 frames 306-318, crop 661,340-1055,937,
blend and DIS both had spike about 1.24. Blend doubled the hitting arm.
DIS did not tear it.
"""
from __future__ import annotations

import base64
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from badminton_coach.__main__ import find_tool, require_ffmpeg

from .bundle import build_bundle
from .timeline import build_review, review_picks

OUT_W = 1920
OUT_H = 1080
PANE_W = 1380
PANEL_W = 540
SLOWMO = "dis"
_FACE = (0, 1, 2)
_ANKLES = (15, 16)
_GRIDS: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
_FONTS: dict[tuple[int, bool], ImageFont.ImageFont] = {}


def export_review(video, doc, report, out_mp4, *, picks=None, limit_s=30):
    bundle = build_bundle(doc, report, video_url=str(video))
    clips = _clips(report, picks, limit_s)
    duration_s = sum(clip.timeline.duration_s for clip in clips)
    fps = float(report.fps)
    frames = max(1, round(duration_s * fps))
    poses = _poses(bundle)
    cuts = _cut_starts(doc, report.n_frames)
    style = bundle["style"]
    names = list(bundle.get("joints") or [])
    reader = _Reader(video)
    flows = _Flows()
    out = Path(out_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample = reader.get(0)
    video_size = (sample.shape[1], sample.shape[0])
    source_size = _source_size(bundle, video_size)
    geom = _contain(video_size[0], video_size[1], PANE_W, OUT_H)
    clock: dict[tuple, float] = {}
    ffmpeg, _ffprobe = _tools()
    proc = None
    started = time.perf_counter()
    proc = subprocess.Popen(
        [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{OUT_H}",
            "-r", str(fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "21",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for index in range(frames):
            wall = min(duration_s, index / fps) if duration_s > 0 else 0.0
            canvas = _frame_at(
                wall, clips, reader, flows, poses, cuts, style, names,
                report, bundle, video_size, source_size, geom, clock,
            )
            proc.stdin.write(np.ascontiguousarray(canvas).tobytes())
        proc.stdin.close()
        proc.stdin = None
        err = proc.stderr.read()
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited {code}: {err.decode('utf-8', 'replace')}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if proc.stdin is not None:
            proc.stdin.close()
            proc.stdin = None
        if proc.stderr is not None:
            proc.stderr.close()
            proc.stderr = None
    elapsed = time.perf_counter() - started
    per = elapsed / duration_s if duration_s > 0 else elapsed
    print(
        f"render_s={elapsed:.3f} seconds_per_output_second={per:.3f} dis_s={flows.seconds:.3f} frames={frames}",
        file=sys.stderr,
    )
    return {
        "out": str(out),
        "duration_s": duration_s,
        "render_s": elapsed,
        "seconds_per_output_second": per,
        "frames": frames,
    }


class _Clip:
    def __init__(self, finding, primary, best, timeline):
        self.finding = finding
        self.primary = primary
        self.best = best
        self.timeline = timeline


class _Reader:
    def __init__(self, path):
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"could not open {path}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frames = None
        self.cache: dict[int, np.ndarray] = {}
        self.next_index = 0
        self.cap = cap
        if 0 < count <= 64:
            loaded = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                loaded.append(frame)
            cap.release()
            self.cap = None
            if not loaded:
                raise RuntimeError(f"could not read {path}")
            self.frames = loaded

    def get(self, index):
        index = max(0, int(index))
        if self.frames is not None:
            return self.frames[min(index, len(self.frames) - 1)]
        hit = self.cache.get(index)
        if hit is not None:
            return hit
        if index != self.next_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"could not read frame {index} of {self.path}")
        self.next_index = index + 1
        self.cache[index] = frame
        if len(self.cache) > 6:
            del self.cache[min(self.cache)]
        return frame


class _Flows:
    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.order: list[int] = []
        self.seconds = 0.0

    def pair(self, index, left, right):
        hit = self.cache.get(index)
        if hit is not None:
            return hit
        started = time.perf_counter()
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        hit = (self.dis.calc(gray_left, gray_right, None), self.dis.calc(gray_right, gray_left, None))
        self.seconds += time.perf_counter() - started
        self.cache[index] = hit
        self.order.append(index)
        if len(self.order) > 8:
            del self.cache[self.order.pop(0)]
        return hit


def _tools():
    require_ffmpeg()
    ffmpeg = find_tool("ffmpeg")
    ffprobe = find_tool("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise FileNotFoundError("ffmpeg and ffprobe are required")
    return ffmpeg, ffprobe


def _clips(report, picks, limit_s):
    timelines = build_review(report, picks, report.fps, limit_s=limit_s)
    chosen = list(review_picks(report) if picks is None else picks)
    clips = []
    index = 0
    for finding in chosen:
        if index >= len(timelines):
            break
        timeline = timelines[index]
        freeze = next((segment for segment in timeline.segments if segment.kind == "freeze"), None)
        issues = finding.issue_moments
        if issues:
            if freeze is None or freeze.start_frame != issues[0].key_frame or freeze.show != "issue":
                break
            clips.append(_Clip(finding, issues[0], finding.best_moment, timeline))
            index += 1
            continue
        best = finding.best_moment
        if best is None:
            continue
        if freeze is None or freeze.start_frame != best.key_frame or freeze.show != "best":
            break
        clips.append(_Clip(finding, best, None, timeline))
        index += 1
    if index != len(timelines):
        raise RuntimeError(f"paired {index} of {len(timelines)} review timelines")
    return clips


def _locate(clips, wall):
    if not clips:
        return None, 0.0
    cursor = 0.0
    for clip in clips:
        span = clip.timeline.duration_s
        if wall < cursor + span or clip is clips[-1]:
            local = min(max(0.0, wall - cursor), span)
            return clip, local
        cursor += span
    last = clips[-1]
    return last, last.timeline.duration_s


def _frame_at(wall, clips, reader, flows, poses, cuts, style, names, report, bundle, video_size, source_size, geom, clock):
    clip, local = _locate(clips, wall)
    if clip is None:
        source = reader.get(0)
        pane = _zoom(_fit(source, geom), (PANE_W / 2, OUT_H / 2), 1.0)
        panel = _panel(None, None, "issue", [], style, report, bundle, poses, 0, 0.0)
    else:
        sample = clip.timeline.sample_at(local)
        show = sample["show"]
        moment = clip.best if show == "best" and clip.best is not None else clip.primary
        source_index = float(sample["frame"])
        pixels, held, alpha = _pixels(reader, flows, source_index, cuts, report.n_frames)
        alphas = {
            role: _joint_alpha(held, alpha, poses, role, cuts, report.n_frames)
            for role in poses
        }
        _draw_poses(pixels, poses, held, alphas, style, names, clip.finding, video_size, source_size)
        dropped = _draw_marks(
            pixels, moment, show, source_index, wall, clock, poses, held, alphas,
            style, video_size, source_size,
        )
        origin = _zoom_origin(poses, _role(clip.finding.player), moment.key_frame, video_size, source_size, geom)
        pane = _zoom(_fit(pixels, geom), origin, float(sample["zoom"]))
        panel = _panel(
            clip.finding, moment, show, dropped, style, report, bundle, poses, held,
            alphas.get(_role(clip.finding.player), 0.0),
        )
    canvas = np.concatenate([pane, panel], axis=1)
    if canvas.shape != (OUT_H, OUT_W, 3):
        raise RuntimeError(f"frame shape {canvas.shape}")
    return canvas


def _pixels(reader, flows, source_index, cuts, n_frames):
    if n_frames <= 1 or source_index <= 0:
        return reader.get(0).copy(), 0, 0.0
    last = n_frames - 1
    if source_index >= last:
        return reader.get(last).copy(), last, 0.0
    held = int(math.floor(source_index))
    alpha = source_index - held
    if alpha <= 1e-4 or (held + 1) in cuts:
        return reader.get(held).copy(), held, 0.0
    left = reader.get(held)
    right = reader.get(held + 1)
    if SLOWMO == "blend":
        return cv2.addWeighted(left, 1.0 - alpha, right, alpha, 0.0), held, alpha
    flow_ab, flow_ba = flows.pair(held, left, right)
    warped_left = _warp(left, flow_ab, alpha)
    warped_right = _warp(right, flow_ba, 1.0 - alpha)
    return cv2.addWeighted(warped_left, 1.0 - alpha, warped_right, alpha, 0.0), held, alpha


def _joint_alpha(held, alpha, poses, role, cuts, n_frames):
    # A cut or an unmeasured neighbour is not a joint we are allowed to invent.
    if alpha <= 0.0 or held + 1 >= n_frames or (held + 1) in cuts:
        return 0.0
    pose = poses.get(role)
    if pose is None:
        return 0.0
    measured = pose["measured"]
    if held >= len(measured) or not measured[held] or not measured[held + 1]:
        return 0.0
    return alpha


def _warp(image, flow, amount):
    height, width = image.shape[:2]
    grid_x, grid_y = _grid(width, height)
    return cv2.remap(
        image,
        grid_x - flow[..., 0] * amount,
        grid_y - flow[..., 1] * amount,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _grid(width, height):
    key = (width, height)
    hit = _GRIDS.get(key)
    if hit is None:
        hit = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        _GRIDS[key] = hit
    return hit


def _draw_poses(frame, poses, held, alphas, style, names, finding, video_size, source_size):
    focus_role = _role(finding.player)
    focus = {_joint_index(names, name) for name in finding.focus_joints}
    focus.discard(-1)
    sx, sy = _scale(video_size, source_size)
    dim = float(style["dim"])
    for role, pose in poses.items():
        role_alpha = alphas.get(role, 0.0)
        measured = held < len(pose["measured"]) and bool(pose["measured"][held])
        faded = role_alpha > 1e-4 or not measured
        fade = float(style["interpolated"]["alpha"]) if faded else 1.0
        dash = bool(style["interpolated"]["dash"]) and faded
        snapshot = frame.copy()
        dim_layer = snapshot.copy()
        focus_layer = snapshot.copy()
        xy, conf = _sample_joints(pose, held, role_alpha)
        box = _sample_box(pose, held, role_alpha)
        role_focus = focus if role == focus_role else set()
        _draw_role(dim_layer, xy, conf, box, style, role, role_focus, dash, sx, sy, focus_pass=False)
        _draw_role(focus_layer, xy, conf, box, style, role, role_focus, dash, sx, sy, focus_pass=True)
        _stamp_drawn(frame, dim_layer, snapshot, fade * dim)
        _stamp_drawn(frame, focus_layer, snapshot, fade)


def _draw_role(layer, xy, conf, box, style, role, focus, dash, sx, sy, focus_pass):
    gate = float(style["confidence_gate"])
    color = _bgr(style["roles"].get(role, style.get("untracked", "#D7D7D7")))
    outline = _bgr(style["outline"])
    joint = _bgr(style["joint"])
    limb = _limb(box, style, sy)
    points = []
    for index in range(len(conf)):
        if conf[index] < gate or not np.isfinite(xy[index]).all():
            points.append(None)
            continue
        points.append((int(round(xy[index, 0] * sx)), int(round(xy[index, 1] * sy))))
    for start, end in style["bones"]:
        if start >= len(points) or end >= len(points):
            continue
        touch = start in focus or end in focus
        if touch != focus_pass:
            continue
        if points[start] is None or points[end] is None:
            continue
        width = limb + int(style["outline_extra_px"])
        _segment(layer, points[start], points[end], outline, width, dash)
        _segment(layer, points[start], points[end], color, limb, dash)
    for index, point in enumerate(points):
        if point is None or (index in focus) != focus_pass:
            continue
        radius = max(2, limb // 2 + 2)
        if index in focus:
            radius = max(radius + 1, int(round(radius * float(style["focus_scale"]))))
            cv2.circle(layer, point, radius + 5, color, 2, cv2.LINE_AA)
        cv2.circle(layer, point, radius + 2, outline, -1, cv2.LINE_AA)
        cv2.circle(layer, point, radius, joint, -1, cv2.LINE_AA)
        cv2.circle(layer, point, max(1, radius - 2), color, -1, cv2.LINE_AA)


def _segment(image, start, end, color, width, dash):
    if dash:
        _dashed(image, start, end, color, max(1, width))
    else:
        cv2.line(image, start, end, color, max(1, width), cv2.LINE_AA)


def _dashed(image, start, end, color, width):
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    if length < 1:
        return
    dash, gap = 10, 7
    step = dash + gap
    pos = 0.0
    while pos < length:
        end_pos = min(length, pos + dash)
        cv2.line(
            image,
            _along(start, end, pos / length),
            _along(start, end, end_pos / length),
            color, width, cv2.LINE_AA,
        )
        pos += step


def _along(start, end, t):
    return (
        int(round(start[0] + (end[0] - start[0]) * t)),
        int(round(start[1] + (end[1] - start[1]) * t)),
    )


def _limb(box, style, sy):
    if box is None or not np.isfinite(box).all():
        return int(style["min_limb_px"])
    height = abs(box[3] - box[1]) * sy
    return max(int(style["min_limb_px"]), int(round(height / float(style["limb_per_box_height"]))))


def _draw_marks(frame, moment, show, source_index, wall, clock, poses, held, alphas, style, video_size, source_size):
    if moment is None or (moment.kind == "best" and show != "best"):
        return []
    if moment.kind != "best" and show == "best":
        return []
    displayed = int(round(source_index))
    sx, sy = _scale(video_size, source_size)
    boxes = _boxes(poses, held, alphas, sx, sy)
    faces = _faces(poses, held, alphas, sx, sy)
    font = _font(max(16, int(round(video_size[1] * 0.03))), bold=False)
    used = []
    dropped = []
    for mark_index, mark in enumerate(moment.marks):
        visible = abs(displayed - mark.frame) <= 1
        key = (id(moment), mark_index)
        if not visible:
            clock.pop(key, None)
            continue
        entered = clock.setdefault(key, wall)
        opacity = min(1.0, (wall - entered) / 0.35)
        spec = style["marks"].get(mark.style, style["marks"]["neutral"])
        text = (mark.label or "").strip()
        anchor = _video_point(mark.points[0], sx, sy) if mark.points else None
        placed = None
        if text and anchor is not None:
            placed = _place_label(anchor, text, font, boxes, faces, used, video_size)
            if placed is None:
                dropped.append(text)
        def paint(layer, mark=mark, spec=spec, placed=placed, text=text, anchor=anchor):
            _paint_mark(layer, mark, spec, sx, sy)
            if placed is not None and anchor is not None:
                rect, leader, text_xy = placed
                if leader:
                    cv2.line(layer, anchor, _nearest(rect, anchor), _bgr(spec["stroke"]), 1, cv2.LINE_AA)
                _pill(layer, rect, text, font, spec, text_xy)
        _stamp(frame, paint, opacity)
        if placed is not None:
            used.append(placed[0])
    return dropped


def _paint_mark(layer, mark, spec, sx, sy):
    stroke = _bgr(spec["stroke"])
    fill = _bgr(spec["fill"])
    width = max(1, int(spec["width"]))
    dash = bool(spec["dash"])
    points = [_video_point(point, sx, sy) for point in mark.points]
    kind = mark.kind
    if kind == "dot" and points:
        cv2.circle(layer, points[0], 14, stroke, 2, cv2.LINE_AA)
        cv2.circle(layer, points[0], 6, fill, -1, cv2.LINE_AA)
    elif kind == "line" and len(points) == 2:
        _segment(layer, points[0], points[1], stroke, width, dash)
    elif kind == "bracket" and len(points) == 2:
        _segment(layer, points[0], points[1], stroke, width, False)
        _caps(layer, points[0], points[1], stroke, width)
    elif kind == "band" and len(points) == 2:
        _band(layer, points[0], points[1], fill, stroke, width, dash)
    elif kind == "angle" and len(points) == 3:
        _segment(layer, points[1], points[0], stroke, width, dash)
        _segment(layer, points[1], points[2], stroke, width, dash)
        _arc(layer, points[1], points[0], points[2], stroke, width, dash)
    elif kind == "trail" and len(points) >= 2:
        _trail(layer, points, stroke, fill, width)
    elif kind == "label":
        return


def _caps(image, start, end, color, width):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    px, py = -dy / length * 8, dx / length * 8
    for point in (start, end):
        cv2.line(
            image,
            (int(round(point[0] - px)), int(round(point[1] - py))),
            (int(round(point[0] + px)), int(round(point[1] + py))),
            color, width, cv2.LINE_AA,
        )


def _band(image, a, b, fill, stroke, width, dash):
    x0, x1 = sorted((a[0], b[0]))
    y0, y1 = sorted((a[1], b[1]))
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), fill, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.28, image, 0.72, 0.0, dst=image)
    if dash:
        _dashed(image, (x0, y0), (x1, y0), stroke, width)
        _dashed(image, (x1, y0), (x1, y1), stroke, width)
        _dashed(image, (x1, y1), (x0, y1), stroke, width)
        _dashed(image, (x0, y1), (x0, y0), stroke, width)
    else:
        cv2.rectangle(image, (x0, y0), (x1, y1), stroke, width, cv2.LINE_AA)


def _arc(image, vertex, a, b, color, width, dash):
    va = math.atan2(a[1] - vertex[1], a[0] - vertex[0])
    vb = math.atan2(b[1] - vertex[1], b[0] - vertex[0])
    sweep = (vb - va + math.pi) % (2 * math.pi) - math.pi
    radius = max(12.0, 0.35 * min(math.hypot(a[0] - vertex[0], a[1] - vertex[1]), math.hypot(b[0] - vertex[0], b[1] - vertex[1])))
    pts = []
    for step in range(13):
        ang = va + sweep * step / 12
        pts.append((
            int(round(vertex[0] + radius * math.cos(ang))),
            int(round(vertex[1] + radius * math.sin(ang))),
        ))
    for start, end in zip(pts, pts[1:]):
        _segment(image, start, end, color, width, dash)


def _trail(image, points, stroke, fill, width):
    count = len(points)
    for index in range(1, count):
        tint = tuple(int(channel * index / (count - 1)) for channel in stroke)
        cv2.line(image, points[index - 1], points[index], tint, width, cv2.LINE_AA)
    cv2.circle(image, points[-1], max(4, width + 2), fill, -1, cv2.LINE_AA)


def _pill(image, rect, text, font, spec, text_xy):
    x0, y0, x1, y1 = (int(round(v)) for v in rect)
    fill = _bgr(spec["fill"])
    stroke = _bgr(spec["stroke"])
    cv2.rectangle(image, (x0, y0), (x1, y1), fill, -1, cv2.LINE_AA)
    cv2.rectangle(image, (x0, y0), (x1, y1), stroke, 1, cv2.LINE_AA)
    _paint_text(image, [(text_xy, text, font, _ink(fill))])


def _place_label(anchor, text, font, boxes, faces, used, bounds):
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = probe.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 8
    pw, ph = tw + pad * 2, th + pad * 2
    width, height = bounds
    ax, ay = anchor
    for gap in (14, 32, 56, 88, 128):
        candidates = (
            (ax + gap, ay - ph / 2),
            (ax - pw - gap, ay - ph / 2),
            (ax - pw / 2, ay - ph - gap),
            (ax - pw / 2, ay + gap),
        )
        for left, top in candidates:
            rect = (left, top, left + pw, top + ph)
            if rect[0] < 4 or rect[1] < 4 or rect[2] > width - 4 or rect[3] > height - 4:
                continue
            if any(_overlaps(rect, box, 6) for box in boxes):
                continue
            if any(_point_in(rect, point, 8) for point in faces):
                continue
            if any(_overlaps(rect, other, 4) for other in used):
                continue
            text_xy = (int(round(rect[0] + pad - bbox[0])), int(round(rect[1] + pad - bbox[1])))
            return rect, gap > 14, text_xy
    return None


def _overlaps(rect, other, pad):
    return not (
        rect[2] < other[0] - pad or rect[0] > other[2] + pad
        or rect[3] < other[1] - pad or rect[1] > other[3] + pad
    )


def _point_in(rect, point, pad):
    return rect[0] - pad <= point[0] <= rect[2] + pad and rect[1] - pad <= point[1] <= rect[3] + pad


def _nearest(rect, point):
    return (
        int(round(min(max(point[0], rect[0]), rect[2]))),
        int(round(min(max(point[1], rect[1]), rect[3]))),
    )


def _panel(finding, moment, show, dropped, style, report, bundle, poses, held, alpha):
    bg = _rgb(style["panel"]["background"])
    text = _rgb(style["panel"]["text"])
    muted = _rgb(style["panel"]["muted"])
    image = Image.new("RGB", (PANEL_W, OUT_H), bg)
    draw = ImageDraw.Draw(image)
    title = _font(34, bold=True)
    body = _font(22, bold=False)
    kicker = _font(16, bold=True)
    number = _font(32, bold=True)
    court = _minimap(finding, moment, bundle, poses, held, alpha, style) if finding is not None else None
    floor = OUT_H - 28 - (court.size[1] + 16 if court is not None else 0)
    y = 48
    x = 36
    width = PANEL_W - 72
    if finding is None:
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    y = _write(draw, finding.headline, title, text, x, y, width, floor, 42)
    y = _write(draw, finding.explain, body, muted, x, y + 8, width, floor, 30)
    y += 18
    metrics = list(finding.metrics)
    if metrics:
        if metrics[0].value is not None:
            y = _write(draw, "Your number, clip summary", kicker, muted, x, y, width, floor, 22)
            y = _write(draw, _value(metrics[0]), number, text, x, y + 4, width, floor, 40)
            y = _write(draw, metrics[0].label, body, text, x, y, width, floor, 28)
            y = _write(draw, _confidence(metrics[0].confidence), body, muted, x, y, width, floor, 28)
            if _basis(metrics[0]) == "projected_2d":
                y = _write(draw, "projected", kicker, muted, x, y + 6, width, floor, 22)
                y = _write(draw, "rough estimate from the 2D picture", body, muted, x, y, width, floor, 28)
            valued = _valued_mark(moment)
            if valued is not None:
                y = _write(draw, f"This moment. {valued.label.strip()}", body, text, x, y + 12, width, floor, 28)
        for metric in metrics[1:]:
            if metric.value is None:
                continue
            y = _write(draw, metric.label, body, text, x, y + 8, width, floor, 28)
            y = _write(draw, _value(metric), body, text, x, y, width, floor, 28)
        targeted = [metric for metric in metrics if metric.target is not None]
        if targeted:
            y = _write(draw, "Practice guide", kicker, muted, x, y + 16, width, floor, 22)
            y = _write(draw, _target(targeted[0]), number, text, x, y + 4, width, floor, 40)
            y = _write(draw, "A coaching guide, not a tested personal target.", body, muted, x, y, width, floor, 28)
        best_metric = next((metric for metric in metrics if metric.best is not None), None)
        if best_metric is not None:
            y = _write(draw, "Your best", kicker, muted, x, y + 16, width, floor, 22)
            y = _write(draw, _num(best_metric.best, best_metric.unit), number, text, x, y + 4, width, floor, 40)
    if "preview_estimate" in report.warnings:
        y = _write(draw, "Preview estimate", body, muted, x, y + 12, width, floor, 28)
    if show == "best":
        y = _write(draw, "Your best one", kicker, text, x, y + 16, width, floor, 22)
    for label in dropped:
        y = _write(draw, label, body, text, x, y + 8, width, floor, 28)
    spark = _sparkline(finding, moment, report, style)
    if spark is not None and y + spark.size[1] + 12 < floor:
        image.paste(spark, (x, int(y + 12)))
    if court is not None:
        image.paste(court, (x, OUT_H - court.size[1] - 28))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _write(draw, text, font, fill, x, y, width, floor, line_h):
    if y >= floor:
        return y
    for line in _wrap(draw, str(text), font, width):
        if y + line_h > floor:
            break
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h
    return y


def _wrap(draw, text, font, width):
    words = text.split()
    if not words:
        return []
    lines = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textlength(trial, font=font) <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _sparkline(finding, moment, report, style):
    if not finding.overlay_series:
        return None
    role = _role(finding.player)
    name = finding.overlay_series[0]
    values = report.series.get(role, {}).get(name)
    if not values:
        return None
    start = max(0, moment.start_frame)
    end = min(len(values) - 1, moment.end_frame)
    window = list(values[start:end + 1])
    nums = [value for value in window if value is not None]
    if len(nums) < 2:
        return None
    metric = next((item for item in finding.metrics if item.series == name), None)
    low = min(nums)
    high = max(nums)
    if metric is not None and metric.target is not None:
        low = min(low, metric.target[0])
        high = max(high, metric.target[1])
    span = high - low or 1.0
    width, height = PANEL_W - 72, 72
    image = Image.new("RGB", (width, height), _rgb(style["panel"]["background"]))
    draw = ImageDraw.Draw(image)
    if metric is not None and metric.target is not None:
        top = height - 8 - (metric.target[1] - low) / span * (height - 16)
        bottom = height - 8 - (metric.target[0] - low) / span * (height - 16)
        draw.rectangle((0, top, width, bottom), fill=_rgb(style["marks"]["target"]["fill"]))
    pts = []
    for index, value in enumerate(window):
        if value is None:
            if len(pts) >= 2:
                draw.line(pts, fill=_rgb(style["panel"]["text"]), width=2)
            pts = []
            continue
        x = index / max(1, len(window) - 1) * (width - 1)
        y = height - 8 - (value - low) / span * (height - 16)
        pts.append((x, y))
    if len(pts) >= 2:
        draw.line(pts, fill=_rgb(style["panel"]["text"]), width=2)
    return image


def _minimap(finding, moment, bundle, poses, held, alpha, style):
    if _cat(finding) not in ("footwork", "tactics"):
        return None
    court = bundle.get("court")
    if not court:
        return None
    width_m = float(court["width_m"])
    length_m = float(court["length_m"])
    matrix = np.asarray(court["to_court"], dtype=np.float64)
    map_w, map_h = 220, 420
    pad = 16
    image = Image.new("RGB", (map_w, map_h), _rgb(style["panel"]["background"]))
    draw = ImageDraw.Draw(image)
    ink = _rgb(style["panel"]["text"])

    def plot(mx, my):
        x = pad + mx / width_m * (map_w - 2 * pad)
        y = pad + my / length_m * (map_h - 2 * pad)
        return x, y

    corners = [plot(0, 0), plot(width_m, 0), plot(width_m, length_m), plot(0, length_m)]
    draw.polygon(corners, outline=ink)
    net = [plot(0, length_m / 2), plot(width_m, length_m / 2)]
    draw.line(net, fill=_rgb(style["panel"]["muted"]), width=1)
    for role, pose in poses.items():
        color = _rgb(style["roles"].get(role, style.get("untracked", "#D7D7D7")))
        step = max(1, (moment.end_frame - moment.start_frame) // 30)
        for index in range(moment.start_frame, moment.end_frame + 1, step):
            point = _foot(pose, index, matrix)
            if point is not None:
                x, y = plot(*point)
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        current = _foot(pose, held, matrix)
        if current is not None:
            x, y = plot(*current)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    for mark in moment.marks:
        if mark.kind not in ("dot", "line"):
            continue
        mapped = [_project(matrix, point[0], point[1]) for point in mark.points]
        if any(point is None for point in mapped):
            continue
        plotted = [plot(*point) for point in mapped]
        if mark.kind == "dot":
            x, y = plotted[0]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=ink)
        else:
            draw.line(plotted, fill=ink, width=2)
    return image


def _foot(pose, index, matrix):
    if index < 0 or index >= len(pose["xy"]):
        return None
    pts = []
    for joint in _ANKLES:
        xy = pose["xy"][index, joint]
        if np.isfinite(xy).all():
            pts.append(xy)
    if not pts:
        return None
    mean = np.mean(pts, axis=0)
    return _project(matrix, float(mean[0]), float(mean[1]))


def _project(matrix, x, y):
    point = matrix @ np.array([x, y, 1.0], dtype=np.float64)
    if not np.isfinite(point).all() or abs(point[2]) < 1e-9:
        return None
    return float(point[0] / point[2]), float(point[1] / point[2])


def _contain(video_w, video_h, pane_w, pane_h):
    scale = min(pane_w / video_w, pane_h / video_h)
    nw = max(1, int(round(video_w * scale)))
    nh = max(1, int(round(video_h * scale)))
    ox = (pane_w - nw) // 2
    oy = (pane_h - nh) // 2
    return scale, ox, oy, nw, nh


def _fit(frame, geom):
    scale, ox, oy, nw, nh = geom
    pane = np.empty((OUT_H, PANE_W, 3), np.uint8)
    pane[:] = (24, 24, 24)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (nw, nh), interpolation=interp)
    pane[oy:oy + nh, ox:ox + nw] = resized
    return pane


def _zoom(pane, origin, zoom):
    if abs(zoom - 1.0) < 1e-3:
        return pane
    matrix = cv2.getRotationMatrix2D((float(origin[0]), float(origin[1])), 0.0, float(zoom))
    return cv2.warpAffine(
        pane, matrix, (PANE_W, OUT_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(24, 24, 24),
    )


def _zoom_origin(poses, role, key_frame, video_size, source_size, geom):
    pose = poses.get(role)
    if pose is None or key_frame >= len(pose["bbox"]) or not np.isfinite(pose["bbox"][key_frame]).all():
        return (PANE_W / 2, OUT_H / 2)
    box = pose["bbox"][key_frame]
    sx, sy = _scale(video_size, source_size)
    scale, ox, oy, _nw, _nh = geom
    cx = (box[0] + box[2]) / 2 * sx * scale + ox
    cy = (box[1] + box[3]) / 2 * sy * scale + oy
    return (cx, cy)


def _source_size(bundle, video_size):
    source = bundle.get("source") or {}
    width = source.get("width") or video_size[0]
    height = source.get("height") or video_size[1]
    return (float(width), float(height))


def _scale(video_size, source_size):
    return video_size[0] / source_size[0], video_size[1] / source_size[1]


def _video_point(point, sx, sy):
    return (int(round(point[0] * sx)), int(round(point[1] * sy)))


def _boxes(poses, held, alphas, sx, sy):
    boxes = []
    for role, pose in poses.items():
        box = _sample_box(pose, held, alphas.get(role, 0.0))
        if box is None or not np.isfinite(box).all():
            continue
        boxes.append((box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy))
    return boxes


def _faces(poses, held, alphas, sx, sy):
    points = []
    for role, pose in poses.items():
        xy, conf = _sample_joints(pose, held, alphas.get(role, 0.0))
        for index in _FACE:
            if index < len(conf) and conf[index] >= 0.5 and np.isfinite(xy[index]).all():
                points.append((float(xy[index, 0] * sx), float(xy[index, 1] * sy)))
    return points


def _sample_joints(pose, held, alpha):
    xy = pose["xy"]
    conf = pose["conf"]
    held = int(np.clip(held, 0, len(xy) - 1))
    left = xy[held].copy()
    scores = conf[held].copy()
    nxt = min(held + 1, len(xy) - 1)
    if alpha > 0 and nxt != held:
        right = xy[nxt]
        known = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
        left[known] = left[known] * (1.0 - alpha) + right[known] * alpha
        scores[known] = np.minimum(scores[known], conf[nxt][known])
    return left, scores


def _sample_box(pose, held, alpha):
    boxes = pose["bbox"]
    if len(boxes) == 0:
        return None
    held = int(np.clip(held, 0, len(boxes) - 1))
    left = boxes[held]
    if alpha <= 0 or not np.isfinite(left).all():
        return left
    right = boxes[min(held + 1, len(boxes) - 1)]
    if not np.isfinite(right).all():
        return left
    return left * (1.0 - alpha) + right * alpha


def _poses(bundle):
    decoded = {}
    for role, packed in bundle["poses"].items():
        xy = _decode(packed["xy"])
        xy = xy.astype(np.float64)
        unknown = packed["xy"].get("unknown", -32768)
        xy[xy == unknown] = np.nan
        xy *= float(packed["xy"].get("scale", 0.1))
        conf = _decode(packed["conf"]).astype(np.float64) * float(packed["conf"].get("scale", 0.01))
        measured = _decode(packed["measured"]).astype(bool)
        bbox = _decode(packed["bbox"]).astype(np.float64)
        box_unknown = packed["bbox"].get("unknown", -32768)
        bbox[bbox == box_unknown] = np.nan
        bbox *= float(packed["bbox"].get("scale", 0.1))
        decoded[role] = {"xy": xy, "conf": conf, "measured": measured, "bbox": bbox}
    return decoded


def _decode(packed):
    dtype = {"int16": "<i2", "uint8": "|u1"}[packed["dtype"]]
    raw = np.frombuffer(base64.b64decode(packed["b64"]), dtype=np.dtype(dtype))
    return raw.reshape(packed["shape"]).copy()


def _cut_starts(doc, n_frames):
    starts = set()
    for cut in doc.get("cuts") or ():
        if isinstance(cut, bool) or not isinstance(cut, (int, float, np.integer, np.floating)):
            continue
        number = float(cut)
        if not math.isfinite(number) or not number.is_integer():
            continue
        cut = int(number)
        if 0 < cut < n_frames:
            starts.add(cut)
    return starts


def _joint_index(names, name):
    try:
        return names.index(name)
    except ValueError:
        return -1


def _role(player):
    return player.value if hasattr(player, "value") else str(player)


def _cat(finding):
    category = finding.category
    return category.value if hasattr(category, "value") else str(category)


def _valued_mark(moment):
    if moment is None:
        return None
    for mark in moment.marks:
        if mark.value is not None and (mark.label or "").strip():
            return mark
    return None


def _basis(metric):
    basis = metric.basis
    return basis.value if hasattr(basis, "value") else str(basis)


def _confidence(value):
    if value >= 0.75:
        return "high"
    if value >= 0.5:
        return "medium"
    return "low"


def _num(value, unit):
    return f"{value:.3g} {unit}".strip()


def _value(metric):
    return _num(metric.value, metric.unit)


def _target(metric):
    low, high = metric.target
    return f"{low:.3g} to {high:.3g} {metric.unit}".strip()


def _rgb(hex_color):
    text = hex_color.lstrip("#")
    return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))


def _bgr(hex_color):
    red, green, blue = _rgb(hex_color)
    return (blue, green, red)


def _ink(fill_bgr):
    blue, green, red = fill_bgr
    lum = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return (28, 28, 28) if lum > 160 else (255, 255, 255)


def _font(size, bold):
    key = (size, bold)
    hit = _FONTS.get(key)
    if hit is not None:
        return hit
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / name
    hit = ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()
    _FONTS[key] = hit
    return hit


def _paint_text(image, items):
    if not items:
        return
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for xy, text, font, color in items:
        blue, green, red = color
        draw.text(xy, text, font=font, fill=(red, green, blue))
    image[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _stamp(base, paint, alpha):
    if alpha <= 0:
        return
    if alpha >= 0.999:
        paint(base)
        return
    layer = base.copy()
    paint(layer)
    _stamp_array(base, layer, alpha)


def _stamp_array(base, layer, alpha):
    _stamp_drawn(base, layer, base, alpha)


def _stamp_drawn(base, layer, origin, alpha):
    if alpha <= 0:
        return
    changed = np.any(layer != origin, axis=2)
    if not np.any(changed):
        return
    if alpha >= 0.999:
        base[changed] = layer[changed]
        return
    mixed = cv2.addWeighted(layer, float(alpha), base, 1.0 - float(alpha), 0.0)
    base[changed] = mixed[changed]
