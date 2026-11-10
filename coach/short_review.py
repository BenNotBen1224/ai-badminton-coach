"""Vertical teaching excerpts: measured examples, not automated pro-player grading."""
from __future__ import annotations

import argparse
import copy
import html
import json
import math
import os
import subprocess
import tempfile
import time
import uuid
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .app.backend import PROJECT, FastBackend, sha256_file
from .app.pipeline import analyze_video
from .player_input import player_inputs
from .coaching_points import PRIORITIES, corpus_description, evidence_candidates, priorities, rank_candidates
from .coaching_corpus.corpus_learn import HEDGE_ZH
from .complete_review import VOICE, _voice, _wrap_zh
from .export import _tools
from .play_state import exchange_mask
from .schema import Role
from .tracks import JOINT_INDEX, load_tracks

FPS = 30
WIDTH, HEIGHT = 1080, 1920
SAFE = (64, 210, 900, 1510)
VIEW = (40, 440, 920, 1170)
FREEZE_S = 2.0
MAX_FREEZE_S = 3.5
MAX_PAUSED_SHARE = 0.35
SPEECH_SPEED = 1.12
HOOK = "这一跨，学三个关键"
ATTRIBUTION = (
    "原片：Badminton Sphere｜China Open 2025 女单决赛剪辑\n"
    "CC BY 3.0｜Wikimedia Commons\n"
    "https://commons.wikimedia.org/wiki/File:Wang_Zhiyi_triumph_in_all-Chinese_China_Open_finals_I_China_Open_2025.webm\n"
    "https://www.youtube.com/watch?v=CKErzNzPSvQ\n"
    "许可：https://creativecommons.org/licenses/by/3.0/\n"
    "改编：节选、竖屏重构、慢放、放大、模型标注、中文神经旁白。\n"
    "原片含多个回合及切镜，并非连续整场比赛。无球员或赛事背书。\n"
    "Commons 的许可标注不等于上游赛事转播权已获清理；公开发布前需自行确认权利。"
)
SOURCE_COMMONS = "https://commons.wikimedia.org/wiki/File:Wang_Zhiyi_triumph_in_all-Chinese_China_Open_finals_I_China_Open_2025.webm"
DEFAULT_CREDIT = "Badminton Sphere · CC BY 3.0\nWikimedia Commons · 剪辑与标注"
CHANGED_SOURCES = (
    "coach/short_review.py", "coach/verify_short.py", "coach/coaching_points.py",
    "coach/player_input.py", "coach/events.py", "coach/signals_body.py",
    "coach/analyze_body.py", "coach/analyze_move.py", "coach/play_state.py",
    "coach/tracks.py", "coach/synth.py", "coach/app/pipeline.py",
    "coach/coaching_corpus/coaching_priorities.json", "coach/coaching_corpus/make_priorities.py",
    "coach/coaching_corpus/corpus_learn.py", "sample-videos/china-open-source-attribution.json",
    "coach/coaching_corpus/describe_corpus.py", "coach/coaching_corpus/provenance_counts.json",
    "coach/tests/test_player_input.py", "coach/tests/test_lunge_stride.py",
    "coach/tests/test_coaching_points.py", "coach/tests/test_short_review.py",
    "coach/tests/test_body.py", "coach/tests/test_complete_review.py", "coach/README.md",
    "coach/chat/engine.py", "coach/tests/test_chat.py",
)


def source_credit(video, attribution_file=None):
    video = Path(video)
    if attribution_file is not None:
        credit = json.loads(Path(attribution_file).read_text("utf-8"))
        for key in ("author", "license", "source_url", "license_url"):
            if not isinstance(credit.get(key), str) or not credit[key].strip():
                raise ValueError(f"Attribution JSON requires a nonempty {key}")
        return {
            "endcard": f'{credit["author"]} · {credit["license"]}\n节选、慢放、放大、模型标注',
            "full": (f'原片：{credit["author"]}\n{credit["source_url"]}\n'
                     f'{credit["license"]} · {credit["license_url"]}\n'
                     '改编：节选、竖屏重构、慢放、放大、模型标注、中文神经旁白。\n'
                     '来源许可信息由导出者提供，不代表上游赛事转播权已获清理；公开发布前需确认权利。'),
        }
    known = video.parent / "china-open-source-attribution.json"
    if known.is_file():
        metadata = json.loads(known.read_text("utf-8"))
        if (metadata.get("source", {}).get("commons_page") == SOURCE_COMMONS
                and any(clip.get("file") == video.name for clip in metadata.get("clips", []))):
            return {"full": ATTRIBUTION, "endcard": DEFAULT_CREDIT}
    raise FileNotFoundError("Source credit unavailable; supply --attribution-json with author, license, source_url and license_url")


def _dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")


def _supported(tracks, active, role, frame, joints):
    player = tracks.players[role]
    return bool(active[frame] and player.measured[frame]
                and all(player.joint_visible(name)[frame] for name in joints))


def _crop(box, width, height):
    x1, y1, x2, y2 = (float(value) for value in box)
    if not np.isfinite(box).all() or x2 <= x1 or y2 <= y1:
        raise ValueError("Selected player has no usable measured bounding box")
    crop_h = max((y2 - y1) * 1.8, (x2 - x1) * 1.6 / (880 / 730))
    crop_w = crop_h * 880 / 730
    scale = min(1.0, width / crop_w, height / crop_h)
    crop_w, crop_h = crop_w * scale, crop_h * scale
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(cx - crop_w / 2, 0), width - crop_w)
    top = min(max(cy - crop_h / 2, 0), height - crop_h)
    rect = (int(math.floor(left)), int(math.floor(top)),
            int(math.ceil(left + crop_w)), int(math.ceil(top + crop_h)))
    if not (rect[0] <= x1 <= x2 <= rect[2] and rect[1] <= y1 <= y2 <= rect[3]):
        raise ValueError("Focus crop cannot contain the player's complete box")
    return rect


def select_points(doc, report):
    tracks = load_tracks(doc)
    selected = rank_candidates(evidence_candidates(doc, report))
    for point in selected:
        point["freeze_crop"] = list(_crop(point["bbox"], tracks.width, tracks.height))
        value = point["value"]
        if point["concept"] == "footwork":
            side = "右" if point["hand"] == "right" else "左"
            point.update({
                "title": f"持拍脚（{side}脚）跨出去",
                "evidence": f"跨步脚距 {value:.2f} 身高",
                "clauses": [f"关键是让持拍脚先跨出去，",
                            f"看{side}脚越过另一只脚，落地后接住身体，",
                            "后脚留在身后，把这一进一退的步法练顺。"],
                "teaching_principle": "持拍脚领跨，观察前后脚与身体的衔接；脚距是此帧投影脚距，不是整步位移或个人标准。",
            })
        elif point["concept"] == "knee_bend":
            side = "右" if point["cycle"]["lead_side"] == "right" else "左"
            arm_cue = ("同时把非持拍手展开，" if point.get("balance_arm")
                       else "让上身跟上腿部的下降，")
            finish = (f"看她{side}膝弯下去，另一只手保持伸展，让跨步和上身衔接起来。"
                      if point.get("balance_arm") else f"看{side}膝弯下去，把跨步和身体下降连起来练。")
            point.update({
                "title": "前膝屈曲，身体下降",
                "evidence": f"{side}膝屈曲 {value:.0f}°",
                "clauses": ["关键是前腿屈膝，身体跟着下降，",
                            arm_cue, finish],
                "teaching_principle": "前膝与躯干一起下降，非持拍臂伸展是平衡教学线索；二维姿态不能直接测量重心、受力或平衡效果。",
            })
        elif point["concept"] == "recovery":
            point.update({
                "title": "起身回位，接下一步",
                "evidence": f"前膝回伸 {value:.2f} 秒",
                "clauses": ["关键是跨出去后马上起身回位，衔接下一步，",
                            f"看前膝从深屈到伸展，用时约{value:.2f}秒，",
                            "接着又向另一侧移动，把起身和下一步连起来练。"],
                "teaching_principle": "练习跨步后的起身与下一次移动的连续性；计时是前膝恢复到20°以内，不是触球后回中时间，也不是测得蹬地力。",
            })
        else:
            title = point["priority"]["label_zh"]
            point.update({"title": title, "evidence": f'{value:.2f} {point["unit"]}',
                          "clauses": [f"关键是把{title}和下一次移动连起来练，",
                                      "先看这一刻身体的位置，再看动作前后的衔接。"],
                          "teaching_principle": "一般教学线索，单帧数值不是个人动作标准。"})
        point["cue"] = "".join(point["clauses"])
        if HEDGE_ZH.search(point["cue"]):
            raise RuntimeError("Teaching narration contains hedge text; keep limitations in the written analysis")
    return selected


def synthesize(points, model_dir):
    tts = _voice(model_dir)
    rows = []
    for point in points:
        parts = []
        for text in point["clauses"]:
            generated = tts.generate(text, sid=0, speed=SPEECH_SPEED)
            samples = np.asarray(generated.samples, dtype=np.float32)
            rate = int(generated.sample_rate)
            if not samples.size or not np.isfinite(samples).all() or np.max(np.abs(samples)) < 0.01:
                raise RuntimeError("Local neural voice returned silent or invalid speech")
            # Captions use the actual non-silent boundaries of each synthesized clause.
            voiced = np.flatnonzero(np.abs(samples) > 0.003)
            start = max(0, int(voiced[0]) - round(0.03 * rate))
            end = min(len(samples), int(voiced[-1]) + round(0.06 * rate))
            samples = samples[start:end]
            parts.append({"text": text, "samples": samples, "rate": rate})
        rows.append(parts)
    if len({part["rate"] for parts in rows for part in parts}) != 1:
        raise RuntimeError("Neural voice sample rates do not agree")
    return rows


def make_schedule(points, speech_lengths, source_fps, source_n):
    points = copy.deepcopy(points)
    frame_map, states, segments, captions = [], [], [], []

    def append(point_index, source, phase):
        frame_map.append(int(source))
        states.append({"point": point_index, "phase": phase})

    for _ in range(round(1.2 * FPS)):
        append(0, points[0]["source_frame"], "hook")
    for index, (point, lengths) in enumerate(zip(points, speech_lengths)):
        start_s = len(frame_map) / FPS
        audio_s = sum(lengths) + 0.08 * (len(lengths) - 1)
        duration = max(9.4, audio_s + 0.35)
        start, key, end = point["source_start_frame"], point["source_frame"], point["source_end_frame"]
        if not (0 <= start < key < end < source_n and point["shot"][0] <= start <= end <= point["shot"][1]):
            raise ValueError("Teaching point must have action on both sides within one camera shot")
        freeze_count = round(FREEZE_S * FPS)
        total = math.ceil(duration * FPS)
        moving = total - freeze_count
        inbound = max(round(0.8 * FPS), round(moving * (key - start) / (end - start)))
        outbound = moving - inbound
        if outbound < 1:
            raise ValueError("No motion remains after the freeze")
        for frame in np.linspace(start, key, inbound, endpoint=False):
            append(index, math.floor(frame), "motion")
        freeze_start = len(frame_map) / FPS
        for _ in range(freeze_count):
            append(index, key, "freeze")
        freeze_end = len(frame_map) / FPS
        for frame in np.linspace(key, end, outbound + 1)[1:]:
            append(index, math.floor(frame), "motion")
        end_s = len(frame_map) / FPS
        cursor = start_s + 0.10
        for text, length in zip(point["clauses"], lengths):
            captions.append({"point": index, "text": text, "start_s": cursor,
                             "end_s": cursor + length, "speech_start_s": cursor,
                             "speech_end_s": cursor + length})
            cursor += length + 0.08
        point.update({"output_start_s": start_s, "output_end_s": end_s,
                      "freeze_start_s": freeze_start, "freeze_end_s": freeze_end})
        segments.append({"kind": "point", "point": index, "start_s": start_s, "end_s": end_s,
                         "freeze_start_s": freeze_start, "freeze_end_s": freeze_end})
    for _ in range(round(1.3 * FPS)):
        append(len(points) - 1, frame_map[-1], "credits")
    paused = sum(state["phase"] != "motion" for state in states) / FPS
    duration = len(frame_map) / FPS
    if not 30 <= duration <= 45:
        raise RuntimeError(f"Short review duration {duration:.2f}s is outside 30-45s")
    if paused / duration > MAX_PAUSED_SHARE:
        raise RuntimeError("Short review exceeds the paused-time cap")
    for point in points:
        if point["freeze_end_s"] - point["freeze_start_s"] > MAX_FREEZE_S:
            raise RuntimeError("Short review freeze exceeds its cap")
    return {
        "fps": FPS, "width": WIDTH, "height": HEIGHT, "planned_duration_s": duration,
        "planned_output_frames": len(frame_map), "paused_s": paused,
        "moving_s": duration - paused, "paused_share": paused / duration,
        "points": points, "segments": segments, "captions": captions,
        "frame_map": frame_map, "states": states, "text_safe_rect": list(SAFE),
        "caption_rect": [72, 1285, 892, 1495],
    }


def _srt_time(seconds):
    ms = round(seconds * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def _text(draw, text, xy, font, fill, width=810, line_h=66):
    x, y = xy
    boxes = []
    for line in _wrap_zh(draw, text, font, width):
        box = draw.textbbox((x, y), line, font=font)
        if not (SAFE[0] <= box[0] and SAFE[1] <= box[1] and box[2] <= SAFE[2] and box[3] <= SAFE[3]):
            raise RuntimeError(f"Text would enter platform UI zones: {line!r} {box}")
        draw.text((x, y), line, font=font, fill=fill, stroke_width=1)
        boxes.append(list(box))
        y += line_h
    return boxes


def _focus_overlay(raw, player, frame, point):
    image = raw.copy()
    color = (110, 245, 214)
    width = max(3, round((player.bbox[frame, 3] - player.bbox[frame, 1]) / 55))
    sides = ("left", "right")
    for side in sides:
        names = [f"{side}_{joint}" for joint in ("hip", "knee", "ankle")]
        for left, right in zip(names, names[1:]):
            if not (player.joint_visible(left)[frame] and player.joint_visible(right)[frame]):
                continue
            a, b = [tuple(np.rint(player.joint(name)[frame]).astype(int)) for name in (left, right)]
            cv2.line(image, a, b, (10, 20, 25), width + 4, cv2.LINE_AA)
            cv2.line(image, a, b, color, width, cv2.LINE_AA)
        for name in names:
            if player.joint_visible(name)[frame]:
                xy = tuple(np.rint(player.joint(name)[frame]).astype(int))
                cv2.circle(image, xy, width + 2, (10, 20, 25), -1, cv2.LINE_AA)
                cv2.circle(image, xy, width, color, -1, cv2.LINE_AA)
    if point["concept"] == "footwork" and all(player.joint_visible(j)[frame] for j in ("left_ankle", "right_ankle")):
        a, b = [tuple(np.rint(player.joint(j)[frame]).astype(int)) for j in ("left_ankle", "right_ankle")]
        cv2.line(image, a, b, (255, 220, 100), width, cv2.LINE_AA)
    if point["concept"] == "knee_bend":
        for side in (point["cycle"]["lead_side"],):
            if not all(player.joint_visible(f"{side}_{j}")[frame] for j in ("hip", "knee", "ankle")):
                continue
            a, vertex, b = [player.joint(f"{side}_{j}")[frame] for j in ("hip", "knee", "ankle")]
            extension = vertex - a
            start = math.degrees(math.atan2(*extension[::-1]))
            finish = math.degrees(math.atan2(*(b - vertex)[::-1]))
            span = (finish - start) % 360
            if span > 180:
                start, span = finish, 360 - span
            radius = float(min(np.linalg.norm(a - vertex), np.linalg.norm(b - vertex)) * 0.45)
            extension_end = vertex + extension / np.linalg.norm(extension) * radius * 1.5
            cv2.line(image, tuple(np.rint(vertex).astype(int)), tuple(np.rint(extension_end).astype(int)),
                     (255, 220, 100), max(1, width // 2), cv2.LINE_AA)
            arc = np.array([vertex + radius * np.array([math.cos(t), math.sin(t)])
                            for t in np.linspace(math.radians(start), math.radians(start + span), 40)])
            cv2.polylines(image, [np.rint(arc).astype(np.int32)], False, (255, 220, 100), width, cv2.LINE_AA)
        if point.get("balance_arm"):
            side = point["balance_arm"]["side"]
            for left, right in ((f"{side}_shoulder", f"{side}_elbow"), (f"{side}_elbow", f"{side}_wrist")):
                if all(player.joint_visible(name)[frame] for name in (left, right)):
                    a, b = [tuple(np.rint(player.joint(name)[frame]).astype(int)) for name in (left, right)]
                    cv2.line(image, a, b, (10, 20, 25), width + 4, cv2.LINE_AA)
                    cv2.line(image, a, b, (255, 220, 100), width, cv2.LINE_AA)
                    cv2.circle(image, b, width, (255, 220, 100), -1, cv2.LINE_AA)
    return image


def _fit_view(raw, crop, zoom):
    left, top, right, bottom = crop
    if zoom > 0:
        height, width = raw.shape[:2]
        rect = np.array([0, 0, width, height]) * (1 - zoom) + np.array(crop) * zoom
        left, top, right, bottom = np.rint(rect).astype(int)
        raw = raw[top:bottom, left:right]
    target_w, target_h = VIEW[2] - VIEW[0], VIEW[3] - VIEW[1]
    scale = min(target_w / raw.shape[1], target_h / raw.shape[0])
    size = (round(raw.shape[1] * scale), round(raw.shape[0] * scale))
    pane = np.zeros((target_h, target_w, 3), np.uint8)
    resized = cv2.resize(raw, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    x, y = (target_w - size[0]) // 2, (target_h - size[1]) // 2
    pane[y:y + size[1], x:x + size[0]] = resized
    return pane


def render_frame(raw, tracks, point, state, wall, captions, fonts, credit_text=DEFAULT_CREDIT):
    player = tracks.players[Role(point["player"])]
    source = state["source_frame"]
    phase = state["phase"]
    annotated = _focus_overlay(raw, player, source, point)
    crop = _crop(player.bbox[source], tracks.width, tracks.height) if player.present[source] else point["freeze_crop"]
    if phase in ("hook", "freeze"):
        zoom = 1.0
    elif phase == "motion":
        distance = min(abs(wall - point["freeze_start_s"]), abs(wall - point["freeze_end_s"]))
        zoom = max(0.0, 1 - distance / 0.65)
    else:
        zoom = 0.0
    canvas = np.full((HEIGHT, WIDTH, 3), (30, 22, 14), np.uint8)
    canvas[VIEW[1]:VIEW[3], VIEW[0]:VIEW[2]] = _fit_view(annotated, crop, zoom)
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    title, subtitle, body, small = fonts
    boxes = []
    corpus = corpus_description()
    header = (f'{corpus["coaching_videos"]}个教学视频'
              if phase == "hook" else "向高手学步法 · 技术拆解")
    boxes += _text(draw, header, (72, 215), small, "#6ef5d6")
    boxes += _text(draw, HOOK if phase == "hook" else point["title"], (72, 275), title, "#f5f8ff")
    index = state["point"] + 1
    boxes += _text(draw, f"{index}/3  ·  {'定格看结构' if phase == 'freeze' else '慢放看衔接'}", (72, 380), small, "#aec5d5")
    if point.get("balance_arm") and phase == "freeze":
        arm = point["balance_arm"]
        side = "左" if arm["side"] == "left" else "右"
        boxes += _text(draw, f'{side}臂伸展 {arm["elbow_deg"]:.0f}° · 2D估计',
                       (72, 1100), small, "#98dfff")
    evidence = point["evidence"] if phase in ("freeze", "hook") else point["title"]
    boxes += _text(draw, evidence, (72, 1194), subtitle, "#6ef5d6", line_h=52)
    boxes += _text(draw, "2D估计", (72, 1260), small, "#aec5d5", line_h=40)
    caption = next((row for row in captions if row["start_s"] <= wall < row["end_s"]), None)
    if phase == "hook":
        text = "跨出去 → 降下来 → 接下一步"
    elif phase == "credits":
        text = credit_text
    else:
        text = caption["text"] if caption else ""
    boxes += _text(draw, text, (72, 1320), body if phase != "credits" else small, "#ffffff", line_h=70)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), boxes


def export_short(video, doc, report, out_mp4, *, model_dir=VOICE, attribution_file=None):
    out = Path(out_mp4)
    folder = out.parent
    folder.mkdir(parents=True, exist_ok=True)
    credit = source_credit(video, attribution_file)
    tracks = load_tracks(doc)
    points = select_points(doc, report)
    speeches = synthesize(points, model_dir)
    lengths = [[len(part["samples"]) / part["rate"] for part in parts] for parts in speeches]
    schedule = make_schedule(points, lengths, tracks.fps, tracks.n)
    schedule["credit"] = credit
    frame_map, states = schedule.pop("frame_map"), schedule.pop("states")
    _dump(folder / "schedule.json", schedule)
    _dump(folder / "frame-map.json", {"fps": FPS, "source_frame_per_output_frame": frame_map,
                                    "states": states})
    _dump(folder / "points.json", {"points": schedule["points"], "source": report.source,
                                  "source_sha256": sha256_file(Path(video)),
                                  "framing": "按教学语料优先级，学高手的持拍脚跨步、前膝屈伸与下一步衔接。",
                                  "corpus": corpus_description(),
                                  "priorities_sha256": sha256_file(PRIORITIES),
                                  "limits": ["单摄像机二维投影，不是三维、力或医疗结论。",
                                             "未追踪羽毛球，不确认触球或分腿垫步时机。",
                                             "前膝回伸时间不是触球后回中时间；单帧值不冒充片段汇总。",
                                             "原片是多回合剪辑，节选不跨切镜。"]})
    print(json.dumps({"planned_video": str(out.resolve()),
                      "planned_duration_s": schedule["planned_duration_s"],
                      "output_frames": len(frame_map)}), flush=True)
    rate = speeches[0][0]["rate"]
    sound = np.zeros(math.ceil(schedule["planned_duration_s"] * rate), np.float32)
    for caption, part in zip(schedule["captions"], [part for parts in speeches for part in parts]):
        start = round(caption["start_s"] * rate)
        samples = part["samples"]
        sound[start:start + len(samples)] = samples
    (folder / "captions.srt").write_text("\n\n".join(
        f'{index}\n{_srt_time(row["start_s"])} --> {_srt_time(row["end_s"])}\n{row["text"]}'
        for index, row in enumerate(schedule["captions"], 1)) + "\n", "utf-8")
    (folder / "post.txt").write_text(
        "向高手学这一次跨步：持拍脚跨出去，前膝屈曲降下来，起身就接下一步。\n"
        "结合慢放与真实模型关键点，讲一般教学原则，不给职业选手判错。\n"
        "数字为单摄像机二维估计，不是个人标准；未确认触球。\n"
        "#羽毛球 #技术分析 #步法 #运动学习\n\n" + credit["full"], "utf-8")
    cards = []
    for point in schedule["points"]:
        captions = [row for row in schedule["captions"] if row["point"] == schedule["points"].index(point)]
        cards.append(
            f'<article><h2>{html.escape(point["title"])}</h2><p>{html.escape(point["evidence"])} · 2D estimate</p>'
            f'<p>语料优先级 #{point["priority"]["rank"]}：{point["priority"]["coaching_videos_mentioning"]} 个教学视频提到，'
            f'{point["priority"]["coaching_videos_stressing"]} 个强调；分组加权强调占比 '
            f'{point["priority"]["channel_stressed_share"]:.1%}。有真实动作证据才纳入，频次不是教学有效性的临床证明。</p>'
            f'<p>{html.escape(point["teaching_principle"])}</p>'
            f'<p>源帧 {point["source_frame"]} · 原片 {point["source_time_s"]:.2f}s · '
            f'输出 {point["output_start_s"]:.2f}–{point["output_end_s"]:.2f}s · '
            f'定格 {point["freeze_start_s"]:.2f}–{point["freeze_end_s"]:.2f}s</p>'
            f'<p>此帧值 {point["value"]:.5g} {point["unit"]}，n={point["n"]}，模型置信度 {point["confidence"]:.2f}。'
            '置信度不是教学建议正确率。此例是一个动作周期，原始关节点和周期端点见 points.json。</p>'
            + "".join(f'<p>{row["start_s"]:.2f}–{row["end_s"]:.2f}s：{html.escape(row["text"])}</p>'
                      for row in captions) + '</article>')
    (folder / "analysis.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>向高手学脚下</title>'
        '<style>body{background:#0e181e;color:#edf6ff;font:18px/1.7 "Microsoft YaHei",sans-serif;'
        'max-width:950px;margin:32px auto;padding:24px}video{height:75vh;max-width:100%}'
        'article{padding:24px;background:#1a2c35;margin:24px 0;border-radius:12px}a{color:#6ef5d6}</style>'
        '<h1>向高手学脚下 · 三个技术观察</h1><video controls src="tiktok.mp4" poster="cover.jpg"></video>'
        '<p><a download href="tiktok.mp4">下载 MP4</a> · <a download href="points.json">证据</a> · '
        '<a download href="captions.srt">字幕</a> · <a download href="post.txt">发布文案与署名</a></p>'
        + "".join(cards)
        + '<h2>为什么换掉上一版的三个点</h2><p>开放转录数据获取819条，其中254条与羽毛球相关，'
        '筛出100个教学视频：17个已知频道名称，另有23个视频缺少频道信息，合计40个加权分组。'
        '分组数不等于已核实的频道数。YouTube限制批量获取，没有达到1000个。'
        '依照 coaching_priorities.json 的顺序，仅选择有本片动作证据的概念。站距排第25，不能抢在有证据的步法、屈膝、回位之前。</p>'
        '<h2>这段镜头看不到的高优先级</h2><p>握拍（第1）、落点与战术（第2）、前臂旋转与手腕（第3）'
        '在语料中更常被强调，但本片球拍模糊，也未追踪羽毛球，不能用姿态骨架补猜这些信息。</p>'
        '<h2>修正了两处根因</h2><p>原先按手腕快速运动猜持拍手，误把左侧平衡臂当作持拍臂。'
        '现在使用逐球员的显式输入：本片近、远端均为右手，依据CORPUS/checks/near_swings.jpg与far_swings.jpg。'
        '未声明时为unknown，不输出持拍臂测量或挥拍定时。</p>'
        '<p>原先只用髋高下降找弓步，容易被高机位与透视缩短影响。新增独立的前膝屈曲、脚距、'
        '领跨脚运动及可见回伸周期。此例自动找到右脚落步、右膝深屈与起身；没有把膝角假写成髋高下降。'
        '0.33秒仅为膝屈曲峰值到恢复20°以内的时间，不等于回到场地中心，未测蹬地力，也未确认触球。</p>'
        '<h2>边界</h2><p>单摄像机二维投影，不是三维、力或医疗结论。'
        '未追踪羽毛球，不确认触球。一般教学原则不是职业选手动作对错的判决。'
        '前膝数值是单侧屈曲而非内角；非持拍手伸展只是平衡教学线索。'
        '原片是多回合剪辑，不是整场比赛。</p>'
        '<p>此例不代表六片段质量与速度验收通过，聊天安全问题也未在此验收。</p>'
        '<h2>来源与许可</h2><pre style="white-space:pre-wrap">' + html.escape(credit["full"]) + '</pre></html>', "utf-8")
    font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    fonts = tuple(ImageFont.truetype(str(font_dir / "msyh.ttc"), size) for size in (58, 50, 44, 30))
    ffmpeg, ffprobe = _tools()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot read source {video}")
    text_boxes = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="short-", dir=folder) as tmp:
        tmp = Path(tmp)
        with wave.open(str(tmp / "voice.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes((np.clip(sound, -1, 1) * 32767).astype("<i2").tobytes())
        with (tmp / "encoder.log").open("wb") as errors:
            proc = subprocess.Popen([
                str(ffmpeg), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", "1080x1920", "-r", str(FPS), "-i", "pipe:0", "-i", str(tmp / "voice.wav"),
                "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out),
            ], stdin=subprocess.PIPE, stderr=errors)
            cached = -1
            try:
                for index, (source, state) in enumerate(zip(frame_map, states)):
                    if source != cached:
                        if source != cached + 1:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, source)
                        ok, raw = cap.read()
                        if not ok:
                            raise RuntimeError(f"Could not decode frame {source}")
                        cached = source
                    point = schedule["points"][state["point"]]
                    image, boxes = render_frame(raw, tracks, point, {**state, "source_frame": source},
                                               index / FPS, schedule["captions"], fonts, credit["endcard"])
                    if index == 0:
                        if not cv2.imwrite(str(folder / "cover.jpg"), image):
                            raise RuntimeError("Cannot write cover")
                    if state["phase"] == "freeze" and (
                            index == 0 or states[index - 1]["phase"] != "freeze"):
                        cv2.imwrite(str(folder / f'point-{state["point"] + 1}.jpg'), image)
                    text_boxes.extend(boxes)
                    proc.stdin.write(image.tobytes())
                proc.stdin.close()
                if proc.wait() != 0:
                    raise RuntimeError((tmp / "encoder.log").read_text("utf-8", errors="replace"))
            finally:
                cap.release()
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        subprocess.run([str(ffmpeg), "-v", "error", "-i", str(out), "-f", "null", "-"],
                       check=True, capture_output=True)
        probe = json.loads(subprocess.check_output([
            str(ffprobe), "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(out)]))
        decoded = subprocess.check_output([str(ffmpeg), "-v", "error", "-i", str(out),
                                           "-map", "0:a:0", "-f", "f32le", "-ac", "1", "pipe:1"])
    audio = np.frombuffer(decoded, dtype="<f4")
    video_stream = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    audio_stream = next(stream for stream in probe["streams"] if stream["codec_type"] == "audio")
    checks = {
        "full_decode": True, "non_silent_audio": bool(audio.size and np.max(np.abs(audio)) > 0.01),
        "output_shape": [video_stream["width"], video_stream["height"]],
        "video_frames": int(video_stream["nb_read_frames"]),
        "video_duration_s": float(video_stream["duration"]), "audio_duration_s": float(audio_stream["duration"]),
        "captions_within_measured_speech": all(row["speech_start_s"] <= row["start_s"] < row["end_s"] <= row["speech_end_s"]
                                             for row in schedule["captions"]),
        "text_safe_rect": list(SAFE), "all_text_inside_safe_rect": all(
            SAFE[0] <= b[0] <= b[2] <= SAFE[2] and SAFE[1] <= b[1] <= b[3] <= SAFE[3] for b in text_boxes),
        "paused_s": schedule["paused_s"], "moving_s": schedule["moving_s"],
        "paused_share": schedule["paused_share"], "freeze_cap_s": MAX_FREEZE_S,
        "render_s": time.perf_counter() - started, "sha256": sha256_file(out),
        "voice": {"engine": "sherpa-onnx", "model": "MeloTTS-Chinese", "speed": SPEECH_SPEED,
                  "sample_rate": rate, "local": True, "model_sha256": sha256_file(Path(model_dir) / "model.onnx")},
        "speech_naturalness": "Not independently rated by a human",
        "crop_contains_focus_box_at_freezes": all(
            p["freeze_crop"][0] <= p["bbox"][0] and p["freeze_crop"][1] <= p["bbox"][1]
            and p["freeze_crop"][2] >= p["bbox"][2] and p["freeze_crop"][3] >= p["bbox"][3]
            for p in schedule["points"]),
        "no_point_spans_cut": all(p["shot"][0] <= p["source_start_frame"] <= p["source_frame"]
                                  <= p["source_end_frame"] <= p["shot"][1] for p in schedule["points"]),
    }
    if (checks["output_shape"] != [WIDTH, HEIGHT] or checks["video_frames"] != len(frame_map)
            or abs(checks["audio_duration_s"] - schedule["planned_duration_s"]) > 0.1
            or not checks["non_silent_audio"] or not checks["all_text_inside_safe_rect"]):
        raise RuntimeError("Encoded short review failed media validation")
    _dump(folder / "verification.json", checks)
    return schedule, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--voice-model", type=Path, default=VOICE)
    parser.add_argument("--attribution-json", type=Path)
    parser.add_argument("--near-hand", choices=("left", "right", "unknown"))
    parser.add_argument("--far-hand", choices=("left", "right", "unknown"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Use a new output folder: {args.out}")
    source_credit(args.video.resolve(), args.attribution_json)
    from turbo.accept import acquire_cpu_lock, release_cpu_lock
    lock = PROJECT / "work" / "cpu.lock"
    token = f"short-review {os.getpid()} {uuid.uuid4().hex}"
    acquire_cpu_lock(lock, token)
    try:
        args.out.mkdir(parents=True)
        players = player_inputs(args.video.resolve(), args.attribution_json,
                                {role: hand for role, hand in (("near", args.near_hand), ("far", args.far_hand)) if hand})
        session = analyze_video(args.video.resolve(), FastBackend(), args.out, args.video.stem, players=players)
        if session.error:
            raise RuntimeError(session.error.message)
        schedule, checks = export_short(args.video.resolve(), session.doc, session.report,
                                        args.out / "tiktok.mp4", model_dir=args.voice_model.resolve(),
                                        attribution_file=args.attribution_json)
        _dump(args.out / "run.json", {"fresh_inference": True, "backend": "fast", "timings": session.timings})
        _dump(args.out / "handoff.json", {
            "status": "rendered", "output": str((args.out / "tiktok.mp4").resolve()),
            "planned_duration_s": schedule["planned_duration_s"],
            "moving_s": checks["moving_s"], "paused_s": checks["paused_s"],
            "points": schedule["points"],
            "artifacts": {p.name: {"path": str(p.resolve()), "sha256": sha256_file(p)}
                          for p in args.out.iterdir() if p.is_file() and p.name != "handoff.json"},
            "changed_source_files": {
                name: {"path": str((PROJECT / name).resolve()), "sha256": sha256_file(PROJECT / name)}
                for name in CHANGED_SOURCES},
            "limits": [
                "CLI format only; existing complete-review app route is unchanged.",
                "General teaching cues, not verified causes of elite performance or personal norms.",
                "Source is edited multi-point footage; not a whole match.",
                "Human Chinese naturalness rating and upstream broadcast rights are not established.",
                "Broader six-clip quality/speed acceptance and chat safety remain outside this example.",
            ],
        })
        print(json.dumps({"out": str(args.out.resolve()), "duration_s": schedule["planned_duration_s"],
                          "moving_s": checks["moving_s"], "paused_s": checks["paused_s"]}))
    finally:
        release_cpu_lock(lock, token)


if __name__ == "__main__":
    main()
