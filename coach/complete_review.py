"""Continuous, locally narrated review and its evidence-bound Chinese report."""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .app.backend import PROJECT, FastBackend, sha256_file
from .app.pipeline import analyze_video
from .bundle import build_bundle
from .export import _draw_poses, _fit, _contain, _poses, _tools
from .play_state import exchange_mask
from .tracks import load_tracks

VOICE = PROJECT / "coach" / "models" / "vits-melo-tts-zh_en"
LABELS = {
    "technique.stay_lower": "等待时的投影髋高",
    "technique.knee_bend": "等待时的投影膝屈曲",
    "technique.lunge_depth": "跨步时的投影髋高变化",
    "technique.stance_width": "等待时的投影脚距",
    "technique.trunk_lean": "躯干的投影侧倾",
    "contact.reach_height": "抬臂时的投影手腕高度",
    "contact.arm_extension": "抬臂时的投影肘角",
    "contact.hand_speed": "相对肩部的手腕运动速度",
    "footwork.split_step": "对方挥臂附近的小跳观察",
    "footwork.recovery": "回到等待位置的时间",
    "footwork.first_step": "对方挥臂后的启动时间",
    "footwork.step_pattern": "跨步前的小步数量",
    "footwork.coverage": "观察到的场地覆盖",
    "tactics.base_discipline": "挥臂时与等待位置的距离",
    "tactics.tempo": "相邻挥臂的间隔",
    "tactics.initiative": "挥臂时靠近球网的次数",
}
PLAYERS = {"near": "近端球员", "far": "远端球员"}
METRIC_LABELS = {
    "split_step_share": "挥臂附近观察到小跳的比例",
    "split_step_timing_s": "小跳落地相对挥臂的时间",
    "recovery_s": "回到等待位置的时间",
    "first_step_s": "挥臂后的启动时间",
    "steps_before_lunge": "跨步前的小步数量",
    "lunge_launch_m": "跨步开始时距等待位置的距离",
    "half_coverage": "使用本方半场的比例",
    "weaker_side_share": "在较少活动的一侧停留的比例",
    "habitual_base_gap_m": "挥臂时距惯常等待位置的距离",
    "fixed_base_gap_m": "挥臂时距固定参考点的距离",
    "exchange_gap_s": "相邻挥臂间隔",
    "net_ahead_share": "挥臂时更靠近球网的比例",
}
INTRO = "保留整段原视频，慢放三个观察点。骨架来自本次模型推理，二维画面不能单独证明技术好坏。"
OUTRO = "复盘时，把跨步、回收和下一次移动连起来看。击球接触尚未确认，不据此要求压低重心或改变关节角度。"
LIMITS = (
    "这是一段剪辑，不是整场比赛。镜头切换处不连接速度，也不把发球前的等待当作回合中的准备姿势。",
    "回合支持仅来自同一镜头内双方交替挥臂；未追踪羽毛球，不能确认发球、触球、落点或战术结果。",
    "骨架是模型估计。低置信度关节不画，重复或插值姿态不算新测量；画得好看不能验证建议。",
    "角度、髋高和手腕速度是单摄像机二维投影；不是三维关节角、真实重心或球拍速度。",
    "练习范围不是经过验证的个人标准。没有证据证明姿势有错，所以不输出压低髋部、伸直手臂等纠错指令。",
    "髋高下降少于零点三秒的完整周期不纳入跨步复盘；这是保守的选择规则，不是经过验证的动作分类器。",
    "仅完成这一例；六片段速度与质量验收尚未通过，亚秒目标未达成。聊天安全性与整场比赛支持未在此例验收。",
)


def _number(metric, value=None):
    value = metric.value if value is None else value
    if value is None:
        return "未测得"
    if metric.unit in ("fraction", "heights", "share"):
        return f"{value * 100:.0f}%"
    if metric.unit == "deg":
        return f"{value:.0f} 度"
    if metric.unit == "heights/s":
        return f"{value:.1f} 身高/秒"
    if metric.unit == "s":
        return f"{value:.2f} 秒"
    if metric.unit == "m":
        return f"{value:.2f} 米"
    if metric.unit == "count":
        return f"{value:.0f} 次"
    return f"{value:.3g} {metric.unit}"


def select_observations(doc, report, count=3):
    tracks = load_tracks(doc)
    active = exchange_mask(tracks, report.events)
    priority = ("technique.lunge_depth", "contact.arm_extension", "contact.hand_speed",
                "technique.knee_bend")
    pool = sorted(
        (finding for finding in report.findings
         if finding.id in priority and finding.moments and finding.metrics[0].value is not None),
        key=lambda finding: (priority.index(finding.id), -finding.metrics[0].confidence,
                             finding.player.value),
    )
    selected = []
    for finding in pool:
        moment = finding.moments[0]
        key = moment.key_frame
        player = tracks.players[finding.player]
        if not active[key] or not player.measured[key] or not moment.marks:
            continue
        if any(abs(key - row["key_frame"]) < 2 * report.fps for row in selected):
            continue
        if any(row["finding_id"] == finding.id and row["player"] == finding.player.value for row in selected):
            continue
        if len(selected) == 1 and selected[0]["player"] == finding.player.value:
            continue
        valued = [mark.value for mark in moment.marks if mark.value is not None and mark.style == "actual"]
        if not valued:
            continue
        value = valued[0]
        metric = finding.metrics[0]
        role = finding.player.value
        label = LABELS[finding.id]
        observation = f"{PLAYERS[role]}：{label}约 {_number(metric, value)}。"
        if finding.id == "technique.lunge_depth":
            cue = "先看跨步下降，再看后续移动。这只是画面中的髋高变化，不是深度达标评分。"
        elif finding.id == "contact.arm_extension":
            cue = "慢放看肩、肘、腕的连续变化。触球未确认，不能把这一帧当作击球接触。"
        elif finding.id == "contact.hand_speed":
            cue = "看手腕相对肩部的运动。不是球拍速度，也不能据此判断击球质量。"
        else:
            cue = "看回合中等待到移动的衔接。投影角度不等于三维角度，不据此要求蹲低。"
        start = max(moment.start_frame, key - round(report.fps))
        end = min(moment.end_frame, key + round(report.fps))
        selected.append({
            "finding_id": finding.id, "player": role, "key_frame": key,
            "source_start_frame": start, "source_end_frame": end,
            "source_key_s": key / report.fps, "label": label,
            "observation": observation, "cue": cue,
            "narration": re.sub(r"(\d+)%", r"百分之\1", observation.replace("：", "，").replace("身高/秒", "个身高每秒")) + cue,
            "local_value": value, "clip_summary_value": metric.value,
            "unit": metric.unit, "basis": metric.basis.value,
            "confidence": metric.confidence, "n_obs": metric.n_obs,
            "marks": [mark.to_dict() if hasattr(mark, "to_dict") else {
                "kind": mark.kind, "frame": mark.frame, "points": list(mark.points),
                "value": mark.value, "style": mark.style,
            } for mark in moment.marks],
        })
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(f"Only {len(selected)} supported, separated observations; need {count}. No fabricated replacement.")
    return sorted(selected, key=lambda row: row["key_frame"])


def _voice(model_dir):
    try:
        import sherpa_onnx
    except ModuleNotFoundError as exc:
        raise RuntimeError(r"Local Chinese neural TTS unavailable: install coach\requirements-narration.txt") from exc

    folder = Path(model_dir)
    for name in ("model.onnx", "lexicon.txt", "tokens.txt"):
        if not (folder / name).is_file():
            raise FileNotFoundError(f"Local Chinese neural TTS is unavailable: {folder / name}")
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(folder / "model.onnx"), lexicon=str(folder / "lexicon.txt"),
                tokens=str(folder / "tokens.txt"),
            ), num_threads=2,
        ),
        rule_fsts=",".join(str(folder / name) for name in ("number.fst", "date.fst")),
    )
    if not config.validate():
        raise RuntimeError("Local Chinese neural TTS configuration is invalid")
    return sherpa_onnx.OfflineTts(config)


def _speak(tts, text):
    audio = tts.generate(text, sid=0, speed=1.0)
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.size == 0 or not np.isfinite(samples).all() or float(np.max(np.abs(samples))) < 0.01:
        raise RuntimeError("Neural TTS produced empty, invalid or silent narration")
    return samples, int(audio.sample_rate)


def make_schedule(n, fps, observations, durations):
    frames = []
    entries = []
    intro_count = math.ceil((durations[0] + 0.4) * fps)
    entries.append({"kind": "intro", "start_s": 0.0, "end_s": intro_count / fps})
    frames.extend([0] * intro_count)
    for source in range(n):
        row = next((row for row in observations
                    if row["source_start_frame"] <= source <= row["source_end_frame"]), None)
        if row and source == row["source_start_frame"]:
            row["output_start_s"] = len(frames) / fps
        frames.extend([source] * (2 if row else 1))
        if row and source == row["key_frame"]:
            row["output_freeze_s"] = len(frames) / fps
            index = observations.index(row)
            motion_s = 2 * (row["source_end_frame"] - row["source_start_frame"] + 1) / fps
            hold_s = max(2.0, durations[index + 1] + 0.4 - motion_s)
            frames.extend([source] * math.ceil(hold_s * fps))
            row["output_freeze_end_s"] = len(frames) / fps
        if row and source == row["source_end_frame"]:
            row["output_end_s"] = len(frames) / fps
            entries.append({"kind": "observation", "finding_id": row["finding_id"],
                            "player": row["player"], "key_frame": row["key_frame"],
                            "start_s": row["output_start_s"], "end_s": row["output_end_s"]})
    entries.append({"kind": "outro", "start_s": len(frames) / fps})
    frames.extend([n - 1] * math.ceil((durations[-1] + 0.4) * fps))
    entries[-1]["end_s"] = len(frames) / fps
    return frames, entries


def _wrap_zh(draw, text, font, width):
    lines, line = [], ""
    for character in text:
        if character == "\n":
            lines.append(line)
            line = ""
        elif line and draw.textlength(line + character, font=font) > width:
            lines.append(line)
            line = character
        else:
            line += character
    if line:
        lines.append(line)
    return lines


def _panel(source, fps, duration, row, phase, fonts):
    image = Image.new("RGB", (540, 1080), "#101d2b")
    draw = ImageDraw.Draw(image)
    title, body, small = fonts
    draw.text((36, 36), "羽毛球 · 完整视频复盘", font=title, fill="#edf6ff")
    draw.text((36, 100), f"原片 {source / fps:.2f} / {duration:.2f} 秒", font=small, fill="#93b4ca")
    y = 185
    if row:
        text = row["observation"] + "\n\n" + row["cue"]
        draw.text((36, 140), "慢放观察 · 不作技术好坏评分", font=small, fill="#66e8bd")
    else:
        text = INTRO if phase == "intro" else OUTRO if phase == "outro" else "连续观看原片\n\n近端：粉色骨架\n远端：黄色骨架\n\n保留动作前后，不用静止截图代替分析。"
    for line in _wrap_zh(draw, text, body, 464):
        draw.text((36, y), line, font=body, fill="#edf6ff")
        y += 44
    note = "二维模型估计 · 未确认触球\n缺失关节不补画 · 非医疗建议"
    for i, line in enumerate(_wrap_zh(draw, note, small, 464)):
        draw.text((36, 920 + i * 32), line, font=small, fill="#93b4ca")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def report_html(analysis):
    esc = html.escape
    cards = []
    for row in analysis["observations"]:
        cards.append(
            f'<article><h2>{esc(row["observation"])}</h2><p>{esc(row["cue"])}</p>'
            f'<p class="meta">源帧 {row["key_frame"]} · 原片 {row["source_key_s"]:.2f}s · '
            f'输出 {row["output_start_s"]:.2f}–{row["output_end_s"]:.2f}s · '
            f'定格 {row["output_freeze_s"]:.2f}–{row["output_freeze_end_s"]:.2f}s</p>'
            f'<p>此刻值 {row["local_value"]:.4g}；片段汇总 {row["clip_summary_value"]:.4g} {esc(row["unit"])}'
            f'（{row["n_obs"]} 个样本）。模型置信度 {row["confidence"]:.2f}，不是建议正确率。</p></article>'
        )
    rows = []
    for finding in analysis["findings"]:
        for metric in finding["metrics"]:
            rows.append(f'<tr><td>{esc(PLAYERS[finding["player"]])}</td>'
                        f'<td>{esc(METRIC_LABELS.get(metric["name"], LABELS[finding["id"]]))}</td>'
                        f'<td>{esc(_number(SimpleNamespace(**metric)))}</td>'
                        f'<td>{metric["n_obs"]}</td><td>{metric["confidence"]:.2f}</td></tr>')
    limits = "".join(f"<li>{esc(text)}</li>" for text in analysis["limitations"])
    transcript = "".join(f'<p><b>{entry["start_s"]:.2f}–{entry["audio_end_s"]:.2f}s</b> {esc(entry["text"])}</p>'
                         for entry in analysis["narration"])
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>羽毛球完整视频分析</title><style>
body{{font:18px/1.75 "Microsoft YaHei",sans-serif;background:#0b1520;color:#e4eff8;max-width:1060px;margin:40px auto;padding:20px}}
h1,h2{{line-height:1.5}}article{{background:#152638;border-left:4px solid #63e5bb;padding:24px;margin:24px 0;border-radius:12px}}
.meta{{color:#97b5cc}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border-bottom:1px solid #345}}a{{color:#63e5bb}}
video{{width:100%;background:black}}li{{margin:12px 0}}</style>
<h1>完整观看 + 三个慢放观察点</h1><p>{esc(analysis["overview"])}</p>
<video controls preload="metadata" src="review.mp4"></video>
<p><a href="review.mp4" download>下载分析视频</a> · <a href="analysis.json" download>下载分析与证据</a> · <a href="narration.txt" download>下载旁白</a></p>
{''.join(cards)}<h2>这次能说什么，不能说什么</h2><ul>{limits}</ul>
<h2>全部测量汇总</h2><p>以下是数值观察，不是技术好坏评分；“未测得”不是零。此刻值与片段汇总分开列出。</p>
<table><tr><th>球员</th><th>观察</th><th>汇总</th><th>样本数</th><th>模型置信度</th></tr>{''.join(rows)}</table>
<h2>同步旁白全文</h2>{transcript}<h2>来源与许可</h2><pre style="white-space:pre-wrap">{esc(analysis["attribution"])}</pre>
<p class="meta">全程本地模型推理与神经语音合成；未上传视频。速度、整场比赛和聊天安全性不在这一例中验收。</p></html>"""


def export_complete(video, doc, report, out_mp4, *, model_dir=VOICE):
    started = time.perf_counter()
    out = Path(out_mp4)
    folder = out.parent
    folder.mkdir(parents=True, exist_ok=True)
    observations = select_observations(doc, report)
    texts = [INTRO, *(row["narration"] for row in observations), OUTRO]
    tts = _voice(model_dir)
    speeches = [_speak(tts, text) for text in texts]
    rates = {rate for _samples, rate in speeches}
    if len(rates) != 1:
        raise RuntimeError("Narration sample rates disagree")
    rate = rates.pop()
    fps = report.fps
    frame_map, entries = make_schedule(report.n_frames, fps, observations,
                                      [len(samples) / rate for samples, _rate in speeches])
    duration = len(frame_map) / fps
    (folder / "schedule.json").write_text(json.dumps({
        "fps": fps, "planned_duration_s": duration, "planned_output_frames": len(frame_map),
        "entries": entries, "observations": observations,
    }, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
    narration = np.zeros(math.ceil(duration * rate), np.float32)
    bindings = []
    for entry, text, (samples, _rate) in zip(entries, texts, speeches):
        start = round(entry["start_s"] * rate)
        end = start + len(samples)
        if end > round(entry["end_s"] * rate):
            raise RuntimeError("Narration does not fit its evidence window")
        narration[start:end] += samples
        bindings.append({**entry, "text": text, "audio_end_s": end / rate})
    font_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    font_path = font_dir / "msyh.ttc"
    if not font_path.is_file():
        raise FileNotFoundError(f"Chinese font unavailable: {font_path}")
    fonts = tuple(ImageFont.truetype(str(font_path), size) for size in (30, 28, 20))
    bundle = build_bundle(doc, report, video_url=str(video))
    poses = _poses(bundle)
    style = bundle["style"]
    names = bundle["joints"]
    findings = {(f.id, f.player.value): f for f in report.findings}
    neutral = next(iter(findings.values()))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode source {video}")
    ffmpeg, ffprobe = _tools()
    with tempfile.TemporaryDirectory(prefix="complete-", dir=folder) as tmp:
        tmp = Path(tmp)
        audio_path = tmp / "narration.wav"
        with wave.open(str(audio_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes((np.clip(narration, -1, 1) * 32767).astype("<i2").tobytes())
        err_path = tmp / "encoder.log"
        with err_path.open("wb") as errors:
            proc = subprocess.Popen([
                str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x1080",
                "-r", str(fps), "-i", "pipe:0", "-i", str(audio_path),
                "-c:v", "libx264", "-preset", "fast", "-crf", "21",
                "-c:a", "aac", "-b:a", "160k", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(out),
            ], stdin=subprocess.PIPE, stderr=errors)
            cached_source = -1
            try:
                for index, source in enumerate(frame_map):
                    if source != cached_source:
                        if source != cached_source + 1:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, source)
                        ok, raw = cap.read()
                        if not ok:
                            raise RuntimeError(f"Source decode failed at frame {source}")
                        cached_source = source
                        painted = raw.copy()
                        row = next((row for row in observations if row["source_start_frame"] <= source <= row["source_end_frame"]), None)
                        finding = findings[(row["finding_id"], row["player"])] if row else neutral
                        size = (raw.shape[1], raw.shape[0])
                        _draw_poses(painted, poses, source, {role: 0.0 for role in poses},
                                    style, names, finding, size, (doc["source"]["width"], doc["source"]["height"]))
                        base = _fit(painted, _contain(size[0], size[1], 1380, 1080))
                        zoom_base = base
                        if row and source == row["key_frame"]:
                            box = poses[row["player"]]["bbox"][source]
                            if not np.isfinite(box).all():
                                raise RuntimeError("Selected freeze has no measured player box")
                            x1, y1, x2, y2 = box
                            pad = max(x2 - x1, y2 - y1) * 0.6
                            left, top = max(0, int(x1 - pad)), max(0, int(y1 - pad))
                            right, bottom = min(size[0], math.ceil(x2 + pad)), min(size[1], math.ceil(y2 + pad))
                            crop = painted[top:bottom, left:right]
                            zoom_base = _fit(crop, _contain(crop.shape[1], crop.shape[0], 1380, 1080))
                    wall = index / fps
                    row = next((row for row in observations if row["output_start_s"] <= wall < row["output_end_s"]), None)
                    phase = "intro" if wall < entries[0]["end_s"] else "outro" if wall >= entries[-1]["start_s"] else "play"
                    panel = _panel(source, fps, report.n_frames / fps, row, phase, fonts)
                    pane = zoom_base if row and row["output_freeze_s"] <= wall < row["output_freeze_end_s"] else base
                    proc.stdin.write(np.concatenate((pane, panel), axis=1).tobytes())
                proc.stdin.close()
                if proc.wait() != 0:
                    raise RuntimeError(err_path.read_text("utf-8", errors="replace"))
            finally:
                cap.release()
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        subprocess.run([str(ffmpeg), "-v", "error", "-i", str(out), "-f", "null", "-"],
                       check=True, capture_output=True)
        probe = json.loads(subprocess.check_output([
            str(ffprobe), "-v", "error", "-count_frames", "-show_streams", "-show_format",
            "-of", "json", str(out),
        ]))
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    audio_stream = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    if int(video_stream["nb_read_frames"]) != len(frame_map) or abs(float(audio_stream["duration"]) - duration) > 0.1:
        raise RuntimeError("Encoded video/audio duration or frame count disagrees with schedule")
    attribution_path = Path(video).with_suffix(".attribution.json")
    if not attribution_path.is_file():
        for path in sorted(Path(video).parent.glob("*attribution*.json")):
            candidate = json.loads(path.read_text("utf-8"))
            if any(clip.get("file") == Path(video).name for clip in candidate.get("clips", [])):
                attribution_path = path
                break
    attribution = attribution_path.read_text("utf-8") if attribution_path.is_file() else "Input attribution must be supplied by the uploader."
    attribution += "\n本次新增：模型骨架、慢放、定格放大、中文旁白和分析文字。未获球员、赛事或创作者背书。"
    analysis = {
        "schema_version": 1, "overview": INTRO, "source": report.source,
        "source_sha256": sha256_file(Path(video)), "pose_tier": doc.get("tier", "refined"),
        "observations": observations, "findings": report.to_dict()["findings"],
        "narration": bindings, "limitations": list(LIMITS), "warnings": list(report.warnings),
        "attribution": attribution, "tts": {"engine": "sherpa-onnx", "model": "MeloTTS-Chinese",
                                           "model_sha256": sha256_file(Path(model_dir) / "model.onnx"),
                                           "sample_rate": rate, "speaker": 0, "local": True},
        "verification": {"full_decode": True, "video_frames": len(frame_map),
                         "audio_duration_s": float(audio_stream["duration"]),
                         "duration_s": duration, "all_source_frames_preserved": sorted(set(frame_map)) == list(range(report.n_frames)),
                         "monotonic_source_mapping": all(a <= b for a, b in zip(frame_map, frame_map[1:]))},
        "render_s": time.perf_counter() - started,
    }
    (folder / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
    (folder / "frame-map.json").write_text(json.dumps({"fps": fps, "source_frame_per_output_frame": frame_map}), "utf-8")
    (folder / "narration.txt").write_text("\n\n".join(
        f'{row["start_s"]:.2f}–{row["audio_end_s"]:.2f}s\n{row["text"]}' for row in bindings), "utf-8")
    (folder / "analysis.html").write_text(report_html(analysis), "utf-8")
    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--voice-model", type=Path, default=VOICE)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Use a NEW output folder: {args.out}")
    from turbo.accept import acquire_cpu_lock, release_cpu_lock
    import uuid

    lock = PROJECT / "work" / "cpu.lock"
    token = f"complete-review {os.getpid()} {uuid.uuid4().hex}"
    acquire_cpu_lock(lock, token)
    try:
        args.out.mkdir(parents=True)
        session = analyze_video(args.video.resolve(), FastBackend(), args.out, args.video.stem)
        if session.error:
            raise RuntimeError(session.error.message)
        result = export_complete(args.video.resolve(), session.doc, session.report,
                                 args.out / "review.mp4", model_dir=args.voice_model.resolve())
        (args.out / "run.json").write_text(json.dumps({
            "fresh_inference": True, "backend": "fast", "timings": session.timings,
            "verification": result["verification"],
        }, indent=2), "utf-8")
        print(json.dumps({"out": str(args.out.resolve()), "verification": result["verification"]}))
    finally:
        release_cpu_lock(lock, token)


if __name__ == "__main__":
    main()
