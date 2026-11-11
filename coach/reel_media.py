"""Kokoro commentary rendering and independent encoded-output checks."""
from __future__ import annotations

import math
import json
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from .app.backend import sha256_file

EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
RATE, FPS = 24000, 30


def checked_voice_log(stderr, count):
    log = stderr.decode("utf-8", "replace").replace("\r\n", "\n")
    for index in range(count):
        begin, end = f"@@VOICE {index}\n", f"@@ENDVOICE {index}\n"
        if begin not in log or end not in log:
            raise ValueError(f"Voice worker did not report completed pronunciation check for line {index}")
        segment = log.split(begin, 1)[1].split(end, 1)[0].lower()
        if "skip unknown phonemes" in segment or "unknown token" in segment:
            raise ValueError(f"Kokoro dropped unknown phonemes in line {index}; review the supplied name or wording")
    return {"lines_checked": count, "dropped_phonemes": 0,
            "method": "Native stderr captured in child process around each generated line",
            "human_listening": False}


def voice_lines(plan, model, folder):
    texts = [line["say"] for moment in plan["moments"] for line in moment["lines"]]
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp)/"voices.npz"
        child = subprocess.run(
            [sys.executable, "-B", "-m", "coach.reel_media", "--voices", str(model), str(target)],
            input=json.dumps(texts, ensure_ascii=False).encode("utf-8"), capture_output=True, check=True)
        checked = checked_voice_log(child.stderr, len(texts))
        with np.load(target) as samples:
            speeches = [samples[f"line_{i}"].copy() for i in range(len(texts))]
    lexicon = dict(row.split(maxsplit=1) for row in (model/"lexicon-zh.txt").read_text(encoding="utf-8").splitlines()
                   if len(row.split(maxsplit=1)) == 2)
    names = plan.get("names_at_first_reliable_rally", {})
    intended = {name: {c: lexicon.get(c) for c in name if ord(c) > 127} for name in names.values()}
    (folder/"voice-check.json").write_text(json.dumps({**checked, "texts": texts,
                                          "supplied_name_lexicon": intended,
                                          "pronunciation_scope": "Intended dictionary phonemes, not a human listening check"},
                                          ensure_ascii=False, indent=2)+"\n",
                                          encoding="utf-8")
    return speeches


def synthesize_voice_worker(texts, model):
    import sherpa_onnx

    required = ("model.onnx", "voices.bin", "tokens.txt", "lexicon-us-en.txt", "lexicon-zh.txt")
    if any(not (model/name).is_file() for name in required):
        raise FileNotFoundError("Kokoro voice inputs missing; run python scripts/get_models.py kokoro")
    if sha256_file(model/"model.onnx") != "99db0bc8fd0c5033d249fe922db16768b71b4a6b26cfd93d206fbf66c76efb5f":
        raise ValueError("Kokoro model differs from the reference voice; fetch the recorded model")
    if sha256_file(model/"voices.bin") != "e64a5a581d8c2a350d848f51c3121657cd83aa07ed6109172177345874a7244c":
        raise ValueError("Kokoro speaker table differs from the reference sid 60 voice")
    config = sherpa_onnx.OfflineTtsKokoroModelConfig(
        model=str(model/"model.onnx"), voices=str(model/"voices.bin"), tokens=str(model/"tokens.txt"),
        lexicon=f"{model/'lexicon-us-en.txt'},{model/'lexicon-zh.txt'}",
        data_dir=str(model/"espeak-ng-data"), dict_dir=str(model/"dict"),
    )
    engine = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(kokoro=config, num_threads=2)))
    speeches = []
    for index, text in enumerate(texts):
        print(f"@@VOICE {index}", file=sys.stderr, flush=True)
        generated = engine.generate(text, sid=60, speed=1.)
        print(f"@@ENDVOICE {index}", file=sys.stderr, flush=True)
        samples = np.asarray(generated.samples, dtype=np.float32)
        rate = int(generated.sample_rate)
        if not len(samples) or not np.isfinite(samples).all() or np.max(np.abs(samples)) < .01:
            raise ValueError(f"Voice returned invalid or silent speech: {text}")
        if rate != RATE:
            raise ValueError(f"Reference Kokoro voice must produce {RATE} Hz audio, got {rate}")
        speeches.append(samples)
    return speeches


def schedule(plan, speeches):
    slots, durations = [], []
    number = 0
    for moment in plan["moments"]:
        lines = moment["lines"]
        boundaries = [moment["start_frame"]]
        boundaries.extend(max(lines[i-1]["cue_frame"]+1, row["cue_frame"]-7)
                          for i, row in enumerate(lines[1:], 1))
        boundaries.append(moment["end_frame"]+1)
        for line, first, last in zip(lines, boundaries[:-1], boundaries[1:], strict=True):
            samples = speeches[number]
            before_cue = line["cue_frame"]-first
            after_cue = max(last-line["cue_frame"], math.ceil((len(samples)/RATE+.2)*FPS))
            count = before_cue+after_cue
            slots.append({"line": line, "first": first, "last": last, "count": count,
                          "before_cue": before_cue, "speech": number})
            durations.append(count)
            number += 1
    minimum = sum(durations)
    if minimum > 900:
        raise ValueError("Edited plan exceeds 30 seconds; shorten lines or source windows before rendering")
    target = max(plan["target_frames"], minimum)
    extra = target-minimum
    for i, slot in enumerate(slots):
        before = round(extra*sum(durations[:i])/minimum)
        after = round(extra*sum(durations[:i+1])/minimum)
        slot["count"] += after-before
    rows, captions = [], []
    for slot in slots:
        start = len(rows)
        cue = slot["line"]["cue_frame"]
        sources = [*range(slot["first"], cue),
                   *np.linspace(cue, slot["last"]-1, slot["count"]-slot["before_cue"])]
        for source in sources:
            rows.append({"source_frame": round(float(source)), "speech": slot["speech"]})
        cue_output = start+slot["before_cue"]
        captions.append({
            "start_frame": start, "end_frame": len(rows),
            "voice_start_frame": cue_output+2,
            "voice_duration_s": len(speeches[slot["speech"]])/RATE,
            "cue_frame": slot["line"]["cue_frame"],
            "cue_output_frame": cue_output,
            "text": slot["line"]["say"], "caption": slot["line"]["caption"],
            "tag": slot["line"]["tag"], "source_start_frame": slot["first"],
            "source_end_frame": slot["last"]-1,
        })
    mix = np.zeros(math.ceil(len(rows)/FPS*RATE), np.float32)
    for row, samples in zip(captions, speeches, strict=True):
        start = round(row["voice_start_frame"]/FPS*RATE)
        if start+len(samples) > round(row["end_frame"]/FPS*RATE):
            raise ValueError("Internal schedule error: measured voice exceeds constructed slot")
        mix[start:start+len(samples)] = samples
    peak = float(np.max(np.abs(mix)))
    if peak >= .95:
        mix *= .94/peak
    return rows, captions, mix


@lru_cache(maxsize=256)
def fitted_font(path, text, largest, width):
    for size in range(largest, 17, -2):
        font = ImageFont.truetype(str(path), size)
        if font.getlength(text) <= width:
            return font
    raise ValueError("Text does not fit; shorten the edited plan")


def compose(raw, pose, plan, line, font_path, show_caption=True):
    raw = raw.copy()
    for player in pose["players"]:
        if player.get("role") not in ("near", "far"):
            continue
        joints = player["joints"]
        colour = (181, 85, 255) if player["role"] == "near" else (80, 225, 168)
        for a, b in EDGES:
            if min(joints[a][2], joints[b][2]) >= .5:
                cv2.line(raw, tuple(round(v) for v in joints[a][:2]),
                         tuple(round(v) for v in joints[b][:2]), colour, 3, cv2.LINE_AA)
    canvas = Image.new("RGB", (1080, 1920), "#07161d")
    image = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
    scale = min(1068/image.width, 601/image.height)
    display = image.resize((round(image.width*scale), round(image.height*scale)))
    canvas.paste(display, (6+(1068-display.width)//2, 350+(601-display.height)//2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 296, 1079, 299), fill="#a8e255")
    for i, title in enumerate(plan["title"]):
        font = fitted_font(font_path, title, 52, 1000)
        draw.text(((1080-font.getlength(title))/2, 148+i*68), title, font=font,
                  fill=("#f3f6f7", "#a8e255")[i])
    draw.rectangle((0, 1040, 1079, 1043), fill="#a8e255")
    tag_font = fitted_font(font_path, line["tag"], 40, 976)
    draw.text((52, 1076), line["tag"], font=tag_font, fill="#a8e255")
    if show_caption:
        font = fitted_font(font_path, line["caption"], 64, 976)
        draw.text((52, 1255-round(font.size*.66)), line["caption"], font=font, fill="#f3f6f7")
    footer = fitted_font(font_path, plan["footer"], 22, 976)
    draw.text((52, 1390), plan["footer"], font=footer, fill="#a5b4be")
    credit = plan["attribution"]
    for index, text in enumerate((f"{credit['author']} · {credit['license']}",
                                  "剪辑 / 估算骨架 / AI 配音 · 不代表来源方认可")):
        font = fitted_font(font_path, text, 24, 976)
        draw.text((52, 1520+index*42), text, font=font, fill="#a5b4be")
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def pictures(video, pose, plan, rows, font_path):
    lines = [line for m in plan["moments"] for line in m["lines"]]
    cap = cv2.VideoCapture(str(video))
    previous, raw = -2, None
    try:
        for row in rows:
            frame = row["source_frame"]
            if frame != previous:
                if 0 < frame-previous <= 5:
                    for _ in range(frame-previous):
                        ok, raw = cap.read()
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, raw = cap.read()
                if not ok:
                    raise ValueError(f"Cannot decode planned source frame {frame}")
                previous = frame
            yield (compose(raw, pose["frames"][frame], plan, lines[row["speech"]], font_path),
                   compose(raw, pose["frames"][frame], plan, lines[row["speech"]], font_path, False))
    finally:
        cap.release()


def render(video, pose, plan, folder, model, font_path, ffmpeg):
    speeches = voice_lines(plan, model, folder)
    rows, captions, mix = schedule(plan, speeches)
    voice = folder/"voice.wav"
    with wave.open(str(voice), "wb") as stream:
        stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes(np.rint(mix*32767).astype("<i2").tobytes())
    refs, captions_ref, controls_ref = [], [], []
    output = folder/"tiktok.mp4"
    with (folder/"encode.log").open("wb") as errors:
        proc = subprocess.Popen([
            str(ffmpeg), "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", "1080x1920", "-r", "30", "-i", "pipe:0", "-i", str(voice),
            "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", "19",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
            "-movflags", "+faststart", str(output),
        ], stdin=subprocess.PIPE, stderr=errors)
        try:
            for picture, no_caption in pictures(video, pose, plan, rows, font_path):
                proc.stdin.write(picture.tobytes())
                refs.append(cv2.resize(picture, (108, 192), interpolation=cv2.INTER_AREA))
                captions_ref.append(cv2.resize(picture[1175:1305], (270, 65), interpolation=cv2.INTER_AREA))
                controls_ref.append(cv2.resize(no_caption[1175:1305], (270, 65), interpolation=cv2.INTER_AREA))
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError("Reel encoding failed; see encode.log")
        finally:
            if not proc.stdin.closed:
                proc.stdin.close()
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
    np.savez_compressed(folder/"verification-inputs.npz", picture=np.asarray(refs),
                        caption=np.asarray(captions_ref), no_caption=np.asarray(controls_ref), audio=mix)
    contact_sheet(video, plan, folder/"contact-sheet.jpg", font_path)
    return rows, captions


def contact_sheet(video, plan, destination, font_path):
    cap = cv2.VideoCapture(str(video))
    lines = [line for m in plan["moments"] for line in m["lines"]]
    sheet = Image.new("RGB", (1080, math.ceil(len(lines)/2)*350), "#07161d")
    for i, line in enumerate(lines):
        cap.set(cv2.CAP_PROP_POS_FRAMES, line["cue_frame"])
        ok, image = cap.read()
        if not ok:
            raise ValueError("Cannot decode contact sheet cue")
        x, y = (i % 2)*540, (i//2)*350
        tile = ImageOps.contain(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), (540, 304))
        sheet.paste(tile, (x+(540-tile.width)//2, y+(304-tile.height)//2))
        draw = ImageDraw.Draw(sheet)
        label = f"{line['cue_frame']/30:.2f}s {line['caption']}"
        draw.text((x+8,y+312), label, font=fitted_font(font_path,label,24,524), fill="#f3f6f7")
    cap.release()
    sheet.save(destination)


def evidence_sheets(video, hits, folder):
    named = [hit for hit in hits if hit["family"] is not None]
    cap = cv2.VideoCapture(str(video))
    for start in range(0, len(named), 6):
        sheet = np.zeros((6*362, 1800, 3), np.uint8)
        for row, hit in enumerate(named[start:start+6]):
            for column, offset in enumerate((-5, 0, 5)):
                frame = max(0, hit["frame"]+offset)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f"Cannot decode named-shot review frame {frame}")
                for point in hit.get("shuttle_path", [])[:12]:
                    cv2.circle(image, tuple(round(v) for v in point["xy_px"]), 5, (0,255,255), -1)
                cv2.circle(image, tuple(round(v) for v in hit["xy_px"]), 12, (0,0,255), 3)
                tile = ImageOps.contain(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), (600, 338))
                x, y = column*600+(600-tile.width)//2, row*362+(338-tile.height)//2
                sheet[y:y+tile.height, x:x+tile.width] = cv2.cvtColor(np.asarray(tile), cv2.COLOR_RGB2BGR)
                label = f"{hit['time_s']:.3f}s {hit['hitter']} {hit['family']} frame {frame}"
                cv2.putText(sheet,label,(column*600+8,row*362+355),0,.55,(255,255,255),1)
        if not cv2.imwrite(str(folder/f"named-hits-{start//6+1:02d}.jpg"),sheet):
            raise RuntimeError("Cannot write shot evidence sheet")
    cap.release()


def verify(movie, references, rows, captions, ffmpeg):
    import av

    with np.load(references) as archive:
        expected = {key: archive[key] for key in ("picture", "caption", "no_caption", "audio")}
    errors, caption_errors, missing_errors = [], [], []
    with av.open(str(movie)) as container:
        video = container.streams.video[0]
        video.codec_context.thread_count = 4
        video.thread_type = "FRAME"
        shape = [video.width, video.height]
        if shape != [1080, 1920] or float(video.average_rate) != 30:
            raise ValueError("Encoded geometry or frame rate differs from plan")
        if not container.streams.audio:
            raise ValueError("Encoded output is missing audio")
        audio_codec = container.streams.audio[0].codec_context.name
        for i, decoded in enumerate(container.decode(video=0)):
            if i >= len(rows):
                raise ValueError("Encoded output contains unplanned frames")
            image = decoded.to_ndarray(format="bgr24")
            small = cv2.resize(image,(108,192),interpolation=cv2.INTER_AREA)
            caption = cv2.resize(image[1175:1305],(270,65),interpolation=cv2.INTER_AREA)
            errors.append(float(np.abs(small.astype(float)-expected["picture"][i]).mean()))
            for name, lo, hi in (("title", 14, 28), ("green tag", 107, 114)):
                observed = small[lo:hi].astype(float)
                reference = expected["picture"][i, lo:hi].astype(float)
                error = float(np.abs(observed-reference).mean())
                missing = float(np.abs(observed-np.array([29, 22, 7])).mean())
                if error >= missing*.5:
                    raise ValueError(f"Encoded {name} is missing or differs from the planned text")
            caption_errors.append(float(np.abs(caption.astype(float)-expected["caption"][i]).mean()))
            missing_errors.append(float(np.abs(caption.astype(float)-expected["no_caption"][i]).mean()))
    if len(errors) != len(rows) or max(errors) > 4:
        raise ValueError("Encoded frame count or picture content does not match planned frames")
    if any(a >= b*.5 for a,b in zip(caption_errors,missing_errors)):
        raise ValueError("Encoded white spoken captions are missing or differ from the planned text")
    audio = subprocess.run([str(ffmpeg),"-v","error","-i",str(movie),"-vn","-ac","1","-ar",str(RATE),
                            "-f","f32le","pipe:1"],capture_output=True,check=True)
    actual = np.frombuffer(audio.stdout,dtype="<f4")
    correlations = []
    for caption in captions:
        cue = caption["cue_output_frame"]
        if (not 0 <= cue < len(rows) or rows[cue]["source_frame"] != caption["cue_frame"]
                or caption["voice_start_frame"]-cue != 2):
            raise ValueError("Narration is not aligned to its planned action cue")
        start = round(caption["voice_start_frame"]/FPS*RATE)
        end = start+round(caption["voice_duration_s"]*RATE)
        if end > len(actual):
            raise ValueError("Encoded audio ends before measured narration")
        reference, observed = expected["audio"][start:end], actual[start:end]
        if np.sqrt(np.mean(observed**2)) < .005:
            raise ValueError("Encoded narration is silent in a planned voice slot")
        correlation = float(np.corrcoef(reference,observed)[0,1])
        if not math.isfinite(correlation) or correlation < .95:
            raise ValueError("Encoded narration does not match the synthesized speech")
        correlations.append(correlation)
    return {
        "full_decode": True, "picture_frames_checked": len(errors), "caption_frames_checked": len(errors),
        "voice_slots_checked": len(correlations), "white_caption_controls_distinguished": True,
        "title_frames_checked": len(errors), "green_tag_frames_checked": len(errors),
        "voice_start_after_action_frames": 2,
        "voice_fits_measured_slots": True, "non_silent_audio": True,
        "geometry": shape, "fps": FPS, "frames": len(rows), "duration_s": len(rows)/FPS,
        "audio_codec": audio_codec, "picture_mae_max": max(errors),
        "caption_mae_max": max(caption_errors), "voice_correlation_min": min(correlations),
        "human_listening": False, "naturalness_verified": False,
    }


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--voices":
        raise SystemExit("Internal voice worker: --voices MODEL OUTPUT")
    speeches = synthesize_voice_worker(json.loads(sys.stdin.buffer.read()), Path(sys.argv[2]))
    np.savez(sys.argv[3], **{f"line_{i}": speech for i, speech in enumerate(speeches)})
