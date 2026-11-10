import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import psutil
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "english-cut-20261109"))
import probe_voices as voices
import sherpa_onnx

ROOT = HERE.parents[3]
OLD = ROOT / "badminton-coach" / "outputs" / "tactics-original-editorial-20261105" / "tiktok-viral-v3" / "render"
SOURCE = ROOT / "obj2" / "work" / "H" / "downloads" / "china-open-2025-ws-final.webm"
FACTS = ROOT / "obj2" / "work" / "SHUTTLE" / "beat-facts.json"
NARRATION = HERE / "narration"
OUT = HERE / "video"
MOVIE = OUT / "editorial-chinese-technical.mp4"
FF = ROOT / "tools" / "ffmpeg.exe"
PROBE = ROOT / "tools" / "ffprobe.exe"
RATE = 48000
FRAMES = 798
OLD_SHA = "415dec061188a08339b89cd69799c93deb8654709dc8e74c6361336e5e9bc012"
SOURCE_SHA = "2500471d4852a4a73e0f4eb25ad28171b3c54967f58e1cc7620af16b46ccbb52"
FONT = Path(r"C:\Windows\Fonts\msyh.ttc")
CARD_Y = 1040
HOOK_Y = 130
HOOK_HEIGHT = 170
BURNED_HOOK_FRAMES = 38
SPEECH_LEAD = 0.08
SPEECH_GAP = 0.02


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Refusing to overwrite an existing artifact: {path}")
    else:
        path.write_text(payload, encoding="utf-8")


def run(args):
    return subprocess.run([str(value) for value in args], capture_output=True, check=True, timeout=300).stdout


def audio(path):
    with wave.open(str(path), "rb") as handle:
        if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) != (RATE, 1, 2):
            raise ValueError(f"Unexpected audio format: {path}")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768


def write_wave(path, samples):
    if float(np.max(np.abs(samples))) >= 0.99:
        raise ValueError(f"Audio would clip: {path}")
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        handle.writeframes(np.rint(samples * 32767).astype("<i2").tobytes())


def lines_of(script):
    return [line for beat in script["beats"] for line in beat["lines"]]


def card_size(texts):
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    for size in (64, 60, 56, 52):
        font = ImageFont.truetype(str(FONT), size)
        if all(probe.textlength(text, font=font) <= 976 for text in texts):
            return size
    raise ValueError("A subtitle would overflow one card line")


def card(path, tag, text, footer, size):
    image = Image.new("RGB", (1080, 390), "#07161d")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1079, 3), fill="#a8e255")
    tag_font = ImageFont.truetype(str(FONT), 40)
    if draw.textlength(tag, font=tag_font) > 976:
        raise ValueError(f"Tag would overflow: {tag}")
    draw.text((52, 36), tag, font=tag_font, fill="#a8e255")
    font = ImageFont.truetype(str(FONT), size)
    draw.text((52, 215 - round(size * 0.66)), text, font=font, fill="#f3f6f7")
    draw.text((52, 350), footer, font=ImageFont.truetype(str(FONT), 22), fill="#a5b4be")
    image.save(path)


def hook(path, lines):
    image = Image.new("RGB", (1080, HOOK_HEIGHT), "#07161d")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT), 52)
    draw.rectangle((0, HOOK_HEIGHT - 4, 1079, HOOK_HEIGHT - 1), fill="#a8e255")
    colors = ("#f3f6f7", "#a8e255")
    for index, line in enumerate(lines):
        colour = colors[index % len(colors)]
        width = draw.textlength(line, font=font)
        if width > 1000:
            raise ValueError("Hook line would overflow")
        draw.text(((1080 - width) / 2, 18 + index * 68), line, font=font, fill=colour)
    image.save(path)


def synthesize_child(sid, path):
    tts = voices.engine()
    text = sys.stdin.buffer.read().decode("utf-8")
    os.write(2, b"@@GENERATE\n")
    generated = tts.generate(text, sid=sid, speed=1.0)
    voices.write(path, np.asarray(generated.samples, dtype=np.float32), generated.sample_rate)
    os.write(2, f"@@RATE {generated.sample_rate}\n".encode())


def synthesize(text, sid, path):
    # The native TTS library keeps its own stderr handle, so in-process fd redirection cannot see its warnings.
    child = subprocess.run([sys.executable, "-B", "-u", __file__, "--synthesize", str(sid), str(path)],
                           input=text.encode("utf-8"), capture_output=True, check=True, timeout=300)
    log = child.stderr.decode("utf-8", "replace").split("@@GENERATE\n", 1)[1]
    dropped = sum("Skip unknown phonemes" in line or "Unknown token" in line for line in log.splitlines())
    with wave.open(str(path), "rb") as handle:
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768
        return samples, handle.getframerate(), dropped


def recognizer_zh():
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(voices.ASR / "model.int8.onnx"), tokens=str(voices.ASR / "tokens.txt"),
        num_threads=2, provider="cpu", language="zh", use_itn=True,
    )


def narrate(script):
    staging = NARRATION.with_name(NARRATION.name + ".partial")
    staging.mkdir()
    sid = script["voice"]["sid"]
    asr = recognizer_zh()
    samples = []
    for line in lines_of(script):
        raw, rate, dropped = synthesize(line["say"], sid, staging / f"{line['id']}-native.wav")
        if dropped:
            raise ValueError(f"Kokoro dropped {dropped} phonemes in {line['id']}: {line['say']}")
        voice = resample_poly(raw, RATE, rate)
        path = staging / f"{line['id']}.wav"
        write_wave(path, voice / max(1.0, float(np.max(np.abs(voice))) / 0.95))
        samples.append({"id": line["id"], "tts_text": line["say"], "dropped_phonemes": dropped,
                        "seconds": len(voice) / RATE, "sha256": sha(path),
                        "asr_proxy": voices.transcribe(asr, path)})
    save(staging / "result.json", {"status": "no_dropped_phonemes",
                                   "detector": "per-line child process; stderr after model load",
                                   "voice": script["voice"], "samples": samples})
    staging.rename(NARRATION)


def frames(path, crop, count):
    data = run([FF, "-v", "error", "-threads", "2", "-i", path, "-an", "-vf", f"crop={crop}",
                "-filter_threads", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
    width, height = (int(value) for value in crop.split(":")[:2])
    return np.frombuffer(data, dtype=np.uint8).reshape((count, height, width, 3))


def verify():
    plan = read(OUT / "planned-schedule.json")
    if sha(OLD / "editorial.mp4") != OLD_SHA or sha(SOURCE) != SOURCE_SHA or sha(FACTS) != plan["facts_sha256"]:
        raise ValueError("An existing source changed")
    info = json.loads(run([PROBE, "-v", "error", "-count_frames", "-show_streams", "-of", "json", MOVIE]))
    video = next(item for item in info["streams"] if item["codec_type"] == "video")
    sound = next(item for item in info["streams"] if item["codec_type"] == "audio")
    if (video["width"], video["height"], video["avg_frame_rate"], int(video["nb_read_frames"])) != (1080, 1920, "30/1", FRAMES):
        raise ValueError("Encoded picture geometry or frame count changed")
    if abs(float(video["duration"]) - 26.6) > 1 / 48000 or abs(float(sound["duration"]) - 26.6) > 1 / 48000:
        raise ValueError("Encoded audio and video durations must both be 26.6 seconds")

    def small(path):
        data = run([FF, "-v", "error", "-threads", "2", "-i", path, "-an",
                    "-vf", "crop=1068:600:6:350,scale=178:100", "-filter_threads", "1",
                    "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
        return np.frombuffer(data, dtype=np.uint8).reshape((FRAMES, 100, 178, 3)).astype(np.int16)
    before, after = small(OLD / "editorial.mp4"), small(MOVIE)
    errors = np.mean(np.abs(before - after), axis=(1, 2, 3))
    wrong = float(np.mean(np.abs(before[30:] - after[:-30])))
    if float(errors.max()) > 3 or float(errors.mean()) >= wrong / 2:
        raise ValueError("The court-picture comparison failed or could not discriminate a one-second offset")

    hook_crop = f"1080:{HOOK_HEIGHT}:0:{HOOK_Y}"
    top, base_top = (frames(path, hook_crop, FRAMES).astype(np.int16) for path in (MOVIE, OLD / "editorial.mp4"))
    hook_png = np.asarray(Image.open(OUT / "hook.png").convert("RGB"), dtype=np.int16)
    hook_end = round(plan["hook"]["display_span_s"][1] * 30)
    hook_errors = np.mean(np.abs(top[:hook_end] - hook_png), axis=(1, 2, 3))
    hook_control = np.mean(np.abs(base_top[:hook_end] - hook_png), axis=(1, 2, 3))
    after_hook = np.mean(np.abs(top[hook_end:] - base_top[hook_end:]), axis=(1, 2, 3))
    if (hook_end < BURNED_HOOK_FRAMES or float(hook_errors.max()) > 4
            or float(hook_errors.max()) >= float(hook_control.min()) / 2 or float(after_hook.max()) > 3):
        raise ValueError("The Chinese hook does not cover the burned-in hook exactly when planned")

    card_crop = f"1080:390:0:{CARD_Y}"
    bottom, base_bottom = (frames(path, card_crop, FRAMES).astype(np.int16) for path in (MOVIE, OLD / "editorial.mp4"))
    coverage, covered = [], np.zeros(FRAMES, dtype=bool)
    lines = [line for beat in plan["beats"] for line in beat["lines"]]
    pngs = {line["id"]: np.asarray(Image.open(OUT / f"{line['id']}.png").convert("RGB"), dtype=np.int16) for line in lines}
    for line in lines:
        a, b = line["card_span_frames"]
        mae = np.mean(np.abs(bottom[a:b] - pngs[line["id"]]), axis=(1, 2, 3))
        control = np.mean(np.abs(base_bottom[a:b] - pngs[line["id"]]), axis=(1, 2, 3))
        neighbours = [np.mean(np.abs(bottom[a:b] - pngs[other["id"]]), axis=(1, 2, 3)).min()
                      for other in lines if other["id"] != line["id"]]
        if (float(mae.max()) > 4 or float(mae.max()) >= float(control.min()) / 2
                or float(mae.max()) >= float(min(neighbours)) / 2):
            raise ValueError(f"Card {line['id']} is not the one shown during its span")
        covered[a:b] = True
        coverage.append({"id": line["id"], "card_mae_max": float(mae.max()),
                         "base_caption_control_mae_min": float(control.min()),
                         "other_card_control_mae_min": float(min(neighbours))})
    if not covered.all():
        raise ValueError("Some frames have no planned card")

    decoded = np.frombuffer(run([FF, "-v", "error", "-threads", "2", "-i", MOVIE,
                                 "-vn", "-ac", "1", "-ar", str(RATE), "-f", "f32le", "pipe:1"]), dtype="<f4")
    expected = audio(OUT / "mix.wav")
    correlations = []
    asr = recognizer_zh()
    for line in lines:
        a, b = (round(value * RATE) for value in line["speech_span_s"])
        correlation = float(np.corrcoef(expected[a:b], decoded[a:b])[0, 1])
        if not math.isfinite(correlation) or correlation < 0.98:
            raise ValueError(f"Encoded narration does not match the mix: {line['id']}")
        stream = asr.create_stream()
        stream.accept_waveform(16000, resample_poly(decoded[a:b], 1, 3).astype(np.float32))
        asr.decode_stream(stream)
        correlations.append({"id": line["id"], "correlation": correlation,
                             "final_mix_asr_proxy": stream.result.text})
    result = {
        "status": "technical_media_verified",
        "video_sha256": sha(MOVIE), "plan_sha256": sha(OUT / "planned-schedule.json"),
        "frames": FRAMES, "duration_seconds": 26.6,
        "picture_mae_max": float(errors.max()), "picture_mae_mean": float(errors.mean()),
        "wrong_one_second_offset_mae": wrong,
        "hook_cover_mae_max": float(hook_errors.max()), "hook_chinese_control_mae_min": float(hook_control.min()),
        "after_hook_vs_base_mae_max": float(after_hook.max()),
        "card_coverage": coverage, "encoded_audio": correlations,
        "dropped_phonemes": 0,
        "naturalness_verified": False, "human_listening": False,
        "asr_note": "Chinese ASR is a rough transcript check, not a human listening or naturalness review.",
        "numbers_note": "On-screen numbers are TrackNetV3/pose-projection estimates; shot names come from frame review.",
    }
    save(OUT / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def render():
    if sha(OLD / "editorial.mp4") != OLD_SHA or sha(SOURCE) != SOURCE_SHA:
        raise ValueError("Source identity mismatch")
    script = read(HERE / "script.json")
    schedule = read(OLD / "planned-schedule.json")
    if not NARRATION.exists():
        narrate(script)
    generated = read(NARRATION / "result.json")
    observed = {item["id"]: item for item in generated["samples"]}
    if [(item["id"], item["tts_text"]) for item in generated["samples"]] != [(line["id"], line["say"]) for line in lines_of(script)]:
        raise ValueError("Cached narration does not match the script lines")
    OUT.mkdir()
    size = card_size([line["show"] for line in lines_of(script)])
    hook(OUT / "hook.png", script["hook"]["lines"])
    mix = np.zeros(round(schedule["duration_s"] * RATE), dtype=np.float32)
    beats = []
    for beat, planned in zip(schedule["beats"], script["beats"], strict=True):
        if beat["id"] != planned["id"]:
            raise ValueError("Commentary beats must follow the base schedule order")
        a, b = beat["display_span_s"]
        if abs(a * 30 - round(a * 30)) > 1e-9 or abs(b * 30 - round(b * 30)) > 1e-9:
            raise ValueError(f"Beat {beat['id']} is not frame-aligned")
        starts = [line["at_frame"] for line in planned["lines"]] + [round(b * 30)]
        if starts[0] != round(a * 30) or any(x >= y for x, y in zip(starts, starts[1:])):
            raise ValueError(f"Lines must tile {beat['id']} from its start in order")
        source_a, source_b = beat["range"]
        if abs(source_b - source_a - (b - a)) > 1e-6:
            raise ValueError("This edition must retain real-time footage")
        court = np.frombuffer(run([FF, "-v", "error", "-threads", "2", "-ss", source_a, "-t", source_b - source_a,
                                   "-i", SOURCE, "-vn", "-ac", "1", "-ar", RATE, "-f", "f32le", "pipe:1"]), dtype="<f4")
        if len(court) != round((b - a) * RATE):
            raise ValueError("Court-audio extraction duration differs from the scene")
        mix[round(a * RATE):round(b * RATE)] += court * 0.08
        lines = []
        for line, end in zip(planned["lines"], starts[1:]):
            path = NARRATION / f"{line['id']}.wav"
            if sha(path) != observed[line["id"]]["sha256"] or observed[line["id"]]["tts_text"] != line["say"]:
                raise ValueError(f"Generated narration changed: {line['id']}")
            samples = audio(path)
            speech_start = line["at_frame"] / 30 + SPEECH_LEAD
            speech_end = speech_start + len(samples) / RATE
            if speech_end > end / 30 - SPEECH_GAP:
                raise ValueError(f"{line['id']} runs past its slot; do not speed up the voice to fit")
            gain = min(0.12 / float(np.sqrt(np.mean(samples * samples))), 0.85 / float(np.max(np.abs(samples))))
            first = round(speech_start * RATE)
            mix[first:first + len(samples)] += samples * gain
            card_path = OUT / f"{line['id']}.png"
            card(card_path, line["tag"], line["show"], script["footer"], size)
            lines.append({"id": line["id"], "say": line["say"], "show": line["show"], "tag": line["tag"],
                          "card_span_frames": [line["at_frame"], end],
                          "card_span_s": [line["at_frame"] / 30, end / 30], "speech_span_s": [speech_start, speech_end],
                          "voice_sha256": sha(path), "gain_only": gain, "voice_speed": 1.0,
                          "card_sha256": sha(card_path)})
        beats.append({"id": beat["id"], "replaces": planned["replaces"], "source_span_s": beat["range"],
                      "display_span_s": [a, b], "lines": lines})
    write_wave(OUT / "mix.wav", mix)
    save(OUT / "planned-schedule.json", {
        "fps": 30, "frames": FRAMES, "duration_s": 26.6, "beats": beats,
        "hook": {**script["hook"], "png_sha256": sha(OUT / "hook.png"), "y": HOOK_Y},
        "card_font_px": size, "base_video_sha256": OLD_SHA, "source_sha256": SOURCE_SHA,
        "facts_sha256": sha(FACTS), "court_audio_gain": 0.08, "mix_sha256": sha(OUT / "mix.wav"),
        "renderer_sha256": sha(Path(__file__)), "script_sha256": sha(HERE / "script.json"),
        "voice": "Kokoro v1.1 Chinese (sid 60), local sherpa-onnx, speed 1.0",
    })
    cards = [line for beat in beats for line in beat["lines"]]
    arguments = [FF, "-hide_banner", "-loglevel", "error", "-n", "-threads", "2",
                 "-i", OLD / "editorial.mp4", "-i", OUT / "mix.wav"]
    for line in cards:
        arguments += ["-loop", "1", "-framerate", "30", "-i", OUT / f"{line['id']}.png"]
    arguments += ["-loop", "1", "-framerate", "30", "-i", OUT / "hook.png"]
    hook_end = round(script["hook"]["display_span_s"][1] * 30)
    filters = [f"[0:v][{len(cards) + 2}:v]overlay=0:{HOOK_Y}:enable='lt(n,{hook_end})'[v0]"]
    for index, line in enumerate(cards):
        a, b = line["card_span_frames"]
        filters.append(f"[v{index}][{index + 2}:v]overlay=0:{CARD_Y}:enable='gte(n,{a})*lt(n,{b})'[v{index + 1}]")
    arguments += ["-filter_complex_threads", "1", "-filter_complex", ";".join(filters),
                  "-map", f"[v{len(cards)}]", "-map", "1:a", "-t", "26.6",
                  "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-threads", "2",
                  "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", MOVIE]
    run(arguments)
    save(OUT / "attribution.json", {
        "existing_footage_attribution": read(OLD / "attribution.json"),
        "shuttle_model": "TrackNetV3 official checkpoint (qaz812345/TrackNetV3, MIT), local inference",
        "speech_model": "Kokoro v1.1 multi-lang, Chinese sid 60 (sherpa-onnx), local inference",
        "asr_proxy_model": "SenseVoice int8 2024-07-17 (sherpa-onnx), local inference",
    })
    verify()
    run([FF, "-v", "error", "-threads", "2", "-ss", "10.5", "-i", MOVIE, "-frames:v", "1", "-update", "1", OUT / "poster.png"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--narrate-only", action="store_true")
    parser.add_argument("--synthesize", nargs=2, metavar=("SID", "PATH"))
    args = parser.parse_args()
    if args.synthesize:
        synthesize_child(int(args.synthesize[0]), Path(args.synthesize[1]))
        return
    if psutil.virtual_memory().available < 4 * 1024**3:
        raise RuntimeError("Not enough available memory for this bounded render")
    lease = ROOT / "obj2" / "work" / "cpu.lock"
    marker = {"owner": "chinese-technical-render", "pid": os.getpid(), "id": str(uuid.uuid4())}
    with lease.open("x") as handle:
        json.dump(marker, handle)
    try:
        if args.verify_only:
            verify()
        elif args.narrate_only:
            narrate(read(HERE / "script.json"))
            print((NARRATION / "result.json").read_text(encoding="utf-8"), flush=True)
        else:
            render()
    finally:
        error = sys.exception()
        if error is not None and OUT.exists():
            save(OUT / "failure.json", {"type": type(error).__name__, "message": str(error)})
        if read(lease) != marker:
            raise RuntimeError("CPU lease ownership changed")
        lease.unlink()


if __name__ == "__main__":
    main()
