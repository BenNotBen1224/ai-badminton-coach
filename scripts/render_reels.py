"""Rebuild the three published reels from source footage and frozen editorial inputs."""
import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studio.editorial import render as editorial

REELS = ROOT / "studio" / "reels"
SOURCE_SHA = "2500471d4852a4a73e0f4eb25ad28171b3c54967f58e1cc7620af16b46ccbb52"
FPS, RATE = 30, 48000
EDITIONS = ("technical", "base", "coaching")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, doc):
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    result = subprocess.run(list(map(str, args)), capture_output=True, timeout=600)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-4000:])
    return result.stdout


def tool(value, name):
    if value:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path.resolve())
    for path in (ROOT / "tools" / f"{name}.exe", ROOT / "tools" / name):
        if path.is_file():
            return str(path)
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(f"{name} not found; supply --{name} or add it to PATH")
    return found


def audio(ffmpeg, path):
    return np.frombuffer(run([ffmpeg, "-v", "error", "-threads", "2", "-i", path, "-vn",
                              "-ac", "1", "-ar", RATE, "-f", "f32le", "pipe:1"]), dtype="<f4")


def write_wave(path, samples):
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) >= 0.99:
        raise ValueError(f"Invalid or clipping audio for {path.name}")
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
        stream.writeframes(np.rint(samples * 32767).astype("<i2").tobytes())


def checked_voices(inputs, edition, manifest):
    rows = manifest["editions"][edition]
    for row in rows:
        path = inputs / row["file"]
        if not path.is_file() or sha(path) != row["sha256"]:
            raise ValueError(f"Missing or changed narration input: {row['file']}")
    return rows


def synthesize_voices(edition, manifest, models, target):
    import sherpa_onnx
    target.mkdir(parents=True, exist_ok=False)
    if edition == "coaching":
        from coach.editorial_short import _synthesize
        generated = _synthesize([(r["id"], r["text"]) for r in manifest["editions"][edition]],
                                models / "vits-melo-tts-zh_en")
        for row, speech in zip(manifest["editions"][edition], generated, strict=True):
            with wave.open(str(target / f"{row['id']}.wav"), "wb") as stream:
                stream.setparams((1, 2, speech["rate"], 0, "NONE", "not compressed"))
                stream.writeframes(np.rint(speech["samples"] * 32767).astype("<i2").tobytes())
    else:
        folder = models / "kokoro-multi-lang-v1_1"
        model = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=str(folder / "model.onnx"), voices=str(folder / "voices.bin"),
            tokens=str(folder / "tokens.txt"),
            lexicon=f"{folder / 'lexicon-us-en.txt'},{folder / 'lexicon-zh.txt'}",
            data_dir=str(folder / "espeak-ng-data"), dict_dir=str(folder / "dict"),
        )
        engine = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(kokoro=model, num_threads=2)))
        for row in manifest["editions"][edition]:
            chunks = row.get("chunks", [row["text"]])
            pieces = []
            for index, text in enumerate(chunks):
                generated = engine.generate(text, sid=60, speed=1.0)
                pieces.append(np.asarray(generated.samples, dtype=np.float32))
                if index + 1 < len(chunks):
                    pieces.append(np.zeros(round(0.06 * generated.sample_rate), np.float32))
            samples = np.concatenate(pieces)
            write_rate = generated.sample_rate
            with wave.open(str(target / f"{row['id']}.wav"), "wb") as stream:
                stream.setparams((1, 2, write_rate, 0, "NONE", "not compressed"))
                stream.writeframes(np.rint(samples * 32767).astype("<i2").tobytes())
    return [{**row, "file": f"{row['id']}.wav", "sha256": sha(target / f"{row['id']}.wav")}
            for row in manifest["editions"][edition]]


def mix_audio(args, plan, rows, inputs, frames):
    mix = np.zeros(round(frames / FPS * RATE), dtype=np.float32)
    if args.edition != "coaching":
        for beat in plan["beats"]:
            start, end = beat["range"]
            court = np.frombuffer(run([
                args.ffmpeg, "-v", "error", "-threads", "2", "-ss", start, "-i", args.source,
                "-t", end - start, "-vn", "-ac", "1", "-ar", RATE, "-f", "f32le", "pipe:1",
            ]), dtype="<f4")
            first, last = (round(t * RATE) for t in beat["display_span_s"])
            if len(court) != last - first:
                raise ValueError("Court sound length differs from the planned source interval")
            mix[first:last] += court * 0.08
    voice_checks = []
    for row in rows:
        samples = audio(args.ffmpeg, inputs / row["file"])
        start = round(row["start_s"] * RATE)
        end = start + len(samples)
        if not len(samples) or np.max(np.abs(samples)) < 0.01:
            raise ValueError(f"Empty or silent narration: {row['id']}")
        if end > round(row["slot_end_s"] * RATE) or end > len(mix):
            raise ValueError(f"Narration does not fit its unchanged slot: {row['id']}")
        gain = 0.9 if args.edition != "technical" else min(
            0.12 / float(np.sqrt(np.mean(samples * samples))),
            0.85 / float(np.max(np.abs(samples))),
        )
        mix[start:end] += samples * gain
        voice_checks.append({"id": row["id"], "text": row["text"], "start_sample": start, "end_sample": end,
                             "voice_sha256": sha(inputs / row["file"]), "gain": gain})
    peak = float(np.max(np.abs(mix)))
    if peak > 0.96:
        mix *= 0.96 / peak
    write_wave(args.out / "mix.wav", mix)
    return voice_checks


@lru_cache(maxsize=16)
def font(path, size):
    return ImageFont.truetype(path, size)


def technical_cards(script, font_path):
    cards, controls = {}, {}
    texts = [line["show"] for beat in script["beats"] for line in beat["lines"]]
    sizes = [size for size in (64, 60, 56, 52)
             if all(font(font_path, size).getlength(text) <= 976 for text in texts)]
    if not sizes:
        raise ValueError("Technical commentary overflows the card")
    for beat in script["beats"]:
        for line in beat["lines"]:
            image = Image.new("RGB", (1080, 390), "#07161d")
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 1079, 3), fill="#a8e255")
            draw.text((52, 36), line["tag"], font=font(font_path, 40), fill="#a8e255")
            draw.text((52, 350), script["footer"], font=font(font_path, 22), fill="#a5b4be")
            controls[line["id"]] = image.copy()
            draw.text((52, 215 - round(sizes[0] * 0.66)), line["show"],
                      font=font(font_path, sizes[0]), fill="#f3f6f7")
            cards[line["id"]] = image
    hook = Image.new("RGB", (1080, 170), "#07161d")
    draw = ImageDraw.Draw(hook)
    draw.rectangle((0, 166, 1079, 169), fill="#a8e255")
    for index, text in enumerate(script["hook"]["lines"]):
        f = font(font_path, 52)
        width = f.getlength(text)
        if width > 1000:
            raise ValueError("Hook overflows")
        draw.text(((1080 - width) / 2, 18 + index * 68), text, font=f,
                  fill=("#f3f6f7", "#a8e255")[index % 2])
    return cards, controls, hook


def source_pairs(args, moments):
    for moment, group in moments:
        first = round(moment["range"][0] * 60)
        last = round(moment["range"][1] * 60)
        with (args.out / f"decode-{moment['id']}.log").open("wb") as errors:
            decoder = subprocess.Popen([
                args.ffmpeg, "-v", "error", "-threads", "2", "-ss", str(first / 60),
                "-i", str(args.source), "-frames:v", str(last - first), "-an",
                "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
            ], stdout=subprocess.PIPE, stderr=errors)
            cache, position = {}, first - 1
            try:
                for row in group:
                    a, b = row["picture_frames"]
                    if not first <= a <= b < last:
                        raise ValueError(f"Source frame outside {moment['id']}: {a}, {b}")
                    while position < b:
                        data = decoder.stdout.read(1920 * 1080 * 3)
                        if len(data) != 1920 * 1080 * 3:
                            raise ValueError("Source decoder ended before the planned frame")
                        position += 1
                        cache[position] = np.frombuffer(data, np.uint8).reshape((1080, 1920, 3))
                    if a not in cache:
                        raise ValueError("Source frame plan is not monotonic")
                    weight = row["picture_weight"]
                    image = cache[a].copy() if a == b else cv2.addWeighted(cache[a], 1 - weight, cache[b], weight, 0)
                    yield row, image
                    cache = {key: value for key, value in cache.items() if key >= a}
                while decoder.stdout.read(1024 * 1024):
                    pass
                if decoder.wait() != 0:
                    raise RuntimeError(f"Source decoding failed for {moment['id']}")
            finally:
                decoder.stdout.close()
                if decoder.poll() is None:
                    decoder.terminate()
                decoder.wait()


def current_frames(args, plan, mapping):
    editorial.FONT = Path(args.font)
    editorial.VERTICAL = True
    editorial.WIDTH, editorial.HEIGHT = 1080, 1920
    editorial.PICTURE_BOX = (6, 350, 1068, 601)
    editorial._FONT_CACHE.clear()
    script = read(REELS / "technical-script.json")
    cards, no_caption, hook = technical_cards(script, args.font)
    lines = [line for beat in script["beats"] for line in beat["lines"]]
    ends = [line["at_frame"] for line in lines[1:]] + [798]
    spans = [(line, end) for line, end in zip(lines, ends, strict=True)]
    moments = read(ROOT / "results" / "editorial-base-v3" / "script.json")["moments"]
    poses = {}
    for name in ("segment-02", "segment-05", "segment-10"):
        doc = read(ROOT / "studio" / "shuttle" / "engine" / "pose_cache" / f"{name}.json")
        poses[f"{name}.json"] = {frame["frame_index"]: frame for frame in doc["frames"]}
    beats = {b["id"]: b for b in plan["beats"]}
    moment_groups = [(moment, [r for r in mapping["frames"] if r["moment"] == moment["id"]])
                     for moment in moments]
    for row, raw in source_pairs(args, moment_groups):
        moment = row["moment"]
        name = row["pose_cache"].replace("\\", "/").rsplit("/", 1)[-1]
        pose = poses[name][row["pose_frame"]]
        if abs(row["pose_source_time_s"] - row["source_position"] / 60) > 1 / 60 + 0.0002:
            raise ValueError("Pose/source sync exceeds the frozen tolerance")
        if row["skeleton_drawn"]:
            editorial.overlay_pose(raw, pose, moment)
        x0, y0, x1, y1 = row["picture_crop_xyxy"]
        focused = cv2.resize(raw[y0:y1, x0:x1], (1068, 601), interpolation=cv2.INTER_AREA)
        output = editorial.vertical_frame(focused, beats[row["beat"]], row["caption_text"], row["output_frame"] / FPS)
        if args.edition == "technical":
            image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            line = next(line for line, end in spans if line["at_frame"] <= row["output_frame"] < end)
            image.paste(cards[line["id"]], (0, 1040))
            if row["output_frame"] < round(script["hook"]["display_span_s"][1] * FPS):
                image.paste(hook, (0, 130))
            output = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            card_id = line["id"]
            control_image = image.copy()
            control_image.paste(no_caption[line["id"]], (0, 1040))
            control = cv2.cvtColor(np.asarray(control_image), cv2.COLOR_RGB2BGR)
        else:
            card_id = row["caption_text"]
            control = None
        if card_id and args.edition == "base":
            control = editorial.vertical_frame(focused, beats[row["beat"]], None, row["output_frame"] / FPS)
        yield output, {"frame": row["output_frame"], "source_position_60fps": row["source_position"],
                       "card": card_id, "skeleton_drawn": row["skeleton_drawn"]}, control


def coaching_frames(args, plan, mapping):
    from coach.editorial_short import render_frame
    from coach.tracks import load_tracks
    tracks = load_tracks(read(ROOT / "results" / "tiktok-coaching-editorial-20261103-v4" / "pose.json"))
    fonts = tuple(font(args.font, size) for size in (72, 42, 44, 28, 52))
    evidence = {chapter["key"]: chapter["evidence"] for chapter in plan["chapters"]}
    story = {"fonts": fonts, "evidence": evidence, "tactics": evidence["tactics"]}
    clip = args.out / "source-clip04.mp4"
    run([args.ffmpeg, "-v", "error", "-threads", "2", "-ss", "20", "-i", args.source,
         "-t", "30", "-an", "-vf", "fps=30", "-c:v", "libx264", "-threads", "2", "-crf", "18",
         "-frames:v", "900", "-pix_fmt", "yuv420p", clip])
    cap = cv2.VideoCapture(str(clip))
    try:
        for index, state in enumerate(mapping["states"]):
            frame = state["source_frame"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, raw = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode coaching source frame {frame}")
            output, boxes, crop = render_frame(raw, tracks, state, plan, story, fonts,
                                               plan["credit"]["endcard"], index / FPS)
            caption = next((row["text"] for row in plan["captions"]
                            if row["start_s"] <= index / FPS < row["end_s"]), None)
            control = None
            if caption:
                control, _, _ = render_frame(raw, tracks, state, {**plan, "captions": []}, story, fonts,
                                             plan["credit"]["endcard"], index / FPS)
            yield output, {"frame": index, "source_frame_clip04": frame, "chapter": state["chapter"],
                           "card": caption, "text_boxes": boxes, "crop": crop}, control
    finally:
        cap.release()


def tiny_picture(image):
    return cv2.resize(image, (180, 320), interpolation=cv2.INTER_AREA)


def tiny_card(image):
    return cv2.resize(image[1040:1430], (540, 195), interpolation=cv2.INTER_AREA)


def caption_strip(image):
    return cv2.resize(image[1115:1590, 44:1036], (496, 238), interpolation=cv2.INTER_AREA)


def verify(args, expected_frames, references, voice_checks):
    movie = args.out / "reel.mp4"
    info = json.loads(run([args.ffprobe, "-v", "error", "-count_frames", "-show_streams", "-of", "json", movie]))
    picture = next(s for s in info["streams"] if s["codec_type"] == "video")
    sound = next(s for s in info["streams"] if s["codec_type"] == "audio")
    if (picture["width"], picture["height"], picture["avg_frame_rate"], int(picture["nb_read_frames"])) != (
            1080, 1920, "30/1", expected_frames):
        raise ValueError("Encoded geometry, frame rate or frame count differs")
    duration = expected_frames / FPS
    if abs(float(picture["duration"]) - duration) > 0.001 or abs(float(sound["duration"]) - duration) > 0.05:
        raise ValueError("Encoded audio/video duration differs")
    cap = cv2.VideoCapture(str(movie))
    errors, card_errors, card_controls = [], [], []
    caption_errors, missing_caption_controls = [], []
    try:
        for index in range(expected_frames):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Encoded frame {index} is missing")
            errors.append(float(np.abs(tiny_picture(image).astype(float) - references["picture"][index]).mean()))
            if args.edition == "technical":
                card = tiny_card(image).astype(float)
                expected_id = references["card_ids"][index]
                expected = references["cards"][expected_id]
                card_errors.append(float(np.abs(card - expected).mean()))
                card_controls.append(min(float(np.abs(card - other).mean())
                                         for key, other in references["cards"].items() if key != expected_id))
            if references["has_caption"][index]:
                actual = caption_strip(image).astype(float)
                expected, control = references["caption"][index], references["caption_control"][index]
                mask = np.max(np.abs(expected.astype(float) - control), axis=2) > 30
                if int(mask.sum()) < 50:
                    raise ValueError(f"Caption check cannot discriminate missing text at {index}")
                error = float(np.abs(actual[mask] - expected[mask]).mean())
                missing = float(np.abs(actual[mask] - control[mask]).mean())
                if error > 12 or error >= missing / 2:
                    raise ValueError(f"Encoded caption is missing or changed at {index}")
                caption_errors.append(error)
                missing_caption_controls.append(missing)
    finally:
        cap.release()
    if max(errors) > 5:
        raise ValueError(f"Decoded pictures differ from generated frames: {max(errors)}")
    if card_errors and (max(card_errors) > 4 or any(a >= b / 2 for a, b in zip(card_errors, card_controls))):
        raise ValueError("Technical card coverage failed to distinguish wrong or missing cards")
    expected_audio = audio(args.ffmpeg, args.out / "mix.wav")
    decoded = audio(args.ffmpeg, movie)
    correlations = []
    for row in voice_checks:
        a, b = row["start_sample"], row["end_sample"]
        correlation = float(np.corrcoef(expected_audio[a:b], decoded[a:b])[0, 1])
        if not math.isfinite(correlation) or correlation < 0.98:
            raise ValueError(f"Encoded voice does not match planned sound: {row['id']}")
        correlations.append({**row, "encoded_mix_correlation": correlation})
    return {"status": "verified_regenerated_media", "edition": args.edition, "frames": expected_frames,
            "duration_s": duration, "dimensions": [1080, 1920], "fps": FPS,
            "picture_mae_max": max(errors), "technical_card_mae_max": max(card_errors) if card_errors else None,
            "technical_wrong_card_mae_min": min(card_controls) if card_controls else None,
            "picture_frames_checked": len(errors), "card_frames_checked": len(card_errors),
            "caption_frames_checked": len(caption_errors),
            "caption_mask_mae_max": max(caption_errors) if caption_errors else None,
            "missing_caption_control_mae_min": min(missing_caption_controls) if missing_caption_controls else None,
            "voice_coverage": correlations, "source_sha256": sha(args.source), "video_sha256": sha(movie),
            "renderer_sha256": sha(Path(__file__)), "voice_mode": args.voice_mode,
            "fresh_pose_inference": False, "byte_identical_reference_claimed": False,
            "human_listening": False, "naturalness_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition", choices=EDITIONS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="directory extracted by get_reel_inputs.py")
    parser.add_argument("--out", type=Path, required=True, help="new directory, never overwritten")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    parser.add_argument("--font", default=str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "msyh.ttc"))
    parser.add_argument("--voice-mode", choices=("frozen", "synthesize"), default="frozen")
    parser.add_argument("--models", type=Path, help="voice model parent; defaults to models or coach/models")
    parser.add_argument("--verify-only", action="store_true", help="decode and recheck an existing render")
    args = parser.parse_args()
    if args.models is None:
        args.models = ROOT / "coach" / "models" if args.edition == "coaching" else ROOT / "models"
    if sha(args.source) != SOURCE_SHA:
        raise ValueError("Source SHA256 differs; use scripts/get_footage.py --no-cut")
    if not Path(args.font).is_file():
        raise FileNotFoundError("Font is missing; supply --font with a Chinese TrueType/OpenType font")
    args.ffmpeg, args.ffprobe = tool(args.ffmpeg, "ffmpeg"), tool(args.ffprobe, "ffprobe")
    manifest = read(REELS / "inputs-manifest.json")
    if args.voice_mode == "frozen":
        rows = checked_voices(args.inputs, args.edition, manifest)
    if args.verify_only:
        expected = np.load(args.out / "expected-frames.npz", allow_pickle=False)
        binding = read(args.out / "render-inputs.json")
        if binding["edition"] != args.edition or binding["voice_mode"] != args.voice_mode:
            raise ValueError("Verification edition/voice mode differs from the render")
        references = {"picture": expected["picture"], "card_ids": binding["card_ids"],
                      "cards": {key: expected[f"card-{key}"] for key in set(binding["card_ids"])},
                      "caption": expected["caption"], "caption_control": expected["caption_control"],
                      "has_caption": binding["has_caption"]}
        result = verify(args, binding["frames"], references, binding["voices"])
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    if args.voice_mode == "synthesize":
        rows = synthesize_voices(args.edition, manifest, args.models, args.out / "voices")
        voice_inputs = args.out / "voices"
    else:
        voice_inputs = args.inputs
    if args.edition == "coaching":
        plan, mapping = read(REELS / "coaching-plan.json"), read(REELS / "coaching-frame-map.json")
        count = plan["planned_output_frames"]
        pictures = coaching_frames(args, plan, mapping)
    else:
        plan, mapping = read(REELS / "base-plan.json"), read(REELS / "base-frame-map.json")
        count = plan["frame_count"]
        pictures = current_frames(args, plan, mapping)
    voice_checks = mix_audio(args, plan, rows, voice_inputs, count)
    references = {"picture": [], "cards": {}, "card_ids": [],
                  "caption": [], "caption_control": [], "has_caption": []}
    frame_map = []
    with (args.out / "encode.log").open("wb") as errors:
        encoder = subprocess.Popen([
            args.ffmpeg, "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", "1080x1920", "-r", "30", "-i", "pipe:0", "-i", str(args.out / "mix.wav"),
            "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-threads", "2", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
            "-t", str(count / FPS), "-movflags", "+faststart", str(args.out / "reel.mp4"),
        ], stdin=subprocess.PIPE, stderr=errors)
        try:
            for index, (picture, binding, control) in enumerate(pictures):
                encoder.stdin.write(picture.tobytes())
                references["picture"].append(tiny_picture(picture))
                if args.edition == "technical":
                    key = binding["card"]
                    references["cards"].setdefault(key, tiny_card(picture))
                    references["card_ids"].append(key)
                references["has_caption"].append(control is not None)
                references["caption"].append(caption_strip(picture) if control is not None else np.zeros((1, 1, 3), np.uint8))
                references["caption_control"].append(caption_strip(control) if control is not None else np.zeros((1, 1, 3), np.uint8))
                frame_map.append(binding)
                if index % 210 == 0:
                    cv2.imwrite(str(args.out / f"frame-{index:04d}.jpg"), picture)
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("Encoder failed; see encode.log")
        finally:
            if not encoder.stdin.closed:
                encoder.stdin.close()
            if encoder.poll() is None:
                encoder.terminate()
            encoder.wait()
    if len(frame_map) != count:
        raise ValueError("Generated frame coverage differs from plan")
    save(args.out / "frame-map.json", {"fps": FPS, "frames": frame_map})
    for key in ("caption", "caption_control"):
        references[key] = [row if row.shape != (1, 1, 3) else np.zeros((238, 496, 3), np.uint8)
                           for row in references[key]]
    np.savez_compressed(args.out / "expected-frames.npz", picture=np.asarray(references["picture"]),
                        caption=np.asarray(references["caption"]), caption_control=np.asarray(references["caption_control"]),
                        **{f"card-{key}": image for key, image in references["cards"].items()})
    save(args.out / "render-inputs.json", {"edition": args.edition, "voice_mode": args.voice_mode,
                                         "frames": count, "card_ids": references["card_ids"],
                                         "has_caption": references["has_caption"],
                                         "voices": voice_checks})
    verified = verify(args, count, references, voice_checks)
    verified["font_sha256"] = sha(Path(args.font))
    verified["ffmpeg_version"] = run([args.ffmpeg, "-version"]).decode().splitlines()[0]
    verified["input_sha256"] = {name: sha(REELS / name) for name in
                               ("inputs-manifest.json", "base-plan.json", "base-frame-map.json",
                                "coaching-plan.json", "coaching-frame-map.json", "technical-script.json")}
    save(args.out / "verification.json", verified)
    print(json.dumps({key: value for key, value in verified.items() if key != "voice_coverage"}, indent=2))


if __name__ == "__main__":
    main()
