"""Reproducible, output-only original-reel editorial render; never imports coach."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
TACTICS = HERE.parent
ROOT = TACTICS.parents[2]
APP = ROOT / "badminton-coach"
OUT = Path(os.environ.get(
    "TACTICS_EDITORIAL_OUT",
    str(APP / "outputs" / "tactics-original-editorial-20261105"),
)).resolve()
SCRIPT_PATH = Path(os.environ.get("TACTICS_EDITORIAL_SCRIPT", str(HERE / "script.json"))).resolve()
SOURCE = ROOT / "obj2" / "work" / "H" / "downloads" / "china-open-2025-ws-final.webm"
FF = ROOT / "tools" / "ffmpeg.exe"
PROBE = ROOT / "tools" / "ffprobe.exe"
MODEL = ROOT / "obj2" / "work" / "NATURAL" / "voice" / "models" / "kokoro-multi-lang-v1_1"
FPS, SOURCE_FPS = 30, 60
VERTICAL = os.environ.get("TACTICS_EDITORIAL_FORMAT") == "vertical_tiktok"
SOURCE_WIDTH, SOURCE_HEIGHT = 1920, 1080
WIDTH, HEIGHT, TOP = (1080, 1920, 350) if VERTICAL else (1920, 1440, 120)
PICTURE_BOX = (6, TOP, 1068, 601) if VERTICAL else (0, TOP, WIDTH, 1080)
EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
FONT = Path(r"C:\Windows\Fonts\msyh.ttc")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_reviewed_script(script):
    if script.get("format") == "vertical_tiktok":
        if not VERTICAL:
            raise ValueError("Set TACTICS_EDITORIAL_FORMAT=vertical_tiktok for this script")
        expected = {
            "space": [57.8, 64.8],
            "time": [126.6, 134.0],
            "options": [272.8, 285.0],
        }
        moments = {moment["id"]: moment for moment in script["moments"]}
        beats = {beat["id"]: beat for beat in script["beats"]}
        if list(moment["id"] for moment in script["moments"]) != list(expected):
            raise ValueError("Vertical edition must preserve the three original source moments in order")
        if set(beats) != set(expected):
            raise ValueError("Vertical edition must have one narration beat per source moment")
        for moment_id, source_range in expected.items():
            moment = moments[moment_id]
            beat = beats[moment_id]
            if moment["range"] != source_range or beat["range"] != source_range or beat["moment"] != moment_id:
                raise ValueError(f"Vertical edition changed the approved source mapping: {moment_id}")
            chunks = beat.get("caption_chunks", [])
            if not beat.get("text") or len(chunks) < 2 or "".join(chunks) != beat["text"]:
                raise ValueError(f"Caption chunks must reproduce the spoken line: {moment_id}")
            if not beat.get("intent_cue") or not beat.get("technique_cue"):
                raise ValueError(f"Vertical edition is missing its coaching content: {moment_id}")
            if beat.get("hold", 0) != 0:
                raise ValueError("Vertical edition must not add freezes")
            emphasis = moments[moment_id].get("emphasis")
            if emphasis:
                a, b = emphasis["source_range_s"]
                if not beat["range"][0] <= a < b <= beat["range"][1]:
                    raise ValueError(f"Emphasis falls outside its continuous passage: {moment_id}")
                if not 1.0 < emphasis["zoom"] <= 1.15 or emphasis["target_role"] not in ("near", "far"):
                    raise ValueError(f"Unsafe or ungrounded emphasis settings: {moment_id}")
        if "若继续前压，可以打身后" not in beats["options"]["text"]:
            raise ValueError("The behind-the-opponent option must remain explicitly conditional")
        if "被动先过渡" not in beats["time"]["text"] or "蹬地回位" not in beats["options"]["text"]:
            raise ValueError("Use familiar, natural coaching terminology")
        return
    spec_path=TACTICS/"coaching-common-terms-spec.json"
    spec=read(spec_path)
    if script["title_zh"]!=spec["title_zh"]:
        raise ValueError("Script title differs from the parent-reviewed wording spec")
    if len(script["beats"])!=len(spec["beats"]) or script["beats"][-1]["id"]=="summary":
        raise ValueError("Script beat count or closing summary differs from the reviewed spec")
    moments={m["id"]:m for m in script["moments"]}
    approved_moments={b["moment"]:b for b in spec["beats"]}
    for moment_id,approved in approved_moments.items():
        moment=moments[moment_id]
        if moment["range"]!=approved["range"] or moment["title"]!=approved["title"]:
            raise ValueError(f"Moment title/range differs from reviewed spec: {moment_id}")
    approved_beats={b["id"]:b for b in spec["beats"]}
    for beat in script["beats"]:
        approved=approved_beats.get(beat["id"])
        if approved is None or any(beat.get(key)!=approved[key] for key in ("range","title","text","cue")):
            raise ValueError(f"Beat title, spoken text, cue or range differs from reviewed spec: {beat['id']}")


def command(args, log=None):
    p = subprocess.run([str(x) for x in args], capture_output=True)
    if log is not None:
        log.write_bytes(p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError(f"Command failed {p.returncode}: {args}\n{p.stderr.decode('utf-8', 'replace')[-3500:]}")
    return p.stdout


def snapshot():
    paths = [APP / "coach" / name for name in ("tactics.py", "tests/test_tactics.py", "README.md")]
    v3 = APP / "outputs" / "tactics-engine-20261106-v3"
    paths.extend(p for p in v3.rglob("*") if p.is_file())
    return {str(p): sha(p) for p in paths}


def assert_frozen():
    expected = read(OUT / "frozen-v3.json")
    current = snapshot()
    if current != expected:
        raise RuntimeError("Frozen V3 source or output changed; do not overwrite or repair it here.")


def wav_read(path):
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2:
            raise ValueError(f"Expected mono PCM16: {path}")
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768, stream.getframerate()


def wav_write(path, data, rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((np.clip(data, -1, 1) * 32767).astype("<i2").tobytes())


def synth(script):
    import sherpa_onnx
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(MODEL / "model.onnx"), voices=str(MODEL / "voices.bin"),
                tokens=str(MODEL / "tokens.txt"),
                lexicon=f"{MODEL / 'lexicon-us-en.txt'},{MODEL / 'lexicon-zh.txt'}",
                data_dir=str(MODEL / "espeak-ng-data"), dict_dir=str(MODEL / "dict"),
            ), num_threads=4,
        ),
        rule_fsts=",".join(str(MODEL / x) for x in ("date-zh.fst", "number-zh.fst")),
    )
    if not config.validate():
        raise RuntimeError("Local Kokoro config is invalid")
    tts = sherpa_onnx.OfflineTts(config)
    timings = {}
    previous = read(OUT/"voice.json") if (OUT/"voice.json").exists() else {}
    for beat in script["beats"]:
        path = OUT / "voice" / f"{beat['id']}.wav"
        speed=float(beat.get("voice_speed",1.0))
        if not .95<=speed<=1.05:
            raise ValueError(f"Narration speed outside the natural editorial range: {beat['id']} {speed}")
        caption_texts=beat.get("caption_chunks")
        if (beat["id"] in previous and previous[beat["id"]]["text"]==beat["text"]
                and previous[beat["id"]].get("speed")==speed and path.exists()
                and (not VERTICAL or (
                    previous[beat["id"]].get("caption_texts")==caption_texts
                    and previous[beat["id"]].get("caption_gap_s")==.06
                ))):
            if sha(path)!=previous[beat["id"]]["sha256"]:
                raise ValueError(f"Existing voice changed: {path}")
            timings[beat["id"]]=previous[beat["id"]]
            continue
        if VERTICAL:
            chunks = []
            offset = 0.0
            rate = None
            gap_s = .06
            for index, text in enumerate(beat["caption_chunks"], 1):
                audio = tts.generate(text, sid=60, speed=speed)
                samples = np.asarray(audio.samples, dtype=np.float32)
                active = np.flatnonzero(np.abs(samples) > 0.004)
                if not len(active):
                    raise RuntimeError(f"No speech generated: {beat['id']} caption {index}")
                a = max(0, int(active[0]) - round(audio.sample_rate * .02))
                b = min(len(samples), int(active[-1]) + round(audio.sample_rate * .04))
                samples = samples[a:b]
                if rate is not None and rate != audio.sample_rate:
                    raise RuntimeError("Caption TTS sample rates differ within one narration beat")
                rate = audio.sample_rate
                chunk_path = OUT / "voice" / f"{beat['id']}-caption-{index:02}.wav"
                wav_write(chunk_path, samples, rate)
                chunks.append({
                    "text": text, "path": str(chunk_path), "sha256": sha(chunk_path),
                    "duration_s": len(samples) / rate,
                    "relative_span_s": [offset, offset + len(samples) / rate],
                })
                offset += len(samples) / rate + gap_s
                if index < len(beat["caption_chunks"]):
                    chunks[-1]["relative_span_s"][1] = offset - gap_s
            combined = np.zeros(round((offset-gap_s) * rate), dtype=np.float32)
            cursor = 0
            for index, chunk in enumerate(chunks):
                samples, chunk_rate = wav_read(chunk["path"])
                if chunk_rate != rate:
                    raise RuntimeError("Caption WAV sample rate changed after writing")
                count = len(samples)
                combined[cursor:cursor+count] = samples
                cursor += count + (round(gap_s*rate) if index + 1 < len(chunks) else 0)
            wav_write(path, combined, rate)
            timings[beat["id"]] = {
                "path": str(path), "sha256": sha(path), "sample_rate": int(rate),
                "samples": len(combined), "duration_s": len(combined) / rate,
                "speaker": 60, "speed": speed, "text": beat["text"],
                "caption_chunks": chunks, "caption_texts":caption_texts,"caption_gap_s": gap_s,
            }
            print("VOICE", beat["id"], round(timings[beat["id"]]["duration_s"], 3),
                  "seconds in", len(chunks), "caption-timed chunks", flush=True)
            continue
        audio = tts.generate(beat["text"], sid=60, speed=speed)
        samples = np.asarray(audio.samples, dtype=np.float32)
        active = np.flatnonzero(np.abs(samples) > 0.004)
        if not len(active):
            raise RuntimeError(f"No speech generated: {beat['id']}")
        # Retain short leading/trailing room; do not change voice rate.
        a = max(0, int(active[0]) - round(audio.sample_rate * .05))
        b = min(len(samples), int(active[-1]) + round(audio.sample_rate * .10))
        samples = samples[a:b]
        wav_write(path, samples, audio.sample_rate)
        timings[beat["id"]] = {
            "path": str(path), "sha256": sha(path), "sample_rate": int(audio.sample_rate),
            "samples": len(samples), "duration_s": len(samples) / audio.sample_rate,
            "speaker": 60, "speed": speed, "text": beat["text"],
        }
        print("VOICE", beat["id"], round(timings[beat["id"]]["duration_s"], 3), flush=True)
    write(OUT / "voice.json", timings)
    return timings


def make_schedule(script, voice):
    frames, beats = [], []
    for beat in script["beats"]:
        a, b = [round(x * SOURCE_FPS) for x in beat["range"]]
        hold = round(beat["hold"] * FPS)
        n = max(round((b-a) / SOURCE_FPS * FPS), math.ceil((voice[beat["id"]]["duration_s"] + .24) * FPS))
        moving = n - hold
        if moving < 2:
            raise ValueError("Too many held frames")
        requested_speed = float(beat.get("target_speed", 1.0))
        if not .75 <= requested_speed <= 1.05:
            raise ValueError(f"Editorial speed outside the natural range: {beat['id']} {requested_speed}")
        planned_moving = round(((b-a) / SOURCE_FPS / requested_speed) * FPS)
        n = max(n, planned_moving + hold)
        moving = n - hold
        speed = ((b-a) / SOURCE_FPS) / (moving / FPS)
        if speed < .75:
            raise ValueError(f"Shorten speech before slowing footage excessively: {beat['id']} needs {speed:.3f}x")
        start = len(frames)
        for i in range(hold):
            frames.append({"output_frame": len(frames), "beat": beat["id"], "moment": beat["moment"],
                           "source_position": float(a), "kind": "hold"})
        for i in range(moving):
            p = a + (b-a) * i / moving
            frames.append({"output_frame": len(frames), "beat": beat["id"], "moment": beat["moment"],
                           "source_position": p, "kind": "play"})
        speech_span = [start / FPS + .08, start / FPS + .08 + voice[beat["id"]]["duration_s"]]
        caption_spans = []
        for chunk in voice[beat["id"]].get("caption_chunks", []):
            chunk_start = speech_span[0] + chunk["relative_span_s"][0]
            chunk_end = speech_span[0] + chunk["relative_span_s"][1]
            caption_spans.append({
                **chunk, "display_span_s": [chunk_start, chunk_end],
                "first_frame": max(start, math.floor(chunk_start * FPS)),
                "end_frame_exclusive": min(start + n, math.ceil(chunk_end * FPS)),
            })
        beats.append({
            **beat, "first_frame": start, "end_frame_exclusive": len(frames),
            "display_span_s": [start / FPS, len(frames) / FPS],
            "speech_span_s": speech_span,
            "caption_spans": caption_spans,
            "moving_speed": speed, "hold_frames": hold,
            "voice_sha256": voice[beat["id"]]["sha256"],
        })
    positions = [f["source_position"] for f in frames]
    if not all(b >= a for a, b in zip(positions, positions[1:])):
        raise ValueError("Source timeline goes backward")
    left, top, picture_width, picture_height = PICTURE_BOX
    return {"fps": FPS, "width": WIDTH, "height": HEIGHT,
            "picture_box": [left, top, picture_width, picture_height],
            "duration_s": len(frames)/FPS, "frame_count": len(frames), "beats": beats, "frames": frames}


def stage_evidence_and_inputs(script, *, create_frozen=False):
    draft = read(TACTICS / "video-script.json")
    if sha(SOURCE) != draft["source_sha256"]:
        raise ValueError("Original source hash mismatch")
    if create_frozen:
        write(OUT / "frozen-v3.json", snapshot())
    write(OUT / "script.json", script)
    if SCRIPT_PATH != HERE/"script.json":
        write(OUT/"source-script.json",script)
    evidence = []
    bindings = []
    for moment in script["moments"]:
        path = TACTICS / "evidence" / moment["evidence"] / "evidence.json"
        document = read(path)
        if document["source_sha256"] != draft["source_sha256"]:
            raise ValueError("Evidence and source differ")
        for cache, digest in document["cache_hashes"].items():
            if sha(Path(cache)) != digest:
                raise ValueError("Pose cache changed")
        bindings.append({"path": str(path), "sha256": sha(path)})
        for sample in document["samples"]:
            evidence.append({
                **sample, "id": f"E{sample['source_frame']:06d}", "moment": moment["id"],
                "manifest": str(path), "manifest_sha256": sha(path),
                "cache_sha256": document["cache_hashes"][sample["cache"]],
                "observation_kind": "cached_2d_pose_and_visible_picture_not_racket_contact",
            })
    write(OUT / "evidence.json", evidence)
    inputs = [SOURCE, TACTICS/"video-script.json", TACTICS/"codebook.json", TACTICS/"codebook_verified.json",
              SCRIPT_PATH, HERE/"render.py", TACTICS/"script-review"/"result"/"initial-preference.json"]
    for reviewed_spec in (TACTICS/"coaching-content-spec.json",TACTICS/"coaching-common-terms-spec.json"):
        if reviewed_spec.exists():
            inputs.append(reviewed_spec)
    if "TACTICS_EDITORIAL_OUT" in os.environ:
        original=APP/"outputs"/"tactics-original-editorial-20261105"/"editorial.mp4"
        inputs.append(original)
        if VERTICAL:
            inputs.append(APP/"outputs"/"tactics-original-editorial-20261105"/"tiktok-viral-v1"/"editorial.mp4")
    write(OUT / "inputs.json", {"inputs": {str(p): sha(p) for p in inputs}, "evidence_manifests": bindings,
                              "voice_model": {str(p): sha(p) for p in MODEL.iterdir() if p.is_file()}})


def prepare():
    if OUT.exists():
        raise FileExistsError(f"New output directory required: {OUT}")
    OUT.mkdir(parents=True)
    script = read(SCRIPT_PATH)
    validate_reviewed_script(script)
    stage_evidence_and_inputs(script, create_frozen=True)
    voice = synth(script)
    schedule = make_schedule(script, voice)
    write(OUT / "planned-schedule.json", schedule)
    print("PLANNED", schedule["duration_s"], "seconds;", schedule["frame_count"], "frames", flush=True)
    for beat in schedule["beats"]:
        print(beat["id"], beat["display_span_s"], beat["speech_span_s"], "speed", round(beat["moving_speed"],3), flush=True)


def revise():
    assert_frozen()
    if (OUT/"picture.mp4").exists():
        raise RuntimeError("Picture already encoded; do not revise its plan")
    backup=OUT/"revisions"/f"schedule-revision-{time.time_ns()}"
    backup.mkdir(parents=True,exist_ok=False)
    for name in ("script.json","voice.json","planned-schedule.json"):
        if (OUT/name).exists():
            shutil.copy2(OUT/name,backup/name)
    if (OUT/"voice").is_dir():
        shutil.copytree(OUT/"voice",backup/"voice")
    script=read(SCRIPT_PATH)
    stage_evidence_and_inputs(script)
    voice=synth(script)
    plan=make_schedule(script,voice)
    write(OUT/"planned-schedule.json",plan)
    write(OUT/"pre-encode-revision.json",{
        "reason":"Bounded pre-encode wording and timing revision; no source or inference changed.",
        "corrections":["Merged the possible-reply and observed-reply narration to fit one continuous play sequence."],
        "previous_plan":str(backup/"planned-schedule.json"),
        "previous_plan_sha256":sha(backup/"planned-schedule.json") if (backup/"planned-schedule.json").exists() else None,
        "current_plan_sha256":sha(OUT/"planned-schedule.json"),
        "narration_all_fits":True,
    })
    print("REVISED PRE-ENCODE PLAN",plan["duration_s"],plan["frame_count"],flush=True)


def extracted(moment):
    start, end = [round(x*SOURCE_FPS) for x in moment["range"]]
    path = HERE / f"{moment['id']}-source.mkv"
    if not path.exists():
        command([FF, "-hide_banner", "-loglevel", "error", "-nostdin",
                 "-threads", "2", "-ss", f"{start/SOURCE_FPS:.9f}", "-i", SOURCE,
                 "-map", "0:v:0", "-frames:v", end-start, "-an", "-c:v", "ffv1", "-threads", "2", path])
    return path, start, end


def load_frames(moment):
    path, start, end = extracted(moment)
    frames = []
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) != end-start:
        raise ValueError(f"Extracted wrong frame count: {moment['id']} {len(frames)} != {end-start}")
    return frames, start, end


def knots_for(frames, start):
    duplicates = []
    previous = None
    for i, frame in enumerate(frames):
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (480,270), interpolation=cv2.INTER_AREA)
        if previous is not None:
            diff = cv2.absdiff(gray, previous)
            if float(diff.mean()) < .35 and float(np.percentile(diff,99)) < 4:
                duplicates.append(i+start)
        previous = gray
    return [i+start for i in range(len(frames)) if i+start not in set(duplicates)], duplicates


def picture_pair(p, knots):
    if abs(p-round(p)) < 1e-6:
        return round(p), round(p), 0.0
    q = float(np.interp(p, knots, np.arange(len(knots))))
    j = math.floor(q)
    return knots[j], knots[min(j+1,len(knots)-1)], q-j


def picture(frames, start, pair):
    a,b,w = pair
    if a==b or w==0:
        return frames[a-start].copy()
    return cv2.addWeighted(frames[a-start], 1-w, frames[b-start], w, 0)


def inspect():
    assert_frozen()
    script = read(OUT/"script.json")
    evidence = read(OUT/"evidence.json")
    docs = pose_documents(evidence)
    sample_times = {
        "space": [57.8,58.4,59.0,59.6,60.2,60.65,61.35,62.05,62.6,63.3,64.0,64.6],
        "time": [126.6,127.3,128.05,128.3167,128.9,129.45,130.0167,130.6,131.9833,132.6167,133.02,133.8],
        "options": [272.8,273.5,274.0333,274.6,275.0,275.6,276.3,276.8,277.4,278.5,280.0,282.0,284.6],
    }
    manifest = []
    for moment in script["moments"]:
        frames,start,end = load_frames(moment)
        cells = []
        for requested in sample_times[moment["id"]]:
            p = min(end-1,max(start,round(requested*SOURCE_FPS)))
            path,pose,t = nearest_pose(p,docs)
            image = frames[p-start]
            output = HERE/"inspection"/f"source-{p:06d}.png"
            output.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(output),image)
            manifest.append({"id":f"C{p:06d}","moment":moment["id"],"source_frame":p,
                             "source_time_s":p/SOURCE_FPS,"image":str(output),"image_sha256":sha(output),
                             "pose_cache":str(path),"pose_cache_sha256":sha(path),
                             "pose_frame":pose["frame_index"],"pose_time_s":t,
                             "pose_sync_error_s":abs(t-p/SOURCE_FPS),"players":pose["players"]})
            cell = np.zeros((390,640,3),dtype=np.uint8)
            cell[:360] = cv2.resize(image,(640,360),interpolation=cv2.INTER_AREA)
            cv2.putText(cell,f"source {p/SOURCE_FPS:.4f}s | F{p}",(12,382),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
            cells.append(cell)
        sheet = np.zeros((math.ceil(len(cells)/3)*390,1920,3),dtype=np.uint8)
        for i,cell in enumerate(cells):
            sheet[i//3*390:(i//3+1)*390,i%3*640:(i%3+1)*640] = cell
        cv2.imwrite(str(HERE/"inspection"/f"{moment['id']}-sheet.jpg"),sheet)
    write(OUT/"context-evidence.json",manifest)
    print("SOURCE INSPECTION",len(manifest),"picture/pose bindings",flush=True)


def pose_documents(evidence):
    documents = {}
    for entry in evidence:
        path = Path(entry["cache"])
        if sha(path) != entry["cache_sha256"]:
            raise ValueError("Pose cache hash changed")
        if path not in documents:
            doc = read(path)
            if Path(doc["source"]["original_file"]).resolve() != SOURCE.resolve():
                raise ValueError("Cached pose belongs to different source")
            documents[path] = doc
    return documents


def nearest_pose(p, docs):
    t = p/SOURCE_FPS
    choices = [(path,doc) for path,doc in docs.items()
               if doc["source"]["reel_start_s"] <= t < doc["source"]["reel_end_s"]]
    if len(choices)!=1:
        raise ValueError(f"Pose source interval not unique at {t}")
    path,doc = choices[0]
    start = doc["source"]["reel_start_s"]
    frame = min(doc["frames"],key=lambda f:abs(start+f["timestamp_seconds"]-t))
    pose_t = start+frame["timestamp_seconds"]
    if abs(pose_t-t) > 1/60+.0002:
        raise ValueError(f"Pose/source sync failure {pose_t} vs {t}")
    return path,frame,pose_t


def overlay_pose(image, frame, moment):
    color_for = {"near": (90,110,255) if moment!="options" else (215,240,90),
                 "far": (215,240,90) if moment!="options" else (90,110,255)}
    for player in frame["players"]:
        role = player.get("role")
        if role not in color_for or player.get("box_confidence",0)<.70:
            continue
        joints = player["joints"]
        valid = {i for i,(x,y,c) in enumerate(joints) if c>=.70 and
                 math.isfinite(x) and math.isfinite(y) and 0<=x<SOURCE_WIDTH and 0<=y<SOURCE_HEIGHT}
        for a,b in EDGES:
            if a in valid and b in valid:
                cv2.line(image,tuple(round(x) for x in joints[a][:2]),
                         tuple(round(x) for x in joints[b][:2]),color_for[role],2,cv2.LINE_AA)
        for i in valid & set(range(5,17)):
            cv2.circle(image,tuple(round(x) for x in joints[i][:2]),3,color_for[role],-1,cv2.LINE_AA)


def text_lines(text, font, max_width):
    lines, line = [], ""
    for ch in text:
        if font.getlength(line+ch)>max_width:
            lines.append(line)
            line=ch
        else:
            line+=ch
    if line:
        lines.append(line)
    return lines


def panel(beat, moment, speed, held):
    canvas = Image.new("RGB",(WIDTH,HEIGHT),(13,22,32))
    d=ImageDraw.Draw(canvas)
    title=ImageFont.truetype(str(FONT),42)
    font=ImageFont.truetype(str(FONT),58)
    small=ImageFont.truetype(str(FONT),27)
    d.text((42,12),moment["title"],font=title,fill=(245,249,251))
    tag="观察暂停" if held else ("原速" if abs(speed-1)<.015 else f"{speed:.2f}× 慢放")
    d.text((1500,22),tag,font=small,fill=(125,225,222))
    d.text((42,76),"选段讲解 · 红衣 / 灰衣 · 保留原画面 · 非自动识别验收",font=small,fill=(168,185,196))
    lines=text_lines(beat["text"],font,WIDTH-84)
    if len(lines)>2:
        raise ValueError(f"Caption exceeds two lines: {beat['id']}")
    for i,line in enumerate(lines):
        d.text((42,1224+i*64),line,font=font,fill=(246,249,252))
    d.text((42,1374),beat["cue"],font=small,fill=(119,221,214))
    return cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)


def emphasis_picture(image, pose, moment, source_time):
    settings=moment.get("emphasis")
    if not settings:
        return cv2.resize(image,(PICTURE_BOX[2],PICTURE_BOX[3]),interpolation=cv2.INTER_AREA), {
            "zoom":1.0,"crop_xyxy":[0,0,SOURCE_WIDTH,SOURCE_HEIGHT],"focus_bbox":None,
        }
    start,end=settings["source_range_s"]
    ramp=.18
    amount=max(0.0,min(1.0,(source_time-start)/ramp,(end-source_time)/ramp))
    zoom=1+(settings["zoom"]-1)*amount
    if zoom<=1.0001:
        return cv2.resize(image,(PICTURE_BOX[2],PICTURE_BOX[3]),interpolation=cv2.INTER_AREA), {
            "zoom":zoom,"crop_xyxy":[0,0,SOURCE_WIDTH,SOURCE_HEIGHT],"focus_bbox":None,
        }
    candidates=[p for p in pose["players"] if p.get("role")==settings["target_role"]
                and p.get("box_confidence",0)>=.70]
    if not candidates:
        raise ValueError(f"Missing reliable cached pose for emphasis at {source_time}")
    bbox=candidates[0]["bbox_xyxy"]
    crop_width=round(SOURCE_WIDTH/zoom); crop_height=round(SOURCE_HEIGHT/zoom)
    cx=(bbox[0]+bbox[2])/2; cy=(bbox[1]+bbox[3])/2
    x0=max(0,min(SOURCE_WIDTH-crop_width,round(cx-crop_width/2)))
    y0=max(0,min(SOURCE_HEIGHT-crop_height,round(cy-crop_height/2)))
    x1=x0+crop_width; y1=y0+crop_height
    focused=cv2.resize(image[y0:y1,x0:x1],(PICTURE_BOX[2],PICTURE_BOX[3]),interpolation=cv2.INTER_AREA)
    return focused,{"zoom":zoom,"crop_xyxy":[x0,y0,x1,y1],"focus_bbox":bbox}


_FONT_CACHE={}


def font(size):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size]=ImageFont.truetype(str(FONT),size)
    return _FONT_CACHE[size]


def vertical_frame(image, beat, caption, output_time):
    small=cv2.resize(image,(1707,960),interpolation=cv2.INTER_AREA)
    left=(small.shape[1]-540)//2
    background=cv2.resize(small[:,left:left+540],(WIDTH,HEIGHT),interpolation=cv2.INTER_LINEAR)
    background=cv2.GaussianBlur(background,(41,41),0)
    background=cv2.convertScaleAbs(background,alpha=.34,beta=0)
    canvas=Image.fromarray(cv2.cvtColor(background,cv2.COLOR_BGR2RGB))
    sharp=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    canvas.paste(sharp,(PICTURE_BOX[0],PICTURE_BOX[1]))
    draw=ImageDraw.Draw(canvas,"RGBA")
    if beat.get("hook") and output_time<1.25:
        hook_font=font(62)
        hook=beat.get("hook_text","后场怎么调？")
        box=draw.textbbox((0,0),hook,font=hook_font,stroke_width=1)
        width=box[2]-box[0]
        x=(WIDTH-width)//2
        draw.text((x,165),hook,font=hook_font,fill=(255,255,255,255),
                  stroke_width=2,stroke_fill=(8,15,23,230))
    if caption:
        caption_font=font(59)
        lines=text_lines(caption,caption_font,930)
        if len(lines)>2:
            raise ValueError(f"Phone caption is too long: {caption}")
        draw.rounded_rectangle((44,1115,1036,1385),radius=28,fill=(7,14,22,204),
                               outline=(255,255,255,36),width=2)
        line_height=76
        total_height=len(lines)*line_height
        y=1115+(270-total_height)//2
        for line in lines:
            box=draw.textbbox((0,0),line,font=caption_font,stroke_width=1)
            x=(WIDTH-(box[2]-box[0]))//2
            draw.text((x,y),line,font=caption_font,fill=(255,255,255,255),
                      stroke_width=2,stroke_fill=(0,0,0,235))
            y+=line_height
    return cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)


def encode():
    started=time.monotonic()
    assert_frozen()
    if (OUT/"editorial.mp4").exists() or (OUT/"picture.mp4").exists():
        raise FileExistsError("Refusing to overwrite encoded picture/final media")
    script=read(OUT/"script.json"); plan=read(OUT/"planned-schedule.json")
    evidence=read(OUT/"evidence.json"); docs=pose_documents(evidence)
    write(OUT/"run.json",{
        "interpreter":sys.executable,"source":str(SOURCE),"source_sha256":sha(SOURCE),
        "helper":str(Path(__file__).resolve()),"helper_sha256_at_encode":sha(Path(__file__).resolve()),
        "script_sha256_at_encode":sha(OUT/"script.json"),
        "planned_schedule_sha256_before_encode":sha(OUT/"planned-schedule.json"),
        "frozen_v3_snapshot_sha256":sha(OUT/"frozen-v3.json"),
        "source_fps":SOURCE_FPS,"output_fps":FPS,"dimensions":[WIDTH,HEIGHT],
        "vertical_layout":VERTICAL,"picture_box":list(PICTURE_BOX),
        "pose_inference_run":False,"tts":"local Kokoro sid60 speed1.0",
        "cpu_lock_owner_at_encode":(ROOT/"obj2"/"work"/"cpu.lock").read_text(encoding="utf-8"),
        "independent_acceptance":"pending","engine_accuracy_gate":"failed_recall_unchanged",
    })
    moments={m["id"]:m for m in script["moments"]}
    beats={b["id"]:b for b in plan["beats"]}
    ff=subprocess.Popen([str(FF),"-hide_banner","-loglevel","error","-nostdin",
                         "-f","rawvideo","-pix_fmt","bgr24","-s",f"{WIDTH}x{HEIGHT}","-r",str(FPS),
                         "-i","pipe:0","-an","-c:v","libx264","-preset","fast","-crf","18",
                         "-threads","4","-pix_fmt","yuv420p","-movflags","+faststart",str(OUT/"picture.mp4")],
                        stdin=subprocess.PIPE,stderr=open(OUT/"picture-encode.log","wb"))
    frame_map=[]; duplicates=[]; current=None; panels={}
    try:
        for row in plan["frames"]:
            if row["moment"]!=current:
                current=row["moment"]; frames,start,end=load_frames(moments[current])
                knots,dups=knots_for(frames,start)
                duplicates.append({"moment":current,"source_range_frames":[start,end],"duplicates":dups,"knots":knots})
                # Check source extraction against independently decoded evidence pictures, using JPEG tolerance.
                for e in evidence:
                    if e["moment"]!=current:
                        continue
                    reference=cv2.imread(e["source_image"])
                    actual=frames[e["source_frame"]-start]
                    mae=float(np.abs(reference.astype(float)-actual.astype(float)).mean())
                    if mae>4:
                        raise ValueError(f"Extracted source frame disagrees with evidence: {e['id']} MAE={mae}")
                    e["extraction_reference_mae"]=mae
            beat=beats[row["beat"]]; moment=moments[current]; p=row["source_position"]
            pair=picture_pair(p,knots); image=picture(frames,start,pair)
            path,pose,pose_time=nearest_pose(p,docs)
            cue=any(abs(p/SOURCE_FPS-anchor)<.16 for anchor in moment["anchors"])
            if cue:
                overlay_pose(image,pose,current)
            held=row["kind"]=="hold"
            left,top,picture_width,picture_height=PICTURE_BOX
            if VERTICAL:
                focused,focus_info=emphasis_picture(image,pose,moment,p/SOURCE_FPS)
                output_time=row["output_frame"]/FPS
                caption=next((c["text"] for c in beat["caption_spans"]
                              if c["first_frame"]<=row["output_frame"]<c["end_frame_exclusive"]),None)
                output=vertical_frame(focused,beat,caption,output_time)
            else:
                focus_info={"zoom":1.0,"crop_xyxy":[0,0,SOURCE_WIDTH,SOURCE_HEIGHT],"focus_bbox":None}
                key=(beat["id"],held)
                if key not in panels:
                    panels[key]=panel(beat,moment,beat["moving_speed"],held)
                output=panels[key].copy()
                focused=image
            if not VERTICAL:
                output[top:top+picture_height,left:left+picture_width]=image
            ff.stdin.write(output.tobytes())
            frame_map.append({
                **row, "picture_frames":list(pair[:2]),"picture_weight":pair[2],
                "pose_cache":str(path),"pose_frame":pose["frame_index"],
                "pose_source_time_s":pose_time,"pose_sync_error_s":abs(pose_time-p/SOURCE_FPS),
                "skeleton_drawn":cue,"pose_score_min":.70,
                "caption_text":caption if VERTICAL else beat["text"],
                "emphasis_zoom":focus_info["zoom"],"picture_crop_xyxy":focus_info["crop_xyxy"],
                "emphasis_focus_bbox":focus_info["focus_bbox"],
            })
            if len(frame_map)%300==0:
                print("ENCODED",len(frame_map),"/",plan["frame_count"],flush=True)
        ff.stdin.close()
        if ff.wait()!=0:
            raise RuntimeError("Picture encoder failed; see picture-encode.log")
    except BaseException:
        if ff.stdin and not ff.stdin.closed:
            ff.stdin.close()
        ff.wait()
        raise
    write(OUT/"frame-map.json",{"picture_model":"unique_blend_v1_local_source_knots",
                               "fps":FPS,"source_fps":SOURCE_FPS,"frames":frame_map})
    write(OUT/"duplicate-frame-audit.json",duplicates)
    write(OUT/"evidence.json",evidence)
    mix_audio(plan, moments)
    command([FF,"-hide_banner","-loglevel","error","-nostdin","-i",OUT/"picture.mp4",
             "-i",OUT/"audio.wav","-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","aac",
             "-b:a","192k","-t",f"{plan['duration_s']:.9f}","-movflags","+faststart",OUT/"editorial.mp4"],
            OUT/"mux.log")
    run=read(OUT/"run.json")
    run["encode_and_mix_elapsed_s"]=round(time.monotonic()-started,3)
    run["video_sha256"]=sha(OUT/"editorial.mp4")
    run["local_tts_startup_warnings"]=[
        "sherpa-onnx: dict_dir is ignored from v1.12.15",
        "bundled lexicon reports unknown token U+2753; none of the spoken input lines contain that token",
    ]
    write(OUT/"run.json",run)
    print("FINAL ENCODED",sha(OUT/"editorial.mp4"),flush=True)


def mix_audio(plan, moments):
    rate=24000; total=round(plan["duration_s"]*rate)
    court=np.zeros(total,dtype=np.float32)
    # Keep real court ambience in timeline order. Nearest-time resampling is labelled;
    # no fake hits or pitch-corrected impact audio is introduced.
    for m in moments.values():
        path=HERE/f"{m['id']}-court.wav"
        a,b=m["range"]
        command([FF,"-hide_banner","-loglevel","error","-nostdin","-ss",a,"-i",SOURCE,
                 "-t",b-a,"-vn","-ac","1","-ar",rate,"-c:a","pcm_s16le","-y",path])
        samples,sr=wav_read(path)
        rows=[f for f in plan["frames"] if f["moment"]==m["id"]]
        for row in rows:
            first=round(row["output_frame"]*rate/FPS); last=round((row["output_frame"]+1)*rate/FPS)
            # Continuous sample mapping follows the same source-time step as the picture.
            i=row["output_frame"]
            nxt=plan["frames"][i+1] if i+1<len(plan["frames"]) else None
            step=(nxt["source_position"]-row["source_position"])/SOURCE_FPS if nxt and nxt["moment"]==m["id"] else 1/FPS
            if row["kind"]=="hold":
                # Never loop an impact or a short piece of audience audio during a freeze.
                continue
            positions=(row["source_position"]/SOURCE_FPS-a+np.arange(last-first)/(last-first)*step)*sr
            data=np.interp(positions,np.arange(len(samples)),samples)
            court[first:last]=data*.08
    voice_mix=np.zeros(total,dtype=np.float32)
    for beat in plan["beats"]:
        voice,sr=wav_read(OUT/"voice"/f"{beat['id']}.wav")
        if sr!=rate:
            voice=np.interp(np.arange(round(len(voice)*rate/sr))*sr/rate,np.arange(len(voice)),voice)
        start=round(beat["speech_span_s"][0]*rate); end=start+len(voice)
        if end>round(beat["display_span_s"][1]*rate):
            raise ValueError("Speech exceeds evidence window")
        voice_mix[start:end]+=voice
    mix=court+voice_mix*.90
    peak=float(np.abs(mix).max())
    if peak>.96:
        mix*=.96/peak
    wav_write(OUT/"audio.wav",mix,rate)
    write(OUT/"audio-mix.json",{"sample_rate":rate,"samples":len(mix),"duration_s":len(mix)/rate,
                              "voice_model":"local Kokoro sid60 speed1.0","court_audio_gain":.08,
                              "court_time_model":"frame_map_linear_resampling_slowed_pitch_not_contact_confirmation",
                              "digital_zero_samples":int(np.count_nonzero(mix==0)),"peak":float(np.abs(mix).max()),
                              "time_padding":False,"synthetic_impacts":False})


def spans(indices):
    if not indices:
        return []
    result=[]; start=last=indices[0]
    for i in indices[1:]:
        if i!=last+1:
            result.append([start/FPS,(last+1)/FPS]);start=i
        last=i
    result.append([start/FPS,(last+1)/FPS])
    return result


def documents():
    script=read(OUT/"script.json"); plan=read(OUT/"planned-schedule.json")
    evidence=read(OUT/"evidence.json"); mapping=read(OUT/"frame-map.json")["frames"]
    context=read(OUT/"context-evidence.json")
    evidence_ids={e["id"] for e in evidence}|{e["id"] for e in context}
    for beat in plan["beats"]:
        beat["evidence_ids"]=[
            next((candidate for candidate in (f"E{x:06d}",f"C{x:06d}") if candidate in evidence_ids),
                 f"E{x:06d}")
            for x in beat["evidence"]
        ]
        beat["context_evidence_ids"]=[e["id"] for e in context if e["moment"]==beat["moment"]
                                      and beat["range"][0] <= e["source_time_s"] <= beat["range"][1]]
    for e in context:
        e["encoded_display_spans_s"]=spans([r["output_frame"] for r in mapping
                                           if r["moment"]==e["moment"]
                                           and abs(r["source_position"]/SOURCE_FPS-e["source_time_s"])<.05])
    for e in evidence:
        e["encoded_cue_spans_s"]=spans([r["output_frame"] for r in mapping
                                       if r["moment"]==e["moment"] and r["skeleton_drawn"]
                                       and abs(r["source_position"]/SOURCE_FPS-e["pose_time_s"])<.16])
    limits=[
        "人工选择的原始集锦选段；不是整场、不是已通过验收的自动分析。",
        "V3 确认候选在父评估中为 32/35 精度、32/96 召回；80% 召回门槛未通过。",
        "没有跟踪球路、测量落点或读取球员意图；2D 姿态不证明拍球接触或精确击球类型。",
        "低置信或遮挡关节隐藏；不将腾空脚踝投影当作精确场地位置。",
        "本地 Kokoro 神经中文语音，没有人类听音质量结论；最终独立证据/视频验收待定。",
        "保留原始水印。素材标注 CC BY 3.0 不等于上游转播权已获公开再分发许可。",
    ]
    coach_analysis=script.get("coach_analysis")
    terminology=script.get("terminology",[])
    usage_references=script.get("usage_references",[])
    if coach_analysis:
        limits.extend(coach_analysis["global_limitations"])
    else:
        limits.insert(3,"回网、挑后场是条件性选择，不是事先预测成绩；后续动作是回看观察。")
    if coach_analysis:
        claim_review={
            "review_kind":"source-picture inspection by rendering agent; tactical readings are conditional, not known intent",
            "court_position_method":"visible feet relative to floor markings over the sequence; not hip height or the raised net tape",
            "observed":[{"id":f"O{i+1}","claim_zh":beat["observed"],"evidence_ids":beat["evidence_ids"]}
                        for i,beat in enumerate(coach_analysis["beats"])],
            "conditional":[{"id":f"I{i+1}","claim_zh":beat["tactical_read"],"evidence_ids":beat["evidence_ids"]}
                           for i,beat in enumerate(coach_analysis["beats"])],
            "coach_cues":[{"id":beat["id"],"cue_zh":beat["technique_cue"],"prepare_for_zh":beat["prepare_for"],
                           "evidence_ids":beat["evidence_ids"],"limitations":beat["limitations"]}
                          for beat in coach_analysis["beats"]],
            "omitted_claims":["known player intention","racket-shuttle contact","exact shot family",
                              "shuttle trajectory or landing","measured time gain","point winner"],
            "caption_review":"Video captions are one short spoken chunk at a time; the written report separates observations from conditional coaching advice.",
            "voice_review":"No human listening or independent voice naturalness acceptance is claimed.",
        }
    else:
        claim_review={
            "review_kind":"source-picture inspection by rendering agent, not independent final acceptance",
            "court_position_method":"visible feet relative to floor markings over the sequence; not hip height or the raised net tape",
            "observed":[
                {"id":"O1","claim_zh":"红衣移动到画面右后场，随后向画面左前方赶球。","evidence_ids":["E003639","E003723"]},
                {"id":"O2","claim_zh":"红衣的低位救球后，双方继续来回。","evidence_ids":["E003723","C003756","C003798","C003840","C003876"]},
                {"id":"O3","claim_zh":"红衣低位之后起身；灰衣下一次后场头顶挥拍前，红衣已站起来。","evidence_ids":["E007699","C007734","E007801"]},
                {"id":"O4","claim_zh":"灰衣继续挥拍，红衣还要处理后续来球。","evidence_ids":["E007801","C007836","E007919","C007981"]},
                {"id":"O5","claim_zh":"最后选段近端灰衣低位回球，远端红衣随后前移；不把前移等同于已到网前。","evidence_ids":["E016442","C016476","E016500"]},
                {"id":"O6","claim_zh":"灰衣随后退后举拍，红衣继续回应；未判定精确击球类型或得分归属。","evidence_ids":["E016578","C016608","C016644","C016710"]}
            ],
            "conditional":[
                {"id":"I1","claim_zh":"若对手仍在后场，可以考虑前场空当；这是战术选择，不是测得的球路或球员意图。","evidence_ids":["E003639","E003723"],"rule":"rear_then_front"},
                {"id":"I2","claim_zh":"若自己还没站稳，又要面对对手后续进攻，回球应兼顾恢复；不声称某拍量化争取了多少时间。","evidence_ids":["E007699","E007801","E007919"],"rule":"buy_recovery_time"},
                {"id":"I3","claim_zh":"对手前移低位处理球时，可同时准备短回球和挑后场；不声称只能或必然挑球。","evidence_ids":["E016442","E016500","E016578"],"rule":"net_reply_options"}
            ],
            "omitted_claims":["two measured shuttle landing placements","universal recover instead of counterattack rule",
                              "measured time gained","exact block/lift/downward shot families",
                              "intention or point winner","arrival at the net inferred from tape/hip image height"],
            "caption_review":"Decoded contact sheet inspected: Chinese captions are readable, in the margin, with both players/court/watermarks preserved.",
            "voice_review":"No human listening or independent voice naturalness acceptance is claimed.",
        }
    write(OUT/"claim-review.json",claim_review)
    analysis={"title_zh":script["title_zh"],"mode":script["mode"],"source":{"path":str(SOURCE),"sha256":sha(SOURCE)},
              "duration_s":plan["duration_s"],"evidence":evidence,"context_evidence":context,"beats":plan["beats"],
              "claim_checks":script["claim_checks"],"limitations":limits,
              "format":{"canvas_px":[WIDTH,HEIGHT],"picture_box":list(PICTURE_BOX),
                        "caption_style":"separate speech-synchronized phrase chunks; no persistent report cards",
                        "emphasis_frames":[{"output_frame":row["output_frame"],
                                            "source_time_s":row["source_position"]/SOURCE_FPS,
                                            "zoom":row["emphasis_zoom"],
                                            "crop_xyxy":row["picture_crop_xyxy"]}
                                           for row in mapping if row["emphasis_zoom"]>1.001]},
              "attribution":{"credit":"Badminton Sphere / Wikimedia Commons, China Open 2025 women's singles final, CC BY 3.0",
                             "upstream_broadcast_rights":"not independently cleared"},
              "codebook":{"path":str(TACTICS/"codebook_verified.json"),"sha256":sha(TACTICS/"codebook_verified.json"),
                          "use":"conditional coaching rationale; not a new literature review or prediction evaluation"},
              "observed_claims":claim_review["observed"],"conditional_claims":claim_review["conditional"],
              "coach_analysis":coach_analysis,
              "terminology":terminology,"usage_references":usage_references,
              "independent_acceptance":"pending","engine_acceptance":"failed_recall_gate_unchanged"}
    write(OUT/"analysis.json",analysis)
    write(OUT/"attribution.json",analysis["attribution"])
    (OUT/"narration.txt").write_text("\n".join(
        f"{beat['id']} [{chunk['display_span_s'][0]:.3f}–{chunk['display_span_s'][1]:.3f}s] {chunk['text']}"
        for beat in plan["beats"] for chunk in (beat.get("caption_spans") or [{
            "text":beat["text"],"display_span_s":beat["speech_span_s"],
        }]))+"\n",encoding="utf-8")
    def srt(t):
        ms=round(t*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    subtitle_rows=[]
    for beat in plan["beats"]:
        chunks=beat.get("caption_spans") or [{
            "text":beat["text"],"display_span_s":beat["speech_span_s"],
        }]
        subtitle_rows.extend(chunks)
    (OUT/"narration.srt").write_text("\n\n".join(
        f"{i}\n{srt(chunk['display_span_s'][0])} --> {srt(chunk['display_span_s'][1])}\n{chunk['text']}"
        for i,chunk in enumerate(subtitle_rows,1))+"\n",encoding="utf-8")
    (OUT/"narration.vtt").write_text("WEBVTT\n\n"+"\n\n".join(
        f"{srt(chunk['display_span_s'][0]).replace(',', '.')} --> {srt(chunk['display_span_s'][1]).replace(',', '.')}\n{chunk['text']}"
        for chunk in subtitle_rows)+"\n",encoding="utf-8")
    labels={"observed":"画面观察","conditional":"条件性解读","observed_then_conditional":"观察＋条件建议",
            "coach_read":"战术读法＋教练要点","coach_summary":"教练总结",
            "conditional_then_observed":"条件建议＋观察","conditional_alternatives":"两种可能回球",
            "conditional_alternatives_then_observed":"可能回球＋实际后续",
            "conditional_summary":"条件性总结"}
    coach_cards=[]
    if coach_analysis:
        coach_by_id={item["id"]:item for item in coach_analysis["beats"]}
        for beat in plan["beats"]:
            item=coach_by_id[beat["id"]]
            coach_cards.append(
                f"<article><h3>{html.escape(beat['id'])} · 画面观察</h3>"
                f"<p>{html.escape(item['observed'])}</p><h3>条件性战术解读（非已知意图）</h3>"
                f"<p>{html.escape(item['tactical_read'])}</p><h3>教练要点</h3>"
                f"<p>{html.escape(item['technique_cue'])}</p><h3>下一步准备</h3>"
                f"<p>{html.escape(item['prepare_for'])}</p><p>证据：{', '.join(html.escape(x) for x in item['evidence_ids'])}"
                f" · 限制：{html.escape('；'.join(item['limitations']))}</p></article>")
    cards=[]
    for moment in script["moments"]:
        bs=[b for b in plan["beats"] if b["moment"]==moment["id"]]
        items="".join(f"<tr><td>{b['speech_span_s'][0]:.2f}–{b['speech_span_s'][1]:.2f}s</td>"
                      f"<td>{html.escape(labels[b['type']])}</td><td>{html.escape(b['text'])}</td><td>"
                      +", ".join(html.escape(x) for x in b["evidence_ids"])+"</td></tr>" for b in bs)
        cards.append(f"<section><h2>{moment['title']}</h2><p>原片 {moment['range'][0]}–{moment['range'][1]}s · 一次向前播放，无动作重放。</p>"
                     f"<table><tr><th>成片语音</th><th>性质</th><th>讲解</th><th>证据</th></tr>{items}</table></section>")
    terminology_rows="".join(
        "<tr><td>"+html.escape(t["previous"])+"</td><td>"+html.escape(t["current"])+"</td><td>"+
        html.escape(t.get("boundary",t.get("note","")))+"</td></tr>" for t in terminology)
    reference_items="".join(
        '<li><a href="'+html.escape(r["url"],quote=True)+'">'+html.escape(r["title"])+
        "</a>："+html.escape(r["evidence"])+"</li>" for r in usage_references)
    terminology_html=(
        '<section><h2>术语说明</h2><table><tr><th>此前说法</th><th>本版常用说法</th><th>边界</th></tr>'+
        terminology_rows+'</table><h3>术语参考</h3><ul>'+reference_items+'</ul></section>'
        if terminology else "")
    page=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{script['title_zh']}</title>
<style>body{{margin:36px auto;max-width:1180px;padding:0 24px;background:#0d1620;color:#eaf1f7;font:18px/1.7 "Microsoft YaHei",sans-serif}}h1{{font-size:36px}}h2{{color:#7dddD6}}video{{width:100%;max-height:76vh}}section{{margin:32px 0}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #344252}}.notice{{padding:16px;background:#382b18;border-left:4px solid #efb15b}}a{{color:#7dddd6}}</style>
<h1>{script['title_zh']}</h1><p>{html.escape(script.get('display_label','先看对手的位置和状态，再考虑选择；看完实际回球，再判断。'))}</p>
<p class="notice">这是有来源证据的人工选段讲解。V3 自动识别的召回验收未通过，不能用这条视频替代模型验收。最终独立视频验收仍待定。</p>
<video controls preload="metadata" src="editorial.mp4"><track kind="subtitles" srclang="zh" label="中文" src="narration.vtt"></video>
<p><a download href="editorial.mp4">下载 MP4</a> · <a href="analysis.json">完整证据与时间绑定</a> · <a href="narration.txt">语音文本</a> · <a href="verification.json">技术与对齐检查</a></p>
{''.join(cards)}<section><h2>证据怎样对应画面</h2><p>细线关节来自真实缓存，只在对应状态附近短暂出现；其时间与显示的原片位置相差不超过一个 60 fps 原片帧。画面不裁掉另一位球员、球场或水印。完整逐帧绑定见 frame-map.json。</p>
<p>画面观察与条件性战术建议分开呈现，不能当作测到的落点、意图或因果关系。三个片段按原顺序连续播放，不重复已发生的动作。</p></section>
{('<section><h2>逐段教练分析</h2>'+''.join(coach_cards)+'</section>') if coach_analysis else ''}
{terminology_html}
<section><h2>限制与署名</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in limits)}</ul><p>{analysis['attribution']['credit']}</p></section></html>"""
    (OUT/"analysis.html").write_text(page,encoding="utf-8")
    initial=read(OUT/"inputs.json")
    final_inputs={path:sha(Path(path)) for path in initial["inputs"]}
    write(OUT/"inputs-final.json",{
        "initial_inputs_note":"Input hashes captured at preparation and rechecked for this output edition.",
        "inputs":final_inputs,"source_sha256":sha(SOURCE),
        "script_sha256":sha(OUT/"script.json"),"planned_schedule_sha256":sha(OUT/"planned-schedule.json"),
        "helper_sha256":sha(Path(__file__).resolve()),
        "changes_since_initial_prepare":[path for path,digest in initial["inputs"].items() if digest!=final_inputs[path]],
        "evidence_manifests":initial["evidence_manifests"],"voice_model":initial["voice_model"],
    })


def verify():
    started=time.monotonic()
    assert_frozen()
    plan=read(OUT/"planned-schedule.json"); mapping=read(OUT/"frame-map.json")["frames"]
    script=read(OUT/"script.json"); evidence=read(OUT/"evidence.json")
    validate_reviewed_script(script)
    assert len(mapping)==plan["frame_count"]
    assert [r["source_position"] for r in mapping]==[r["source_position"] for r in plan["frames"]]
    assert all(b["source_position"]>=a["source_position"] for a,b in zip(mapping,mapping[1:]))
    assert max(r["pose_sync_error_s"] for r in mapping)<=1/60+.0002
    voice=read(OUT/"voice.json")
    for beat in plan["beats"]:
        assert beat["speech_span_s"][1]<=beat["display_span_s"][1]
        assert beat["text"]==voice[beat["id"]]["text"]
        assert sha(Path(voice[beat["id"]]["path"]))==voice[beat["id"]]["sha256"]
    decoded_audio=HERE/"decoded-final-audio.wav"
    command([FF,"-hide_banner","-loglevel","error","-nostdin","-i",OUT/"editorial.mp4",
             "-vn","-ac","1","-ar","24000","-c:a","pcm_s16le","-y",decoded_audio])
    audio,sr=wav_read(decoded_audio)
    speech_checks=[]
    for beat in plan["beats"]:
        samples,rate=wav_read(OUT/"voice"/f"{beat['id']}.wav")
        if rate!=sr:
            samples=np.interp(np.arange(round(len(samples)*sr/rate))*rate/sr,np.arange(len(samples)),samples)
        first=round(beat["speech_span_s"][0]*sr)
        actual=audio[first:first+len(samples)]
        if len(actual)!=len(samples):
            raise RuntimeError(f"Truncated speech: {beat['id']}")
        corr=float(np.corrcoef(samples,actual)[0,1])
        if not math.isfinite(corr) or corr<.97:
            raise RuntimeError(f"Encoded speech mismatch: {beat['id']} correlation={corr}")
        speech_checks.append({"beat":beat["id"],"voice_correlation":corr,
                              "rms":float(np.sqrt(np.mean(actual**2))),"speech_span_s":beat["speech_span_s"]})
    chunk_speech_checks=[]
    for beat in plan["beats"]:
        chunks=voice[beat["id"]].get("caption_chunks",[])
        scheduled=beat.get("caption_spans",[])
        if VERTICAL and len(chunks)!=len(scheduled):
            raise RuntimeError(f"Caption/audio chunk count differs for {beat['id']}")
        for chunk,span in zip(chunks,scheduled):
            if chunk["text"]!=span["text"] or span["text"] not in beat["text"]:
                raise RuntimeError(f"Caption text does not match the narration: {beat['id']}")
            samples,rate=wav_read(chunk["path"])
            if sha(Path(chunk["path"]))!=chunk["sha256"]:
                raise RuntimeError(f"Caption voice chunk hash mismatch: {chunk['path']}")
            if rate!=sr:
                samples=np.interp(np.arange(round(len(samples)*sr/rate))*rate/sr,np.arange(len(samples)),samples)
            first=round((beat["speech_span_s"][0]+chunk["relative_span_s"][0])*sr)
            actual=audio[first:first+len(samples)]
            corr=float(np.corrcoef(samples,actual)[0,1]) if len(actual)==len(samples) else float("nan")
            if not math.isfinite(corr) or corr<.97:
                raise RuntimeError(f"Caption is not synchronized to its spoken phrase: {beat['id']} / {span['text']}")
            chunk_speech_checks.append({"beat":beat["id"],"text":span["text"],
                                        "display_span_s":span["display_span_s"],
                                        "voice_correlation":corr})
    cap=cv2.VideoCapture(str(OUT/"editorial.mp4"))
    current=None; checks=[]; control=[]; sheets=[]; decoded=0; phone_preview=None
    preview_span=next((c for b in plan["beats"] for c in b.get("caption_spans",[])
                       if b["id"]=="space" and c["text"]=="动作小、手上轻。"),None)
    preview_index=(preview_span["first_frame"]+preview_span["end_frame_exclusive"])//2 if preview_span else plan["frame_count"]//2
    for row in mapping:
        ok,frame=cap.read()
        if not ok:
            raise RuntimeError(f"Video stops at frame {decoded}")
        decoded+=1
        if row["moment"]!=current:
            current=row["moment"]
            moment=next(m for m in script["moments"] if m["id"]==current)
            frames,start,end=load_frames(moment)
            knots,_=knots_for(frames,start)
        expected_pair=picture_pair(row["source_position"],knots)
        if list(expected_pair[:2])!=row["picture_frames"] or abs(expected_pair[2]-row["picture_weight"])>1e-8:
            raise RuntimeError(f"Planned source position and stored picture pair disagree: {row['output_frame']}")
        beat=next(b for b in plan["beats"] if b["id"]==row["beat"])
        expected_caption=next((c["text"] for c in beat.get("caption_spans",[])
                               if c["first_frame"]<=row["output_frame"]<c["end_frame_exclusive"]),None)
        if VERTICAL and row.get("caption_text")!=expected_caption:
            raise RuntimeError(f"Rendered caption timeline mismatch at frame {row['output_frame']}")
        if row["output_frame"]==preview_index:
            phone_preview=cv2.resize(frame,(360,640),interpolation=cv2.INTER_AREA)
        if row["output_frame"]%15!=0 and row["kind"]!="hold":
            continue
        expected=picture(frames,start,(*row["picture_frames"],row["picture_weight"]))
        left,top,picture_width,picture_height=PICTURE_BOX
        actual=frame[top:top+picture_height,left:left+picture_width]
        crop_x0,crop_y0,crop_x1,crop_y1=row["picture_crop_xyxy"]
        expected=expected[crop_y0:crop_y1,crop_x0:crop_x1]
        expected=cv2.resize(expected,(picture_width,picture_height),interpolation=cv2.INTER_AREA)
        # Thin pose strokes occupy little area. Quantile trimming excludes them without
        # using a player box to hide large portions of the actual picture.
        error=np.abs(actual.astype(np.int16)-expected.astype(np.int16)).mean(axis=2)
        trimmed=float(np.mean(error[error<=np.percentile(error,98)]))
        if trimmed>4.0:
            raise RuntimeError(f"Decoded source binding failed: frame {row['output_frame']} MAE {trimmed}")
        wrong_index=min(end-1,max(start,round(row["source_position"])+60))-start
        wrong_frame=frames[wrong_index][crop_y0:crop_y1,crop_x0:crop_x1]
        wrong=cv2.resize(wrong_frame,(picture_width,picture_height),interpolation=cv2.INTER_AREA)
        ctrl=float(np.abs(actual.astype(np.int16)-wrong.astype(np.int16)).mean())
        checks.append({"output_frame":row["output_frame"],"trimmed_picture_mae":trimmed,"wrong_plus_1s_mae":ctrl})
        control.append(ctrl)
        if row["output_frame"]%150==0:
            sample_size=(360,640) if VERTICAL else (640,480)
            small=cv2.resize(frame,sample_size,interpolation=cv2.INTER_AREA)
            sheets.append(small)
    more,_=cap.read();cap.release()
    if more:
        raise RuntimeError("Encoded media has extra unplanned frames")
    if VERTICAL:
        if phone_preview is None:
            raise RuntimeError("Could not decode the requested phone-size caption preview")
        cv2.imwrite(str(OUT/"phone-preview.jpg"),phone_preview,[cv2.IMWRITE_JPEG_QUALITY,94])
    if not len(checks) or np.median(control)<=np.median([x["trimmed_picture_mae"] for x in checks])+1:
        raise RuntimeError("Wrong-source control not discriminative")
    if sheets:
        cols=3; thumb_width,thumb_height=sheets[0].shape[1],sheets[0].shape[0]
        rows=math.ceil(len(sheets)/cols)
        sheet=np.zeros((rows*thumb_height,cols*thumb_width,3),dtype=np.uint8)
        for i,image in enumerate(sheets):
            sheet[(i//cols)*thumb_height:(i//cols+1)*thumb_height,
                  (i%cols)*thumb_width:(i%cols+1)*thumb_width]=image
        cv2.imwrite(str(OUT/"decoded-contact-sheet.jpg"),sheet)
    audit=command([sys.executable,"-B",ROOT/"obj2"/"tools"/"audit_narrated_media.py",OUT/"editorial.mp4",
                   "--ffmpeg",FF,"--expected-duration",plan["duration_s"]],OUT/"independent-media-audit.log")
    media=json.loads(audit)
    if not media["technical_media_pass"]:
        raise RuntimeError("Independent media check failed")
    write(OUT/"media-audit.json",media)
    result={"status":"passed","artifact_sha256":sha(OUT/"editorial.mp4"),"decoded_frames":decoded,
            "expected_frames":plan["frame_count"],"planned_duration_s":plan["duration_s"],
            "speech_window_checks":speech_checks,"source_picture_checks":checks,
            "caption_voice_checks":chunk_speech_checks,
            "source_position_map_matches_plan":True,"monotonic_source_no_replay":True,
            "picture_recipe_checked_frames":len(mapping),"decoded_pixel_samples":len(checks),
            "pose_max_sync_error_s":max(r["pose_sync_error_s"] for r in mapping),
            "frozen_v3_unchanged":True,"independent_media_pass":True,
            "voice_human_listening_review":False,"independent_final_acceptance":"pending",
            "engine_acceptance":"failed_recall_gate_unchanged",
            "limits":"Pixel binding, source audio and synthetic speech checks are not a human coaching/voice verdict."}
    if VERTICAL:
        result["caption_chunks_match_voice_and_display"]=True
        result["phone_preview"]=str(OUT/"phone-preview.jpg")
    result["verification_elapsed_s"]=round(time.monotonic()-started,3)
    write(OUT/"verification.json",result)
    documents()
    run=read(OUT/"run.json")
    run["helper_sha256_final"]=sha(Path(__file__).resolve())
    run["source_script_sha256_final"]=sha(OUT/"script.json")
    run["planned_schedule_sha256_final"]=sha(OUT/"planned-schedule.json")
    write(OUT/"run.json",run)
    handoff={"output_directory":str(OUT),"video":str(OUT/"editorial.mp4"),"video_sha256":sha(OUT/"editorial.mp4"),
             "planned_duration_s":plan["duration_s"],"verification":result|{"source_picture_checks_count":len(checks)},
             "files":[{"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(OUT.rglob("*")) if p.is_file() and p.name!="handoff.json"]}
    write(OUT/"handoff.json",handoff)
    print("VERIFIED",json.dumps({k:v for k,v in result.items() if k!="source_picture_checks"},ensure_ascii=True),flush=True)


def selftest():
    frames=[np.zeros((2,2,3),np.uint8),np.zeros((2,2,3),np.uint8),np.full((2,2,3),100,np.uint8)]
    assert picture_pair(1,[0,2])==(1,1,0.0)
    assert picture_pair(.5,[0,2])==(0,2,.25)
    assert np.all(picture(frames,0,picture_pair(.5,[0,2]))==25)
    assert spans([0,1,3])==[[0,2/30],[3/30,4/30]]
    fake={"beats":[{"id":"a","moment":"x","range":[1,2],"hold":0.0}]}
    p=make_schedule(fake,{"a":{"duration_s":1.0,"sha256":"test"}})
    assert p["frames"][0]["source_position"]==60
    assert p["beats"][0]["speech_span_s"][1]<p["duration_s"]
    if OUT.exists():
        write(OUT/"tests.json",{
            "status":"passed","interpreter":sys.executable,
            "command":f'python -B "{Path(__file__).resolve()}" selftest',
            "assertions_passed":6,"assertions_failed":0,
            "cases":["integer source position remains raw","unique-knot interpolation",
                     "expected blended pixels","contiguous display spans",
                     "source start position","speech fits the planned interval"],
            "application_tests":"Not rerun: application source is frozen and unchanged; this task changes output-only helpers.",
        })
    print("SELFTEST passed: raw integer binding, unique blend, display spans, monotonic plan, speech fit")


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["prepare","revise","inspect","encode","verify","selftest"])
    args=parser.parse_args()
    start=time.monotonic()
    {"prepare":prepare,"revise":revise,"inspect":inspect,"encode":encode,"verify":verify,"selftest":selftest}[args.stage]()
    print("STAGE",args.stage,"elapsed_s",round(time.monotonic()-start,3),flush=True)
