import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import wave

import numpy as np
import psutil
import sherpa_onnx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TTS = ROOT / "obj2" / "work" / "NATURAL" / "voice" / "models" / "kokoro-multi-lang-v1_1"
ASR = HERE.parent / "models" / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
FF = ROOT / "tools" / "ffmpeg.exe"
VOICES = {0: "af_maple", 1: "af_sol", 2: "bf_vale"}


def spoken(text):
    return text


def engine():
    return sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(TTS / "model.onnx"), voices=str(TTS / "voices.bin"), tokens=str(TTS / "tokens.txt"),
                lexicon=f"{TTS / 'lexicon-us-en.txt'},{TTS / 'lexicon-zh.txt'}",
                data_dir=str(TTS / "espeak-ng-data"), dict_dir=str(TTS / "dict"),
            ),
            num_threads=2,
        ),
    ))


def recognizer():
    return sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(ASR / "model.int8.onnx"), tokens=str(ASR / "tokens.txt"),
        num_threads=2, provider="cpu", language="en", use_itn=True,
    )


def write(path, samples, rate):
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        handle.writeframes(np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def transcribe(asr, path):
    pcm = subprocess.run([str(FF), "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "pipe:1"], capture_output=True, check=True, timeout=60).stdout
    stream = asr.create_stream()
    stream.accept_waveform(16000, np.frombuffer(pcm, dtype="<f4"))
    asr.decode_stream(stream)
    return stream.result.text


def main():
    out = HERE / "voice-probe"
    out.mkdir(exist_ok=True)
    if psutil.virtual_memory().available < 3 * 1024**3:
        raise RuntimeError("Not enough free memory")
    lease = ROOT / "obj2" / "work" / "cpu.lock"
    marker = {"owner": "english-cut-voice-probe", "pid": os.getpid(), "id": str(uuid.uuid4())}
    with lease.open("x") as handle:
        json.dump(marker, handle)
    try:
        script = json.loads((HERE / "script.json").read_text(encoding="utf-8"))
        tts, asr = engine(), recognizer()
        rows = []
        for sid, name in VOICES.items():
            for beat in script["beats"]:
                audio = tts.generate(spoken(beat["caption"]), sid=sid, speed=1.0)
                path = out / f"{name}-{beat['id']}.wav"
                write(path, np.asarray(audio.samples, dtype=np.float32), audio.sample_rate)
                row = {"voice": name, "sid": sid, "beat": beat["id"],
                       "seconds": round(len(audio.samples) / audio.sample_rate, 3),
                       "asr": transcribe(asr, path)}
                rows.append(row)
                print(json.dumps(row), flush=True)
        (out / "probe.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    finally:
        if json.loads(lease.read_text()) != marker:
            raise RuntimeError("CPU lease ownership changed")
        lease.unlink()


if __name__ == "__main__":
    main()
