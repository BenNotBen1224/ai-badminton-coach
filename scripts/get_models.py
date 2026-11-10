"""Download or rebuild the models. None of them are committed to the repo.

    python scripts/get_models.py                # the app: YOLO26 detectors + MeloTTS voice
    python scripts/get_models.py --studio       # also Kokoro, SenseVoice and TrackNetV3 for studio/
    python scripts/get_models.py yolo melo      # only the named groups

Groups: yolo, melo, kokoro, sensevoice, tracknet. A group is skipped when its file is already there, so the
script can be re-run after a failure. Downloads are checked against the SHA-256 recorded when the project was
built. Licences are listed in docs/models-and-licences.md.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
WEIGHTS = ROOT / "fast" / "weights"
SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
DRIVE = "https://drive.usercontent.google.com/download?id=1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA&export=download&confirm=t"
REPRO = ("https://github.com/ychenfen/badminton-pipeline-repro", "1666675c7fcca8ed1c1428ce36232e9a3c8aa0e1")

# name, export size (h, w), source .pt sha256, sha256 and export date of the ONNX the recorded runs used
DETECTORS = [
    ("yolo26n", (384, 640), "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
     "11b9c86466d31b704f2cde8dc994172d53ba510687d22fceb2b11180e95f13a4", "2026-10-26T20:51:12.597570"),
    ("yolo26s", (448, 768), "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b",
     "65c41b422fd432070886cbff8a8f4e6b42da678eb2f827b9ecc766a42a6b2aea", "2026-10-26T20:51:31.304235"),
]
# group -> (archive url, extract into, checked file, its sha256)
ARCHIVES = {
    "melo": (f"{SHERPA}/tts-models/vits-melo-tts-zh_en.tar.bz2", ROOT / "coach" / "models",
             "vits-melo-tts-zh_en/model.onnx", "bf30582eb1b012250a35b1a4a80e7dfbcf8485e7bb9de0d95efbbeef0e4ad86d"),
    "kokoro": (f"{SHERPA}/tts-models/kokoro-multi-lang-v1_1.tar.bz2", MODELS,
               "kokoro-multi-lang-v1_1/model.onnx", "99db0bc8fd0c5033d249fe922db16768b71b4a6b26cfd93d206fbf66c76efb5f"),
    "sensevoice": (f"{SHERPA}/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2", MODELS,
                   "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx",
                   "c71f0ce00bec95b07744e116345e33d8cbbe08cef896382cf907bf4b51a2cd51"),
    "tracknet": (DRIVE, MODELS / "TrackNetV3", "ckpts/TrackNet_best.pt",
                 "df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a"),
}
APP = ["yolo", "melo"]
STUDIO = ["kokoro", "sensevoice", "tracknet"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ai-badminton-coach/1.0"})
    print(f"  downloading {url}", flush=True)
    with urllib.request.urlopen(request) as response, part.open("wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    part.replace(dest)


def archive(group):
    url, into, checked, expected = ARCHIVES[group]
    target = into / checked
    if target.is_file():
        print(f"{group}: have {target.relative_to(ROOT)}")
        return
    download = into / ("download.zip" if url == DRIVE else Path(url).name)
    fetch(url, download)
    if download.suffix == ".zip":
        with zipfile.ZipFile(download) as z:
            z.extractall(into)
    else:
        with tarfile.open(download) as t:
            t.extractall(into, filter="data")
    download.unlink()
    got = sha256(target)
    if got != expected:
        sys.exit(f"{group}: {target.relative_to(ROOT)} sha256 {got}, expected {expected}")
    print(f"{group}: saved {target.relative_to(ROOT)}")


def tracknet_code():
    repo = MODELS / "badminton-pipeline-repro"
    if (repo / "scripts" / "tracknet_runtime" / "predict.py").is_file():
        print(f"tracknet: have {repo.relative_to(ROOT)}")
        return
    url, commit = REPRO
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")  # its own TrackNet_best.pt is not used
    subprocess.run(["git", "clone", "--quiet", url, str(repo)], check=True, env=env)
    subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", commit], check=True, env=env)
    print(f"tracknet: cloned {url} at {commit[:12]}")


def yolo():
    import onnx
    from ultralytics import YOLO  # only needed here; see requirements.txt

    work = MODELS / "yolo26"
    work.mkdir(parents=True, exist_ok=True)
    for name, size, pt_sha, onnx_sha, date in DETECTORS:
        out = WEIGHTS / f"{name}-{size[0]}x{size[1]}.onnx"
        if out.is_file():
            print(f"yolo: have {out.relative_to(ROOT)}")
            continue
        pt = work / f"{name}.pt"
        model = YOLO(str(pt))
        if sha256(pt) != pt_sha:
            sys.exit(f"yolo: {pt.relative_to(ROOT)} is not the checkpoint the project used")
        exported = Path(model.export(format="onnx", imgsz=size, batch=1, half=False, int8=False, dynamic=False,
                                     simplify=True, opset=17, nms=False))
        graph = onnx.load(str(exported))
        for prop in graph.metadata_props:
            if prop.key == "date":  # the export time is the only difference from the recorded file
                prop.value = date
        onnx.save(graph, str(out))
        exported.unlink()
        if sha256(out) != onnx_sha:
            print(f"yolo: WARNING {out.relative_to(ROOT)} differs from the recorded export; "
                  "check the ultralytics, torch and onnx versions in requirements.txt")
        else:
            print(f"yolo: exported {out.relative_to(ROOT)}, same bytes as the recorded export")


def main(argv):
    if {"-h", "--help"} & set(argv):
        sys.exit(__doc__)
    groups = [a for a in argv if not a.startswith("--")] or APP + (STUDIO if "--studio" in argv else [])
    unknown = set(groups) - set(APP + STUDIO)
    if unknown:
        sys.exit(f"unknown group: {', '.join(sorted(unknown))}")
    for group in groups:
        if group == "yolo":
            yolo()
        else:
            if group == "tracknet":
                tracknet_code()
            archive(group)


if __name__ == "__main__":
    main(sys.argv[1:])
