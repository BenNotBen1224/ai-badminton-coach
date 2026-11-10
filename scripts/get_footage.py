"""Download the China Open 2025 women's singles final from Wikimedia Commons and recut the sample clips.

    python scripts/get_footage.py            # download + cut clips 04-10
    python scripts/get_footage.py --no-cut   # download only

Source: "Wang Zhiyi triumph in all-Chinese China Open finals I China Open 2025", Badminton Sphere,
Wikimedia Commons, CC BY 3.0. Upstream broadcast rights are not independently cleared.

The recut clips use the same start times and settings as the originals but are not byte-identical,
so numbers from them can differ slightly from the recorded runs. Clips 01-03 and 11 came from other
sources; see sample-videos/*attribution*.json.
"""

import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "Wang_Zhiyi_triumph_in_all-Chinese_China_Open_finals_I_China_Open_2025.webm"
URL = f"https://commons.wikimedia.org/wiki/Special:FilePath/{NAME}"
SHA256 = "2500471d4852a4a73e0f4eb25ad28171b3c54967f58e1cc7620af16b46ccbb52"
WEBM = ROOT / "footage" / "china-open-2025-ws-final.webm"
CLIPS = {  # file -> start second in the source
    "04-china-open-ws-a.mp4": 20,
    "06-china-open-ws-c.mp4": 70,
    "05-china-open-ws-b.mp4": 110,
    "07-china-open-ws-d.mp4": 180,
    "08-china-open-ws-f.mp4": 250,
    "10-china-open-ws-h.mp4": 320,
    "09-china-open-ws-g.mp4": 365,
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download():
    if WEBM.is_file() and sha256(WEBM) == SHA256:
        print(f"have {WEBM.relative_to(ROOT)}")
        return
    WEBM.parent.mkdir(exist_ok=True)
    part = WEBM.with_suffix(".part")
    request = urllib.request.Request(URL, headers={"User-Agent": "ai-badminton-coach/1.0 (github.com/BenNotBen1224)"})
    print(f"downloading {URL}")
    with urllib.request.urlopen(request) as response, part.open("wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    got = sha256(part)
    if got != SHA256:
        part.unlink()
        sys.exit(f"sha256 mismatch: expected {SHA256}, got {got}")
    part.replace(WEBM)
    print(f"saved {WEBM.relative_to(ROOT)}")


def ffmpeg():
    for candidate in (ROOT / "tools" / "ffmpeg.exe", ROOT / "tools" / "ffmpeg"):
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("ffmpeg")
    if not found:
        sys.exit("ffmpeg not found: put it in tools/ or on PATH")
    return found


def cut():
    exe = ffmpeg()
    out_dir = ROOT / "sample-videos"
    for name, start in CLIPS.items():
        out = out_dir / name
        if out.is_file():
            print(f"have sample-videos/{name}")
            continue
        subprocess.run([exe, "-v", "error", "-y", "-ss", str(start), "-i", str(WEBM), "-t", "30", "-an",
                        "-vf", "fps=30", "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
                        "-crf", "18", "-frames:v", "900", "-movflags", "+faststart", str(out)], check=True)
        print(f"cut sample-videos/{name}")


if __name__ == "__main__":
    if {"-h", "--help"} & set(sys.argv[1:]):
        sys.exit(__doc__)
    download()
    if "--no-cut" not in sys.argv:
        cut()
