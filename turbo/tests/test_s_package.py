"""Import turbo.service from PROJECT and load_tracks each real tier output.

Run from any directory. The entry script's cwd is PROJECT, so a bare
`import decode` would fail. This does not change a scoring threshold.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
PY = Path(sys.executable)
FFMPEG = PROJECT / "tools" / "ffmpeg.exe"
CLIP = PROJECT / "sample-videos" / "03-womens-singles-rally.mp4"
SHORT = PROJECT / "work" / "S" / "clip03-1s.mp4"

ENTRY = r"""
import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(os.environ["PROJECT"])
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

import turbo.service
from coach.tracks import load_tracks

assert turbo.service.decode.__name__ == "turbo.decode", turbo.service.decode.__name__

weights = PROJECT / "fast" / "weights"
video = os.environ["SHORT"]
lock = PROJECT / "work" / "cpu.lock"
deadline = time.time() + 30 * 60
while True:
    try:
        handle = open(lock, "x", encoding="utf-8")
    except FileExistsError:
        if time.time() > deadline:
            raise SystemExit("cpu.lock still held")
        time.sleep(15)
        continue
    handle.write(f"S {os.getpid()} package-entry\n")
    handle.close()
    break
try:
    svc = turbo.service.TurboService(weights, decode_method="serial")
    assert callable(svc._refined), type(svc._refined)
    svc.warm_up()
    assert callable(svc._refined), type(svc._refined)
    preview, _ = svc.analyze(video, tier="preview")
    refined, _ = svc.analyze(video, tier="refined")
    base = json.loads(json.dumps(preview))
    base["windows_refined"] = [{"start": 20, "end": 24, "s": 0.0}]
    before = json.dumps(base)
    window, _ = svc.analyze(video, tier="window", windows=[(0, 7)], base_doc=base)
    if json.dumps(base) != before:
        raise SystemExit("base_doc was mutated")
finally:
    text = lock.read_text(encoding="utf-8")
    if text.startswith(f"S {os.getpid()} "):
        lock.unlink()

for name, doc in (("preview", preview), ("refined", refined), ("window", window)):
    tracks = load_tracks(doc)
    print(name, tracks.tier.value, len(doc["frames"]), flush=True)

if preview["tier"] != "preview":
    raise SystemExit(f"preview tier {preview['tier']}")
if refined["tier"] != "refined":
    raise SystemExit(f"refined tier {refined['tier']}")
if window["tier"] != "preview":
    raise SystemExit(f"window kept tier {window['tier']!r}, expected preview")
if "windows_refined" not in window:
    raise SystemExit("window operation was not recorded")

if window["windows_refined"][0]["start"] != 20 or window["windows_refined"][-1]["start"] != 0 or window["windows_refined"][-1]["end"] != 7:
    raise SystemExit(f"windows_refined did not keep the inclusive input: {window['windows_refined']}")
if window["source"] != preview["source"] or window.get("court_lines_normalized") != preview.get("court_lines_normalized"):
    raise SystemExit("source or court changed")
outside = list(range(8, len(window["frames"])))
kept = False
for i in outside:
    if window["frames"][i] != preview["frames"][i]:
        raise SystemExit(f"frame {i} outside the window changed")
    for p in window["frames"][i]["players"]:
        if p.get("src") == "interpolated":
            kept = True
if not kept:
    raise SystemExit("no untouched interpolated row survived the window pass")
inside = window["frames"][0]["players"]
if any(p.get("src") == "interpolated" for p in inside):
    raise SystemExit("window row still says interpolated")
print("package-entry ok", flush=True)
"""


class PackageEntryTests(unittest.TestCase):
    @unittest.skipUnless(CLIP.is_file() and FFMPEG.is_file()
                         and (PROJECT / "fast" / "weights" / "yolo26n-384x640.onnx").is_file(),
                         "needs sample clip 03, tools/ffmpeg.exe and the YOLO26 exports (see docs/reproducibility.md)")
    def test_import_and_real_tiers_from_project(self):
        if not SHORT.is_file():
            SHORT.parent.mkdir(parents=True, exist_ok=True)
            subprocess.check_call([
                str(FFMPEG), "-y", "-i", str(CLIP), "-t", "1", "-an",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(SHORT),
            ])
        env = os.environ.copy()
        env["PROJECT"] = str(PROJECT)
        env["SHORT"] = str(SHORT)
        env["PYTHONPATH"] = str(PROJECT)
        subprocess.check_call(
            [str(PY), "-c", ENTRY],
            cwd=PROJECT,
            env=env,
        )


if __name__ == "__main__":
    unittest.main()
