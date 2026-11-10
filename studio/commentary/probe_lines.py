import json
from pathlib import Path
import subprocess
import sys
import wave

HERE = Path(__file__).resolve().parent
CHILD = HERE / "render_commentary.py"


def measure(text, path):
    child = subprocess.run([sys.executable, "-B", "-u", str(CHILD), "--synthesize", "60", str(path)],
                           input=text.encode("utf-8"), capture_output=True, check=True, timeout=300)
    log = child.stderr.decode("utf-8", "replace").split("@@GENERATE\n", 1)[1]
    dropped = sum("Skip unknown phonemes" in line or "Unknown token" in line for line in log.splitlines())
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate(), dropped


lines = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for index, text in enumerate(lines):
    seconds, dropped = measure(text, HERE / "probe" / f"{index:02d}.wav")
    print(f"{seconds:5.2f}s dropped={dropped} {text}", flush=True)
