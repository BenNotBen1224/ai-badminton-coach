"""Run every Python test suite from the folder it expects and print one summary.

    python scripts/run_tests.py

Tests that need footage, weights or ffmpeg skip themselves when those are missing.
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = [
    ("coach", ROOT, ["-m", "unittest", "discover", "-s", "coach/tests", "-t", "."]),
    ("core", ROOT, ["-m", "unittest", "discover", "-s", "tests"]),
    ("turbo", ROOT / "turbo", ["-m", "unittest", "discover", "-s", "tests"]),
    ("comparison", ROOT / "comparison", ["-m", "unittest", "test_runner"]),
    ("studio-tactics", ROOT / "studio" / "tactics", ["-m", "unittest", "test_eval_strict"]),
]


def run(name, cwd, args):
    start = time.monotonic()
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run([sys.executable, "-B", *args], cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)
    seconds = time.monotonic() - start
    out = proc.stderr + proc.stdout
    ran = re.search(r"Ran (\d+) tests?", out)
    skipped = re.search(r"skipped=(\d+)", out)
    return {"suite": name, "ok": proc.returncode == 0, "ran": int(ran.group(1)) if ran else 0,
            "skipped": int(skipped.group(1)) if skipped else 0, "seconds": round(seconds, 1), "log": out}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # failure logs contain Chinese text
    results = [run(*suite) for suite in SUITES]
    for r in results:
        if not r["ok"]:
            print(f"===== {r['suite']} =====\n{r['log'][-6000:]}")
    print(f"{'suite':16}{'result':8}{'ran':>6}{'skipped':>9}{'seconds':>9}")
    for r in results:
        print(f"{r['suite']:16}{'pass' if r['ok'] else 'FAIL':8}{r['ran']:>6}{r['skipped']:>9}{r['seconds']:>9}")
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
