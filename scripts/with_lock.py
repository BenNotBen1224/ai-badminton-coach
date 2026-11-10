"""Run one command while holding work\\cpu.lock.

Usage: python -B with_lock.py "<purpose>" -- <command> [args...]
Takes the lock by exclusive create, retries every 30 s for up to 30 minutes, and on exit removes the lock
only if it still holds our own text. It never touches another owner's lock. The exit code is the command's.
"""
import os
import argparse
import subprocess
import sys
import time
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "work" / "cpu.lock"
POLL_S, GIVE_UP_S = 30, 1800


def acquire(text):
    start = time.time()
    while True:
        try:
            with open(LOCK, "x", encoding="utf-8") as fh:
                fh.write(text)
            return
        except FileExistsError:
            if time.time() - start > GIVE_UP_S:
                raise TimeoutError(f"cpu.lock still held after {GIVE_UP_S}s")
            print(f"waiting for cpu.lock held by: {LOCK.read_text('utf-8')[:120]!r}", flush=True)
            time.sleep(POLL_S)


def main():
    global LOCK
    if len(sys.argv) > 1 and sys.argv[1] == "--lock":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--lock", type=Path, required=True)
        args = parser.parse_args(sys.argv[1:3])
        LOCK = args.lock
        sys.argv = [sys.argv[0], *sys.argv[3:]]
    if "--" not in sys.argv[2:]:
        sys.exit(__doc__)
    cut = sys.argv.index("--")
    purpose, cmd = sys.argv[1], sys.argv[cut + 1:]
    text = f"nat-audit {os.getpid()} {purpose}"
    acquire(text)
    try:
        return subprocess.run(cmd).returncode
    finally:
        if LOCK.exists() and LOCK.read_text("utf-8") == text:
            LOCK.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
