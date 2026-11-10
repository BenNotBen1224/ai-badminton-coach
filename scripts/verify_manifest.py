"""Check every file in the repo against results/MANIFEST.json.

    python scripts/verify_manifest.py
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = json.loads((ROOT / "results" / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    bad = []
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file():
            bad.append(f"missing  {row['path']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            bad.append(f"changed  {row['path']}")
    for line in bad:
        print(line)
    print(f"{len(rows) - len(bad)} of {len(rows)} files match")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
