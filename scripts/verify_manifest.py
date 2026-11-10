"""Check every file in the repo against results/MANIFEST.json.

    python scripts/verify_manifest.py
"""

import hashlib
import json
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="maintainer: refresh hashes for all indexed files")
    args = parser.parse_args()
    manifest_path = ROOT / "results" / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["files"]
    if args.write:
        previous = {row["path"]: row for row in rows}
        names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8").split("\0")
        rows = []
        for name in sorted(name for name in names if name and name != "results/MANIFEST.json"):
            path = ROOT / name
            data = path.read_bytes()
            row = dict(previous.get(name, {"path": name}))
            digest = hashlib.sha256(data).hexdigest()
            if "sha256" in row and row["sha256"] != digest:
                row.setdefault("sha256_before_reproduction_package", row["sha256"])
            row.update(bytes=len(data), sha256=digest)
            rows.append(row)
        manifest["files"] = rows
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
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
