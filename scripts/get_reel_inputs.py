"""Fetch or unpack the hash-checked frozen narration bundle for render_reels.py."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unpack(archive, out, manifest):
    if sha(archive) != manifest["archive"]["sha256"]:
        raise ValueError("Reproduction archive SHA256 mismatch")
    expected = {row["file"]: row for rows in manifest["editions"].values() for row in rows}
    with zipfile.ZipFile(archive) as source:
        if set(source.namelist()) != set(expected) or len(source.namelist()) != len(expected):
            raise ValueError("Archive members differ from the explicit input list")
        for name, row in expected.items():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError(f"Unsafe archive member: {name}")
            data = source.read(name)
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"Changed archive member: {name}")
        for name in expected:
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = source.read(name)
            if target.exists() and target.read_bytes() != data:
                raise FileExistsError(f"Refusing to overwrite changed input: {target}")
            target.write_bytes(data)
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="already downloaded release asset")
    parser.add_argument("--out", type=Path, default=ROOT / "work" / "reel-inputs")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "studio" / "reels" / "inputs-manifest.json").read_text(encoding="utf-8"))
    archive = args.archive
    if archive is None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        archive = args.out.parent / manifest["archive"]["filename"]
        if not archive.is_file() or sha(archive) != manifest["archive"]["sha256"]:
            part = archive.with_suffix(".part")
            request = urllib.request.Request(manifest["archive"]["url"], headers={"User-Agent": "ai-badminton-coach/1.1"})
            with urllib.request.urlopen(request) as response, part.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if sha(part) != manifest["archive"]["sha256"]:
                raise ValueError("Downloaded reproduction archive SHA256 mismatch")
            part.replace(archive)
    count = unpack(archive, args.out, manifest)
    print(f"Verified and unpacked {count} narration files into {args.out}")


if __name__ == "__main__":
    main()
