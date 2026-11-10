import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path


def api(repo, endpoint):
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/{endpoint}",
        headers={"User-Agent": "AI-Badminton-Coach-release-check"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Verify a public release's actual tag and every downloaded asset."
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--asset", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tag = urllib.parse.quote(args.tag, safe="")
    reference = api(args.repo, f"git/ref/tags/{tag}")["object"]
    while reference["type"] == "tag":
        reference = api(args.repo, f"git/tags/{reference['sha']}")["object"]
    if reference["type"] != "commit" or reference["sha"] != args.commit:
        raise ValueError(f"Release tag does not point to expected commit {args.commit}")

    release = api(args.repo, f"releases/tags/{tag}")
    if release["draft"]:
        raise ValueError("Release is still a draft")
    expected = {path.name: path for path in args.asset}
    if len(expected) != len(args.asset):
        raise ValueError("Duplicate expected asset names")
    published = {asset["name"]: asset for asset in release["assets"]}
    if set(expected) != set(published):
        raise ValueError(
            f"Release assets differ: expected {sorted(expected)}, got {sorted(published)}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, local in expected.items():
        asset = published[name]
        expected_size = local.stat().st_size
        expected_hash = digest(local)
        if asset["size"] != expected_size:
            raise ValueError(f"{name}: published size does not match local file")
        target = args.out / name
        if target.resolve() == local.resolve():
            raise ValueError("Download directory must not overwrite original assets")
        request = urllib.request.Request(
            asset["browser_download_url"],
            headers={"User-Agent": "AI-Badminton-Coach-release-check"},
        )
        total = 0
        with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > expected_size:
                    raise ValueError(f"{name}: download exceeded its expected size")
                handle.write(chunk)
        if total != expected_size or digest(target) != expected_hash:
            raise ValueError(f"{name}: anonymous download does not match original bytes")
        results.append({
            "name": name,
            "bytes": total,
            "sha256": expected_hash,
            "url": asset["browser_download_url"],
        })

    report = {
        "status": "verified",
        "repository": args.repo,
        "release_url": release["html_url"],
        "tag": args.tag,
        "commit": args.commit,
        "authentication": "none",
        "assets": results,
    }
    (args.out / "public-verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
