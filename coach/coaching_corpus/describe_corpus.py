"""Describe observed channel metadata separately from the learner's weighting groups."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def describe(videos):
    channels = {row["channel"] for row in videos if row.get("channel")}
    groups = {row.get("channel") or row["video_id"] for row in videos}
    return {
        "coaching_videos": len(videos),
        "known_channel_names": len(channels),
        "videos_without_channel_metadata": sum(not row.get("channel") for row in videos),
        "weighting_groups": len(groups),
        "grouping_rule": "Each named channel is one group; each video without channel metadata uses its video_id.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("learned", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    raw = args.learned.read_bytes()
    result = describe(json.loads(raw)["coaching_videos"])
    result["source"] = {"file": args.learned.name, "sha256": hashlib.sha256(raw).hexdigest(),
                        "field": "coaching_videos", "derivation": "describe_corpus.py"}
    args.out.write_text(json.dumps(result, indent=2) + "\n", "utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
