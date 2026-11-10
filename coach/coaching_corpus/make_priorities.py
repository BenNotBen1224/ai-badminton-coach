"""learned.json -> coaching_priorities.json, the corpus ranking the video pipeline reads.

Usage: python make_priorities.py [corpus_dir]
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

# What footage must show before a concept can be checked. Only "pose" and "pose+hand" are usable from a
# broadcast pose track; "pose+hand" needs the player's racket hand, "pose+opponent_contact" needs the
# opponent's hit time, the rest need racket close-ups, shuttle tracking, or are not visual at all.
OBSERVE = {
    "footwork": "pose", "knee_bend": "pose", "recovery": "pose", "lunge": "pose",
    "stance_width": "pose", "ready_position": "pose", "body_rotation": "pose_3d",
    "non_racket_arm": "pose+hand", "contact_point": "pose+hand", "arm_extension": "pose+hand",
    "first_step": "pose+opponent_contact", "split_step": "pose+opponent_contact",
    "grip": "racket_closeup", "forearm_wrist": "racket_closeup", "racket_up": "racket",
    "follow_through": "racket", "deception": "racket+shuttle", "placement": "shuttle",
    "timing": "shuttle", "watch_shuttle": "gaze", "consistency": "many_rallies",
    "relax_power": "not_visible", "fitness": "not_visible", "mental": "not_visible",
    "equipment": "not_visible",
}
VISIBLE = {"pose", "pose+hand"}
LICENCE = {
    "youtube-commons": "CC-BY 4.0 transcripts (PleIAs/YouTube-Commons); quotes allowed with attribution",
    "vidchapters": "VidChapters-7M ASR, research use; counts only, no quotes",
    "yodas2": "YODAS2 transcripts of CC-licensed videos; counts only",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(corpus_dir="."):
    root = Path(corpus_dir).resolve()
    learned = json.loads((root / "learned.json").read_text(encoding="utf-8"))
    with open(root / "counts.tsv", encoding="utf-8", newline="") as f:
        sources = {r["source"]: r for r in csv.DictReader(f, delimiter="\t")}
    videos = {v["video_id"]: v for v in learned["coaching_videos"]}
    n = learned["corpus"]["coaching_videos"]
    concepts = []
    for row in learned["coaching"]:
        quotes = []
        for vid, text in row["quotes"]:
            v = videos.get(vid)
            if v and v["source"] == "youtube-commons" and len(quotes) < 2:
                quotes.append({"video_id": vid, "channel": v["channel"], "text": text.strip()})
        concepts.append({
            "rank": row["rank"], "key": row["key"], "label": row["label"], "label_zh": row["label_zh"],
            "observe": OBSERVE[row["key"]], "visible_in_match_video": OBSERVE[row["key"]] in VISIBLE,
            "coaching_videos_mentioning": row["docs"], "coaching_videos_stressing": row["stressed_docs"],
            "channel_stressed_share": row["channel_stressed_share"], "most_important_hits": row["most_important_hits"],
            "cue_quotes": quotes,
        })
    concepts.sort(key=lambda c: c["rank"])
    out = {
        "what": "Badminton coaching points ranked by how often coaches stress them (channel-weighted).",
        "provenance": {
            "learner": "corpus_learn.py", "learner_sha256": sha256(root / "corpus_learn.py"),
            "learned_sha256": sha256(root / "learned.json"),
            "coaching_videos": n, "coaching_channels": learned["corpus"]["coaching_channels"],
            "relevant_videos": learned["corpus"]["relevant_videos"],
            "sources": {k: {"fetched": int(v["fetched"]), "relevant": int(v["relevant"]),
                            "licence": LICENCE.get(k, "")} for k, v in sources.items() if k != "total"},
            "why_not_1000": "YouTube blocks bulk transcript download; only open transcript datasets were used.",
        },
        "visible_order": [c["key"] for c in concepts if c["visible_in_match_video"]],
        "concepts": concepts,
    }
    (root / "coaching_priorities.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("visible order:", ", ".join(out["visible_order"]))
    print("sha256", sha256(root / "coaching_priorities.json"))


if __name__ == "__main__":
    main(*sys.argv[1:])
