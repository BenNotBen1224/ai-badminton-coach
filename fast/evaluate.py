"""Score a keypoints.json against the clip-01 reference.

Only the two labeled players count. For each player-frame the best-matching
candidate person is chosen (one candidate per player). Missing persons and
missing joints count as misses.

usage: python evaluate.py result.json [more.json ...] [--ref other_reference.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

REF = Path(__file__).resolve().parent / "reference" / "consensus.json"
REFERENCES = ("joints", "sapiens", "posewarper")


def arr(joints):
    return np.array([[np.nan, np.nan] if j is None else j[:2] for j in joints], float)


def load_reference(path=REF):
    frames = json.loads(Path(path).read_text("utf-8"))["frames"]
    out = []
    for f in frames:
        roles = {}
        for p in f["people"]:
            if p["role"]:
                b = p["bbox_xyxy"]
                roles[p["role"]] = {"h": b[3] - b[1], **{k: arr(p[k]) for k in REFERENCES}}
        out.append(roles)
    return out


def score(path, reference):
    frames = json.loads(Path(path).read_text("utf-8"))["frames"]
    assert len(frames) == len(reference), (path, len(frames))
    hits = {k: {a: [0, 0] for a in (0.05, 0.1)} for k in REFERENCES}
    agreed = [0, 0]
    by_role, role_ok = [0, 0], [0, 0]
    found = total = 0
    per_role = {"near": [0, 0], "far": [0, 0]}
    role_hits = {r: {a: [0, 0] for a in (0.05, 0.1)} for r in per_role}
    ids = {"near": [], "far": []}
    for f, roles in zip(frames, reference):
        cands = [(p, arr(p["joints"])) for p in f["players"]]
        taken = set()
        for role, ref in sorted(roles.items(), key=lambda r: -r[1]["h"]):
            target = ref["joints"]
            best, best_key = None, (0, -np.inf)
            for i, (p, c) in enumerate(cands):
                if i in taken:
                    continue
                d = np.linalg.norm(c - target, axis=1)
                ok = np.isfinite(d)
                key = ((d[ok] < 0.1 * ref["h"]).sum(), -np.nanmean(d) if ok.any() else -np.inf)
                if key > best_key:
                    best, best_key = i, key
            valid = np.isfinite(target).all(1)
            matched = best is not None and best_key[0] >= 0.5 * valid.sum()
            total += 1
            per_role[role][1] += 1
            if matched:
                taken.add(best)
                found += 1
                per_role[role][0] += 1
                ids[role].append(cands[best][0].get("track_id", cands[best][0].get("role")))
            cand = cands[best][1] if matched else np.full((17, 2), np.nan)
            for k in REFERENCES:
                tgt = ref[k]
                v = np.isfinite(tgt).all(1)
                d = np.linalg.norm(cand - tgt, axis=1)
                for a in (0.05, 0.1):
                    hits[k][a][0] += int((d[v] < a * ref["h"]).sum())
                    hits[k][a][1] += int(v.sum())
                    if k == "joints":
                        role_hits[role][a][0] += int((d[v] < a * ref["h"]).sum())
                        role_hits[role][a][1] += int(v.sum())
            # joints all three references place within 5% of body height of each other
            r = [ref[k] for k in REFERENCES]
            tol = 0.05 * ref["h"]
            agree = np.ones(17, bool)
            for i in range(3):
                for j in range(i + 1, 3):
                    with np.errstate(invalid="ignore"):
                        agree &= np.linalg.norm(r[i] - r[j], axis=1) < tol
            d = np.linalg.norm(cand - target, axis=1)
            agreed[0] += int((d[agree] < tol).sum())
            agreed[1] += int(agree.sum())
            # pipelines that name the players: score only the person they called `role`
            labeled = [c for p, c in cands if p.get("role") == role]
            if any("role" in p for p, _ in cands):
                lc = labeled[0] if labeled else np.full((17, 2), np.nan)
                v = np.isfinite(target).all(1)
                by_role[0] += int((np.linalg.norm(lc - target, axis=1)[v] < 0.05 * ref["h"]).sum())
                by_role[1] += int(v.sum())
                role_ok[0] += bool(labeled) and matched and cands[best][0].get("role") == role
                role_ok[1] += 1
    switches = {r: sum(1 for x, y in zip(s, s[1:]) if x != y) for r, s in ids.items()}
    roles = {"role_correct": round(role_ok[0] / role_ok[1], 4),
             "pck05_by_role": round(by_role[0] / by_role[1], 4)} if role_ok[1] else {}
    return {
        "file": str(path),
        "player_frames_found": round(found / total, 4),
        "near_found": round(per_role["near"][0] / per_role["near"][1], 4),
        "far_found": round(per_role["far"][0] / per_role["far"][1], 4),
        "pck05": round(hits["joints"][0.05][0] / hits["joints"][0.05][1], 4),
        "pck10": round(hits["joints"][0.1][0] / hits["joints"][0.1][1], 4),
        "pck10_vs_sapiens": round(hits["sapiens"][0.1][0] / hits["sapiens"][0.1][1], 4),
        "pck10_vs_posewarper": round(hits["posewarper"][0.1][0] / hits["posewarper"][0.1][1], 4),
        "pck05_exact": hits["joints"][0.05][0] / hits["joints"][0.05][1],
        "pck10_exact": hits["joints"][0.1][0] / hits["joints"][0.1][1],
        "player_frames_found_exact": found / total,
        "counts": {
            "pck05": hits["joints"][0.05],
            "pck10": hits["joints"][0.1],
            "player_frames_found": [found, total],
        },
        "pck_by_role": {r: {f"pck{int(a * 100):02}": hit / total if total else None
                           for a, (hit, total) in h.items()} for r, h in role_hits.items()},
        "pck05_agreed": round(agreed[0] / agreed[1], 4),
        "agreed_joints": agreed[1],
        "id_switches": switches,
        "player_frames": total,
        **roles,
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    ref = REF
    if "--ref" in args:
        i = args.index("--ref")
        ref = Path(args[i + 1])
        del args[i:i + 2]
    reference = load_reference(ref)
    rows = [score(p, reference) for p in args]
    for r in rows:
        print(json.dumps(r))
