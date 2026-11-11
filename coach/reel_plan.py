"""Editable, source-bound commentary plans; uncertain shot families abstain."""
from __future__ import annotations

import math
import re
from itertools import combinations

import cv2
import numpy as np

from .reel_analysis import player_frames

FOOTER = "数据：TrackNetV3 羽毛球追踪 + 姿态模型 · 模型估算 · AI 配音"


def draft_plan(analysis, names, attribution, source_sha, assets):
    return {
        "schema_version": 1, "review_status": "draft_requires_review",
        "source_sha256": source_sha, "analysis_fps": 30, "normalized_frames": analysis["frames"],
        "names_at_first_reliable_rally": names, "attribution": attribution, "assets": assets,
        "reviewer_kind": None, "reviewer": "", "review_note": "",
        "title": [f"{names['near']}对上{names['far']}", "这几拍，攻防节奏拉满！"],
        "footer": FOOTER, "target_frames": 798, "moments": [],
        "edit_instructions": [
            "Inspect normalized.mp4 and analysis.json; contact candidates and families are not ground truth.",
            "Copy moment_template into moments, choose source windows and cue frames at normalized 30 fps.",
            "Supply the actual player name at each cue; initial near/far names may change after a change of ends.",
            "Write short Chinese speech/captions and estimate-labelled green tags; do not speak numbers.",
            "Keep caption_chunks joined exactly equal to caption. Use null for an uncertain shot_family.",
            "Set each line claim_source to reviewed_plan and describe the visual evidence in evidence.review_note.",
            "Add a top-level review_note, reviewer_kind (agent or human), and reviewer name.",
            "Change review_status to reviewed_plan, then use --plan with a NEW output directory.",
            "Use reviewer_kind human only for an actual human attestation; agent frame review is not human ground truth or listening approval.",
        ],
        "moment_template": {
            "id": "reviewed-moment", "start_frame": None, "end_frame": None,
            "lines": [{"cue_frame": None, "hitter": None, "player": "", "say": "", "caption": "",
                       "caption_chunks": [], "tag": "", "shot_family": None, "claim_source": "reviewed_plan",
                       "evidence": {"review_note": ""}}],
        },
        "candidate_contacts": [{"frame": h["frame"], "time_s": h["time_s"], "hitter": h["hitter"],
                                "model_family": h["family"], "confidence": h["confidence"]}
                               for h in analysis["hits"]],
    }


def jersey(image, player):
    joints = np.asarray(player["joints"], float)
    torso = joints[[5, 6, 12, 11], :2]
    lo, hi = torso.min(0).astype(int), torso.max(0).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [image.shape[1], image.shape[0]])
    crop = image[lo[1]:hi[1], lo[0]:hi[0]]
    if not crop.size:
        raise ValueError("Cannot establish player kit identity from measured torso")
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0][hsv[:, :, 1] < 32] = 0
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull((torso-lo).astype(np.int32)), 255)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    if not mask.any():
        raise ValueError("Torso evidence is too small to establish kit identity")
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [18, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return hist/max(np.linalg.norm(hist), 1e-6)


def identities(video, pose, hits, names, sides=None, reference_frames=None):
    cap = cv2.VideoCapture(str(video))
    frames = player_frames(pose)
    references = reference_frames or [hits[0]["frame"]] if hits else []
    prototypes = {role: [] for role in names}
    for i in references:
        if len(frames[i]) != 2:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, image = cap.read()
        if not ok:
            raise ValueError(f"Cannot read initial-rally identity evidence frame {i}")
        for role, player in frames[i].items():
            prototypes[role].append(jersey(image, player))
    assignments = {}
    for hit in hits:
        i = hit["frame"]
        if len(frames[i]) != 2:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, image = cap.read()
        if not ok:
            raise ValueError(f"Cannot read identity evidence frame {i}")
        observed = {role: jersey(image, p) for role, p in frames[i].items()}
        direct_scores = [max(float(observed[role] @ ref) for ref in prototypes[role]) for role in names]
        swapped_scores = [max(float(observed[role] @ ref)
                              for ref in prototypes["far" if role == "near" else "near"]) for role in names]
        direct, swapped = sum(direct_scores), sum(swapped_scores)
        reviewed = next((row for row in sides or [] if row["start_s"] <= i/30 < row["end_s"]), None)
        if reviewed:
            mapping = {role: reviewed[role] for role in names}
            if set(mapping.values()) != set(names.values()):
                raise ValueError("Reviewed side assignments must use exactly the two supplied names")
            method = "reviewed side assignment"
        elif (abs(direct-swapped) >= .15 and max(direct, swapped) >= 1.0
              and min(direct_scores if direct > swapped else swapped_scores) >= .45):
            mapping = names if direct > swapped else {"near": names["far"], "far": names["near"]}
            method = "torso-colour match to first reliable rally; not face/OCR identification"
        else:
            continue
        assignments[i] = {"near": mapping["near"], "far": mapping["far"], "method": method,
                          "direct_similarity": direct/2, "swapped_similarity": swapped/2,
                          "reference_frames": references}
    cap.release()
    return assignments


def wording(hit, who):
    if hit["family"] == "clear":
        return f"{who}高远球压后场！", "高远球"
    if hit["family"] == "lift":
        return f"{who}挑后场，拆连贯！", "挑后场"
    target = hit.get("receiver_court_m")
    origin = hit["hitter_court_m"]
    if target and abs(target[0]-origin[0]) > 2.4:
        return f"{who}对角调动！", "对角调动"
    if target:
        deep = target[1] < 2.4 if hit["hitter"] == "near" else target[1] > 11.0
        if deep and abs(target[0]-3.05) > 1.3:
            return f"{who}压底角！", "压底角"
        if deep:
            return f"{who}压住后场！", "后场压制"
        depth = target[1] if hit["hitter"] == "near" else 13.4-target[1]
        if depth > 4.8:
            return f"{who}网前调动！", "网前调动"
        previous = hit.get("receiver_before_court_m")
        if previous and abs(previous[0]-target[0]) > 1.8:
            return f"{who}拉开角度！", "横向调动"
    return None


def sequence_context(moments):
    previous = {}
    for moment in moments:
        for line in moment["lines"]:
            who, hit = line["player"], line["evidence"]
            earlier, rally = previous.get(who, (None, None))
            previous[who] = (line, moment.get("rally_anchor_frame"))
            if (earlier is None or rally != moment.get("rally_anchor_frame")
                    or not 0 < (line["cue_frame"]-earlier["cue_frame"])/30 <= 8):
                continue
            before, after = earlier["evidence"].get("receiver_court_m"), hit.get("receiver_court_m")
            if before is None or after is None:
                continue
            depth_before = before[1] if earlier["hitter"] == "near" else 13.4-before[1]
            depth_after = after[1] if line["hitter"] == "near" else 13.4-after[1]
            text = None
            if depth_before < 2.8 and depth_after > 4.8:
                if earlier["shot_family"] is None:
                    first = f"{who}先压后场！"
                    earlier.update(say=first, caption=first, caption_chunks=[first])
                text, tactic = f"{who}再收网前，前后调动！", "前后调动"
            elif depth_before > 4.8 and depth_after < 2.8 and line["shot_family"] is None:
                text, tactic = f"{who}再顶后场，前后拉开！", "前后调动"
            elif depth_before < 2.8 and depth_after < 2.8 and line["shot_family"] is None:
                text, tactic = f"{who}连续压住底线！", "后场连续施压"
            if text:
                line.update(say=text, caption=text, caption_chunks=[text],
                            tag=f"{tactic} · 飞行约 {hit['flight_s']:.1f} 秒 · 估算",
                            sequence_evidence={"earlier_cue_frame": earlier["cue_frame"],
                                               "earlier_receiver_court_m": before,
                                               "current_receiver_court_m": after,
                                               "scope": "Observed selected placements; not a claim of forced receiver retreat"})


def make_plan(analysis, pose, video, names, attribution, source_sha, sides=None):
    rally_frames = {frame for rally in analysis["rallies"] for frame in rally["hit_frames"]}
    rally_hits = [hit for hit in analysis["hits"] if hit["frame"] in rally_frames]
    if not rally_hits:
        raise ValueError("No reliable alternating centre-court rally with observed shuttle found; no reel produced")
    first_rally = analysis["rallies"][0]
    reference_frames = first_rally["hit_frames"]
    if any(reference_frames[0] < cut <= reference_frames[-1] for cut in analysis["cuts"]):
        raise ValueError("Initial identity reference crosses a camera cut; reviewed side assignments required")
    assignments = identities(video, pose, rally_hits, names, sides, reference_frames)
    supported = [h for h in analysis["hits"] if h["frame"] in assignments
                 and "next_hit_frame" in h and h.get("visible_fraction", 0) >= .55
                 and h["confidence"] >= .45
                 and wording(h, assignments[h["frame"]][h["hitter"]]) is not None]
    candidates = []
    for rally in analysis["rallies"]:
        group = [h for h in supported if h["frame"] in rally["hit_frames"]]
        for hit in group:
            start = max(rally["start_s"], hit["time_s"]-.8)
            end = min(rally["end_s"], max(hit["time_s"]+3, hit["next_hit_frame"]/30+1.2))
            if not any(start*30 < cut < end*30 for cut in analysis["cuts"]):
                quality = (hit["family"] is not None)*2+hit["confidence"]
                candidates.append((quality, start, end, (hit,)))
        for a in range(len(group)-1):
            for b in range(a+1, len(group)):
                first, second = group[a], group[b]
                span = second["time_s"]-first["time_s"]
                if 1.2 <= span <= 3.8:
                    start = max(rally["start_s"], first["time_s"]-.45)
                    end = min(rally["end_s"], second["time_s"]+1.4)
                    if any(start*30 < cut < end*30 for cut in analysis["cuts"]):
                        continue
                    quality = sum((h["family"] is not None)*2+h["confidence"] for h in (first, second))
                    quality += min(len(group), 15)/15
                    candidates.append((quality, start, end, (first, second)))
    pool = sorted(candidates, key=lambda x: -x[0])[:60]
    chosen, best = [], None
    for size in range(1, 4):
        for rows in combinations(pool, size):
            ordered = sorted(rows, key=lambda row: row[1])
            if any(math.ceil(a[2]*30) >= math.floor(b[1]*30) for a, b in zip(ordered, ordered[1:])):
                continue
            selected = [hit for row in rows for hit in row[3]]
            players = {assignments[h["frame"]][h["hitter"]] for h in selected}
            texts = {wording(h, assignments[h["frame"]][h["hitter"]])[0] for h in selected}
            score = (len(players), len(texts), size, sum(row[0] for row in rows))
            if best is None or score > best:
                chosen, best = ordered, score
    if not chosen:
        raise ValueError("No identity-safe, shuttle-supported moment; provide reviewed --sides assignments if ends changed")
    selected_players = {assignments[h["frame"]][h["hitter"]] for row in chosen for h in row[3]}
    if selected_players != set(names.values()):
        raise ValueError("Cannot support commentary for both named players; review the saved plan.draft.json")
    expanded = []
    for index, (quality, start, end, hits) in enumerate(chosen):
        rally = next(r for r in analysis["rallies"] if hits[0]["frame"] in r["hit_frames"])
        before = (chosen[index-1][2]+start)/2 if index else 0
        after = (end+chosen[index+1][1])/2 if index+1 < len(chosen) else analysis["frames"]/30
        start = max(rally["start_s"], start-1.5, before+1/30)
        end = min(rally["end_s"], end+1.5, after-1/30)
        expanded.append((quality, start, end, hits))
    chosen = expanded
    moments = []
    for index, (_, start, end, hits) in enumerate(chosen):
        lines = []
        for hit in hits:
            identity = assignments[hit["frame"]]
            who = identity[hit["hitter"]]
            text, label = wording(hit, who)
            lines.append({
                "cue_frame": hit["frame"], "hitter": hit["hitter"], "player": who,
                "say": text, "caption": text, "tag": f"{label} · 飞行约 {hit['flight_s']:.1f} 秒 · 估算",
                "caption_chunks": [text], "shot_family": hit["family"],
                "claim_source": "model_estimate", "identity": identity, "evidence": hit,
            })
        rally_anchor = next(r["hit_frames"][0] for r in analysis["rallies"] if hits[0]["frame"] in r["hit_frames"])
        moments.append({"id": f"moment-{index+1}", "rally_anchor_frame": rally_anchor,
                        "start_frame": math.floor(start*30),
                        "end_frame": math.ceil(end*30), "lines": lines})
    sequence_context(moments)
    plan = {
        "schema_version": 1, "review_status": "automatic_unreviewed",
        "source_sha256": source_sha, "analysis_fps": 30,
        "names_at_first_reliable_rally": names, "attribution": attribution,
        "title": [f"{names['near']}对上{names['far']}", "这几拍，攻防节奏拉满！"],
        "footer": FOOTER, "target_frames": 798, "moments": moments,
        "selection": "Ranked candidate rallies with at least three observed side alternations and per-line shuttle evidence; no manually chosen timestamps",
        "limits": [
            "Names are supplied for near/far at the first reliable rally, not inferred from attribution order.",
            "Torso colour tracks side swaps; uncertain identities abstain and require reviewed side assignments.",
            "Uncertain shot families use broader commentary, never unverified smash/drop/jump/backhand claims.",
            "No human listening or naturalness review is claimed.",
        ],
    }
    validate_plan(plan, source_sha, analysis["frames"])
    return plan


def validate_plan(plan, source_sha, frame_count):
    if plan.get("schema_version") != 1 or plan.get("source_sha256") != source_sha:
        raise ValueError("Plan belongs to a different source or unsupported schema")
    if plan.get("review_status") == "draft_requires_review":
        raise ValueError("Complete the draft moments and reviewer provenance, then set review_status to reviewed_plan")
    if plan.get("review_status") not in ("automatic_unreviewed", "reviewed_plan"):
        raise ValueError("Plan needs an explicit automatic_unreviewed or reviewed_plan status")
    reviewed = plan["review_status"] == "reviewed_plan"
    if reviewed and (plan.get("reviewer_kind") not in ("agent", "human")
                     or not str(plan.get("reviewer", "")).strip()
                     or not str(plan.get("review_note", "")).strip()):
        raise ValueError("A reviewed plan needs reviewer_kind (agent or human), reviewer and review_note")
    if plan.get("analysis_fps") != 30 or not 750 <= plan.get("target_frames", 0) <= 900:
        raise ValueError("Plan must use 30 fps and a 25–30 second target")
    if len(plan.get("title", [])) != 2 or any(not isinstance(s, str) or not s.strip() for s in plan["title"]):
        raise ValueError("Plan needs a nonempty two-line title")
    names = plan.get("names_at_first_reliable_rally", {})
    if (set(names) != {"near", "far"} or any(not isinstance(n, str) or not n.strip() for n in names.values())
            or names["near"] == names["far"]):
        raise ValueError("Plan needs distinct supplied near/far names for the first reliable rally")
    if not plan.get("attribution") or not plan.get("moments"):
        raise ValueError("Plan needs attribution and moments")
    previous = -1
    for moment in plan["moments"]:
        start, end = moment.get("start_frame"), moment.get("end_frame")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end < frame_count:
            raise ValueError("Source window outside normalized footage")
        if start <= previous:
            raise ValueError("Source moments must be ordered and non-overlapping")
        previous = end
        if not isinstance(moment.get("lines"), list) or not moment["lines"]:
            raise ValueError("Each source moment needs at least one commentary line")
        cue = start-1
        for line in moment["lines"]:
            if reviewed and (line.get("claim_source") != "reviewed_plan"
                             or not str(line.get("evidence", {}).get("review_note", "")).strip()):
                raise ValueError("Each reviewed claim needs claim_source and a visual evidence review_note")
            if (not isinstance(line.get("cue_frame"), int) or not start <= line["cue_frame"] <= end
                    or line["cue_frame"] <= cue):
                raise ValueError("Line cues must increase inside their source window")
            cue = line["cue_frame"]
            if line.get("hitter") not in ("near", "far") or line.get("player") not in names.values():
                raise ValueError("Each line needs a measured hitter and supplied player name")
            if not all(isinstance(line.get(k), str) and line[k].strip() for k in ("say", "caption", "tag")):
                raise ValueError("Each line needs speech, caption and green tag")
            spoken = line["say"].replace(line["player"], "")
            measured = (r"(?:[0-9零〇一二三四五六七八九十百千万两点几半]+(?:\.[0-9]+)?)\s*"
                        r"(?:公里每小时|千米每小时|米每秒|公里|千米|厘米|毫米|米|毫秒|秒钟?|分钟|小时|帧|%|％|km/h|m/s)"
                        r"|百分之[0-9零〇一二三四五六七八九十百千万两点几半]+")
            if re.search(measured, spoken, re.IGNORECASE):
                raise ValueError("No spoken numbers for measured stats; use estimated on-screen tags, not distances/times in narration")
            if len(line["say"]) > 32:
                raise ValueError("Use a short commentary line (at most 32 characters)")
            if "".join(line.get("caption_chunks", [])) != line["caption"]:
                raise ValueError("Caption chunks must exactly reproduce the caption")
            if "估" not in line["tag"]:
                raise ValueError("On-screen measurements must stay marked as estimates")
