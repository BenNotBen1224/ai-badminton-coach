"""Scoped agreement with incomplete AI annotations, not whole-video accuracy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from studio.tactics.eval_engine import match, norm_family


def score(analysis, labels):
    annotated = [
        {**hit, "t": float(hit["t"]), "rally_id": rally["id"]}
        for rally in labels["rallies"] for hit in rally["hits"]
    ]
    predicted = [{**hit, "t": float(hit["time_s"]), "family": norm_family(hit.get("family"))}
                 for hit in analysis["hits"]]
    eligible = {
        i for i, hit in enumerate(annotated)
        if hit.get("conf", {}).get("family") == "high"
        and hit.get("conf", {}).get("hitter") == "high"
        and hit.get("conf", {}).get("t") in ("med", "high")
        and hit.get("shuttle_seen") is True
        and norm_family(hit.get("family")) is not None
    }
    rows = []
    for label_i, prediction_i in match(annotated, predicted, .25):
        if label_i not in eligible:
            continue
        label, prediction = annotated[label_i], predicted[prediction_i]
        rows.append({
            "rally_id": label["rally_id"], "label_time_s": label["t"],
            "prediction_time_s": prediction["t"], "hitter": label["hitter"],
            "ai_label_family": norm_family(label["family"]),
            "predicted_family": prediction["family"],
            "abstained": prediction["family"] is None,
            "same": prediction["family"] == norm_family(label["family"]) if prediction["family"] else None,
        })
    named_rows = [row for row in rows if not row["abstained"]]
    named = sum(hit["family"] is not None for hit in predicted)
    return {
        "scope": "Agreement only on matched, confidently family-annotated AI contacts. Not human ground truth or whole-video accuracy.",
        "label_author": labels.get("labeler"),
        "eligibility": "family high, hitter high, timing med/high, shuttle_seen true; same hitter within 0.25 seconds, one-to-one matching",
        "unmatched_predictions": "Not scored as false positives: these AI labels are incomplete.",
        "detected_contact_candidates": len(predicted), "named_family_candidates": named,
        "candidate_family_coverage": named/len(predicted) if predicted else None,
        "eligible_annotations": len(eligible), "matched_eligible_annotations": len(rows),
        "matched_eligible_abstentions": len(rows)-len(named_rows),
        "matched_eligible_named": len(named_rows),
        "matched_eligible_family_agreement": sum(row["same"] for row in named_rows)/len(named_rows) if named_rows else None,
        "rows": rows,
        "visual_review": "Not performed by this scorer; separately inspect every selected narrated claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = score(json.loads(args.analysis.read_text(encoding="utf-8")),
                   json.loads(args.labels.read_text(encoding="utf-8")))
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
