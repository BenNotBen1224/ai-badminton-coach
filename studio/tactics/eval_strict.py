import argparse
import collections
import hashlib
import json
import pathlib

from eval_engine import engine_rows, match, norm_family, norm_zone


CONFIDENT = {"med", "high"}


def score(rallies, predictions, tolerance):
    results = []
    for rally in rallies:
        strict = [h for h in rally["hits"] if h["eval_tier"] == "strict"]
        for hit in strict:
            if hit["evidence"] != "visible_contact" or not all(hit["conf"][k] in CONFIDENT for k in ("t", "hitter")):
                raise ValueError(f"{rally['id']}: inconsistent strict annotation at {hit['t']}")
        pairs = match(strict, predictions, tolerance)
        details = []
        for label_index, prediction_index in pairs:
            label, prediction = strict[label_index], predictions[prediction_index]
            zone = norm_zone(label["next_zone"]) if label["conf"]["zone"] in CONFIDENT else None
            family = norm_family(label["family"]) if label["conf"]["family"] in CONFIDENT else None
            details.append({
                "label_time": label["t"],
                "prediction_time": prediction["t"],
                "hitter": label["hitter"],
                "absolute_error_s": abs(label["t"] - prediction["t"]),
                "zone_agrees": zone == prediction["zone"] if zone and prediction["zone"] else None,
                "family_agrees": family == prediction["family"] if family and prediction["family"] else None,
            })
        matched_labels = {i for i, _ in pairs}
        results.append({
            "id": rally["id"],
            "strict_annotations": len(strict),
            "matched": len(pairs),
            "strict_subset_recall": len(pairs) / len(strict) if strict else None,
            "annotation_roles": dict(collections.Counter(h["hitter"] for h in strict)),
            "matched_roles": dict(collections.Counter(h["hitter"] for h in details)),
            "matches": details,
            "unmatched_annotations": [{"t": h["t"], "hitter": h["hitter"]} for i, h in enumerate(strict) if i not in matched_labels],
            "low_or_inferred_excluded": len(rally["hits"]) - len(strict),
        })
    count = sum(r["strict_annotations"] for r in results)
    matched = sum(r["matched"] for r in results)
    return {
        "tolerance_s": tolerance,
        "strict_annotations": count,
        "matched": matched,
        "strict_subset_recall": matched / count if count else None,
        "precision": None,
        "full_rally_recall": None,
        "limitations": [
            "The labels are incomplete AI-authored visual annotations, not human-certified racket-shuttle contacts.",
            "Strict means a visible swing or reach with medium-or-high timing and hitter confidence, not a confirmed shuttle contact.",
            "Unmatched predictions are not classified as false positives because other real contacts may be unlabelled.",
            "The source quality and near-player bias do not support whole-match or new-player accuracy claims.",
        ],
        "rallies": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=pathlib.Path)
    parser.add_argument("hits", type=pathlib.Path)
    parser.add_argument("--item", required=True)
    parser.add_argument("--exclude-rally", action="append", default=[])
    parser.add_argument("--tol", type=float, default=0.25)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.tol <= 0:
        parser.error("--tol must be positive")
    label_bytes, prediction_bytes = args.labels.read_bytes(), args.hits.read_bytes()
    document = json.loads(label_bytes)
    items = document.get("clips", document.get("matches", []))
    selected = [
        item for item in items
        if item.get("file", item.get("source", "")).replace("\\", "/").rsplit("/", 1)[-1] == args.item
    ]
    if len(selected) != 1:
        parser.error(f"expected exactly one item named {args.item}, found {len(selected)}")
    rallies = [r for r in selected[0]["rallies"] if r["id"] not in args.exclude_rally]
    result = score(rallies, engine_rows(json.loads(prediction_bytes)), args.tol)
    result["input_hashes"] = {
        str(args.labels): hashlib.sha256(label_bytes).hexdigest(),
        str(args.hits): hashlib.sha256(prediction_bytes).hexdigest(),
    }
    result["item"] = args.item
    result["excluded_rallies"] = args.exclude_rally
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"rallies", "input_hashes"}}, indent=2))


if __name__ == "__main__":
    main()
