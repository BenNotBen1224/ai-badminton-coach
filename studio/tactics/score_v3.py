"""Score frozen V3 hits three ways with the unchanged eval_engine.py.

as_emitted: every V3 row; unresolved rows keep hitter=None, so they count as engine rows and never match.
resolved_only: only rows with a confirmed hitter.
role_candidate_diagnostic: unresolved rows borrow pose_role_candidate as hitter. Diagnostic only.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
V3_SHA = "4ff91a6fb316560314950f119f7fad57ff955d81f6f5756c13995a883942f23d"
GATES = {"precision": 0.8, "recall": 0.8, "zone_agreement": 0.7}


def split(per, ids):
    rows = [r for r in per if r["id"] in ids]
    lab = sum(r["label"] for r in rows)
    eng = sum(r["engine"] for r in rows)
    mat = sum(r["matched"] for r in rows)
    zb = sum(r["zone_both"] for r in rows)
    zs = sum(r["zone_same"] for r in rows)
    return {"label": lab, "engine": eng, "matched": mat,
            "precision": mat / eng if eng else None, "recall": mat / lab if lab else None,
            "zone_both": zb, "zone_agreement": zs / zb if zb else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hits", type=pathlib.Path, default=HERE / "inputs" / "hits.json")
    parser.add_argument("--labels", type=pathlib.Path, default=HERE / "inputs" / "labels.json")
    parser.add_argument("--partition", type=pathlib.Path, default=HERE / "inputs" / "partition.json")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--check-reference", action="store_true",
                        help="fail unless every recorded metric equals v3_eval.json")
    args = parser.parse_args()
    raw = args.hits.read_bytes()
    doc = json.loads(raw)
    rows = doc["hits"]
    variants = {
        "as_emitted": rows,
        "resolved_only": [h for h in rows if h.get("hitter")],
        "role_candidate_diagnostic": [dict(h, hitter=h.get("hitter") or h.get("pose_role_candidate")) for h in rows],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    partition = json.loads(args.partition.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    ids = {r["id"] for r in labels["rallies"]}
    development, evaluation = set(partition["development"]), set(partition["evaluation"])
    if development & evaluation or development | evaluation != ids:
        raise ValueError("Partition must cover every labelled rally exactly once")
    summary = {"v3_hits_sha256": V3_SHA, "rows": len(rows),
               "resolved_rows": len(variants["resolved_only"]), "gates": GATES, "variants": {}}
    for name, hits in variants.items():
        hp = args.out / f"{name}.hits.json"
        hp.write_text(json.dumps({"hits": hits}), encoding="utf-8")
        ep = args.out / f"{name}.eval.json"
        subprocess.run([sys.executable, "-B", str(HERE / "eval_engine.py"), str(args.labels),
                        str(hp), "--out", str(ep)], check=True, stdout=subprocess.DEVNULL)
        ev = json.loads(ep.read_text(encoding="utf-8"))
        passed = {k: ev[k] is not None and ev[k] >= v for k, v in GATES.items()}
        summary["variants"][name] = {
            "precision": ev["precision"], "recall": ev["recall"], "zone_agreement": ev["zone_agreement"],
            "zone_coverage": ev["zone_coverage"], "totals": ev["totals"], "gates_passed": passed,
            "all_gates_passed": all(passed.values()),
            "development": split(ev["per_rally"], development),
            "evaluation": split(ev["per_rally"], evaluation),
            "per_rally": [(r["id"], r["label"], r["engine"], r["matched"]) for r in ev["per_rally"]],
        }
    summary = json.loads(json.dumps(summary))
    if args.check_reference:
        expected = json.loads((HERE / "v3_eval.json").read_text(encoding="utf-8"))
        if summary != expected:
            raise ValueError("Re-scored metrics differ from the historical reference")
    summary["input_sha256"] = {
        "hits": hashlib.sha256(raw).hexdigest(),
        "labels": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "partition": hashlib.sha256(args.partition.read_bytes()).hexdigest(),
    }
    summary["mode"] = "re-score frozen predictions; no inference; AI labels are not human ground truth"
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
