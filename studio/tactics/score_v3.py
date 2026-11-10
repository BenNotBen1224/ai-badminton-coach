"""Score frozen V3 hits three ways with the unchanged eval_engine.py.

as_emitted: every V3 row; unresolved rows keep hitter=None, so they count as engine rows and never match.
resolved_only: only rows with a confirmed hitter.
role_candidate_diagnostic: unresolved rows borrow pose_role_candidate as hitter. Diagnostic only.
"""
import hashlib
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
V3 = HERE.parents[2] / "badminton-coach" / "outputs" / "tactics-engine-20261106-v3" / "hits.json"
V3_SHA = "4ff91a6fb316560314950f119f7fad57ff955d81f6f5756c13995a883942f23d"
OUT = HERE / "v3_scoring"
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
    raw = V3.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == V3_SHA, "V3 hits.json changed"
    doc = json.loads(raw)
    rows = doc["hits"]
    variants = {
        "as_emitted": rows,
        "resolved_only": [h for h in rows if h.get("hitter")],
        "role_candidate_diagnostic": [dict(h, hitter=h.get("hitter") or h.get("pose_role_candidate")) for h in rows],
    }
    OUT.mkdir(exist_ok=True)
    partition = json.loads((HERE / "development" / "partition.json").read_text(encoding="utf-8"))
    summary = {"v3_hits_sha256": V3_SHA, "rows": len(rows),
               "resolved_rows": len(variants["resolved_only"]), "gates": GATES, "variants": {}}
    for name, hits in variants.items():
        hp = OUT / f"{name}.hits.json"
        hp.write_text(json.dumps({"hits": hits}), encoding="utf-8")
        ep = OUT / f"{name}.eval.json"
        subprocess.run([sys.executable, "-B", str(HERE / "eval_engine.py"), str(HERE / "labels" / "labels.json"),
                        str(hp), "--out", str(ep)], check=True, stdout=subprocess.DEVNULL)
        ev = json.loads(ep.read_text(encoding="utf-8"))
        passed = {k: ev[k] is not None and ev[k] >= v for k, v in GATES.items()}
        summary["variants"][name] = {
            "precision": ev["precision"], "recall": ev["recall"], "zone_agreement": ev["zone_agreement"],
            "zone_coverage": ev["zone_coverage"], "totals": ev["totals"], "gates_passed": passed,
            "all_gates_passed": all(passed.values()),
            "development": split(ev["per_rally"], set(partition["development"])),
            "evaluation": split(ev["per_rally"], set(partition["evaluation"])),
            "per_rally": [(r["id"], r["label"], r["engine"], r["matched"]) for r in ev["per_rally"]],
        }
    (HERE / "v3_eval.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
