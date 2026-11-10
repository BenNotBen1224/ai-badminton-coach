"""Score engine hits against the by-eye labels.

Usage: python eval_engine.py <labels.json> <engine hits.json> [--tol 0.25] [--out eval.json]

Matching: inside each labelled rally window (serve_t - 0.5 .. end_t + 0.5), a label hit and an
engine hit match when the hitter is the same and the times differ by at most --tol seconds.
The matcher maximizes one-to-one match count, then minimizes total timing error. Precision and
recall are counted over those windows only.
Zone agreement is counted on matched hits where both sides give a 3x3 zone.
"""
import argparse
import json
import pathlib


def norm_zone(z):
    if not isinstance(z, dict):
        return None
    d, s = z.get("depth"), z.get("side")
    if d in ("front", "mid", "rear") and s in ("left", "center", "right"):
        return d, s
    return None


def norm_family(f):
    if not f or f in ("unknown", "other"):
        return None
    return "drive/push" if f in ("drive", "push", "drive/push") else f


def engine_rows(doc):
    rows = doc["hits"] if isinstance(doc, dict) else doc
    out = []
    for h in rows:
        fam = h.get("shot_family", h.get("family"))
        if isinstance(fam, dict):
            fam = fam.get("family")
        out.append({
            "t": float(h["time_s"] if "time_s" in h else h["t"]),
            "hitter": h["hitter"],
            "family": norm_family(fam),
            "zone": norm_zone(h.get("landing_zone")),
        })
    return out


def match(lab, eng, tol):
    labels = sorted(enumerate(lab), key=lambda row: (row[1]["t"], row[0]))
    engine = sorted(enumerate(eng), key=lambda row: (row[1]["t"], row[0]))
    dp = [[(0, 0.0, ()) for _ in range(len(engine) + 1)] for _ in range(len(labels) + 1)]
    for i, (label_i, a) in enumerate(labels, 1):
        for j, (engine_j, b) in enumerate(engine, 1):
            options = [dp[i - 1][j], dp[i][j - 1]]
            error = abs(a["t"] - b["t"])
            if a["hitter"] == b["hitter"] and error <= tol:
                count, cost, pairs = dp[i - 1][j - 1]
                options.append((count + 1, cost + error, pairs + ((label_i, engine_j),)))
            dp[i][j] = min(options, key=lambda result: (-result[0], result[1]))
    return list(dp[-1][-1][2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("hits")
    ap.add_argument("--tol", type=float, default=0.25)
    ap.add_argument("--out")
    a = ap.parse_args()
    labels = json.loads(pathlib.Path(a.labels).read_text(encoding="utf-8"))
    eng_all = engine_rows(json.loads(pathlib.Path(a.hits).read_text(encoding="utf-8")))
    tot = {"label": 0, "engine": 0, "matched": 0, "zone_both": 0, "zone_same": 0,
           "depth_same": 0, "family_both": 0, "family_same": 0}
    per = []
    for r in labels["rallies"]:
        ts = [float(h["t"]) for h in r["hits"]]
        lo = (r.get("serve_t") if r.get("serve_t") is not None else min(ts, default=r["window"][0])) - 0.5
        hi = (r.get("end_t") if r.get("end_t") is not None else max(ts, default=r["window"][1])) + 0.5
        lab = [{"t": float(h["t"]), "hitter": h["hitter"], "family": norm_family(h.get("family")),
                "zone": norm_zone(h.get("next_zone"))} for h in r["hits"]]
        eng = [h for h in eng_all if lo <= h["t"] <= hi]
        m = match(lab, eng, a.tol)
        zb = zs = ds = fb = fs = 0
        for i, j in m:
            lz, ez = lab[i]["zone"], eng[j]["zone"]
            if lz and ez:
                zb += 1
                zs += lz == ez
                ds += lz[0] == ez[0]
            lf, ef = lab[i]["family"], eng[j]["family"]
            if lf and ef:
                fb += 1
                fs += lf == ef
        row = {"id": r["id"], "label": len(lab), "engine": len(eng), "matched": len(m),
               "precision": len(m) / len(eng) if eng else None,
               "recall": len(m) / len(lab) if lab else None,
               "zone_both": zb, "zone_same": zs, "depth_same": ds,
               "family_both": fb, "family_same": fs}
        per.append(row)
        for k in tot:
            tot[k] += row[k]
    res = {
        "tol_s": a.tol,
        "precision": tot["matched"] / tot["engine"] if tot["engine"] else None,
        "recall": tot["matched"] / tot["label"] if tot["label"] else None,
        "zone_agreement": tot["zone_same"] / tot["zone_both"] if tot["zone_both"] else None,
        "depth_agreement": tot["depth_same"] / tot["zone_both"] if tot["zone_both"] else None,
        "family_agreement": tot["family_same"] / tot["family_both"] if tot["family_both"] else None,
        "zone_coverage": tot["zone_both"] / tot["matched"] if tot["matched"] else None,
        "family_coverage": tot["family_both"] / tot["matched"] if tot["matched"] else None,
        "totals": tot,
        "per_rally": per,
    }
    text = json.dumps(res, indent=1)
    if a.out:
        pathlib.Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
