"""Bounded measurements for the preview service.

usage:
  python bench.py check CLIP [CLIP...]
  python bench.py time CLIP
  python bench.py score RESULT.json JUDGE.json BASELINE.json

`check` and `time` take work/cpu.lock. They do not read sealed references.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent / "work"
LOCK = WORK / "cpu.lock"
LOG = WORK / "S" / "iterations.tsv"
PY_SITE = ROOT / ".venv" / "Lib" / "site-packages"


def _bg():
    return psutil.cpu_percent(interval=1.0)


def acquire(purpose):
    """Atomic exclusive create. Never overwrites a lock this process did not make."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 30 * 60
    while True:
        bg = _bg()
        if bg > 10:
            if time.time() > deadline:
                raise SystemExit(f"background CPU stayed at {bg}")
            print(f"waiting, background CPU {bg:.0f}%", flush=True)
            time.sleep(30)
            continue
        try:
            handle = open(LOCK, "x", encoding="utf-8")
        except FileExistsError:
            if time.time() > deadline:
                raise SystemExit(f"cpu.lock still held: {LOCK.read_text(encoding='utf-8')[:200]}")
            print(f"waiting, cpu.lock held: {LOCK.read_text(encoding='utf-8')[:80]}", flush=True)
            time.sleep(30)
            continue
        handle.write(f"S {os.getpid()} {purpose}\n")
        handle.close()
        return

    raise AssertionError("unreachable")


def release():
    try:
        text = LOCK.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    if text.startswith(f"S {os.getpid()} "):
        LOCK.unlink()


def log_row(hypothesis, change, before, after, kept, evidence):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        LOG.write_text("ts\thypothesis\tchange\tbefore\tafter\tkept\tevidence\n", encoding="utf-8")
    row = "\t".join([
        time.strftime("%Y-%m-%dT%H:%M:%S"),
        hypothesis.replace("\t", " "),
        change.replace("\t", " "),
        str(before), str(after), kept, str(evidence),
    ])
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(row + "\n")


def check(clips):
    import decode
    acquire("align-and-conversion")
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        out = WORK / "S" / "align.json"
        reports = []
        for clip in clips:
            report = decode.align_check(clip, limit_gops=2)
            report["conversion"] = decode.conversion_check(clip)
            reports.append(report)
            print(json.dumps(report), flush=True)
        out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        passed = all(r["pass"] and r["conversion"]["pass"] for r in reports)
        log_row(
            "GOP contexts match sequential pixels at a boundary, and a YUV crop matches to_ndarray",
            "align_check DEFAULT and NONREF, 2 GOPs, max 3 contexts; conversion_check one frame",
            "no pixel check",
            "pass" if passed else "fail",
            "Y" if passed else "N",
            out,
        )
        return 0 if passed else 1
    finally:
        release()


def time_one(clip, method="serial"):
    sys.path.insert(0, str(PY_SITE))
    import service
    acquire(f"preview-time-{method}")
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        svc = service.TurboService(ROOT.parent / "fast" / "weights", decode_method=method)
        t0 = time.perf_counter()
        svc._preview_models()
        svc._touch()
        cold_note = round(time.perf_counter() - t0, 3)
        runs = []
        doc = None
        for i in range(3):
            while _bg() > 10:
                time.sleep(5)
            out_json = WORK / "S" / f"preview-{Path(clip).stem}-{method}-{i}.json"
            doc, timings = svc.analyze(clip, out_json_path=out_json, tier="preview")
            runs.append(timings)
            print(json.dumps({"warm_run": i, **timings, "frames": len(doc["frames"])}), flush=True)
        out = WORK / "S" / f"time-preview-{method}.json"
        payload = {"cold_model_load_s": cold_note, "runs": runs, "clip": Path(clip).name, "method": method,
                   "labelled": "warm analyze after preview models were loaded; refined models were not loaded"}
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        med = sorted(r["total_s"] for r in runs)[len(runs) // 2]
        log_row(
            f"{method} NONREF decode overlapped with pose-t stays under 1s on one clip",
            "gop workers=2 decode threads=1 per context pose streams=4 static batch-1 det threads=2",
            3.8648,
            med,
            "Y" if med < 1.0 else "N",
            out,
        )
        return 0
    finally:
        release()


def _pck(hits, total):
    return None if not total else round(hits / total, 4)


def score_files(result, judge, baseline):
    """Role-aware PCK on judge frames that have people. Repeats are not excluded."""
    import numpy as np
    pred = json.loads(Path(result).read_text(encoding="utf-8"))["frames"]
    ref = json.loads(Path(judge).read_text(encoding="utf-8"))["frames"]
    base = json.loads(Path(baseline).read_text(encoding="utf-8"))
    if len(pred) != len(ref):
        raise SystemExit(f"frame count {len(pred)} != judge {len(ref)}")
    stats = {"all": [0, 0], "fast": [0, 0], "found": [0, 0], "pck05": [0, 0]}
    speeds = []
    for i, (pf, rf) in enumerate(zip(pred, ref)):
        people = {p.get("role"): p for p in rf.get("people", []) if p.get("role")}
        if not people:
            continue
        by_role = {p.get("role"): p for p in pf.get("players", [])}
        for role, person in people.items():
            joints = person["joints"]
            h = person["bbox_xyxy"][3] - person["bbox_xyxy"][1]
            target = np.array([[np.nan, np.nan] if j is None else j[:2] for j in joints], float)
            got = by_role.get(role)
            cand = np.full((17, 2), np.nan) if got is None else np.array(
                [[np.nan, np.nan] if j is None else j[:2] for j in got["joints"]], float)
            valid = np.isfinite(target).all(1)
            dist = np.linalg.norm(cand - target, axis=1)
            stats["all"][0] += int((dist[valid] < 0.1 * h).sum())
            stats["all"][1] += int(valid.sum())
            stats["pck05"][0] += int((dist[valid] < 0.05 * h).sum())
            stats["pck05"][1] += int(valid.sum())
            stats["found"][1] += 1
            stats["found"][0] += int(got is not None and valid.sum() and (dist[valid] < 0.1 * h).mean() >= 0.5)
            speeds.append((i, role, target, h))
    fast_ids = _fast_indexes(speeds)
    for i, role, target, h in speeds:
        if (i, role) not in fast_ids:
            continue
        pf = pred[i]
        got = next((p for p in pf.get("players", []) if p.get("role") == role), None)
        cand = np.full((17, 2), np.nan) if got is None else np.array(
            [[np.nan, np.nan] if j is None else j[:2] for j in got["joints"]], float)
        valid = np.isfinite(target).all(1)
        dist = np.linalg.norm(cand - target, axis=1)
        stats["fast"][0] += int((dist[valid] < 0.1 * h).sum())
        stats["fast"][1] += int(valid.sum())
    report = {
        "pck10": _pck(*stats["all"]),
        "pck05": _pck(*stats["pck05"]),
        "fast_pck10": _pck(*stats["fast"]),
        "found": _pck(*stats["found"]),
        "counts": stats,
        "repeats_excluded": False,
        "baseline_keys": sorted(base)[:12],
    }
    print(json.dumps(report, indent=2))
    return report


def _fast_indexes(speeds):
    """Top quarter of reference-frame joint speed, per the harness definition."""
    import numpy as np
    by_role = {}
    for i, role, target, h in speeds:
        by_role.setdefault(role, []).append((i, target, h))
    chosen = set()
    rates = []
    for role, seq in by_role.items():
        seq.sort()
        for (i0, a, h0), (i1, b, h1) in zip(seq, seq[1:]):
            dt = (i1 - i0) / 30.0
            if dt <= 0:
                continue
            d = np.linalg.norm(b - a, axis=1)
            ok = np.isfinite(d)
            if not ok.any():
                continue
            rate = float(np.nanmax(d) / max(h0, 1) / dt)
            rates.append((rate, i1, role))
    if not rates:
        return chosen
    rates.sort(reverse=True)
    keep = max(1, len(rates) // 4)
    for _rate, i, role in rates[:keep]:
        chosen.add((i, role))
    return chosen


def main():
    cmd = sys.argv[1]
    if cmd == "check":
        raise SystemExit(check(sys.argv[2:]))
    if cmd == "time":
        method = sys.argv[3] if len(sys.argv) > 3 else "serial"
        raise SystemExit(time_one(sys.argv[2], method))
    if cmd == "score":
        score_files(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    raise SystemExit("usage: bench.py check|time|score ...")


if __name__ == "__main__":
    main()
