"""Reproducible acceptance run for the fast pipeline: speed gate + accuracy against the model reference.

Every measured run is a fresh `pipeline.py VIDEO OUT.json` process. The timer covers process start
to exit, so it includes Python imports, decode, automatic court finding, inference/tracking and the
JSON save. It excludes one-time setup (hashing, probing, cache listing), the OpenVINO cache
preparation and any MP4 overlay rendering. Each persisted output is kept and validated; a run
passes only when it exited 0, its JSON is complete and well formed, and wall time is under the limit.
Scoring happens after all timing, so nothing competes with the timed processes.

usage (comparison venv):
  python acceptance.py run --suite dev --out runs/acceptance-dev --runs 5 \
      --clip 01|../sample-videos/01-singles-rally.mp4|runs/judge-01.json|runs/baseline-01.json ...
  python acceptance.py merge runs/acceptance-dev/acceptance.json runs/acceptance-clip06/acceptance.json -o runs/acceptance.json
"""
import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMIT_S = 10.0
RUN_TIMEOUT_S = 60
CODE_FILES = ["pipeline.py", "court.py", "detect.py", "pose.py", "ovcache.py", "evaluate.py", "acceptance.py",
              "prepare_cache.py"]
FFPROBE = HERE.parent / "tools" / "ffprobe.exe"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_output(doc, frame_count, fps, size=None):
    """Problems found in a pipeline JSON document; empty list means it is complete and well formed."""
    problems = []
    if not isinstance(doc, dict) or not isinstance(doc.get("frames"), list):
        return ["no frames list"]
    frames = doc["frames"]
    src = doc.get("source") or {}
    if not isinstance(src, dict):
        return ["source is not an object"]
    if src.get("frame_count") != frame_count:
        problems.append(f"header frame_count {src.get('frame_count')} != expected {frame_count}")
    if not _finite(src.get("fps")) or abs(src["fps"] - fps) > 0.01:
        problems.append(f"header fps {src.get('fps')} != expected {fps}")
    if len(frames) != frame_count:
        problems.append(f"{len(frames)} frames saved, expected {frame_count}")
    if size and (src.get("width"), src.get("height")) != tuple(size):
        problems.append(f"source size does not match {tuple(size)}")
    bad = 0
    any_player = False
    for pos, f in enumerate(frames):
        errs = []
        if not isinstance(f, dict):
            problems.append(f"frame {pos} is not an object")
            continue
        if type(f.get("frame_index")) is not int or f["frame_index"] != pos:
            errs.append(f"frame_index {f.get('frame_index')} at position {pos}")
        ts = f.get("timestamp_seconds")
        if not _finite(ts) or abs(ts - pos / fps) > 1e-3:
            errs.append(f"timestamp {ts} at frame {pos}")
        players = f.get("players")
        if not isinstance(players, list):
            errs.append("players not a list")
            players = []
        ids = set()
        for p in players:
            any_player = True
            if not isinstance(p, dict):
                errs.append("player is not an object")
                continue
            tid = p.get("track_id")
            if type(tid) is not int or tid in ids:
                errs.append("missing, invalid or duplicate track_id")
            else:
                ids.add(tid)
            if not _finite(p.get("box_confidence")):
                errs.append("non-finite or missing box confidence")
            box = p.get("bbox_xyxy")
            if not (isinstance(box, list) and len(box) == 4 and all(_finite(v) for v in box)):
                errs.append("bad bbox")
            elif box[0] >= box[2] or box[1] >= box[3]:
                errs.append("empty or inverted bbox")
            joints = p.get("joints")
            if not (isinstance(joints, list) and len(joints) == 17):
                errs.append("joints are not 17 entries")
                continue
            for j in joints:
                if j is not None and not (isinstance(j, list) and len(j) == 3 and all(_finite(v) for v in j)):
                    errs.append("non-finite or malformed joint")
                    break
        if errs:
            bad += 1
            if bad <= 5:
                problems.append(f"frame {pos}: {'; '.join(sorted(set(errs)))}")
    if bad > 5:
        problems.append(f"{bad} frames with problems in total")
    if frames and not any_player:
        problems.append("no player in any frame")
    return problems


def run_one(cmd, out_json, log_dir, tag, frame_count, fps, limit=LIMIT_S, timeout=RUN_TIMEOUT_S, size=None):
    """One fresh-process run. Nothing is deleted; wall time is an unrounded integer of nanoseconds."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    so, se = log_dir / f"{tag}.stdout.txt", log_dir / f"{tag}.stderr.txt"
    rec = {"tag": tag, "command": [str(c) for c in cmd], "output": str(out_json),
           "cwd": str(Path.cwd()),
           "started_utc": datetime.now(timezone.utc).isoformat()}
    timed_out = False
    with open(so, "wb") as fo, open(se, "wb") as fe:
        t0 = time.perf_counter_ns()
        p = subprocess.Popen(cmd, stdout=fo, stderr=fe)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
            rc = p.wait()
        wall_ns = time.perf_counter_ns() - t0
    rec.update(wall_ns=wall_ns, wall_s=wall_ns / 1e9, returncode=rc, timed_out=timed_out,
               stdout_log=str(so), stderr_log=str(se), stdout_sha256=sha256(so), stderr_sha256=sha256(se))
    problems = []
    if timed_out:
        problems.append(f"timed out after {timeout}s")
    elif rc != 0:
        problems.append(f"exit code {rc}")
    out_json = Path(out_json)
    if out_json.is_file():
        rec.update(output_bytes=out_json.stat().st_size, output_sha256=sha256(out_json))
        try:
            doc = json.loads(out_json.read_text("utf-8"))
            problems += validate_output(doc, frame_count, fps, size)
        except (ValueError, OSError) as e:
            problems.append(f"output unreadable: {e}")
    else:
        problems.append("output JSON was not saved")
    try:
        rec["internal"] = json.loads(so.read_text("utf-8").strip().splitlines()[-1])
    except (ValueError, IndexError, OSError):
        rec["internal"] = None
    rec["validation_problems"] = problems
    rec["valid"] = not problems
    rec["under_limit"] = (not timed_out) and wall_ns / 1e9 < limit
    rec["pass"] = rec["valid"] and rec["under_limit"]
    return rec


def gate(runs):
    """The speed/validity gate: every measured run counts, none is filtered."""
    failed = [r["tag"] for r in runs if not r["pass"]]
    return {"runs": len(runs), "failed": len(failed), "failed_tags": failed, "passed": bool(runs) and not failed}


QUALITY_METRICS = ("pck05", "pck10", "player_frames_found")


def quality_gate(runs, baselines):
    """Primary accuracy gate: on every clip, every run's PCK05, PCK10 and found rate vs the judge must be
    >= the baseline's own score vs the same judge. Compared unrounded: evaluate.score returns the raw
    hit and total counts under `counts` for exactly this."""
    fails = []
    for r in runs:
        base = baselines.get(r["clip"], {})
        s = r.get("vs_judge") or {}
        for m in QUALITY_METRICS:
            f, b = s.get(f"{m}_exact"), base.get(f"{m}_exact")
            if not _finite(f) or not _finite(b):
                fails.append(f"{r['tag']} {m}: score missing or non-finite")
                continue
            fc, bc = s.get("counts", {}).get(m), base.get("counts", {}).get(m)
            inferior = f < b
            if fc is not None and bc is not None:
                if not all(type(v) is int and v >= 0 for v in [*fc, *bc]) or fc[1] == 0 or bc[1] == 0:
                    fails.append(f"{r['tag']} {m}: invalid score counts")
                    continue
                inferior = fc[0] * bc[1] < bc[0] * fc[1]
            if inferior:
                fails.append(f"{r['tag']} {m}: {f:.6f} < baseline {b:.6f}")
    return {"passed": bool(runs) and not fails, "failures": fails}


def summarize_walls(runs):
    w = sorted(r["wall_s"] for r in runs)
    if not w:
        return {}
    mid = len(w) // 2
    return {"n": len(w), "min": w[0], "median": w[mid] if len(w) % 2 else (w[mid - 1] + w[mid]) / 2, "max": w[-1]}


def ffprobe_counts(video):
    out = subprocess.run([str(FFPROBE), "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_frames,r_frame_rate,width,height,duration",
                          "-of", "json", str(video)], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    a, b = s["r_frame_rate"].split("/")
    return {"frame_count": int(s["nb_read_frames"]), "fps": int(a) / int(b), "width": s["width"],
            "height": s["height"], "duration_s": float(s["duration"])}


def host_info():
    import psutil
    cpu = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Processor | Select -First 1).Name"],
                         capture_output=True, text=True).stdout.strip()
    return {"cpu": cpu, "logical_cores": psutil.cpu_count(), "physical_cores": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1), "platform": platform.platform()}


def runtime_info():
    import cv2, numpy, onnxruntime, openvino, psutil
    return {"python": sys.version, "executable": sys.executable, "numpy": numpy.__version__, "cv2": cv2.__version__,
            "openvino": openvino.__version__, "onnxruntime": onnxruntime.__version__, "psutil": psutil.__version__}


def load_snapshot():
    import psutil
    procs, unavailable = [], 0
    for p in psutil.process_iter(["name"]):
        if p.pid == 0:
            continue
        try:
            p.cpu_percent()
            procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            unavailable += 1
    cpu = psutil.cpu_percent(interval=1.0)
    busy = []
    for p in procs:
        try:
            usage = p.cpu_percent()
            if usage:
                busy.append((usage, p.info["name"], p.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            unavailable += 1
    return {"cpu_percent": cpu, "sample_seconds": 1.0, "loadavg": psutil.getloadavg(),
            "top_processes": sorted(busy, reverse=True)[:5], "unavailable_processes": unavailable}


def cache_state(cache_dir):
    return {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(Path(cache_dir).glob("*")) if p.is_file()}


def score_all(runs, clips):
    """Per-run score vs the judge, plus the baseline vs the same judge and fast vs baseline."""
    import evaluate
    scored = {}
    for name, c in clips.items():
        ref_judge = evaluate.load_reference(c["judge"])
        c["baseline_vs_judge"] = _safe(lambda: evaluate.score(c["baseline"], ref_judge))
        ref_base, dropped = baseline_reference(c["judge"], c["baseline"])
        c["baseline_reference_players_dropped"] = dropped
        scored[name] = (ref_judge, ref_base)
    for r in runs:
        ref_judge, ref_base = scored[r["clip"]]
        r["vs_judge"] = _safe(lambda: evaluate.score(r["output"], ref_judge))
        r["vs_baseline"] = _safe(lambda: evaluate.score(r["output"], ref_base))


def _safe(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - a scoring failure is recorded, not hidden
        return {"error": f"{type(e).__name__}: {e}"}


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def baseline_reference(judge_path, baseline_path):
    """The judge's players, but with the baseline's skeleton for each (box overlap >= 0.3). Roles the baseline
    never found are dropped, so 'vs baseline' measures agreement where the baseline produced something."""
    import evaluate
    judge = json.loads(Path(judge_path).read_text("utf-8"))
    base = json.loads(Path(baseline_path).read_text("utf-8"))["frames"]
    if len(base) != len(judge["frames"]):
        raise ValueError("baseline and judge have different frame counts")
    dropped = 0
    for f, bf in zip(judge["frames"], base):
        keep = []
        for p in f["people"]:
            best = max(((iou(p["bbox_xyxy"], q["bbox_xyxy"]), q) for q in bf["players"]), default=(0, None),
                       key=lambda t: t[0])
            if best[0] >= 0.3:
                j = [None if x is None else x[:2] for x in best[1]["joints"]]
                keep.append({**p, "joints": j, "sapiens": j, "posewarper": j})
            else:
                dropped += 1
        f["people"] = keep
    tmp = Path(judge_path).with_suffix(".vs-baseline.tmp.json")
    tmp.write_text(json.dumps(judge), "utf-8")
    try:
        return evaluate.load_reference(tmp), dropped
    finally:
        tmp.unlink()


def metric_ranges(runs, key):
    out = {}
    for m in ("pck05", "pck10", "player_frames_found", "near_found", "far_found"):
        v = sorted(r[key][m] for r in runs if key in r and m in r[key])
        if v:
            out[m] = {"min": v[0], "median": v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2,
                      "max": v[-1]}
    return out


def cmd_run(a):
    from pose import DEFAULT_POSE, DEFAULT_SMALL_CONFIDENCE, DEFAULT_SMALL_MIN_JOINT_CONFIDENCE, DEFAULT_SMALL_POSE
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; acceptance results are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    clips = {}
    for spec in a.clip:
        name, video, judge, base = spec.split("|")
        video, judge, base = (Path(x).resolve() for x in (video, judge, base))
        clips[name] = {"video": str(video), "judge": str(judge), "baseline": str(base)}
    result = {"suite": a.suite, "created_utc": datetime.now(timezone.utc).isoformat(), "limit_s": a.limit,
              "timer": "perf_counter_ns around Popen..exit of a fresh `python pipeline.py VIDEO OUT.json`; includes "
                       "imports, decode, automatic court finding, inference/tracking and JSON save; excludes setup, "
                       "OpenVINO cache preparation, overlay rendering",
              "reference_note": "judge = YOLOX-x + RTMPose-x model output, not human ground truth; its three "
                                "joints/sapiens/posewarper slots are duplicates of one model's joints",
              "runs": [], "clips": clips}
    # setup, untimed
    pipeline = HERE / "pipeline.py"
    pose = Path(a.pose).resolve() if a.pose else DEFAULT_POSE
    for path, want in ((pipeline, a.expect_pipeline_sha), (pose, a.expect_pose_sha)):
        if want and sha256(path) != want.upper():
            sys.exit(f"{path} does not match the frozen SHA256 {want}")
    result["setup"] = {"code_sha256": {n: sha256(HERE / n) for n in CODE_FILES},
                       "pose_model": str(pose), "pose_model_sha256": sha256(pose),
                       "frozen_expectations": {"pipeline": a.expect_pipeline_sha, "pose": a.expect_pose_sha},
                       "runtime": runtime_info(), "host": host_info(),
                       "cache_dir": str(HERE / "cache"), "cache_before": cache_state(HERE / "cache"),
                       "warmup_runs": 0}
    result["setup"]["model_sha256"] = {
        str(p): sha256(p) for p in [pose, HERE / "weights" / "yolo26s-448x768.onnx",
                                   HERE / "weights" / "yolo26n-384x640.onnx"]}
    if not a.single_model:
        result["setup"]["model_sha256"][str(DEFAULT_SMALL_POSE)] = sha256(DEFAULT_SMALL_POSE)
    result["setup"]["production_config"] = {
        "primary_pose": str(pose),
        "small_pose": None if a.single_model else str(DEFAULT_SMALL_POSE),
        "small_min_mean_confidence": None if a.single_model else DEFAULT_SMALL_CONFIDENCE,
        "small_min_joint_confidence": None if a.single_model else DEFAULT_SMALL_MIN_JOINT_CONFIDENCE,
        "large_hold_inferences": None if a.single_model else 8,
    }
    source = out / "source"
    source.mkdir()
    for name in CODE_FILES:
        shutil.copy2(HERE / name, source / name)
    for name, c in clips.items():
        c.update(probe=ffprobe_counts(c["video"]), video_sha256=sha256(c["video"]),
                 judge_sha256=sha256(c["judge"]), baseline_sha256=sha256(c["baseline"]))
    names = list(clips)
    order = []
    for r in range(1, a.runs + 1):  # interleaved; the clip order rotates each round
        k = (r - 1) % len(names)
        order += [(n, r) for n in names[k:] + names[:k]]
    result["order"] = [f"{n}-r{r}" for n, r in order]
    path = out / "acceptance.json"
    for n, r in order:
        c = clips[n]
        tag = f"{n}-r{r}"
        (out / "outputs").mkdir(exist_ok=True)
        load = load_snapshot()
        command = [sys.executable, str(pipeline), c["video"], str(out / "outputs" / f"{tag}.json")]
        if a.pose:
            command += ["--pose", str(pose)]
        if a.single_model:
            command += ["--single-model"]
        rec = run_one(command,
                      out / "outputs" / f"{tag}.json", out / "logs", tag, c["probe"]["frame_count"], c["probe"]["fps"],
                      a.limit, size=(c["probe"]["width"], c["probe"]["height"]))
        rec.update(clip=n, run=r, load_before=load)
        result["runs"].append(rec)
        path.write_text(json.dumps(result, indent=1), "utf-8")
        print(json.dumps({k: rec[k] for k in ("tag", "wall_s", "returncode", "valid", "pass")}), flush=True)
    result["setup"]["cache_after"] = cache_state(HERE / "cache")
    result["setup"]["cache_unchanged"] = result["setup"]["cache_before"] == result["setup"]["cache_after"]
    result["setup"]["code_unchanged"] = result["setup"]["code_sha256"] == {
        n: sha256(HERE / n) for n in CODE_FILES}
    result["setup"]["models_unchanged"] = all(sha256(p) == h for p, h in result["setup"]["model_sha256"].items())
    for rec in result["runs"]:
        if rec.get("output_sha256"):
            rec["output_sha256_recheck"] = sha256(rec["output"])
    score_all(result["runs"], clips)
    if "01" in clips:
        import evaluate
        secondary_ref = evaluate.load_reference()
        secondary_base = evaluate.score(clips["01"]["baseline"], secondary_ref)
        secondary_runs = [{"tag": r["tag"], "clip": "01",
                           "vs_judge": evaluate.score(r["output"], secondary_ref)}
                          for r in result["runs"] if r["clip"] == "01"]
        result["secondary_consensus"] = {
            "note": "Separate real earlier Sapiens/PoseWarper reference, not the primary judge",
            "reference": str(evaluate.REF), "reference_sha256": sha256(evaluate.REF),
            "baseline": secondary_base, "runs": secondary_runs,
            "gate": quality_gate(secondary_runs, {"01": secondary_base})}
    result["gate"] = gate(result["runs"])
    result["quality_gate"] = quality_gate(result["runs"], {n: c["baseline_vs_judge"] for n, c in clips.items()})
    result["gate"]["speed_and_validity_passed"] = result["gate"]["passed"]
    result["gate"]["frozen_setup_passed"] = all(result["setup"][k] for k in
                                                ("cache_unchanged", "code_unchanged", "models_unchanged"))
    result["gate"]["passed"] = (result["gate"]["passed"] and result["quality_gate"]["passed"]
                               and result["gate"]["frozen_setup_passed"]
                               and result.get("secondary_consensus", {}).get("gate", {"passed": True})["passed"])
    result["timing_by_clip"] = {n: summarize_walls([r for r in result["runs"] if r["clip"] == n]) for n in names}
    result["timing_all"] = summarize_walls(result["runs"])
    result["accuracy_by_clip"] = {
        n: {"fast_vs_judge": metric_ranges([r for r in result["runs"] if r["clip"] == n], "vs_judge"),
            "fast_vs_baseline": metric_ranges([r for r in result["runs"] if r["clip"] == n], "vs_baseline"),
            "baseline_vs_judge": {m: clips[n]["baseline_vs_judge"].get(m) for m in
                                  ("pck05", "pck10", "player_frames_found", "near_found", "far_found")}}
        for n in names}
    path.write_text(json.dumps(result, indent=1), "utf-8")
    print(json.dumps({"gate": result["gate"], "timing_all": result["timing_all"]}))
    sys.exit(0 if result["gate"]["passed"] else 1)


def cmd_post(a):
    """Exact-count quality gate for a finished suite, plus the clip-01 secondary check against the earlier
    independent Sapiens/PoseWarper consensus (reference/consensus.json), which is NOT the primary judge."""
    import evaluate
    doc = json.loads(Path(a.suite_json).read_text("utf-8"))
    base_scores, judge_refs = {}, {}
    for n, c in doc["clips"].items():
        judge_refs[n] = evaluate.load_reference(c["judge"])
        base_scores[n] = evaluate.score(c["baseline"], judge_refs[n])
    for r in doc["runs"]:
        r["vs_judge"] = evaluate.score(r["output"], judge_refs[r["clip"]])
    post = {"source": str(Path(a.suite_json).resolve()), "source_sha256": sha256(a.suite_json),
            "quality_gate": quality_gate(doc["runs"], base_scores),
            "per_run_vs_judge_exact": {r["tag"]: {m: r["vs_judge"][m + "_exact"] for m in QUALITY_METRICS}
                                       for r in doc["runs"]},
            "baseline_vs_judge_exact": {n: {m: s[m + "_exact"] for m in QUALITY_METRICS}
                                        for n, s in base_scores.items()}}
    if a.consensus_clip and a.consensus_clip in doc["clips"]:
        ref = evaluate.load_reference(evaluate.REF)
        c = doc["clips"][a.consensus_clip]
        keep = ("pck05", "pck10", "pck10_vs_sapiens", "pck10_vs_posewarper", "player_frames_found")
        post["secondary_consensus"] = {
            "note": "clip 01 only; consensus.json holds real earlier Sapiens/PoseWarper predictions (independent "
                    "of the RTMPose-x judge) -- PCK vs 'joints' is the consensus joints slot",
            "reference_sha256": sha256(evaluate.REF),
            "baseline": {k: evaluate.score(c["baseline"], ref)[k] for k in keep},
            "fast_runs": {r["tag"]: {k: evaluate.score(r["output"], ref)[k] for k in keep}
                          for r in doc["runs"] if r["clip"] == a.consensus_clip}}
    Path(a.o).write_text(json.dumps(post, indent=1), "utf-8")
    print(json.dumps({"quality_gate": post["quality_gate"]}))


def merge_checks(docs):
    """Reject mixed revisions/duplicate IDs and recompute all primary gates."""
    if not docs:
        raise ValueError("no suites")
    keys = ("code_sha256", "model_sha256", "production_config")
    frozen = {k: docs[0]["setup"][k] for k in keys}
    suites, tags = set(), set()
    gates = {}
    for doc in docs:
        if {k: doc["setup"][k] for k in keys} != frozen:
            raise ValueError("cannot merge different code/model/config revisions")
        name = doc["suite"]
        if name in suites:
            raise ValueError(f"duplicate suite: {name}")
        suites.add(name)
        for r in doc["runs"]:
            if r["tag"] in tags:
                raise ValueError(f"duplicate run ID: {r['tag']}")
            tags.add(r["tag"])
        speed = gate(doc["runs"])
        quality = quality_gate(doc["runs"], {n: c["baseline_vs_judge"] for n, c in doc["clips"].items()})
        enough = bool(doc["clips"]) and all(
            sum(r["clip"] == n for r in doc["runs"]) >= 5 for n in doc["clips"])
        frozen_ok = all(doc["setup"][k] for k in ("cache_unchanged", "code_unchanged", "models_unchanged"))
        secondary = doc.get("secondary_consensus")
        secondary_ok = not secondary or quality_gate(secondary["runs"], {"01": secondary["baseline"]})["passed"]
        gates[name] = {"passed": speed["passed"] and quality["passed"] and enough and frozen_ok and secondary_ok,
                       "speed_and_validity": speed, "quality": quality,
                       "secondary_passed": secondary_ok if secondary else None,
                       "five_runs_per_clip": enough, "frozen_setup_passed": frozen_ok}
    return gates


def cmd_merge(a):
    parts = []
    for p in a.inputs:
        parts.append({"path": str(Path(p).resolve()), "sha256": sha256(p), "doc": json.loads(Path(p).read_text("utf-8"))})
    checked = merge_checks([d["doc"] for d in parts])
    combined = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "suites": {d["doc"]["suite"]: {"file": d["path"], "sha256": d["sha256"], "gate": d["doc"]["gate"],
                                               "timing_by_clip": d["doc"]["timing_by_clip"],
                                               "accuracy_by_clip": d["doc"]["accuracy_by_clip"]} for d in parts},
                "recomputed_gates": checked,
                "all_gates_passed": all(g["passed"] for g in checked.values()),
                "full_runs": "see each suite file: every run's command, wall_ns, hashes, logs, validation, scores"}
    Path(a.o).write_text(json.dumps(combined, indent=1), "utf-8")
    print(f"wrote {a.o}; all_gates_passed={combined['all_gates_passed']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--suite", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--runs", type=int, default=5)
    r.add_argument("--limit", type=float, default=LIMIT_S)
    r.add_argument("--clip", action="append", required=True, help="NAME|VIDEO|JUDGE.json|BASELINE.json")
    r.add_argument("--expect-pipeline-sha")
    r.add_argument("--expect-pose-sha")
    r.add_argument("--pose", help="explicit candidate pose model; omit to measure the normal CLI defaults")
    r.add_argument("--single-model", action="store_true")
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("post")
    p.add_argument("suite_json")
    p.add_argument("-o", required=True)
    p.add_argument("--consensus-clip")
    p.set_defaults(fn=cmd_post)
    m = sub.add_parser("merge")
    m.add_argument("inputs", nargs="+")
    m.add_argument("-o", required=True)
    m.set_defaults(fn=cmd_merge)
    args = ap.parse_args()
    args.fn(args)
