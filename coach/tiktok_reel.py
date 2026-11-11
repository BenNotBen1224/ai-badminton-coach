"""Generate a source-bound Chinese commentary reel, or re-render an edited plan.

python -m coach.tiktok_reel MATCH OUT --near NAME --far NAME --attribution-json CREDIT
python -m coach.tiktok_reel MATCH NEW_OUT --plan OUT/plan.json
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import platform
from urllib.parse import urlsplit

import cv2

from .app.backend import NoCourt, PROJECT, sha256_file
from .reel_analysis import analyze, normalize
from .reel_media import evidence_sheets, render, verify
from .reel_plan import draft_plan, make_plan, validate_plan
from .export import _tools


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, doc):
    Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def credit(path):
    doc = read(path)
    if "attribution" in doc:
        doc = doc["attribution"]
    result = {
        "author": doc.get("author", doc.get("creator")),
        "license": doc.get("license", doc.get("license_label_on_source")),
        "source_url": doc.get("source_url"),
        "license_url": doc.get("license_url", doc.get("cc_by_url_linked_by_youtube")),
    }
    kind = doc.get("source_kind", "external")
    if kind not in ("external", "self_recorded", "private_permission"):
        raise ValueError("Attribution source_kind must be external, self_recorded or private_permission")
    required = ("author", "license", "source_url", "license_url") if kind == "external" else ("author", "license")
    if any(not isinstance(result[key], str) or not result[key].strip() for key in required):
        raise ValueError("Attribution needs " + ", ".join(required))
    for key in ("source_url", "license_url"):
        if result[key] is None:
            del result[key]
            continue
        value = result[key]
        if not isinstance(value, str) or any(c.isspace() for c in value):
            raise ValueError("Attribution source and licence URLs must use valid HTTPS URLs")
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Attribution source and licence URLs must use valid HTTPS URLs")
    result["source_kind"] = kind
    if kind != "external":
        statement = doc.get("rights_statement")
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError("Local/private attribution needs an explicit rights_statement")
        result["rights_statement"] = statement
    return {**result, "changes": "Excerpt selection, time normalization, slow playback, captions, pose estimates and AI voice",
            "rights_limit": "Rights are supplied by the user, not independently verified; no endorsement implied"}


def run(args):
    start = time.perf_counter()
    code = [Path(__file__), *(Path(__file__).with_name(name) for name in
                              ("reel_analysis.py", "reel_plan.py", "reel_media.py"))]
    code_hashes = {p.name: sha256_file(p) for p in code}
    if not args.analyze_only and not args.font.is_file():
        raise ValueError("Chinese font missing; supply --font with a Chinese TrueType/OpenType font")
    source_sha = sha256_file(args.video)
    ffmpeg, _ = _tools()
    if args.plan:
        plan = read(args.plan)
        validate_plan(plan, source_sha, plan["normalized_frames"])
        assets = {}
        for key in ("normalized", "pose"):
            record = plan["assets"][key]
            asset = (args.plan.parent/record["path"]).resolve()
            if not asset.is_file() or sha256_file(asset) != record["sha256"]:
                raise ValueError(f"Plan {key} evidence missing or changed; cannot rerender without original analysis")
            assets[key] = asset
        normalized, pose = assets["normalized"], read(assets["pose"])
        if pose["source"]["frame_count"] != plan["normalized_frames"]:
            raise ValueError("Plan frame count differs from its bound pose evidence")
        plan["assets"] = {
            key: {"path": os.path.relpath(path, args.out), "sha256": sha256_file(path)}
            for key, path in assets.items()
        }
        timing = {"analysis_reused_from_plan": True}
    else:
        if not args.near or not args.far or args.near == args.far:
            raise ValueError("Supply distinct --near and --far names for the first reliable rally")
        if not args.attribution_json:
            raise ValueError("Supply --attribution-json; unknown videos never inherit sample footage rights")
        attribution = credit(args.attribution_json)
        normalized = args.out/"normalized.mp4"
        normalize_start = time.perf_counter()
        normalize(args.video, normalized, ffmpeg)
        timing = {"normalize_s": time.perf_counter()-normalize_start}
        pose, analysis = analyze(normalized, args.out, args.models, ffmpeg,
                                 lambda text: print(text, flush=True))
        write(args.out/"pose.json", pose)
        write(args.out/"analysis.json", analysis)
        assets = {
            "normalized": {"path": "normalized.mp4", "sha256": sha256_file(normalized)},
            "pose": {"path": "pose.json", "sha256": sha256_file(args.out/"pose.json")},
        }
        write(args.out/"plan.draft.json", draft_plan(
            analysis, {"near": args.near, "far": args.far}, attribution, source_sha, assets))
        evidence_sheets(normalized, analysis["hits"], args.out)
        if args.analyze_only:
            write(args.out/"run.json", {"source_sha256": source_sha, "wall_s": time.perf_counter()-start,
                                       "status": "analysis_only", "timings": timing})
            return
        plan = make_plan(analysis, pose, normalized, {"near": args.near, "far": args.far},
                         attribution, source_sha, read(args.sides) if args.sides else None)
        plan["normalized_frames"] = analysis["frames"]
        plan["assets"] = assets
        timing["analysis"] = analysis["timings"]
    write(args.out/"plan.json", plan)
    stage = time.perf_counter()
    rows, captions = render(normalized, pose, plan, args.out,
                            args.models/"kokoro-multi-lang-v1_1", args.font, ffmpeg)
    timing["voice_render_s"] = time.perf_counter()-stage
    write(args.out/"frame-map.json", {"fps": 30, "frames": rows, "captions": captions})
    bindings = {
        "source_sha256": source_sha, "plan_sha256": sha256_file(args.out/"plan.json"),
        "voice_model_sha256": sha256_file(args.models/"kokoro-multi-lang-v1_1/model.onnx"),
        "voice": {"engine": "sherpa-onnx", "model": "Kokoro v1.1 multi-lang", "sid": 60, "speed": 1.0},
        "voice_inputs_sha256": {name: sha256_file(args.models/"kokoro-multi-lang-v1_1"/name)
                               for name in ("model.onnx", "voices.bin", "tokens.txt", "lexicon-us-en.txt", "lexicon-zh.txt")},
        "code_sha256": code_hashes,
        "environment": {"python": platform.python_version(),
                        "packages": {name: importlib.metadata.version(name) for name in
                                     ("sherpa-onnx", "onnxruntime", "numpy", "torch", "openvino", "av", "opencv-python")}},
    }
    write(args.out/"render-inputs.json", bindings)
    stage = time.perf_counter()
    checked = verify(args.out/"tiktok.mp4", args.out/"verification-inputs.npz", rows, captions, ffmpeg)
    if code_hashes != {p.name: sha256_file(p) for p in code}:
        raise ValueError("Renderer source changed during this run; keep this output as diagnostic only and run again")
    checked["output_sha256"] = sha256_file(args.out/"tiktok.mp4")
    checked["hash_scope"] = "This generated run only; fresh speech and MP4 bytes can differ"
    write(args.out/"verification.json", checked)
    timing["verify_s"] = time.perf_counter()-stage
    write(args.out/"run.json", {"source_sha256": source_sha, "wall_s": time.perf_counter()-start,
                               "status": "encoded_and_verified", "timings": timing,
                               "plan_entry": "supplied_plan" if args.plan else "automatic_selection",
                               "review_status": plan["review_status"],
                               "reviewer_kind": plan.get("reviewer_kind"), "reviewer": plan.get("reviewer")})
    print(json.dumps({"output": str(args.out/"tiktok.mp4"), "frames": checked["frames"],
                      "wall_s": time.perf_counter()-start}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--near")
    parser.add_argument("--far")
    parser.add_argument("--attribution-json", type=Path)
    parser.add_argument("--plan", type=Path, help="Edited source-bound plan; skips analysis, regenerates measured voice")
    parser.add_argument("--sides", type=Path, help="Reviewed list of start_s/end_s/near/far assignments")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--models", type=Path, default=PROJECT/"models")
    parser.add_argument("--font", type=Path, default=Path(os.environ.get("WINDIR", r"C:\Windows"))/"Fonts/msyh.ttc")
    parser.add_argument("--lock", type=Path, default=PROJECT/"work/cpu.lock")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.video, args.out = args.video.resolve(), args.out.resolve()
    if args.out.exists():
        parser.exit(1, "Cannot produce commentary reel: output directory already exists; previous run left unchanged\n")
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    from scripts import with_lock
    with_lock.LOCK = args.lock
    marker = f"tiktok-reel {os.getpid()} {args.out.name}"
    with_lock.acquire(marker)
    created = False
    try:
        args.out.mkdir(parents=True, exist_ok=False)
        created = True
        run(args)
    except (ValueError, FileNotFoundError, FileExistsError, NoCourt, subprocess.CalledProcessError) as exc:
        if created:
            write(args.out/"failure.json", {"error": str(exc), "video_produced": (args.out/"tiktok.mp4").exists()})
        parser.exit(1, f"Cannot produce commentary reel: {exc}\n")
    finally:
        if args.lock.exists() and args.lock.read_text(encoding="utf-8") == marker:
            args.lock.unlink()


if __name__ == "__main__":
    main()
