"""Interleaved dev-only comparisons; retain every process output, failure and score."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import acceptance as ac
import evaluate
from pose import DEFAULT_POSE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--clip", default="03")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--variant", action="append", required=True, help="NAME|extra CLI arguments")
    ap.add_argument("--script", action="append", default=[], help="NAME|saved pipeline source for a historical variant")
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    source = a.out / "source"
    source.mkdir()
    for name in ac.CODE_FILES:
        shutil.copy2(ac.HERE / name, source / name)
    video = next((ac.HERE.parent / "sample-videos").glob(f"{a.clip}-*.mp4"))
    judge = ac.HERE / "runs" / f"judge-{a.clip}.json"
    baseline = ac.HERE / "runs" / f"baseline-{a.clip}.json"
    reference = evaluate.load_reference(judge)
    baseline_score = evaluate.score(baseline, reference)
    probe = ac.ffprobe_counts(video)
    variants = [v.split("|", 1) for v in a.variant]
    scripts = {name: Path(path).resolve() for name, path in (s.split("|", 1) for s in a.script)}
    models = {}
    for name, args in variants:
        args = args.split()
        model = Path(args[args.index("--pose") + 1]).resolve() if "--pose" in args else DEFAULT_POSE
        models[name] = {"pose": str(model), "sha256": ac.sha256(model)}
        if "--pose-small" in args:
            small = Path(args[args.index("--pose-small") + 1]).resolve()
            models[name]["pose_small"] = str(small)
            models[name]["pose_small_sha256"] = ac.sha256(small)
    report = {
        "claim": "Dev-only candidate must reduce end-to-end wall time and not lose baseline-level judge agreement.",
        "clip": a.clip, "video": str(video), "video_sha256": ac.sha256(video),
        "code_sha256": {n: ac.sha256(source / n) for n in ac.CODE_FILES},
        "model_sha256": ac.sha256(ac.HERE / "weights" / "rtmpose-s-ft1.onnx"),
        "variant_models": models,
        "variant_pipeline_sha256": {name: ac.sha256(scripts.get(name, ac.HERE / "pipeline.py"))
                                   for name, _ in variants},
        "runtime": ac.runtime_info(), "host": ac.host_info(),
        "baseline_vs_judge": baseline_score, "runs": [],
    }
    (a.out / "outputs").mkdir()
    for r in range(1, a.runs + 1):
        k = (r - 1) % len(variants)
        for name, args in variants[k:] + variants[:k]:
            tag = f"{name}-r{r}"
            output = a.out / "outputs" / f"{tag}.json"
            load = ac.load_snapshot()
            cache_before = ac.cache_state(ac.HERE / "cache")
            command = [sys.executable, str(ac.HERE / "pipeline.py")]
            if name in scripts:
                # Preserve the live model/cache paths while executing the retained historical source.
                loader = ("import sys;from pathlib import Path;p=Path(sys.argv.pop(1));"
                          "sys.argv[0]=sys.argv.pop(1);"
                          "exec(compile(p.read_text('utf-8'),sys.argv[0],'exec'),"
                          "{'__name__':'__main__','__file__':sys.argv[0]})")
                command = [sys.executable, "-c", loader, str(scripts[name]), str(ac.HERE / "pipeline.py")]
            rec = ac.run_one(
                [*command, str(video), str(output), *args.split()],
                output, a.out / "logs", tag, probe["frame_count"], probe["fps"])
            rec.update(clip=a.clip, variant=name, load_before=load,
                       cache_unchanged=cache_before == ac.cache_state(ac.HERE / "cache"))
            rec["vs_judge"] = ac._safe(lambda: evaluate.score(output, reference))
            rec["quality_gate"] = ac.quality_gate([rec], {a.clip: baseline_score})
            report["runs"].append(rec)
            (a.out / "comparison.json").write_text(json.dumps(report, indent=1), "utf-8")
            print(json.dumps({key: rec[key] for key in
                              ("tag", "wall_s", "valid", "quality_gate", "internal")}), flush=True)
    report["summary"] = {
        name: {
            "wall": ac.summarize_walls([r for r in report["runs"] if r["variant"] == name]),
            "quality": ac.quality_gate([r for r in report["runs"] if r["variant"] == name],
                                      {a.clip: baseline_score}),
        } for name, _ in variants
    }
    (a.out / "comparison.json").write_text(json.dumps(report, indent=1), "utf-8")
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
