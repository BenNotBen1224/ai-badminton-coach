"""Reproducible bounded dev-only weak-joint screening; never reads a holdout."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import acceptance as ac
import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--clips", nargs="+", choices=["01", "02", "03", "04", "05"], default=["01", "03"])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.6, 0.75])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--risk", type=Path)
    ap.add_argument("--hold", type=int, default=8)
    ap.add_argument("--tail-joints", choices=["all", "arms", "wrists"], default="all")
    ap.add_argument("--start-small", action="store_true")
    ap.add_argument("--motion", type=float, default=0)
    ap.add_argument("--mean", type=float, default=1.05)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    source = a.out / "source"
    source.mkdir()
    names = [*ac.CODE_FILES, "tail_candidate.py", "tail_screen.py", "risk_features.py"]
    for name in names:
        shutil.copy2(ac.HERE / name, source / name)
    report = {"scope": "Development screening only; not final acceptance",
              "code_sha256": {n: ac.sha256(source / n) for n in names},
              "host": ac.host_info(), "runtime": ac.runtime_info(), "runs": [],
              "large_hold_inferences": a.hold,
              "tail_joints": a.tail_joints,
              "start_small": a.start_small,
              "motion_margin": a.motion,
              "mean_confidence": a.mean,
              "model_sha256": {str(p): ac.sha256(p) for p in (ac.HERE / "weights").glob("*ft*.onnx")}}
    if a.risk:
        report["risk"] = {"path": str(a.risk.resolve()), "sha256": ac.sha256(a.risk)}
    for repeat in range(a.runs):
        for clip in a.clips:
            video = next((ac.HERE.parent / "sample-videos").glob(f"{clip}-*.mp4"))
            probe = ac.ffprobe_counts(video)
            ref = evaluate.load_reference(ac.HERE / "runs" / f"judge-{clip}.json")
            base = evaluate.score(ac.HERE / "runs" / f"baseline-{clip}.json", ref)
            consensus = evaluate.load_reference() if clip == "01" else None
            for threshold in (a.thresholds if repeat % 2 == 0 else list(reversed(a.thresholds))):
                tag = f"{clip}-tail{threshold}-r{repeat + 1}"
                out = (a.out / f"{tag}.json").resolve()
                load = ac.load_snapshot()
                command = [sys.executable, str(ac.HERE / "tail_candidate.py"),
                           str(video), str(out), "--tail", str(threshold), "--hold", str(a.hold),
                           "--tail-joints", a.tail_joints, "--motion", str(a.motion), "--mean", str(a.mean)]
                if a.risk:
                    command += ["--risk", str(a.risk.resolve())]
                if a.start_small:
                    command += ["--start-small"]
                rec = ac.run_one(command, out, a.out / "logs", tag, probe["frame_count"], probe["fps"],
                                 size=(probe["width"], probe["height"]))
                rec.update(clip=clip, tail=threshold, load_before=load)
                rec["vs_judge"] = ac._safe(lambda: evaluate.score(out, ref))
                rec["quality_gate"] = ac.quality_gate([rec], {clip: base})
                if consensus is not None:
                    rec["secondary"] = evaluate.score(out, consensus)
                report["runs"].append(rec)
                (a.out / "screen.json").write_text(json.dumps(report, indent=1))
                print(json.dumps({k: rec[k] for k in ("tag", "wall_s", "quality_gate")} |
                                 {"secondary_pck05": rec.get("secondary", {}).get("pck05_exact"),
                                  "calls": (rec["internal"] or {}).get("calls")}), flush=True)


if __name__ == "__main__":
    main()
