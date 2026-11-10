"""Recompute saved-output gates and verify frozen assets/render bindings without new inference."""
import argparse
import json
import subprocess
from pathlib import Path

import acceptance as ac
import evaluate


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(freeze_path, suite_paths, render_path):
    freeze = json.loads(freeze_path.read_text("utf-8"))
    for path, digest in freeze["sha256"].items():
        require(ac.sha256(path) == digest, f"Frozen file changed: {path}")
    for path, digest in freeze["source_snapshots"].items():
        require(ac.sha256(path) == digest, f"Frozen source snapshot changed: {path}")
    require(ac.cache_state(ac.HERE / "cache") == freeze["cache"], "Frozen cache changed")
    docs = [json.loads(p.read_text("utf-8")) for p in suite_paths]
    ac.merge_checks(docs)
    tags, probes, checked = set(), {}, []
    for doc in docs:
        primary, baselines = {}, {}
        for clip, c in doc["clips"].items():
            for name in ("video", "judge", "baseline"):
                require(ac.sha256(c[name]) == c[name + "_sha256"], f"{clip} {name} changed")
            if c["video"] not in probes:
                probes[c["video"]] = ac.ffprobe_counts(c["video"])
            require(probes[c["video"]] == c["probe"], f"{clip} decoded video metadata changed")
            primary[clip] = evaluate.load_reference(c["judge"])
            baselines[clip] = evaluate.score(c["baseline"], primary[clip])
        secondary = evaluate.load_reference() if "01" in doc["clips"] else None
        if secondary is not None:
            s = doc["secondary_consensus"]
            require(ac.sha256(s["reference"]) == s["reference_sha256"], "Secondary reference changed")
        secondary_base = evaluate.score(doc["clips"]["01"]["baseline"], secondary) if secondary else None
        rescored, secondary_runs = [], []
        for r in doc["runs"]:
            require(r["tag"] not in tags, f"Duplicate tag: {r['tag']}")
            tags.add(r["tag"])
            require(r["returncode"] == 0 and not r["timed_out"], f"Failed process: {r['tag']}")
            require(r["wall_ns"] < 10_000_000_000, f"Slow process: {r['tag']}")
            require(r["wall_ns"] / 1e9 == r["wall_s"], f"Inconsistent wall seconds: {r['tag']}")
            require(ac.sha256(r["output"]) == r["output_sha256"], f"Output changed: {r['tag']}")
            for name in ("stdout", "stderr"):
                require(ac.sha256(r[name + "_log"]) == r[name + "_sha256"], f"Log changed: {r['tag']}")
            probe = doc["clips"][r["clip"]]["probe"]
            output = json.loads(Path(r["output"]).read_text("utf-8"))
            problems = ac.validate_output(output, probe["frame_count"], probe["fps"],
                                          (probe["width"], probe["height"]))
            require(not problems, f"Invalid persisted output {r['tag']}: {problems}")
            score = evaluate.score(r["output"], primary[r["clip"]])
            require(score["counts"] == r["vs_judge"]["counts"], f"Primary score changed: {r['tag']}")
            rescored.append({**r, "vs_judge": score})
            if secondary is not None and r["clip"] == "01":
                secondary_runs.append({**r, "vs_judge": evaluate.score(r["output"], secondary)})
        quality = ac.quality_gate(rescored, baselines)
        require(quality["passed"], f"Primary quality failed: {quality}")
        secondary_result = ac.quality_gate(secondary_runs, {"01": secondary_base}) if secondary_runs else None
        require(not secondary_result or secondary_result["passed"], "Secondary consensus failed")
        require(doc["gate"]["passed"], f"Saved gate failed: {doc['suite']}")
        checked.append({"suite": doc["suite"], "runs": len(rescored), "quality": quality,
                        "secondary": secondary_result,
                        "timing": ac.summarize_walls(rescored)})
    render = json.loads(render_path.read_text("utf-8"))
    for name in ("source_video", "baseline_json", "accepted_output_json", "acceptance_suite",
                 "renderer", "comparison_mp4", "preview"):
        artifact = render[name]
        require(ac.sha256(artifact["absolute_path"]) == artifact["sha256"], f"Render binding changed: {name}")
    suite = json.loads(Path(render["acceptance_suite"]["absolute_path"]).read_text("utf-8"))
    row = next(r for r in suite["runs"] if r["tag"] == render["accepted_tag"])
    require(row["pass"], "Rendered row was not accepted")
    require(Path(row["output"]).resolve() == Path(render["accepted_output_json"]["absolute_path"]).resolve(),
            "Rendered path is not accepted row")
    require(row["output_sha256"] == render["accepted_output_json"]["sha256"], "Rendered output is not accepted row")
    movie = render["comparison_mp4"]["absolute_path"]
    probe = ac.ffprobe_counts(movie)
    require(probe == render["probe"] and probe["frame_count"] == 900, "Rendered frame metadata differs")
    require(Path(movie).stat().st_size < 64 * 1024 ** 2, "Rendered MP4 exceeds64MiB")
    ffmpeg = ac.FFPROBE.with_name("ffmpeg.exe")
    subprocess.run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", movie,
                    "-f", "null", "-"], check=True, capture_output=True)
    return {"passed": True, "script_sha256": ac.sha256(__file__),
            "freeze_sha256": ac.sha256(freeze_path), "unique_runs": len(tags), "suites": checked,
            "render_bindings_passed": True, "mp4_full_decode_passed": True}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--suite", nargs="+", type=Path, required=True)
    ap.add_argument("--render", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    require(not a.out.exists(), f"Verification output already exists: {a.out}")
    result = verify(a.freeze, a.suite, a.render)
    a.out.write_text(json.dumps(result, indent=1), "utf-8")
    print(json.dumps(result))
