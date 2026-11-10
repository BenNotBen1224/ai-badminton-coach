"""How long a player waits for a coaching session: E2ERunner timed end to end on the eight test clips.

    python -m coach.app.bench_e2e [--backend turbo|fast] [--repeats 5] [--json PATH]

One E2ERunner stays loaded for the whole run. It analyses the warm-up clip once without a timer, then runs every
test clip `--repeats` times, taking the clips in turn. Each run gets a byte-unique copy from turbo.fresh.fresh_copy,
so nothing keyed by the file can be reused, and the whole run() call is timed with perf_counter. The files run()
wrote are checked after the timer stops; a run that fails or writes the wrong files counts as over the limit.
Upload, queue, cold start and warm-up are not in these numbers. Before each run the bench waits up to
QUIET_WAIT_S for the rest of the computer to use less than QUIET_CPU percent CPU and records what it measured.
The clip roles (warm-up, gating) come from turbo/clips.json.

This is a diagnostic. A run passes only when it takes less than LIMIT_S, so exactly 1.000 s fails. The bench checks
only that the three files exist and that pose.json is the returned document, then deletes them, so its verdict is
not acceptance. The acceptance table is a kept `python -m turbo.accept final` record: it keeps every run's output
and also checks that output's validity and quality.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import psutil

from turbo.fresh import fresh_copy

from .backend import DEFAULT_MODELS, PROJECT, FastBackend
from .pipeline import E2ERunner

CLIPS = PROJECT / "turbo" / "clips.json"
LIMIT_S = 1.0
QUIET_CPU = 10.0
QUIET_WAIT_S = 300.0
STAGES = ("pose", "load_tracks", "events", "body", "move", "assemble", "bundle", "write", "total")


def clip_table(path: Path = CLIPS) -> tuple[Path, dict[str, Path]]:
    """The warm-up clip and the test clips, by the roles in turbo/clips.json."""
    clips = json.loads(path.read_text("utf-8"))["clips"]
    warmup = [PROJECT / clip["file"] for clip in clips.values() if clip["role"] == "warmup"]
    tests = {name: PROJECT / clip["file"] for name, clip in sorted(clips.items()) if clip["role"] == "gating"}
    if len(warmup) != 1 or not tests:
        raise SystemExit(f"{path} needs one warm-up clip and at least one gating clip")
    return warmup[0], tests


def background_cpu() -> float:
    """System CPU percent over one second while the bench idles, after waiting up to QUIET_WAIT_S for it to settle."""
    deadline = time.monotonic() + QUIET_WAIT_S
    while True:
        percent = psutil.cpu_percent(interval=1.0)
        if percent < QUIET_CPU or time.monotonic() >= deadline:
            return percent


def check_files(result: dict, work: Path) -> None:
    """pose.json, the report and the bundle are non-empty files in work, and pose.json is the returned document."""
    for key in ("pose_path", "report_path", "bundle_path"):
        path = Path(result[key])
        if path.parent.resolve() != work.resolve() or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"{key} {path} is not a written file in {work}")
    if json.loads(Path(result["pose_path"]).read_text("utf-8")) != result["doc"]:
        raise RuntimeError("pose.json is not the document run() returned")


def bench(runner, warmup: Path, clips: dict[str, Path], repeats: int, scratch: Path, quiet=background_cpu) -> dict:
    start = time.perf_counter()
    runner.warm_up(warmup)
    result = {"warm_up_clip": warmup.name, "warm_up_s": time.perf_counter() - start, "runs": []}
    for repeat in range(1, repeats + 1):
        for name, video in clips.items():
            copy = fresh_copy(video, scratch / "copies")
            work = scratch / f"{name}-{repeat}"
            work.mkdir(parents=True)
            row = {"clip": name, "repeat": repeat, "background_cpu": quiet()}
            start = time.perf_counter()
            try:
                out = runner.run(copy, work)
                row["wall_s"] = time.perf_counter() - start
                check_files(out, work)
                row["timings"] = {stage: out["timings"].get(stage) for stage in STAGES}
                row["pose_stages"] = out["timings"].get("pose_stages")
            except Exception as exc:
                row.setdefault("wall_s", time.perf_counter() - start)
                row["error"] = str(exc) or type(exc).__name__
            finally:
                copy.unlink(missing_ok=True)
                shutil.rmtree(work, ignore_errors=True)
            result["runs"].append(row)
    return result


def summarise(runs: list[dict], limit_s: float = LIMIT_S) -> dict:
    clips = {}
    for name in dict.fromkeys(run["clip"] for run in runs):
        rows = [run for run in runs if run["clip"] == name]
        good = [run for run in rows if "error" not in run]
        walls = [run["wall_s"] for run in good]
        stages = {}
        for stage in STAGES:
            values = [run["timings"][stage] for run in good if run["timings"].get(stage) is not None]
            if values:
                stages[stage] = statistics.median(values)
        clips[name] = {
            "runs": len(rows),
            "failed": len(rows) - len(good),
            "median_s": statistics.median(walls) if walls else None,
            "max_s": max(walls) if walls else None,
            "over": sum(1 for run in rows if "error" in run or run["wall_s"] >= limit_s),
            "stages": stages,
        }
    over = sum(clip["over"] for clip in clips.values())
    return {"limit_s": limit_s, "clips": clips, "runs": len(runs), "over": over,
            "verdict": "PASS" if runs and over == 0 else "FAIL"}


def machine() -> dict:
    memory = psutil.virtual_memory().total / 2**30
    return {"system": platform.platform(), "cpu": cpu_name(), "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(), "ram_gb": round(memory, 1), "python": platform.python_version()}


def cpu_name() -> str:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except (ImportError, OSError):
        return platform.processor()


def _s(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def table(result: dict) -> str:
    summary, host = result["summary"], result["machine"]
    cpu = [run["background_cpu"] for run in result["runs"]]
    lines = [
        f"End-to-end wait for a coaching session: E2ERunner, {result['backend']}",
        f"Machine: {host['system']}, {host['cpu']}, {host['cores']} cores / {host['threads']} threads, "
        f"{host['ram_gb']} GB RAM, Python {host['python']}",
        f"Background CPU before each run: median {statistics.median(cpu):.1f} %, max {max(cpu):.1f} % "
        f"(each run waited up to {QUIET_WAIT_S:.0f} s for under {QUIET_CPU:.0f} %)" if cpu else "No runs.",
        f"Untimed warm-up on {result['warm_up_clip']}: {result['warm_up_s']:.1f} s. "
        "Upload, queue, cold start and warm-up are not in the numbers below.",
        "Seconds per run (the whole run() call), then the median seconds of each stage inside it:",
        "",
        "clip  runs  failed  median    max  >=limit  " + "  ".join(f"{stage:>11}" for stage in STAGES),
    ]
    for name, clip in summary["clips"].items():
        stages = "  ".join(f"{_s(clip['stages'].get(stage)):>11}" for stage in STAGES)
        lines.append(f"{name:<4}  {clip['runs']:>4}  {clip['failed']:>6}  {_s(clip['median_s']):>6}  "
                     f"{_s(clip['max_s']):>6}  {clip['over']:>7}  {stages}")
    errors = sorted({run["error"] for run in result["runs"] if "error" in run})
    limit = f"{summary['limit_s']:.3f} s"
    lines += [""] + [f"Failed run: {error}" for error in errors]
    lines += [f"Runs not under {limit} (failed runs included): {summary['over']} of {summary['runs']}",
              f"Diagnostic verdict, every run under {limit}: {summary['verdict']}",
              "Not acceptance: this bench deletes each run's files after checking only that they exist and that "
              "pose.json matches. The acceptance table is a kept `python -m turbo.accept final` record."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m coach.app.bench_e2e",
                                     description="Time E2ERunner end to end on the test clips (a diagnostic, "
                                                 "not the acceptance table).")
    parser.add_argument("--backend", choices=("turbo", "fast"), default="turbo")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=Path, help="also write every run's numbers to this file")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    warmup, clips = clip_table()
    backend = None if args.backend == "turbo" else FastBackend()
    runner = E2ERunner(DEFAULT_MODELS, backend=backend)
    with tempfile.TemporaryDirectory(prefix="coach-bench-") as scratch:
        result = bench(runner, warmup, clips, args.repeats, Path(scratch))
    result.update(backend=runner.backend.label, machine=machine(), summary=summarise(result["runs"]))
    print(table(result))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, default=str), "utf-8")
    return 0 if result["summary"]["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
