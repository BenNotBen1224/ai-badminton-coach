"""Old way (finder + RTMPose-m) vs the fast pipeline: speed, joint accuracy, player-ID switches.
Accuracy is re-measured with evaluate.py against the judge; times come from runs/gen-timing.csv (timing.py).
Clips 04-05 are a venue the pose model never trained on.

usage: python scoreboard.py OUT_PREFIX   (writes OUT_PREFIX.svg and OUT_PREFIX.png; run with the project .venv)
"""
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
CLIPS = ["01", "02", "03", "04", "05"]
NEW_VENUE = {"04", "05"}
RUN, TIMING = "cut", "gen-timing.csv"
OLD_SECONDS_CLIP01 = 326  # measured once when reproducing the baseline (checkpoint 012)
OLD, NEW = "#9aa0a6", "#1a73e8"


def evaluate(run, clip):
    out = subprocess.run([sys.executable, str(HERE / "evaluate.py"), str(run), "--ref",
                          str(HERE / "runs" / f"judge-{clip}.json")], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def main():
    prefix = Path(sys.argv[1])
    scores = {c: {who: evaluate(HERE / "runs" / f"{who}-{c}.json", c) for who in ("baseline", RUN)} for c in CLIPS}
    walls = {c: [] for c in CLIPS}
    with open(HERE / "runs" / TIMING) as f:
        for row in csv.DictReader(f):
            walls[row["clip"]].append(float(row["wall"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    x = range(len(CLIPS))
    labels = [f"clip {c}" + ("\nnew venue" if c in NEW_VENUE else "") for c in CLIPS]

    ax = axes[0]
    med = [statistics.median(walls[c]) for c in CLIPS]
    top = [max(walls[c]) - m for c, m in zip(CLIPS, med)]
    ax.bar(x, med, color=NEW, width=0.55, yerr=[[0] * len(CLIPS), top], capsize=6, ecolor="#555")
    ax.axhline(10, color="#d93025", ls="--", lw=2)
    ax.text(len(CLIPS) - 0.65, 10.15, "goal: 10 s", color="#d93025", ha="right", fontsize=10)
    for i, m in enumerate(med):
        ax.text(i, m / 2, f"{m:.1f} s", ha="center", color="white", fontsize=12, weight="bold")
    ax.set_ylim(0, 12)
    ax.set_title(f"Seconds per clip (new; old way took {OLD_SECONDS_CLIP01} s on clip 01)", fontsize=11)
    ax.set_ylabel("wall clock, median of 5 (bar = slowest)")

    def pair(ax, key, title, ylim, fmt):
        w = 0.38
        for k, (who, color, name) in enumerate((("baseline", OLD, "old way"), (RUN, NEW, "new fast"))):
            vals = [key(scores[c][who]) for c in CLIPS]
            xs = [i + (k - 0.5) * w for i in x]
            ax.bar(xs, vals, w, color=color, label=name)
            for xi, v in zip(xs, vals):
                ax.text(xi, v + (ylim[1] - ylim[0]) * 0.01, fmt(v), ha="center", fontsize=9)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper right", frameon=False)

    pair(axes[1], lambda s: s["pck05"], "Joints in the right place (PCK@0.05 vs judge)", (0.90, 1.0),
         lambda v: f"{v:.3f}")
    pair(axes[2], lambda s: s["id_switches"]["near"] + s["id_switches"]["far"],
         "Player-ID changes (lower is better)\nclips 04-05 have 3 camera cuts: new fast restarts IDs there", (0, 70), lambda v: f"{v}")
    for ax in axes:
        ax.set_xticks(list(x), labels)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(prefix.with_suffix(f".{ext}"), dpi=110)
    print(json.dumps({c: {who: {k: scores[c][who][k] for k in ("player_frames_found", "pck05", "pck10", "id_switches")}
                          for who in scores[c]} | {"wall_median": statistics.median(walls[c]), "wall_max": max(walls[c])}
                      for c in CLIPS}))


if __name__ == "__main__":
    main()
