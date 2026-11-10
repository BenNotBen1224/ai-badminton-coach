"""Check both players' foot positions, not hip height or the raised net tape."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "badminton-coach" / "outputs" / "tactics-multigame-20261104"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def transform(corners):
    equations = []
    for (x, y), (u, v) in zip(corners, ((0, 0), (6.1, 0), (6.1, 13.4), (0, 13.4))):
        x, y = x * 1920, y * 1080
        equations.extend([
            [x, y, 1, 0, 0, 0, -u * x, -u * y, u],
            [0, 0, 0, x, y, 1, -v * x, -v * y, v],
        ])
    for i in range(8):
        pivot = max(range(i, 8), key=lambda j: abs(equations[j][i]))
        equations[i], equations[pivot] = equations[pivot], equations[i]
        divisor = equations[i][i]
        if abs(divisor) < 1e-10:
            raise ValueError("Degenerate court calibration")
        equations[i] = [value / divisor for value in equations[i]]
        for j in range(8):
            if j != i:
                multiplier = equations[j][i]
                equations[j] = [a - multiplier * b for a, b in zip(equations[j], equations[i])]
    coefficients = [row[-1] for row in equations]

    def project(x, y):
        a, b, c, d, e, f, g, h = coefficients
        divisor = g * x + h * y + 1
        return [(a * x + b * y + c) / divisor, (d * x + e * y + f) / divisor]

    return project


def main():
    cases = (
        ("mens", "ms-source-unscored", 3, (2330, 2370, 2421, 2472, 2547)),
        ("womens", "ws-source-unscored", 1, (650, 669, 692, 723, 762)),
    )
    result = {
        "premise_under_test": (
            "The far player was at the net, so net-reply coaching applies. "
            "This was inferred from image appearance without checking feet against court lines."
        ),
        "reference": {"far_baseline_m": 0, "net_m": 6.7, "near_baseline_m": 13.4,
                      "far_short_service_m": 4.72, "near_short_service_m": 8.68},
        "limits": [
            "Diagnostic projection using saved calibration, not newly validated exact floor coordinates.",
            "Airborne feet, low-confidence joints and calibration error can bias projections.",
            "A hip pixel coordinate or proximity to raised net tape cannot establish court depth.",
            "No shuttle location, shot family or intention is inferred.",
        ],
        "videos": {},
    }
    for name, folder, number, targets in cases:
        index_path = BASE / folder / "pose_cache" / "index.json"
        segment = next(s for s in load(index_path)["segments"] if s["number"] == number)
        path = Path(segment["pose_path"])
        project = transform(segment["court_lines_normalized"])
        frames = {round(f["reel_time_seconds"] * 30): f for f in load(path)["frames"]}
        rows = []
        for target in targets:
            frame = frames[target]
            for player in frame["players"]:
                role = player.get("role")
                if role not in ("near", "far"):
                    continue
                ankles = [player["joints"][joint] for joint in (15, 16)]
                projected = [project(x, y) if confidence >= 0.5 else None
                             for x, y, confidence in ankles]
                rows.append({"normalized_frame": target, "role": role,
                             "ankles_image_xy_and_score": ankles,
                             "diagnostic_projected_feet_m": projected})
        result["videos"][name] = {
            "pose_sha256": digest(path), "index_sha256": digest(index_path),
            "corners_normalized": segment["court_lines_normalized"], "rows": rows,
        }
    output = HERE / "court-position-census.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": sum(
        len(video["rows"]) for video in result["videos"].values())}))


if __name__ == "__main__":
    main()
