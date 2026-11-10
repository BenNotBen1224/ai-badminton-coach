"""Recompute the court-depth diagnostics claimed by court-corrected analysis.json."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "badminton-coach" / "outputs" / "tactics-multigame-20261104"
OUTPUT = ROOT / "badminton-coach" / "outputs" / "tactics-multigame-editorial-court-20261105"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def projector(corners):
    equations = []
    for (x, y), (u, v) in zip(corners, ((0, 0), (6.1, 0), (6.1, 13.4), (0, 13.4))):
        x, y = x * 1920, y * 1080
        equations.extend((
            [x, y, 1, 0, 0, 0, -u * x, -u * y, u],
            [0, 0, 0, x, y, 1, -v * x, -v * y, v],
        ))
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
    a, b, c, d, e, f, g, h = (row[-1] for row in equations)

    def project(x, y):
        divisor = g * x + h * y + 1
        return ((a * x + b * y + c) / divisor, (d * x + e * y + f) / divisor)

    return project


def main():
    analysis_path = OUTPUT / "analysis.json"
    analysis = load(analysis_path)
    if analysis["status"] != "court_feet_checked_editorial_example":
        raise ValueError("Unexpected analysis status")
    configs = {
        "mens": ("ms-source-unscored", 3, 2295, 2676),
        "womens": ("ws-source-unscored", 1, 570, 825),
    }
    results = {}
    for video, (folder, number, start, end) in configs.items():
        index_path = BASE / folder / "pose_cache" / "index.json"
        segment = next(s for s in load(index_path)["segments"] if s["number"] == number)
        pose_path = Path(segment["pose_path"])
        project = projector(segment["court_lines_normalized"])
        frames = {round(f["reel_time_seconds"] * 30): f for f in load(pose_path)["frames"]}
        measured = []
        for claim in analysis["claims"]:
            if claim["video"] != video or "norm_index" not in claim:
                continue
            frame = frames[claim["norm_index"]]
            row = {"id": claim["id"], "norm_index": claim["norm_index"], "roles": {}}
            for role in ("near", "far"):
                player = next(p for p in frame["players"] if p.get("role") == role)
                ankles = [player["joints"][joint] for joint in (15, 16)]
                pixel_points = [[x, y] if confidence >= 0.5 else None
                                for x, y, confidence in ankles]
                xy = [project(x, y) if confidence >= 0.5 else None
                      for x, y, confidence in ankles]
                depth_key, x_key, pixels_key = f"{role}_depth_m", f"{role}_court_x_m", f"{role}_ankles_px"
                if pixels_key in claim:
                    actual = [point for point in pixel_points if point is not None]
                    expected = claim[pixels_key]
                    if len(actual) != 2 or any(
                        abs(actual[i][axis] - expected[i][axis]) > 0.6
                        for i in range(2) for axis in (0, 1)
                    ):
                        raise ValueError(f"{claim['id']}: ankle pixels differ from pose cache")
                if depth_key in claim:
                    actual = [point[1] for point in xy if point is not None]
                    if len(actual) != 2 or any(
                        abs(actual[i] - claim[depth_key][i]) > 0.025 for i in range(2)
                    ):
                        raise ValueError(f"{claim['id']}: depth differs from court calibration")
                if x_key in claim:
                    actual = [point[0] for point in xy if point is not None]
                    if len(actual) != 2 or any(
                        abs(actual[i] - claim[x_key][i]) > 0.025 for i in range(2)
                    ):
                        raise ValueError(f"{claim['id']}: court x differs from court calibration")
                if pixels_key in claim or depth_key in claim or x_key in claim:
                    row["roles"][role] = {
                        "ankles_xy_m": xy, "depth_from_far_baseline_m": [p[1] for p in xy if p],
                        "court_x_m": [p[0] for p in xy if p],
                    }
            measured.append(row)

        valid = []
        far_max_depth = 0.0
        near_min_depth = 13.4
        far_crossed_short_line = 0
        for frame_no in range(start, end + 1, 3):
            frame = frames[frame_no]
            for role in ("near", "far"):
                player = next(p for p in frame["players"] if p.get("role") == role)
                points = [project(*player["joints"][joint][:2])
                          if player["joints"][joint][2] >= 0.7 else None
                          for joint in (15, 16)]
                if not all(points):
                    continue
                valid.append((frame_no, role, points))
                if role == "far":
                    deepest = max(point[1] for point in points)
                    far_max_depth = max(far_max_depth, deepest)
                    far_crossed_short_line += int(deepest >= 4.72)
                else:
                    near_min_depth = min(near_min_depth, *(point[1] for point in points))
        scan_claim = analysis["scan"][video]
        sample_count = sum(1 for frame_no in range(start, end + 1, 3)
                           if all(any(r == role and n == frame_no for n, r, _ in valid)
                                  for role in ("near", "far")))
        if sample_count != scan_claim["sampled"] or far_crossed_short_line != scan_claim["far_feet_at_or_past_short_service"]:
            raise ValueError(f"{video}: scan aggregate differs from reported summary")
        results[video] = {
            "pose_sha256": digest(pose_path), "pose_index_sha256": digest(index_path),
            "claim_anchor_recomputations": measured,
            "scan": {
                "range": [start, end], "step": 3, "valid_both_player_frames": sample_count,
                "far_foot_max_depth_from_far_baseline_m": round(far_max_depth, 3),
                "far_foot_min_distance_from_net_m": round(6.7 - far_max_depth, 3),
                "far_frames_at_or_past_short_service_line": far_crossed_short_line,
                "near_foot_min_depth_from_far_baseline_m": round(near_min_depth, 3),
                "sampled_frames_missing_confident_ankles_for_either_player":
                    len(range(start, end + 1, 3)) - sample_count,
            },
        }
    report = {
        "analysis_sha256": digest(analysis_path),
        "court_geometry": {"far_baseline_m": 0, "net_m": 6.7,
                           "far_short_service_m": 4.72, "near_short_service_m": 8.68,
                           "near_baseline_m": 13.4},
        "videos": results,
        "limits": [
            "Projection is a diagnostic from a single broadcast camera, not surveyed ground truth.",
            "Only ankle detections with confidence at least 0.7 count in the scan.",
            "A court location does not establish shuttle location, shot family, or player intention.",
        ],
    }
    target = HERE / "court-claim-audit.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(target), "videos": {
        name: row["scan"] for name, row in results.items()
    }}))


if __name__ == "__main__":
    main()
