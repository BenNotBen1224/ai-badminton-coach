"""Project both players' feet through the saved court calibration for the three original-render passages."""

import hashlib
import json
import struct
import subprocess
import zlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = ROOT / "badminton-coach" / "outputs" / "tactics-engine-20261104" / "pose_cache"
SOURCE = ROOT / "obj2" / "work" / "H" / "downloads" / "china-open-2025-ws-final.webm"
FFMPEG = ROOT / "tools" / "ffmpeg.exe"
OUT = HERE / "original-audit"
PASSAGES = (("space", 57.8, 64.8, 2), ("time", 126.6, 134.0, 5), ("options", 272.8, 285.0, 10))
COURT = ((0, 0), (6.1, 0), (6.1, 13.4), (0, 13.4))
FOOT_SCORE_MIN = 0.5
STEP_S = 0.25
SHEET_STEP_S = 1.0
WIDTH, HEIGHT = 1920, 1080


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def homography(pairs):
    equations = []
    for (x, y), (u, v) in pairs:
        equations.append([x, y, 1, 0, 0, 0, -u * x, -u * y, u])
        equations.append([0, 0, 0, x, y, 1, -v * x, -v * y, v])
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
        return [(a * x + b * y + c) / divisor, (d * x + e * y + f) / divisor]

    return project


class Canvas:
    def __init__(self):
        self.pixels = bytearray(WIDTH * HEIGHT * 4)

    def dot(self, x, y, color, radius):
        for yy in range(int(y) - radius, int(y) + radius + 1):
            for xx in range(int(x) - radius, int(x) + radius + 1):
                if 0 <= xx < WIDTH and 0 <= yy < HEIGHT:
                    self.pixels[(yy * WIDTH + xx) * 4:(yy * WIDTH + xx) * 4 + 4] = bytes(color)

    def line(self, start, end, color, radius=1):
        steps = int(max(abs(end[0] - start[0]), abs(end[1] - start[1]))) + 1
        for step in range(steps + 1):
            t = step / steps
            self.dot(start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t, color, radius)

    def save(self, path):
        def chunk(tag, data):
            body = tag + data
            return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

        rows = b"".join(b"\x00" + bytes(self.pixels[y * WIDTH * 4:(y + 1) * WIDTH * 4]) for y in range(HEIGHT))
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 6, 0, 0, 0))
                         + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def nearest(frames, seconds):
    return min(frames, key=lambda frame: abs(frame["reel_time_seconds"] - seconds))


def feet(player, to_court):
    rows = []
    for name, joint in (("left_ankle", 15), ("right_ankle", 16)):
        x, y, score = player["joints"][joint]
        rows.append({"joint": name, "image_xy": [round(x, 1), round(y, 1)], "score": round(score, 3),
                     "court_m": [round(value, 2) for value in to_court(x, y)] if score >= FOOT_SCORE_MIN else None})
    return rows


def mean_court(rows):
    valid = [row["court_m"] for row in rows if row["court_m"]]
    if not valid:
        return None
    return [round(sum(point[i] for point in valid) / len(valid), 2) for i in (0, 1)]


def draw_overlay(path, to_image, players, to_court):
    canvas = Canvas()
    for a, b in (((0, 0), (6.1, 0)), ((6.1, 0), (6.1, 13.4)), ((6.1, 13.4), (0, 13.4)), ((0, 13.4), (0, 0))):
        canvas.line(to_image(*a), to_image(*b), (255, 255, 0, 255))
    canvas.line(to_image(0, 6.7), to_image(6.1, 6.7), (255, 0, 255, 255))
    for v in (4.72, 8.68):
        canvas.line(to_image(0, v), to_image(6.1, v), (255, 255, 255, 200))
    canvas.line(to_image(3.05, 0), to_image(3.05, 13.4), (255, 255, 255, 120))
    colors = {"near": (0, 255, 255, 255), "far": (255, 140, 0, 255)}
    for player in players:
        for row in feet(player, to_court):
            if row["court_m"]:
                canvas.dot(row["image_xy"][0], row["image_xy"][1], colors[player["role"]], 6)
    canvas.save(path)


def main():
    OUT.mkdir(exist_ok=True)
    index_path = CACHE / "index.json"
    segments = {segment["number"]: segment for segment in load(index_path)["segments"]}
    result = {
        "purpose": "Check original-render positional wording against feet projected through floor-line calibration.",
        "reference_m": {"far_baseline": 0, "far_short_service": 4.72, "net": 6.7,
                        "near_short_service": 8.68, "near_baseline": 13.4, "width": 6.1},
        "limits": [
            "Diagnostic projection using saved calibration; airborne feet and low-score joints bias it.",
            "depth_m is distance from the far baseline; lateral_m is distance from the screen-left doubles sideline.",
            "Hip image height or net-tape proximity is not used.",
        ],
        "source_sha256": digest(SOURCE),
        "index_sha256": digest(index_path),
        "passages": {},
    }
    for name, start, end, number in PASSAGES:
        segment = segments[number]
        pose_path = Path(segment["pose_path"])
        pose = load(pose_path)
        corners = [(x * WIDTH, y * HEIGHT) for x, y in segment["court_lines_normalized"]]
        to_court = homography(zip(corners, COURT))
        to_image = homography(zip(COURT, corners))
        frames = pose["frames"]
        rows = []
        steps = int(round((end - start) / STEP_S))
        for step in range(steps + 1):
            seconds = start + step * STEP_S
            frame = nearest(frames, seconds)
            row = {"requested_s": round(seconds, 3), "pose_time_s": round(frame["reel_time_seconds"], 4), "players": {}}
            for player in frame["players"]:
                role = player.get("role")
                if role in ("near", "far"):
                    rows_feet = feet(player, to_court)
                    row["players"][role] = {"feet": rows_feet, "mean_lateral_depth_m": mean_court(rows_feet)}
            rows.append(row)
        result["passages"][name] = {
            "segment": number, "pose_sha256": digest(pose_path),
            "corners_normalized": segment["court_lines_normalized"], "rows": rows,
        }
        sheet_dir = OUT / f"census-{name}"
        sheet_dir.mkdir(exist_ok=True)
        count = int((end - start) / SHEET_STEP_S) + 1
        for k in range(count):
            seconds = start + k * SHEET_STEP_S
            frame = nearest(frames, seconds)
            overlay = sheet_dir / f"overlay_{k:02d}.png"
            draw_overlay(overlay, to_image, [p for p in frame["players"] if p.get("role") in ("near", "far")], to_court)
            subprocess.run([str(FFMPEG), "-v", "error", "-y", "-threads", "1", "-ss", f"{frame['reel_time_seconds']:.4f}",
                            "-i", str(SOURCE), "-i", str(overlay), "-filter_complex",
                            "[0:v][1:v]overlay=0:0,scale=640:360", "-frames:v", "1", str(sheet_dir / f"frame_{k:02d}.jpg")],
                           check=True)
        subprocess.run([str(FFMPEG), "-v", "error", "-y", "-framerate", "1", "-i", str(sheet_dir / "frame_%02d.jpg"),
                        "-vf", "tile=4x" + str((count + 3) // 4), "-frames:v", "1", str(OUT / f"census-{name}-sheet.jpg")],
                       check=True)
    output = HERE / "original-court-census.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, passage in result["passages"].items():
        print(name)
        for row in passage["rows"]:
            near = row["players"].get("near", {}).get("mean_lateral_depth_m")
            far = row["players"].get("far", {}).get("mean_lateral_depth_m")
            print(f"  t={row['requested_s']:7.2f} near(lat,depth)={near} far(lat,depth)={far}")


if __name__ == "__main__":
    main()
