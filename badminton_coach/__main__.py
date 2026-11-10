from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import TextIOBase
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import cv2
import numpy as np

from badminton_coach.overlay import annotate_frame, assign_track_slots


MODEL_NAME = "yolo26s-pose.pt"
DEFAULT_IMAGE_SIZE = 1280
NMS_IOU_THRESHOLD = 0.5
COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COLORS = (
    (85, 210, 255), (255, 170, 70), (125, 230, 125),
    (235, 120, 210), (120, 200, 255), (230, 225, 90),
)
PERSON_TRACK_CONFIDENCE = 0.1
DEFAULT_OUTPUT_ROOT = Path("outputs")
WINDOW_NAME = "Badminton Coach - select the court"


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    contour = np.asarray(polygon, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, point, False) >= 0


def validate_polygon(
    points: list[tuple[float, float]], width: int, height: int
) -> list[tuple[float, float]]:
    if len(points) != 4:
        raise ValueError("Select exactly four points around the court.")
    if any(
        not np.isfinite(x) or not np.isfinite(y)
        or x < 0 or y < 0 or x >= width or y >= height
        for x, y in points
    ):
        raise ValueError("Court points must be inside the video frame.")
    contour = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
    if not cv2.isContourConvex(contour):
        raise ValueError("The four points must make a convex shape; select them in order.")
    area = abs(float(cv2.contourArea(contour)))
    if area < width * height * 0.02:
        raise ValueError("The selected court area is too small.")
    return points


def choose_court(frame: np.ndarray) -> list[tuple[float, float]]:
    height, width = frame.shape[:2]
    scale = min(1.0, 1280 / width, 760 / height)
    shown_width = max(1, round(width * scale))
    shown_height = max(1, round(height * scale))
    preview = cv2.resize(frame, (shown_width, shown_height))
    points: list[tuple[float, float]] = []
    display = preview.copy()

    def redraw() -> None:
        nonlocal display
        display = preview.copy()
        cv2.putText(
            display, "Click 4 corners around the court (leave a small margin)",
            (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 245, 245), 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display, "Enter: confirm | Backspace: undo | R: reset | Esc: cancel",
            (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (40, 245, 245), 2,
            cv2.LINE_AA,
        )
        for index, (x, y) in enumerate(points):
            cv2.circle(display, (round(x * scale), round(y * scale)), 6, COLORS[index], -1)
            cv2.putText(
                display, str(index + 1), (round(x * scale) + 8, round(y * scale) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS[index], 2, cv2.LINE_AA,
            )
        if len(points) > 1:
            scaled = np.asarray(
                [(round(x * scale), round(y * scale)) for x, y in points],
                dtype=np.int32,
            )
            cv2.polylines(display, [scaled], len(points) == 4, (40, 245, 245), 2)
        cv2.imshow(WINDOW_NAME, display)

    def on_mouse(event: int, x: int, y: int, _flags: int, _parameter: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))
            redraw()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, shown_width, shown_height)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    redraw()
    try:
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (10, 13):
                if len(points) != 4:
                    print("Select all four corners before pressing Enter.")
                    continue
                return validate_polygon(points, width, height)
            if key in (8, 127):
                if points:
                    points.pop()
                    redraw()
            elif key in (ord("r"), ord("R")):
                points.clear()
                redraw()
            elif key == 27:
                raise KeyboardInterrupt("Court selection cancelled.")
    finally:
        cv2.destroyWindow(WINDOW_NAME)


def find_tool(name: str) -> Path | None:
    environment_value = os.environ.get(f"BADMINTON_{name.upper()}")
    if environment_value:
        candidate = Path(environment_value).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"{name} was not found at BADMINTON_{name.upper()}: {candidate}")
    from_path = shutil.which(name)
    if from_path:
        return Path(from_path).resolve()
    project_dir = Path(__file__).resolve().parents[1]
    workspace_tool = project_dir / "tools" / f"{name}.exe"
    if workspace_tool.is_file():
        return workspace_tool.resolve()
    return None


def require_ffmpeg() -> tuple[Path, Path]:
    ffmpeg = find_tool("ffmpeg")
    ffprobe = find_tool("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise FileNotFoundError(
            "FFmpeg and FFprobe are required for verified H.264 MP4 output. "
            "Install FFmpeg or set BADMINTON_FFMPEG and BADMINTON_FFPROBE."
        )
    return ffmpeg, ffprobe


def probe_video(path: Path, ffprobe: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(ffprobe), "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:"
            "format=duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(completed.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise ValueError(f"No video stream was found in {path}.")
    stream = streams[0]
    if stream["r_frame_rate"] != stream["avg_frame_rate"]:
        raise ValueError(
            "This first version expects a constant-frame-rate video. "
            "Convert the video to constant frame rate before analyzing it."
        )
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    if numerator <= 0 or denominator <= 0:
        raise ValueError("The source video has an invalid frame rate.")
    count_text = stream.get("nb_frames")
    count = int(count_text) if count_text and count_text != "N/A" else 0
    if count <= 0:
        raise ValueError("Could not read the source video frame count.")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps_numerator": numerator,
        "fps_denominator": denominator,
        "fps": numerator / denominator,
        "frame_count": count,
        "duration_seconds": float(data["format"]["duration"]),
    }


def joint_record(raw: np.ndarray, confidence_threshold: float) -> list[list[float] | None]:
    if raw.shape != (17, 3):
        raise ValueError(f"Expected 17 COCO joints; received {raw.shape}.")
    return [
        [round(float(x), 2), round(float(y), 2), round(float(conf), 4)]
        if np.isfinite((x, y, conf)).all() and conf >= confidence_threshold
        else None
        for x, y, conf in raw
    ]


def write_frames_json_start(output: TextIOBase, header: dict[str, Any]) -> None:
    serialized = json.dumps(header, ensure_ascii=False)
    if not serialized.endswith("}"):
        raise ValueError("JSON header must be an object.")
    output.write(serialized[:-1])
    output.write(',\n  "frames": [\n')


def write_frames_json_end(output: TextIOBase) -> None:
    output.write("  ]\n}\n")


def validate_output(
    path: Path, ffprobe: Path, expected_width: int, expected_height: int,
    expected_frames: int,
) -> None:
    completed = subprocess.run(
        [
            str(ffprobe), "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_frames:"
            "format=duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    info = json.loads(completed.stdout)
    if not info.get("streams"):
        raise ValueError(f"FFprobe found no video stream in {path.name}.")
    stream = info["streams"][0]
    checks = {
        "codec": stream["codec_name"] == "h264",
        "width": int(stream["width"]) == expected_width,
        "height": int(stream["height"]) == expected_height,
        "frame count": int(stream["nb_read_frames"]) == expected_frames,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Invalid {path.name}: incorrect {', '.join(failed)}.")
    duration = float(info["format"]["duration"])
    expected_duration = expected_frames / (
        int(stream["avg_frame_rate"].split("/")[0])
        / int(stream["avg_frame_rate"].split("/")[1])
    )
    if abs(duration - expected_duration) > 0.1:
        raise ValueError(f"Invalid duration for {path.name}: {duration:.3f} seconds.")


def report_progress(label: str, frame_index: int, total: int, started: float) -> None:
    done = frame_index + 1
    elapsed = time.monotonic() - started
    rate = done / elapsed if elapsed > 0 else 0
    remaining = (total - done) / rate if rate > 0 else 0
    percent = 100 * done / total
    eta = time.strftime("%H:%M:%S", time.gmtime(remaining))
    sys.stdout.write(f"\r{label}: {done:,}/{total:,}  {percent:5.1f}%  ETA {eta}")
    sys.stdout.flush()


def render_videos(
    keypoints_file: Path, video_path: Path, output_dir: Path, ffmpeg: Path, ffprobe: Path,
) -> None:
    """Draw every skeleton in keypoints.json onto its source video and encode both MP4s."""
    data = json.loads(keypoints_file.read_text(encoding="utf-8"))
    source, frames = data["source"], data["frames"]
    video_info = probe_video(video_path, ffprobe)
    video_facts = (
        video_info["width"], video_info["height"], video_info["fps_numerator"],
        video_info["fps_denominator"], video_info["frame_count"],
    )
    json_facts = (
        source["width"], source["height"], source["fps"]["numerator"],
        source["fps"]["denominator"], source["frame_count"],
    )
    if video_facts != json_facts:
        raise ValueError(
            "This video does not match keypoints.json. "
            f"(width, height, fps numerator, fps denominator, frames): "
            f"video {video_facts}, JSON {json_facts}."
        )
    if [frame["frame_index"] for frame in frames] != list(range(source["frame_count"])):
        raise ValueError("keypoints.json is incomplete or its frames are out of order.")
    if video_path.name != source["file"]:
        print(
            f"Warning: keypoints.json was made from {source['file']}, not {video_path.name}. "
            "Check that the skeletons line up with the people.",
            file=sys.stderr,
        )

    width, height = video_info["width"], video_info["height"]
    slots = assign_track_slots(frames)
    annotated_temp = output_dir / "annotated-temp.mp4"
    comparison_temp = output_dir / "side-by-side-temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    capture = cv2.VideoCapture(str(video_path))
    annotated_writer = cv2.VideoWriter(
        str(annotated_temp), fourcc, video_info["fps"], (width, height),
    )
    comparison_writer = cv2.VideoWriter(
        str(comparison_temp), fourcc, video_info["fps"], (width * 2, height),
    )
    try:
        if not (
            capture.isOpened() and annotated_writer.isOpened() and comparison_writer.isOpened()
        ):
            raise RuntimeError("OpenCV could not open the video or create its temporary MP4 writers.")
        started = time.monotonic()
        for record in frames:
            ok, frame = capture.read()
            if not ok:
                raise ValueError(
                    f"The video ended at frame {record['frame_index']}, "
                    f"but keypoints.json has {len(frames)} frames."
                )
            annotation = annotate_frame(frame, record["players"], slots)
            annotated_writer.write(annotation)
            comparison_writer.write(np.concatenate((frame, annotation), axis=1))
            if record["frame_index"] % 5 == 0 or record["frame_index"] + 1 == len(frames):
                report_progress("Drawing", record["frame_index"], len(frames), started)
        sys.stdout.write("\n")
        sys.stdout.flush()
        if capture.read()[0]:
            raise ValueError("The video has more frames than keypoints.json.")
    finally:
        capture.release()
        annotated_writer.release()
        comparison_writer.release()

    for temp_file, output_name, output_width in (
        (annotated_temp, "annotated.mp4", width),
        (comparison_temp, "side-by-side.mp4", width * 2),
    ):
        output_file = output_dir / output_name
        subprocess.run(
            [
                str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(temp_file), "-an", "-c:v", "libx264",
                "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(output_file),
            ],
            check=True, capture_output=True, text=True,
        )
        temp_file.unlink()
        validate_output(output_file, ffprobe, output_width, height, len(frames))


def analyze_video(
    video_path: Path,
    output_root: Path,
    model_name: str,
    image_size: int,
    joint_threshold: float,
    person_threshold: float,
) -> Path:
    ffmpeg, ffprobe = require_ffmpeg()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    video_path = video_path.resolve()
    video_info = probe_video(video_path, ffprobe)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise ValueError(f"Could not read the first frame of {video_path}.")
    if first_frame.shape[1] != video_info["width"] or first_frame.shape[0] != video_info["height"]:
        capture.release()
        raise ValueError("The decoded frame dimensions do not match the video metadata.")

    print("Click four points around the target court, leaving a small margin.")
    polygon = choose_court(first_frame)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{video_path.stem}-{timestamp}"
    final_dir = output_root.resolve() / run_name
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{run_name}-", dir=output_root.resolve()))
    keypoints_file = staging_dir / "keypoints.json"
    summary_file = staging_dir / "summary.json"

    width, height = video_info["width"], video_info["height"]
    track_counts: dict[int, int] = {}
    tracked_frames = 0
    frame_index = 0
    started = time.monotonic()
    print(f"Loading {model_name} on CPU. First use downloads the pretrained weights.")
    try:
        # Imported here so re-rendering and the unit tests skip the ~3 s ultralytics import.
        from ultralytics import YOLO

        model = YOLO(model_name)
        with keypoints_file.open("w", encoding="utf-8", newline="\n") as output:
            header = {
                "schema_version": 1,
                "source": {
                    "file": video_path.name,
                    "width": width,
                    "height": height,
                    "fps": {
                        "numerator": video_info["fps_numerator"],
                        "denominator": video_info["fps_denominator"],
                    },
                    "frame_count": video_info["frame_count"],
                },
                "model": {
                    "weights": model_name,
                    "tracker": "ByteTrack",
                    "device": "cpu",
                    "image_size": image_size,
                    "nms_iou_threshold": NMS_IOU_THRESHOLD,
                    "joint_threshold": joint_threshold,
                    "person_threshold": person_threshold,
                    "keypoint_layout": "COCO-17",
                    "keypoint_names": COCO17_NAMES,
                },
                "court_region_normalized": [
                    [round(x / width, 6), round(y / height, 6)] for x, y in polygon
                ],
            }
            write_frames_json_start(output, header)
            capture.release()
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise ValueError(f"Could not reopen video: {video_path}")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    device="cpu",
                    imgsz=image_size,
                    iou=NMS_IOU_THRESHOLD,
                    classes=[0],
                    conf=PERSON_TRACK_CONFIDENCE,
                    verbose=False,
                )[0]
                if result.keypoints is None or result.boxes is None:
                    boxes_data = np.empty((0, 6), dtype=np.float32)
                    poses_data = np.empty((0, 17, 3), dtype=np.float32)
                else:
                    boxes_data = np.column_stack((
                        result.boxes.xyxy.cpu().numpy(),
                        result.boxes.conf.cpu().numpy(),
                        result.boxes.cls.cpu().numpy(),
                    ))
                    poses_data = result.keypoints.data.cpu().numpy()
                if len(poses_data) != len(boxes_data):
                    raise ValueError("Pose and person-box counts do not match.")
                if poses_data.ndim != 3 or poses_data.shape[1:] != (17, 3):
                    if len(poses_data):
                        raise ValueError(
                            f"Expected COCO-17 pose output; received {poses_data.shape}."
                        )

                track_ids = (
                    result.boxes.id.int().cpu().tolist()
                    if result.boxes.id is not None else [None] * len(boxes_data)
                )
                person_records: list[dict[str, Any]] = []
                for box_row, pose, track_id in zip(boxes_data, poses_data, track_ids, strict=True):
                    x1, y1, x2, y2, box_confidence, class_id = map(float, box_row[:6])
                    if int(class_id) != 0 or box_confidence < person_threshold:
                        continue
                    footpoint = ((x1 + x2) / 2, y2)
                    if not point_in_polygon(footpoint, polygon):
                        continue
                    joints = joint_record(pose, joint_threshold)
                    track_id_int = int(track_id) if track_id is not None else None
                    if track_id_int is not None:
                        track_counts[track_id_int] = track_counts.get(track_id_int, 0) + 1
                    person_records.append({
                        "track_id": track_id_int,
                        "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "box_confidence": round(box_confidence, 4),
                        "joints": joints,
                    })
                if person_records:
                    tracked_frames += 1
                frame_record = {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(
                        frame_index * video_info["fps_denominator"]
                        / video_info["fps_numerator"], 6,
                    ),
                    "players": person_records,
                }
                output.write(json.dumps(frame_record, ensure_ascii=False))
                output.write(",\n" if frame_index + 1 < video_info["frame_count"] else "\n")
                if frame_index % 5 == 0 or frame_index + 1 == video_info["frame_count"]:
                    report_progress("Detecting", frame_index, video_info["frame_count"], started)
                frame_index += 1
            write_frames_json_end(output)
            capture.release()
            sys.stdout.write("\n")
            sys.stdout.flush()
            if frame_index != video_info["frame_count"]:
                raise ValueError(
                    f"Decoded {frame_index} frames, but the source reports "
                    f"{video_info['frame_count']}. The video may have changed."
                )

        render_videos(keypoints_file, video_path, staging_dir, ffmpeg, ffprobe)
        if len(track_counts) < 2:
            print(
                "Warning: fewer than two tracked people were found inside the selected court. "
                "Check the court selection and original video.",
                file=sys.stderr,
            )
        summary = {
            "source": video_path.name,
            "model": model_name,
            "tracker": "ByteTrack",
            "device": "cpu",
            "image_size": image_size,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
            "frame_count": frame_index,
            "tracked_frames": tracked_frames,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "tracker_ids_and_observation_counts": {
                str(track_id): count
                for track_id, count in sorted(track_counts.items())
            },
            "tracker_id_warning": (
                "IDs are ByteTrack labels for this video only. They are not player names "
                "or guaranteed identity; IDs can change after a long miss or crossing."
            ),
        }
        summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.rename(final_dir)
    except BaseException:
        capture.release()
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    finally:
        capture.release()

    print(f"Done in {time.monotonic() - started:.1f} seconds.")
    print(f"Results: {final_dir}")
    print(f"  Annotated video: {final_dir / 'annotated.mp4'}")
    print(f"  Side by side:    {final_dir / 'side-by-side.mp4'}")
    print(f"  Keypoints:       {final_dir / 'keypoints.json'}")
    print(f"  Run summary:     {final_dir / 'summary.json'}")
    return final_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="badminton-coach",
        description="Run real CPU pose detection on a video and save aligned MP4/JSON outputs.",
    )
    parser.add_argument("video", type=Path, help="local input MP4")
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="folder for new per-run result folders (default: ./outputs)",
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help="pretrained Ultralytics pose weights (default: yolo26s-pose.pt)",
    )
    parser.add_argument(
        "--imgsz", type=int, choices=(640, 960, 1280), default=DEFAULT_IMAGE_SIZE,
        help="inference image size; larger sizes help small distant players but run slower",
    )
    parser.add_argument(
        "--joint-threshold", type=float, default=0.25,
        help="hide individual joints below this confidence (default: 0.25)",
    )
    parser.add_argument(
        "--person-threshold", type=float, default=0.25,
        help="hide people below this confidence after ByteTrack (default: 0.25)",
    )
    args = parser.parse_args()
    for name, value in (
        ("joint threshold", args.joint_threshold),
        ("person threshold", args.person_threshold),
    ):
        if not 0 <= value <= 1:
            parser.error(f"{name} must be between 0 and 1.")
    return args


def main() -> int:
    args = parse_args()
    try:
        analyze_video(
            args.video, args.output_root, args.model,
            args.imgsz, args.joint_threshold, args.person_threshold,
        )
    except KeyboardInterrupt:
        print("\nAnalysis cancelled. The incomplete output was removed.", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        details = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) else str(error)
        print(f"Analysis failed: {details}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
