from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import zipfile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run import HERE, METHODS, RESULTS, ORIGINAL, WEIGHTS, gallery, load, save, sha, validate_data, require_ffmpeg


def main():
    _, probe = require_ffmpeg()
    manifest = load(HERE / "manifest.json")
    if set(manifest["methods"]) != set(METHODS):
        raise ValueError("Candidate list is incomplete")
    panels = []
    evidence = {}
    for name, config in manifest["methods"].items():
        if config["status"] != "complete":
            if config["status"] != "blocked" or not config.get("error"):
                raise ValueError(f"Unfinished candidate {name}: {config['status']}")
            continue
        folder = RESULTS / name
        data = load(folder / "keypoints.json")
        validate_data(data)
        if name in ("crop", "rtmpose", "vitpose", "sapiens", "posewarper") and data["model"].get("acquisition_revision") != 2:
            raise ValueError(f"{name} uses stale detector acquisition")
        if name in ("rtmpose", "vitpose", "sapiens", "posewarper") and data.get("shared_acquisition_sha256") != sha(RESULTS / "crop" / "acquisition.json"):
            raise ValueError(f"{name} used different acquired boxes")
        result = subprocess.run([
            str(probe), "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_read_frames,avg_frame_rate:format=duration",
            "-of", "json", str(folder / "annotated.mp4"),
        ], capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        expected = {"codec_name": "h264", "pix_fmt": "yuv420p", "width": 1920,
                    "height": 1080, "nb_read_frames": "780", "avg_frame_rate": "30/1"}
        for key, value in expected.items():
            if stream.get(key) != value:
                raise ValueError(f"{name}: invalid {key}={stream.get(key)}")
        if abs(float(info["format"]["duration"]) - 26) > .05:
            raise ValueError(f"{name}: duration mismatch")
        capture = cv2.VideoCapture(str(folder / "annotated.mp4"))
        decoded = []
        row = []
        for second in (8, 15, 22):
            capture.set(cv2.CAP_PROP_POS_FRAMES, second * 30)
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode {name} at {second}s")
            decoded.append(hashlib.sha256(image.tobytes()).hexdigest())
            thumb = cv2.resize(image, (640, 360))
            cv2.rectangle(thumb, (0, 0), (640, 36), (15, 15, 15), -1)
            cv2.putText(thumb, f"{name} - {second}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        .65, (255, 255, 255), 1, cv2.LINE_AA)
            row.append(thumb)
        capture.release()
        if len(set(decoded)) != 3:
            raise ValueError(f"{name}: frozen inspection frames")
        panels.append(np.concatenate(row, axis=1))
        unique_poses = {
            json.dumps(p["joints"], separators=(",", ":"))
            for frame in data["frames"] for p in frame["players"]
        }
        evidence[name] = {"ffprobe": info, "unique_observed_display_poses": len(unique_poses),
                          "hard_frame_decoded_image_sha256": decoded, "keypoints_sha256": sha(folder / "keypoints.json"),
                          "video_sha256": sha(folder / "annotated.mp4")}
        if len(unique_poses) < 2:
            raise ValueError(f"{name} contains no varying pose observations")
    if sha(RESULTS / "original" / "annotated.mp4") != sha(ORIGINAL / "annotated.mp4"):
        raise ValueError("Original video has changed")
    if sha(HERE / "sample-videos" / "source-attribution.json") != sha(HERE.parent / "sample-videos" / "source-attribution.json"):
        raise ValueError("Footage attribution copy differs from original")
    cv2.imwrite(str(HERE / "overview.jpg"), np.concatenate(panels, axis=0))
    save(HERE / "verification.json", {"checked_at_unix": time.time(), "methods": evidence,
                                     "scope": "All frames decoded by FFprobe; H264/yuv420p, shape, fps and duration verified. Hard frames decoded at8/15/22s. Unique pose observations detect frozen-pose output, not pose accuracy."})
    common = load(RESULTS / "crop" / "acquisition.json")
    manifest["shared_acquisition"] = {
        "method": "YOLO26s-Pose/ByteTrack + targeted crop retries, >=0.1 before display cutoff",
        "revision": common["model"]["acquisition_revision"],
        "path": "results/crop/acquisition.json",
        "sha256": sha(RESULTS / "crop" / "acquisition.json"),
        "elapsed_seconds": manifest["methods"]["crop"]["inference_seconds"],
        "used_by": ["rtmpose", "vitpose", "sapiens", "posewarper"],
        "timing_note": "Top-down pose inference times exclude this reusable detector pass; add it for uncached pipeline cost.",
    }
    manifest["asset_downloads"] = [load(p) for p in WEIGHTS.glob("*.download.json")]
    manifest["prefetch_timing_note"] = "Initial RTMPose/RTMO/ViTPose/MediaPipe and MoveNet checkpoint downloads were prefetched outside inference and their elapsed times were not instrumented. Model setup timers exclude these prefetches. Later downloads have explicit asset timings."
    manifest["provenance_files"] = {
        "PoseWarper": "posewarper-port.json", "Sapiens": "weights/sapiens-port.json",
    }
    manifest["methods"]["original"]["checkpoint_sha256"] = sha(HERE.parent / "yolo26s-pose.pt")
    manifest["methods"]["original"].pop("elapsed_original_inference_and_render_seconds", None)
    save(HERE / "manifest.json", manifest)
    gallery()
    include = [HERE / n for n in ("index.html", "manifest.json", "setup.json", "verification.json", "overview.jpg",
                                  "posewarper-port.json", "browser-verification.json", "checkpoint-metadata.json", "requirements.txt", "requirements-movenet.txt", "requirements-sapiens.txt")]
    for name, config in manifest["methods"].items():
        folder = RESULTS / name
        for filename in ("annotated.mp4", "keypoints.json", "acquisition.json", "frame-8s.jpg", "frame-15s.jpg", "frame-22s.jpg", "failure.txt"):
            if (folder / filename).exists():
                include.append(folder / filename)
    include += list((HERE / "posewarper_vendor").glob("LICENSE*"))
    include += [p for p in (WEIGHTS / "SAPIENS-LICENSE.txt", WEIGHTS / "sapiens-port.json",
                           WEIGHTS / "sapiens_classes_and_palettes.py",
                           HERE / "sample-videos" / "source-attribution.json") if p.exists()]
    archive = HERE / "badminton-model-comparison.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as out:
        for path in include:
            out.write(path, path.relative_to(HERE).as_posix())
    print(json.dumps({"complete_videos": len(evidence), "gallery": str(HERE / "index.html"),
                      "archive": str(archive), "archive_bytes": archive.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
