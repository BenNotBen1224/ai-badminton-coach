"""Fresh, conservative court/player/shuttle evidence for commentary reels."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from .app.backend import FastBackend, NoCourt, sha256_file
from studio.shuttle.shots import closeness, feet, height

TRACKNET_SHA = "df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a"


def normalize(source, target, ffmpeg):
    result = subprocess.run([
        str(ffmpeg), "-v", "error", "-n", "-threads", "4", "-i", str(source),
        "-vf", "scale=trunc(iw*sar/2)*2:ih,setsar=1,fps=30", "-an", "-c:v", "libx264", "-threads", "4",
        "-preset", "fast", "-crf", "18", str(target),
    ], capture_output=True)
    if result.returncode:
        raise ValueError("Cannot normalize source timestamps: "+result.stderr.decode("utf-8", "replace")[-2000:])


def shuttle_model(models):
    import openvino as ov
    import torch

    checkpoint = models / "TrackNetV3/ckpts/TrackNet_best.pt"
    if not checkpoint.is_file() or sha256_file(checkpoint) != TRACKNET_SHA:
        raise ValueError("Missing or changed TrackNetV3 weights; run python scripts/get_models.py tracknet")
    runtime = models / "badminton-pipeline-repro/scripts/tracknet_runtime/model.py"
    if not runtime.is_file():
        raise FileNotFoundError("TrackNet runtime missing; run python scripts/get_models.py tracknet")
    xml = checkpoint.parent / "reel-tracknet.xml"
    if not xml.exists():
        torch.set_num_threads(8)
        spec = importlib.util.spec_from_file_location("reel_tracknet_runtime", runtime)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["param_dict"]["seq_len"] != 8 or state["param_dict"]["bg_mode"] != "concat":
            raise ValueError("Unsupported TrackNet checkpoint preprocessing")
        model = module.TrackNet(27, 8).eval()
        model.load_state_dict(state["model"])
        converted = ov.convert_model(model, example_input=torch.zeros(1, 27, 288, 512))
        ov.save_model(converted, xml, compress_to_fp16=False)
    core = ov.Core()
    return core.compile_model(str(xml), "CPU", {"INFERENCE_NUM_THREADS": 8, "PERFORMANCE_HINT": "LATENCY"})


def track_shuttle(video, models, progress):
    net = shuttle_model(models)
    cap = cv2.VideoCapture(str(video))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, tall = cap.get(3), cap.get(4)
    background = []
    for frame in np.linspace(0, count - 1, min(160, count)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = cap.read()
        if ok:
            background.append(cv2.resize(image, (512, 288))[..., ::-1])
    if not background:
        raise ValueError("Source has no decodable pictures")
    median = np.median(background, axis=0).astype(np.float32).transpose(2, 0, 1) / 255
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    points = []
    started = time.perf_counter()
    while len(points) < count:
        images = []
        for _ in range(8):
            ok, image = cap.read()
            if not ok:
                break
            images.append(cv2.resize(image, (512, 288))[..., ::-1])
        if not images:
            break
        valid = len(images)
        images.extend([images[-1]] * (8 - valid))
        rgb = np.asarray(images, np.float32).transpose(0, 3, 1, 2).reshape(24, 288, 512) / 255
        data = np.concatenate((median, rgb))[None]
        maps = net([data])[net.output(0)][0]
        for heat in maps[:valid]:
            mask = (heat > .5).astype(np.uint8)
            n, _, stats, centers = cv2.connectedComponentsWithStats(mask)
            if n < 2:
                points.append(None)
                continue
            chosen = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            x, y = centers[chosen]
            points.append([round(x * width / 512, 2), round(y * tall / 288, 2),
                           round(float(heat.max()), 4)])
        if len(points) % 800 == 0:
            progress(f"shuttle {len(points)}/{count}, {time.perf_counter()-started:.1f}s")
    cap.release()
    if len(points) != count:
        raise ValueError(f"Shuttle decoder read {len(points)} of {count} normalized frames")
    return points


def player_frames(doc):
    width, tall = doc["source"]["width"], doc["source"]["height"]
    homographies = {}
    frames = []
    for frame in doc["frames"]:
        corners = frame.get("court_lines_normalized", doc["court_lines_normalized"])
        key = tuple(v for point in corners for v in point)
        if key not in homographies:
            quad = np.float32(corners) * [width, tall]
            homographies[key] = cv2.getPerspectiveTransform(
                quad.astype(np.float32), np.float32([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]]))
        homography = homographies[key]
        roles = {}
        for p in frame["players"]:
            if p.get("role") not in ("near", "far"):
                continue
            joints = np.asarray([j if j is not None else [0, 0, 0] for j in p["joints"]], float)
            if joints.shape != (17, 3) or np.median(joints[[5, 6, 11, 12], 2]) < .5:
                continue
            court = feet(p, homography)
            if not (-.8 <= court[0] <= 6.9 and -.8 <= court[1] <= 14.2):
                continue
            roles[p["role"]] = {**p, "joints": joints.tolist(), "court_m": court.tolist()}
        frames.append(roles)
    return frames


def find_hits(points, frames, cuts):
    """Track turns or closest racket-side approaches; image-plane velocity need not reverse at a hit."""
    xy = np.asarray([p[:2] if p else [np.nan, np.nan] for p in points], float)
    wrist_distances = {role: np.full(len(xy), np.inf) for role in ("near", "far")}
    for i, players in enumerate(frames):
        if not np.isfinite(xy[i]).all():
            continue
        for role, player in players.items():
            joints = np.asarray(player["joints"], float)[[9, 10]]
            wrists = joints[joints[:, 2] >= .5]
            stature = player["bbox_xyxy"][3]-player["bbox_xyxy"][1]
            if len(wrists) and stature > 0:
                wrist_distances[role][i] = np.linalg.norm(wrists[:, :2]-xy[i], axis=1).min()/stature
    candidates = []
    for i in range(4, len(xy) - 5):
        if len(frames[i]) != 2 or not np.isfinite(xy[i]).all() or any(abs(c-i) <= 8 for c in cuts):
            continue
        before = [j for j in range(i-4, i) if np.isfinite(xy[j]).all()]
        after = [j for j in range(i+1, i+5) if np.isfinite(xy[j]).all()]
        if len(before) < 2 or len(after) < 2:
            continue
        incoming = (xy[i] - xy[before[0]]) / (i-before[0])
        outgoing = (xy[after[-1]] - xy[i]) / (after[-1]-i)
        speed = float(np.linalg.norm(incoming) + np.linalg.norm(outgoing))
        if speed < 5 or speed > 300:
            continue
        cos = float(incoming @ outgoing / max(np.linalg.norm(incoming)*np.linalg.norm(outgoing), .001))
        distances = sorted((wrist_distances[role][i], role) for role in frames[i])
        wrist_distance, role = distances[0]
        distance = closeness(frames[i][role], xy[i])
        if distance > 1.05 or distances[1][0] - wrist_distance < .15:
            continue
        player = frames[i][role]
        h = player["bbox_xyxy"][3] - player["bbox_xyxy"][1]
        joints = np.asarray(player["joints"], float)
        wrists = joints[[9, 10]]
        wrists = wrists[wrists[:, 2] >= .5]
        wrist_distance = float(np.linalg.norm(wrists[:, :2]-xy[i], axis=1).min()/h) if len(wrists) else 9
        if wrist_distance > .65:
            continue
        local = wrist_distances[role][max(0, i-4):i+5]
        closest_approach = wrist_distance <= .35 and wrist_distance <= float(local.min())+.02
        if cos > .65 and not closest_approach:
            continue
        contact_height, ratio = height(player, xy[i][1])
        candidates.append({
            "frame": i, "time_s": i/30, "hitter": role, "xy_px": xy[i].tolist(),
            "hitter_court_m": player["court_m"], "contact_height": contact_height,
            "height_ratio": ratio, "wrist_distance_statures": wrist_distance,
            "turn_cosine": cos, "closest_wrist_approach": closest_approach,
            "confidence": round(max(min(1., (1-cos)/2)*max(0., 1-wrist_distance/1.3),
                                    (1-wrist_distance/.65)*.85 if closest_approach else 0), 3),
            "score": max(1-cos, 2*(1-wrist_distance/.65) if closest_approach else 0)/(distance+.3),
            "family": None, "name_evidence": [],
        })
    peaks = []
    for hit in sorted(candidates, key=lambda h: -h["score"]):
        if all(abs(hit["frame"]-other["frame"]) >= 8 for other in peaks):
            peaks.append(hit)
    peaks.sort(key=lambda h: h["frame"])
    hits = []
    for hit in peaks:
        if hits and hit["hitter"] == hits[-1]["hitter"] and hit["frame"]-hits[-1]["frame"] < 65:
            if hit["score"] > hits[-1]["score"]:
                hits[-1] = hit
        else:
            hits.append(hit)
    for a, b in zip(hits, hits[1:]):
        dt = (b["frame"]-a["frame"])/30
        if not .25 <= dt <= 2.8 or a["hitter"] == b["hitter"] or any(a["frame"] < c <= b["frame"] for c in cuts):
            continue
        flight = points[a["frame"]:b["frame"]+1]
        observed = [p for p in flight if p]
        visible = len(observed)/len(flight)
        a.update(next_hit_frame=b["frame"], flight_s=dt, receiver_court_m=b["hitter_court_m"],
                 shuttle_path=[{"frame": a["frame"]+k, "xy_px": p[:2]} for k, p in enumerate(flight) if p],
                 visible_fraction=visible)
        receiver_before = frames[a["frame"]].get(b["hitter"])
        if receiver_before:
            a["receiver_before_court_m"] = receiver_before["court_m"]
        if visible < .65 or a["wrist_distance_statures"] > .45 or a["confidence"] < .6:
            continue
        rear = b["hitter_court_m"][1] < 3.2 if b["hitter"] == "far" else b["hitter_court_m"][1] > 10.2
        # A long, observed high flight ending at the other baseline distinguishes deep returns.
        apex = min(p[1] for p in observed)
        rise = a["xy_px"][1]-apex
        player = frames[a["frame"]][a["hitter"]]
        stature = player["bbox_xyxy"][3]-player["bbox_xyxy"][1]
        if rear and dt >= 1.05 and rise > stature*.55:
            if a["contact_height"] == "overhead" and a["height_ratio"] >= 1.3:
                a["family"] = "clear"
                a["name_evidence"] = ["measured overhead contact", "observed rising arc", "receiver at rear court"]
            elif a["contact_height"] == "low" and a["height_ratio"] <= .25:
                a["family"] = "lift"
                a["name_evidence"] = ["measured low contact", "observed rising arc", "receiver at rear court"]
        a["confidence"] = round(min(a["confidence"], visible), 3)
    return hits


def rally_windows(hits, cuts=()):
    groups = []
    for hit in hits:
        if (not groups or hit["time_s"]-groups[-1][-1]["time_s"] > 3.5
                or any(groups[-1][-1]["frame"] < cut <= hit["frame"] for cut in cuts)):
            groups.append([])
        groups[-1].append(hit)
    return [{"start_s": max(0, group[0]["time_s"]-.5), "end_s": group[-1]["time_s"]+.7,
             "hit_frames": [h["frame"] for h in group]}
            for group in groups if len(group) >= 4
             and sum(a["hitter"] != b["hitter"] for a, b in zip(group, group[1:])) >= 3]


def boundary_is_cut(video, offset):
    if offset == 0:
        return False
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, offset-1)
    signatures = []
    for _ in range(2):
        ok, image = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"Cannot inspect source pictures at analysis boundary {offset}")
        tall = round(image.shape[0]*1280/image.shape[1])
        signatures.append(cv2.resize(image, (1280, tall))[::16, ::16, 1].astype(np.int16))
    cap.release()
    # Match Fast's cut rule without importing its executable pipeline and global path setup.
    return float((np.abs(signatures[0]-signatures[1]) > 30).mean()) > .022


def dense_pose(video, folder, ffmpeg, progress):
    """Bound Fast's decode queues with short downscaled inputs, not an entire match in RAM."""
    cap = cv2.VideoCapture(str(video))
    count, width, tall = int(cap.get(7)), int(cap.get(3)), int(cap.get(4))
    cap.release()
    combined, first_quad = [], None
    cuts, timings = [], []
    for offset in range(0, count, 480):
        size = min(480, count-offset)
        clip = folder / f"pose-input-{offset}.mp4"
        out = folder / f"pose-part-{offset}.json"
        subprocess.run([str(ffmpeg), "-v", "error", "-y", "-threads", "2",
                        "-ss", str(offset/30), "-i", str(video), "-frames:v", str(size),
                        "-vf", "scale=1280:-2", "-an", "-c:v", "libx264", "-threads", "2",
                        "-preset", "ultrafast", "-crf", "16", str(clip)],
                       capture_output=True, check=True)
        try:
            doc, timing = FastBackend().refined(clip, out)
        except NoCourt as exc:
            progress(f"excluding non-court interval {offset/30:.1f}s: {exc}")
            combined.extend({"frame_index": i, "timestamp_seconds": i/30, "players": []}
                            for i in range(offset, offset+size))
            cuts.extend((offset, offset+size))
            clip.unlink()
            continue
        if first_quad is None:
            first_quad = doc["court_lines_normalized"]
        sx, sy = width/doc["source"]["width"], tall/doc["source"]["height"]
        for i, frame in enumerate(doc["frames"]):
            frame["frame_index"], frame["timestamp_seconds"] = offset+i, (offset+i)/30
            frame["court_lines_normalized"] = doc["court_lines_normalized"]
            for p in frame["players"]:
                p["bbox_xyxy"] = [v*(sx if j % 2 == 0 else sy) for j, v in enumerate(p["bbox_xyxy"])]
                p["joints"] = [[j[0]*sx, j[1]*sy, j[2]] if j is not None else [0, 0, 0]
                               for j in p["joints"]]
            combined.append(frame)
        cuts.extend(offset+c for c in doc["cuts"] if c > 0)
        if boundary_is_cut(video, offset):
            cuts.append(offset)
        timings.append(timing)
        clip.unlink()
        out.unlink()
        progress(f"pose {offset+size}/{count}")
    if first_quad is None:
        raise ValueError("No court found; input must show a singles court from behind a baseline")
    if len(combined) != count:
        raise ValueError("Pose frame count does not match normalized source")
    return {"schema_version": 3,
            "source": {"file": Path(video).name, "width": width, "height": tall, "fps": 30, "frame_count": count},
            "court_lines_normalized": first_quad, "cuts": sorted(set(cuts)), "frames": combined}, timings


def analyze(video, folder, models, ffmpeg, progress=print):
    started = time.perf_counter()
    doc, timing = dense_pose(video, folder, ffmpeg, progress)
    frames = player_frames(doc)
    coverage = sum(len(f) == 2 for f in frames)/len(frames)
    if coverage < .15:
        raise ValueError(f"Both centre-court players are not reliable enough (coverage {coverage:.1%})")
    progress("tracking shuttle on all normalized frames")
    points = track_shuttle(video, models, progress)
    hits = find_hits(points, frames, doc["cuts"])
    rallies = rally_windows(hits, doc["cuts"])
    return doc, {
        "fps": 30, "frames": len(frames), "both_players_fraction": coverage,
        "court": doc["court_lines_normalized"], "cuts": doc["cuts"], "shuttle": points,
        "hits": hits, "rallies": rallies, "named_hits": sum(h["family"] is not None for h in hits),
        "timings": {"pose": timing, "total_s": time.perf_counter()-started},
        "model_inputs": {
            "tracknet_weights_sha256": TRACKNET_SHA,
            "tracknet_runtime_sha256": sha256_file(models/"badminton-pipeline-repro/scripts/tracknet_runtime/model.py"),
            "tracknet_openvino_xml_sha256": sha256_file(models/"TrackNetV3/ckpts/reel-tracknet.xml"),
            "tracknet_openvino_bin_sha256": sha256_file(models/"TrackNetV3/ckpts/reel-tracknet.bin"),
        },
        "limits": [
            "Contact candidates and shot families are model estimates, not ground truth.",
            "Only long, well-observed deep clear/lift trajectories receive family names.",
            "Other contacts abstain; no smash/drop/backhand/jump claim from pose alone.",
            "30 fps normalized browser footage can contain repeated or dropped captured frames.",
        ],
    }
