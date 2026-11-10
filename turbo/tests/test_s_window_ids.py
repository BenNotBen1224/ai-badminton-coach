"""Window refinement must not reuse a fresh tracker's local ids.

A tracker that starts again at zero can label the other player as id 0.
label_players then groups that id with the existing track, including across
the window edge. This drives the real window operation with the detection
order reversed. No video and no model compile.
"""
import json
import unittest
from pathlib import Path

import service


NEAR = [400.0, 700.0, 520.0, 980.0]
FAR = [400.0, 80.0, 500.0, 320.0]
STRAY_A = [10.0, 10.0, 40.0, 40.0]
STRAY_B = [1500.0, 10.0, 1560.0, 50.0]
WIDTH, HEIGHT, FPS, N = 1920, 1080, 30.0, 24


class _Compiled:
    def create_infer_request(self):
        return object()


class _Model:
    compiled = _Compiled()


class _Pack:
    pose = _Model()
    det = _Model()


def _joints(box):
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    return [[x, y, 0.9] for _ in range(17)]


def _person(tid, box, fi):
    return service.sampler.player_row(tid, box, 0.9, _joints(box), fi, FPS, "interpolated")


def _base():
    frames = []
    for i in range(N):
        players = []
        # The window interiors have no rows, so identity can come only from the edge.
        if i not in set(range(0, 4)) | set(range(20, 24)):
            players = [_person(0, NEAR, i), _person(1, FAR, i)]
        frames.append({
            "frame_index": i,
            "timestamp_seconds": round(i / FPS, 4),
            "players": players,
        })
    return {
        "schema_version": 3,
        "tier": "preview",
        "source": {
            "file": "clip.mp4", "width": WIDTH, "height": HEIGHT,
            "fps": FPS, "frame_count": N,
        },
        "court_lines_normalized": [[0.2, 0.15], [0.8, 0.15], [0.85, 0.9], [0.15, 0.9]],
        "court_fit": {"ok": True},
        "play_area_normalized": [[0.1, 0.05], [0.9, 0.05], [0.9, 0.95], [0.1, 0.95]],
        "cuts": [],
        "frames": frames,
    }


def _hit(tid, box, fi):
    return (tid, box, 0.9, _joints(box))


class WindowIdentityTests(unittest.TestCase):
    def test_reversed_order_uses_both_edges_and_does_not_alias(self):
        base = _base()
        before = json.dumps(base)
        outside = json.loads(json.dumps(base["frames"][4:20]))
        # Window at the start has only a right edge. Window at the end has only a left edge.
        # Local id 0 is the other person in the second window.
        plans = {
            0: {
                0: [_hit(0, FAR, 0), _hit(1, NEAR, 0), _hit(2, STRAY_A, 0)],
                3: [_hit(0, FAR, 3), _hit(1, NEAR, 3)],
            },
            20: {
                20: [_hit(0, NEAR, 20), _hit(1, FAR, 20), _hit(2, STRAY_B, 20)],
                23: [_hit(0, NEAR, 23), _hit(1, FAR, 23)],
            },
        }
        svc = service.TurboService(Path("unused"))
        svc._refined_models = lambda: _Pack()
        seen = {}

        def fake_span(video, start, end_exclusive):
            seen["start"] = start
            return [None] * (end_exclusive - start), end_exclusive - start

        def fake_track(frames, court, det, pose, ovcache, det_req, pose_req, detect_every):
            rows = plans.pop(seen["start"])
            return rows, 0.0, 0.0, 3, 0.0

        real_span, real_track = service.decode.decode_span, service._track
        service.decode.decode_span = fake_span
        service._track = fake_track
        try:
            doc, _timings = svc.analyze(
                "clip.mp4", tier="window", windows=[(0, 3), (20, 23)], base_doc=base)
        finally:
            service.decode.decode_span = real_span
            service._track = real_track

        self.assertEqual(json.dumps(base), before)
        self.assertEqual(doc["tier"], "preview")
        self.assertEqual(doc["frames"][4:20], outside)
        self.assertEqual(doc["frames"][4], json.loads(before)["frames"][4])
        self.assertEqual(doc["frames"][19], json.loads(before)["frames"][19])

        def ids_for(fi, box):
            hits = [
                p["track_id"] for p in doc["frames"][fi]["players"]
                if service.sampler._iou(p["bbox_xyxy"], box) >= 0.3
            ]
            self.assertEqual(len(hits), 1, (fi, box, doc["frames"][fi]["players"]))
            return hits[0]

        # Right edge of the opening window, local order reversed.
        self.assertEqual(ids_for(0, NEAR), 0)
        self.assertEqual(ids_for(3, FAR), 1)
        # Left edge of the closing window, opposite local order.
        self.assertEqual(ids_for(20, NEAR), 0)
        self.assertEqual(ids_for(23, FAR), 1)
        stray = [
            p["track_id"] for fi in (0, 20) for p in doc["frames"][fi]["players"]
            if p["track_id"] not in (0, 1)
        ]
        self.assertEqual(len(stray), 2)
        self.assertEqual(len(set(stray)), 2)
        self.assertTrue(all(tid > 1 for tid in stray))
        self.assertEqual(doc["windows_refined"][0]["start"], 0)
        self.assertEqual(doc["windows_refined"][0]["end"], 3)
        self.assertEqual(doc["windows_refined"][1]["end"], 23)


if __name__ == "__main__":
    unittest.main()
