import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from PIL import Image

from coach import schema as S
from coach.exercises.library import EQUIPMENT, for_tag, gif_path, load_library, variant
from coach.exercises.poses import (
    FOREARM,
    UPPER_ARM,
    Pt,
    chest_yaw,
    dist3,
    seam_error,
    validate_frame,
    validate_sequence,
    yaw_y,
)
from coach.exercises.render import GIF_DIR, SHEET_DIR, frames_for


def _row(**kw):
    base = dict(
        id="wall_sit",
        name="Wall sit",
        fix_tags=["lower_stance"],
        equipment=[],
        sets_reps="3 x 30 seconds",
        cues=["back flat", "knees over ankles", "breathe"],
        mistakes=["knees cave in", "hips off the wall"],
        safety="Stop if you feel sharp knee pain.",
        difficulty=1,
        gif="gifs/wall_sit.gif",
        duration_s=2.4,
        why="Holds the low ready shape.",
    )
    base.update(kw)
    return base


def _write(rows):
    folder = tempfile.TemporaryDirectory()
    path = Path(folder.name) / "library.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return folder, path


def _exercise(id, tags, difficulty, equipment=()):
    return S.Exercise(
        id=id,
        name=id,
        fix_tags=tags,
        equipment=equipment,
        sets_reps="3 x 8",
        cues=("keep the chest up", "knees out", "breathe out"),
        mistakes=("standing tall", "heels up"),
        safety="Stop if you feel sharp pain.",
        difficulty=difficulty,
        gif="gifs/missing.gif",
        duration_s=2.4,
        why="A stub exercise for the picker.",
    )


class LibraryLoad(unittest.TestCase):
    def test_loads_the_shipped_library(self):
        library = load_library()
        self.assertGreaterEqual(len(library), 16)
        self.assertLessEqual(len(library), 20)
        self.assertEqual(library["wall_sit"].name, "Wall sit")
        self.assertEqual(library["wall_sit"].equipment, ("wall",))
        self.assertEqual(library["sit_low_hold"].equipment, ())
        self.assertEqual(library["goblet_squat"].difficulty, 2)

    def test_unknown_equipment_is_rejected(self):
        folder, path = _write([_row(equipment=["barbell"])])
        self.addCleanup(folder.cleanup)
        with self.assertRaises(ValueError) as caught:
            load_library(path)
        self.assertIn("unknown equipment", str(caught.exception))

    def test_bodyweight_is_not_an_equipment_name(self):
        self.assertNotIn("bodyweight", EQUIPMENT)
        folder, path = _write([_row(equipment=["bodyweight"])])
        self.addCleanup(folder.cleanup)
        with self.assertRaises(ValueError) as caught:
            load_library(path)
        self.assertIn("bodyweight", str(caught.exception))

    def test_a_missing_variant_id_is_rejected(self):
        folder, path = _write([_row(harder="missing")])
        self.addCleanup(folder.cleanup)
        with self.assertRaises(ValueError) as caught:
            load_library(path)
        self.assertIn("not in the library", str(caught.exception))

    def test_a_ladder_edge_with_no_shared_tag_is_rejected(self):
        rows = [
            _row(id="shadow", fix_tags=["base_recovery"], harder="hops"),
            _row(id="hops", difficulty=2, fix_tags=["split_step"]),
        ]
        folder, path = _write(rows)
        self.addCleanup(folder.cleanup)
        with self.assertRaises(ValueError) as caught:
            load_library(path)
        self.assertIn("shares no fix tag", str(caught.exception))

    def test_variant_difficulty_must_move_the_right_way(self):
        rows = [
            _row(id="steep", difficulty=2, easier="light"),
            _row(id="light", difficulty=3, fix_tags=["endurance"]),
        ]
        folder, path = _write(rows)
        self.addCleanup(folder.cleanup)
        with self.assertRaises(ValueError) as caught:
            load_library(path)
        self.assertIn("higher difficulty", str(caught.exception))


class ForTag(unittest.TestCase):
    def setUp(self):
        self.library = {
            "prime_aaa": _exercise("prime_aaa", ("lower_stance", "balance"), 1),
            "prime_easy": _exercise("prime_easy", ("lower_stance",), 1),
            "prime_hard": _exercise("prime_hard", ("lower_stance",), 2),
            "other_easy": _exercise("other_easy", ("balance", "lower_stance"), 1, ("wall",)),
            "too_hard": _exercise("too_hard", ("lower_stance",), 3),
        }

    def test_primary_tag_then_easier_then_id(self):
        picked = [item.id for item in for_tag(self.library, "lower_stance", limit=5)]
        self.assertEqual(picked, ["prime_aaa", "prime_easy", "prime_hard", "too_hard", "other_easy"])

    def test_limit_and_difficulty_cap(self):
        picked = [item.id for item in for_tag(self.library, "lower_stance", limit=2, max_difficulty=1)]
        self.assertEqual(picked, ["prime_aaa", "prime_easy"])

    def test_equipment_filter_skips_what_the_player_does_not_have(self):
        any_gear = [item.id for item in for_tag(self.library, "balance")]
        self.assertEqual(any_gear, ["other_easy", "prime_aaa"])
        bodyweight = [item.id for item in for_tag(self.library, "balance", equipment_ok=())]
        self.assertEqual(bodyweight, ["prime_aaa"])
        with_wall = [item.id for item in for_tag(self.library, "balance", equipment_ok={"wall"})]
        self.assertEqual(with_wall, ["other_easy", "prime_aaa"])

    def test_unknown_tag_is_an_error(self):
        with self.assertRaises(ValueError) as caught:
            for_tag(self.library, "nope")
        self.assertIn("unknown fix tag", str(caught.exception))


class VariantLookup(unittest.TestCase):
    def test_walks_the_shipped_ladder(self):
        library = load_library()
        wall = library["wall_sit"]
        self.assertIsNone(variant(library, wall, "easier"))
        self.assertEqual(variant(library, wall, "harder").id, "sit_low_hold")
        jumps = library["squat_jumps"]
        self.assertEqual(variant(library, jumps, "easier").id, "goblet_squat")
        self.assertIsNone(variant(library, jumps, "harder"))

    def test_rejects_an_unknown_direction(self):
        library = load_library()
        with self.assertRaises(ValueError):
            variant(library, library["wall_sit"], "sideways")


class CoverageAndLadder(unittest.TestCase):
    def test_every_fix_tag_has_two_exercises_and_an_easy_one(self):
        library = load_library()
        gaps = []
        for tag in S.FIX_TAGS:
            hits = [item for item in library.values() if tag in item.fix_tags]
            easy = [item for item in hits if item.difficulty <= 2]
            if len(hits) < 2 or not easy:
                gaps.append((tag, len(hits), len(easy)))
        self.assertEqual(gaps, [])

    def test_ladder_links_exist_and_at_least_ten_exercises_have_one(self):
        library = load_library()
        linked = 0
        bad = []
        for exercise in library.values():
            if exercise.easier is None and exercise.harder is None:
                continue
            linked += 1
            if exercise.easier is not None:
                other = library[exercise.easier]
                if other.difficulty > exercise.difficulty:
                    bad.append(exercise.id)
            if exercise.harder is not None:
                other = library[exercise.harder]
                if other.difficulty < exercise.difficulty:
                    bad.append(exercise.id)
        self.assertEqual(bad, [])
        self.assertGreaterEqual(linked, 10)

    def test_every_ladder_edge_shares_a_fix_tag(self):
        library = load_library()
        bad = []
        for exercise in library.values():
            for which in ("easier", "harder"):
                other_id = getattr(exercise, which)
                if other_id is None:
                    continue
                shared = set(exercise.fix_tags) & set(library[other_id].fix_tags)
                if not shared:
                    bad.append((exercise.id, which, other_id))
        self.assertEqual(bad, [])
        self.assertIn("base_recovery", set(library["six_corner_shadow"].fix_tags) & set(library["pause_lunge_recover"].fix_tags))


class GifFiles(unittest.TestCase):
    def test_every_exercise_gif_exists_and_is_under_1mb(self):
        library = load_library()
        sizes = []
        for exercise in library.values():
            path = gif_path(exercise)
            sizes.append(path.stat().st_size)
            self.assertLess(path.stat().st_size, 1_000_000)
            self.assertEqual(path.name, f"{exercise.id}.gif")
            with Image.open(path) as gif:
                self.assertEqual(gif.size, (480, 360))
            decoded = _gif_delay_sum_ms(path)
            expected = int(round(exercise.duration_s * 100)) * 10
            self.assertEqual(decoded, expected, exercise.id)
        self.assertEqual(len(sizes), len(library))

    def test_gif_path_rejects_a_missing_file(self):
        missing = _exercise("missing", ("balance",), 1)
        with self.assertRaises(FileNotFoundError):
            gif_path(missing)


def _gif_delay_sum_ms(path) -> int:
    """Sum the encoded graphic-control delays. GIF stores those in centiseconds."""
    data = Path(path).read_bytes()
    total_cs = 0
    start = 0
    marker = b"\x21\xF9\x04"
    while True:
        index = data.find(marker, start)
        if index < 0 or index + 6 >= len(data):
            break
        total_cs += data[index + 4] + (data[index + 5] << 8)
        start = index + 6
    return total_cs * 10


def _line_gap(point, a, b):
    dx, dy = b.x - a.x, b.y - a.y
    span = (dx * dx + dy * dy) ** 0.5 or 1.0
    return abs((point.x - a.x) * dy - (point.y - a.y) * dx) / span


def _line_gap3(point, a, b):
    dx, dy, dz = b.x - a.x, b.y - a.y, b.z - a.z
    span = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    cx = (point.y - a.y) * dz - (point.z - a.z) * dy
    cy = (point.z - a.z) * dx - (point.x - a.x) * dz
    cz = (point.x - a.x) * dy - (point.y - a.y) * dx
    return math.sqrt(cx * cx + cy * cy + cz * cz) / span


def _in_chest(frame, point):
    pelvis = frame.points["pelvis"]
    return yaw_y(point, pelvis.x, pelvis.z, -chest_yaw(frame))


def _post_gap(frame, anchor_x, anchor_y):
    wrist = frame.points["l_wrist"]
    return math.hypot(wrist.x - anchor_x, wrist.y - anchor_y)


def _assert_chest_yaw_keeps_the_grip(test, exercise_id, yaw_delta):
    from coach.exercises.poses import joint_angles

    frames = frames_for(exercise_id)
    start = frames[0]
    quiet = ("pelvis", "l_hip", "r_hip", "l_toe", "r_toe", "l_heel", "r_heel")
    local = _in_chest(start, start.points["l_wrist"])
    yaws = []
    for frame in frames:
        yaws.append(chest_yaw(frame))
        test.assertLess(abs(chest_yaw(frame) - frame.yaw), 1.5, frame.phase)
        left = frame.points["l_wrist"]
        right = frame.points["r_wrist"]
        test.assertLess(dist3(left, right), 1.0, frame.phase)
        held = _in_chest(frame, left)
        test.assertLess(dist3(held, local), 1.5, frame.phase)
        for name in quiet:
            test.assertLess(abs(frame.points[name].x - start.points[name].x), 0.5, name)
            test.assertLess(abs(frame.points[name].z - start.points[name].z), 0.5, name)
        angles = joint_angles(frame)
        for side in ("l", "r"):
            shoulder = frame.points[f"{side}_shoulder"]
            elbow = frame.points[f"{side}_elbow"]
            wrist = frame.points[f"{side}_wrist"]
            test.assertLess(_line_gap3(elbow, shoulder, wrist), 0.5, f"{frame.phase} {side}")
            test.assertAlmostEqual(dist3(shoulder, elbow), UPPER_ARM, delta=0.3)
            test.assertAlmostEqual(dist3(elbow, wrist), FOREARM, delta=0.3)
            test.assertLess(angles[f"{side}_elbow"], 12.0, f"{frame.phase} {side}")
    test.assertGreater(max(yaws) - min(yaws), yaw_delta)


def _reflect(point, a, b):
    dx, dy = b.x - a.x, b.y - a.y
    scale = dx * dx + dy * dy
    t = ((point.x - a.x) * dx + (point.y - a.y) * dy) / scale
    return Pt(2 * (a.x + t * dx) - point.x, 2 * (a.y + t * dy) - point.y)


class MotionValidity(unittest.TestCase):
    def test_every_sequence_is_legal_and_the_seam_closes(self):
        library = load_library()
        for exercise_id in library:
            frames = frames_for(exercise_id)
            self.assertEqual(validate_sequence(frames), [], exercise_id)
            self.assertLess(seam_error(frames), 2.0, exercise_id)
            sheet = SHEET_DIR / f"{exercise_id}.png"
            self.assertTrue(sheet.is_file(), exercise_id)
            with Image.open(sheet) as image:
                if exercise_id == "six_corner_shadow":
                    self.assertEqual(image.size[1], 360)
                    self.assertGreater(image.size[0], 480 * 8)
                else:
                    self.assertEqual(image.size, (480 * 8, 360))
            self.assertTrue((GIF_DIR / f"{exercise_id}.gif").is_file())

    def test_split_step_hop_stays_under_8cm(self):
        frames = frames_for("split_step_hops")
        hop = max(frame.points["l_toe"].y for frame in frames)
        self.assertLess(hop, 8.0)
        self.assertGreater(hop, 2.0)

    def test_six_corner_returns_to_base_between_corners(self):
        frames = frames_for("six_corner_shadow")
        order = []
        for frame in frames:
            corner = frame.props.get("corner")
            if corner and (not order or order[-1] != corner):
                order.append(corner)
        named = ["front-left", "front-right", "side-right", "back-right", "back-left", "side-left"]
        expected = []
        for corner in named:
            expected.extend(["base", corner])
        expected.append("base")
        self.assertEqual(order, expected)
        visits = []
        current = []
        for frame in frames:
            if frame.props.get("corner") == "base":
                current.append(frame)
            elif current:
                visits.append(current)
                current = []
        if current:
            visits.append(current)
        self.assertGreaterEqual(len(visits), 6)
        for visit in visits:
            closest = min(visit, key=lambda frame: abs(frame.points["pelvis"].x))
            self.assertLess(abs(closest.points["pelvis"].x), 3.0)
        peaks = {}
        for frame in frames:
            corner = frame.props.get("corner")
            if corner and corner != "base":
                peaks[corner] = max(peaks.get(corner, 0.0), abs(frame.points["pelvis"].x))
        for name in ("front-left", "front-right", "side-left", "side-right"):
            self.assertGreater(peaks[name], 16.0, name)
        for name in ("back-left", "back-right"):
            self.assertGreater(peaks[name], 12.0, name)
        hips = [frame.points["pelvis"].y for frame in frames]
        self.assertGreater(max(hips) - min(hips), 8.0)
        self.assertGreaterEqual(len(frames), 72)
        self.assertGreaterEqual(load_library()["six_corner_shadow"].duration_s, 5.0)

    def test_skater_travels_and_lands_on_the_other_foot(self):
        frames = frames_for("skater_hops")
        xs = [frame.points["pelvis"].x for frame in frames]
        travel = max(xs) - min(xs)
        self.assertGreater(travel, 30.0)
        hop = max(frame.points["l_toe"].y for frame in frames)
        self.assertGreater(travel, hop)
        lands = []
        for frame in frames:
            if frame.phase not in ("left", "right"):
                continue
            left = "l_toe" in frame.support
            right = "r_toe" in frame.support
            if left == right:
                continue
            lands.append((frame, "l" if left else "r"))
        self.assertGreaterEqual(len(lands), 2)
        sides = set()
        for frame, side in lands:
            sides.add(side)
            foot = frame.points[f"{side}_ankle"].x
            self.assertLess(abs(frame.points["pelvis"].x - foot), 12.0)
        self.assertEqual(sides, {"l", "r"})

    def test_front_view_knees_stay_over_the_feet(self):
        for exercise_id in ("sit_low_hold", "split_step_hops", "lateral_lunge", "six_corner_shadow", "skater_hops"):
            for frame in frames_for(exercise_id):
                for side, sign in (("l", -1.0), ("r", 1.0)):
                    knee = frame.points[f"{side}_knee"]
                    ankle = frame.points[f"{side}_ankle"]
                    outside = sign * (knee.x - ankle.x)
                    self.assertLessEqual(outside, 8.0, f"{exercise_id} {frame.phase} {side}")
        laterals = [frame for frame in frames_for("lateral_lunge") if frame.phase == "right"]
        frame = laterals[len(laterals) // 2]
        hip, knee, ankle = frame.points["l_hip"], frame.points["l_knee"], frame.points["l_ankle"]
        self.assertGreater(abs(ankle.x - hip.x), 20.0)
        self.assertLess(_line_gap(knee, hip, ankle), 4.0)

    def test_band_turns_the_chest_around_one_long_grip(self):
        _assert_chest_yaw_keeps_the_grip(self, "band_trunk_rotation", 50.0)
        frames = frames_for("band_trunk_rotation")
        anchor_x, anchor_y = frames[0].props["band_anchor"]
        self.assertLess(anchor_x, -70.0)
        self.assertGreater(anchor_y, 104.0)
        self.assertLess(anchor_y, 124.0)
        start = frames[0]
        turns = [frame for frame in frames if frame.phase == "turn"]
        turn = turns[len(turns) // 2]
        self.assertLess(chest_yaw(start), -20.0)
        self.assertGreater(chest_yaw(turn), 20.0)
        self.assertGreater(_post_gap(turn, anchor_x, anchor_y), _post_gap(start, anchor_x, anchor_y) + 20.0)

    def test_standing_turn_rotates_the_chest_and_keeps_the_feet(self):
        _assert_chest_yaw_keeps_the_grip(self, "standing_trunk_turn", 30.0)
        frames = frames_for("standing_trunk_turn")
        middle = frames[0]
        turns = [frame for frame in frames if frame.phase == "turn"]
        turn = turns[len(turns) // 2]
        self.assertLess(abs(chest_yaw(middle)), 2.0)
        self.assertGreater(abs(chest_yaw(turn)), 25.0)

    def test_split_step_text_is_gym_rebound_not_timing(self):
        from coach.exercises.render import GIF_NOTES

        exercise = load_library()["split_step_hops"]
        self.assertNotIn("opponent", exercise.why.lower())
        self.assertIn("gym", exercise.why.lower())
        self.assertIn("gym rebound", GIF_NOTES["split_step_hops"])

    def test_a_sliding_planted_foot_is_rejected(self):
        frames = frames_for("wall_sit")
        moved = deepcopy(frames[1])
        for name in ("l_toe", "l_heel", "l_ankle"):
            point = moved.points[name]
            moved.points[name] = Pt(point.x + 8.0, point.y)
        issues = validate_sequence([frames[0], moved])
        self.assertTrue(any("slides" in item for item in issues), issues)

    def test_a_flipped_elbow_is_rejected(self):
        frame = frames_for("wall_sit")[0]
        bent = max(frame.points["l_elbow"].y, frame.points["r_elbow"].y)
        self.assertGreater(bent, 40.0)
        bad = deepcopy(frame)
        bad.points["l_elbow"] = _reflect(
            bad.points["l_elbow"], bad.points["l_shoulder"], bad.points["l_wrist"]
        )
        issues = validate_frame(bad)
        self.assertTrue(any("elbow" in item for item in issues), issues)
