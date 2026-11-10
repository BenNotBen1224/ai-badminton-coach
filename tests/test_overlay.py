import unittest

import numpy as np

from badminton_coach.overlay import PALETTE, annotate_frame, assign_track_slots


FAR_PLAYER_BOX = [100, 50, 160, 200]


def player(track_id, joints, box=FAR_PLAYER_BOX):
    return {"track_id": track_id, "bbox_xyxy": box, "box_confidence": 0.9, "joints": joints}


def joints_with(**points):
    joints = [None] * 17
    for index, point in points.items():
        joints[int(index.removeprefix("j"))] = [point[0], point[1], 0.9]
    return joints


def blank():
    return np.zeros((300, 300, 3), np.uint8)


class OverlayVisibilityTests(unittest.TestCase):
    def test_limb_on_a_far_player_is_wide_enough_to_see(self):
        joints = joints_with(j5=(130.0, 80.0), j7=(130.0, 120.0))
        output = annotate_frame(blank(), [player(1, joints)], {1: 0})
        halfway = output[100]
        solid = np.count_nonzero((halfway == PALETTE[0]).all(axis=1))
        dark_edges = np.count_nonzero((halfway == (24, 24, 24)).all(axis=1))
        self.assertGreaterEqual(solid, 4)
        self.assertEqual(2, dark_edges)

    def test_missing_joints_are_never_drawn(self):
        output = annotate_frame(blank(), [player(1, [None] * 17)], {1: 0})
        x1, y1, x2, y2 = FAR_PLAYER_BOX
        self.assertFalse(output[y1 + 6:y2 - 6, x1 + 6:x2 - 6].any())

    def test_a_limb_needs_both_of_its_joints(self):
        output = annotate_frame(blank(), [player(1, joints_with(j5=(130.0, 80.0)))], {1: 0})
        self.assertTrue(output[80, 130].any())
        self.assertFalse(output[110:190, 125:136].any())

    def test_annotating_does_not_change_the_original_frame(self):
        frame = blank()
        annotate_frame(frame, [player(1, joints_with(j5=(130.0, 80.0)))], {1: 0})
        self.assertFalse(frame.any())

    def test_two_players_in_one_frame_get_different_colors(self):
        near = player(1, joints_with(j5=(50.0, 80.0)), [20, 50, 80, 200])
        far = player(2, joints_with(j5=(230.0, 80.0)), [200, 50, 260, 200])
        output = annotate_frame(blank(), [near, far], {1: 0, 2: 1})
        self.assertEqual(PALETTE[0], tuple(output[80, 50]))
        self.assertEqual(PALETTE[1], tuple(output[80, 230]))


class TrackSlotTests(unittest.TestCase):
    def test_players_seen_together_never_share_a_color(self):
        frames = [
            {"players": [{"track_id": 1}, {"track_id": 52}]},
            {"players": [{"track_id": 1}, {"track_id": 52}]},
            {"players": [{"track_id": 1}, {"track_id": 7}]},
            {"players": [{"track_id": 7}, {"track_id": 1}, {"track_id": 9}]},
        ]
        slots = assign_track_slots(frames)
        for first, second in ((1, 52), (1, 7), (1, 9), (7, 9)):
            self.assertNotEqual(slots[first], slots[second])

    def test_the_longest_track_gets_the_first_color(self):
        frames = [{"players": [{"track_id": 2}, {"track_id": 9}]}, {"players": [{"track_id": 9}]}]
        slots = assign_track_slots(frames)
        self.assertEqual({9: 0, 2: 1}, slots)

    def test_an_id_that_splits_keeps_the_same_two_colors(self):
        frames = [{"players": [{"track_id": 1}, {"track_id": 52}]}] * 3
        frames += [{"players": [{"track_id": 1}, {"track_id": 165}]}] * 2
        frames += [{"players": [{"track_id": 1}, {"track_id": 7}]}]
        slots = assign_track_slots(frames)
        self.assertEqual({0, 1}, set(slots.values()))

    def test_untracked_players_do_not_break_slot_assignment(self):
        frames = [{"players": [{"track_id": None}, {"track_id": 4}]}]
        self.assertEqual({4: 0}, assign_track_slots(frames))


if __name__ == "__main__":
    unittest.main()
