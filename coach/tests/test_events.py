"""Swing and lunge detection against synthetic truth. Expected frames are literals."""
from __future__ import annotations

import gc
import unittest
import weakref
from dataclasses import fields

import numpy as np

from coach.events import HIP_GLUED, MIN_TRAVEL, SWING_GAP_S, _at_hip, _dangling, _keep_spaced, _peak_indices, _supported, _swing_from_peak, _travel, detect_events, recovery_windows
from coach.schema import Role
from coach.signals_body import _RATIO_CACHE, _upright_hip_ratio_at, both_hand_speeds, clear_ratio_cache, foot_speed_ratio, hip_height_px, hip_height_ratio, upright_hip_ratio_at
from coach.synth import Lunge, Path, PlayerPlan, Posture, SplitStep, Swing, repeat_frames, script
from coach.tracks import JOINT_INDEX, PlayerTrack, load_tracks


def _rally():
    return script(
        n=80,
        fps=30,
        near=PlayerPlan(swing=Swing(30, 2.05, 170), lunge=Lunge(50, 0.80, 0.4)),
        far=PlayerPlan(court_xy=(3.05, 2.9), swing=Swing(45, 1.90, 160)),
    )


def _swings(events, role):
    return [event for event in events if event.kind == "swing" and event.player == role]


class EventTests(unittest.TestCase):
    def test_clean_swing_contact_is_within_two_frames(self):
        doc, truth = _rally()
        events = detect_events(load_tracks(doc))
        expected = (
            ("near", truth["players"]["near"]["swing"]["contact_frame"], truth["players"]["near"]["swing"]["peak_frame"]),
            ("far", truth["players"]["far"]["swing"]["contact_frame"], truth["players"]["far"]["swing"]["peak_frame"]),
        )
        swings = [event for event in events if event.kind == "swing"]
        hits = 0
        used = set()
        for role, contact, peak in expected:
            match = next((
                event for event in swings
                if event.player.value == role and abs(event.frame - contact) <= 2 and abs(int(event.data["peak_frame"]) - peak) <= 2
            ), None)
            self.assertIsNotNone(match, role)
            used.add(id(match))
            hits += 1
        self.assertGreaterEqual(hits / len(expected), 0.9)
        self.assertGreaterEqual(hits / len(swings), 0.9)
        self.assertEqual(len(used), len(swings))

    def test_lunge_window_is_within_two_frames(self):
        doc, truth = _rally()
        events = detect_events(load_tracks(doc))
        lunges = [event for event in events if event.kind == "lunge" and event.player == Role.NEAR]
        self.assertEqual(len(lunges), 1)
        got = lunges[0]
        want = truth["players"]["near"]["lunge"]
        self.assertLessEqual(abs(got.frame - want["start_frame"]), 2)
        self.assertLessEqual(abs(int(got.data["bottom_frame"]) - want["bottom_frame"]), 2)
        self.assertLessEqual(abs(int(got.data["end_frame"]) - want["end_frame"]), 2)
        self.assertGreater(float(got.data["depth"]), 0.0)

    def test_split_step_and_running_are_not_swings(self):
        split, _ = script(n=40, near=PlayerPlan(split_step=SplitStep(20)))
        run, _ = script(n=40, near=PlayerPlan(path=Path(0, 30, (5.0, 8.0))))
        self.assertEqual(detect_events(load_tracks(split)), ())
        self.assertEqual(detect_events(load_tracks(run)), ())

    def test_repeated_frames_do_not_add_swings(self):
        doc, truth = script(n=60, near=PlayerPlan(swing=Swing(30, 2.05, 170)))
        before = _swings(detect_events(load_tracks(doc)), Role.NEAR)
        repeat_frames(doc, 6, truth)
        after = _swings(detect_events(load_tracks(doc)), Role.NEAR)
        self.assertEqual(len(before), 1)
        self.assertEqual(len(after), 1)
        self.assertLessEqual(abs(after[0].frame - before[0].frame), 2)

    def test_a_swing_does_not_cross_a_cut(self):
        doc, _ = script(n=50, cuts=(28,), near=PlayerPlan(swing=Swing(30, 2.05, 170)))
        for event in detect_events(load_tracks(doc)):
            if event.kind != "swing":
                continue
            start, end = int(event.data["start"]), int(event.data["end"])
            self.assertFalse(start < 28 <= end)

    def test_detection_is_stable(self):
        doc, _ = _rally()
        tracks = load_tracks(doc)
        first = [(event.kind, event.player, event.frame) for event in detect_events(tracks)]
        second = [(event.kind, event.player, event.frame) for event in detect_events(tracks)]
        self.assertEqual(first, second)

    def test_unresolved_drops_are_not_measured_ends(self):
        sustained = _standing(100)
        _drop(sustained, range(30, 100), 0.16)
        _widen(sustained, range(30, 100), 0.18)
        _hide_knees(sustained, range(30, 100))
        self.assertEqual(_near_lunges(sustained), [])
        self.assertTrue(recovery_windows(sustained))

        dropout = _standing(80)
        _drop(dropout, range(30, 70), 0.16)
        _widen(dropout, range(30, 70), 0.18)
        _hide_knees(dropout, range(30, 70))
        _hide(dropout, range(56, 80), ("left_hip", "right_hip"))
        self.assertEqual(_near_lunges(dropout), [])
        self.assertFalse(any(int(event.data["end_frame"]) == 55 for event in detect_events(dropout)))

        short = _standing(50)
        _drop(short, range(30, 50), 0.16)
        _widen(short, range(30, 50), 0.18)
        _hide_knees(short, range(30, 50))
        self.assertEqual(_near_lunges(short), [])
        self.assertFalse(any(int(event.data["end_frame"]) == 49 for event in detect_events(short)))

    def test_return_to_a_low_ready_is_a_lunge(self):
        tracks = _standing(80)
        _drop(tracks, range(20, 80), 0.05)
        _drop(tracks, range(40, 52), 0.14)
        _widen(tracks, range(40, 52), 0.18)
        _hide_knees(tracks, range(20, 80))
        lunges = _near_lunges(tracks)
        self.assertEqual(len(lunges), 1)
        end = int(lunges[0].data["end_frame"])
        self.assertGreaterEqual(end, 51)
        self.assertLessEqual(end, 55)
        self.assertLess(end, 79)

    def test_a_slow_return_is_not_cut_off(self):
        tracks = _standing(110)
        _drop(tracks, range(20, 85), 0.16)
        _widen(tracks, range(20, 85), 0.18)
        _hide_knees(tracks, range(20, 85))
        lunges = _near_lunges(tracks)
        self.assertEqual(len(lunges), 1)
        end = int(lunges[0].data["end_frame"])
        bottom = int(lunges[0].data["bottom_frame"])
        self.assertGreaterEqual(end, 84)
        self.assertLessEqual(end, 88)
        self.assertGreater(end - bottom, 30)

    def test_unknown_stance_does_not_close_the_step(self):
        doc, _ = script(n=80, near=PlayerPlan(lunge=Lunge(40, 0.80, 0.4)))
        tracks = load_tracks(doc)
        before = _near_lunges(tracks)
        self.assertEqual(len(before), 1)
        bottom = int(before[0].data["bottom_frame"])
        _blind_stance(tracks, range(bottom + 1, 80))
        _hide_knees(tracks, range(bottom + 1, 80))
        self.assertEqual(_near_lunges(tracks), [])
        self.assertTrue(recovery_windows(tracks))

    def test_perspective_scale_is_not_a_lunge(self):
        doc, _ = script(n=90, near=PlayerPlan(path=Path(25, 85, (3.05, 4.5))))
        tracks = load_tracks(doc)
        player = tracks.players[Role.NEAR]
        _widen(tracks, range(40, 75), 0.18)
        hip = hip_height_px(player)
        early = hip[:25]
        early = early[np.isfinite(early)]
        late = hip[55:75]
        late = late[np.isfinite(late)]
        self.assertGreater(early.size, 5)
        self.assertGreater(late.size, 5)
        pixel_drop = (float(np.median(early)) - float(np.min(late))) / float(np.median(early))
        self.assertGreater(pixel_drop, 0.06)
        ratio = hip_height_ratio(player)
        stand = upright_hip_ratio_at(player)
        depth = (stand - ratio) / stand
        moving = depth[25:75]
        moving = moving[np.isfinite(moving)]
        self.assertGreater(moving.size, 5)
        self.assertLess(float(np.max(moving)), 0.06)
        self.assertEqual(_near_lunges(tracks), [])

    def test_a_dead_player_does_not_leave_a_shorter_ratio(self):
        clear_ratio_cache()
        short = _rebuild_player(_player_pack(80))
        long = _rebuild_player(_player_pack(90))
        stale = upright_hip_ratio_at(short).copy()
        live = upright_hip_ratio_at(long).copy()
        self.assertEqual(tuple(stale.shape), (80,))
        self.assertEqual(tuple(live.shape), (90,))
        self.assertIn(short, _RATIO_CACHE)
        self.assertIn(long, _RATIO_CACHE)
        self.assertIsNot(_RATIO_CACHE[short], _RATIO_CACHE[long])
        ref = weakref.ref(short)
        del short
        gc.collect()
        self.assertIsNone(ref())
        self.assertEqual(len(_RATIO_CACHE), 1)
        self.assertIn(long, _RATIO_CACHE)
        self.assertTrue(np.array_equal(upright_hip_ratio_at(long), live, equal_nan=True))
        born = _rebuild_player(_player_pack(90))
        got = upright_hip_ratio_at(born)
        self.assertEqual(born.n, 90)
        self.assertEqual(tuple(got.shape), (90,))
        self.assertTrue(np.array_equal(got, _upright_hip_ratio_at(born), equal_nan=True))
        self.assertTrue(np.array_equal(got, live, equal_nan=True))

    def test_a_live_same_length_pose_does_not_share_a_ratio(self):
        clear_ratio_cache()
        packed = _player_pack(40)
        plain = _rebuild_player(packed)
        low = _rebuild_player(packed, drop=40.0)
        stale = upright_hip_ratio_at(plain).copy()
        expected = upright_hip_ratio_at(low).copy()
        self.assertEqual(plain.n, low.n)
        self.assertEqual(stale.shape, expected.shape)
        self.assertFalse(np.array_equal(stale, expected, equal_nan=True))
        self.assertIsNot(_RATIO_CACHE[plain], _RATIO_CACHE[low])
        self.assertFalse(np.array_equal(_RATIO_CACHE[plain], _RATIO_CACHE[low], equal_nan=True))
        ref = weakref.ref(plain)
        del plain
        gc.collect()
        self.assertIsNone(ref())
        self.assertEqual(len(_RATIO_CACHE), 1)
        self.assertIn(low, _RATIO_CACHE)
        self.assertTrue(np.array_equal(upright_hip_ratio_at(low), expected, equal_nan=True))
        born = _rebuild_player(packed, drop=40.0)
        got = upright_hip_ratio_at(born)
        self.assertEqual(got.shape, stale.shape)
        self.assertFalse(np.array_equal(got, stale, equal_nan=True))
        self.assertTrue(np.array_equal(got, expected, equal_nan=True))
        self.assertTrue(np.array_equal(got, _upright_hip_ratio_at(born), equal_nan=True))

    def test_the_same_lunge_matches_at_two_sizes(self):
        depths = []
        for court_xy in ((3.05, 10.5), (3.05, 7.2)):
            doc, _ = script(n=80, near=PlayerPlan(court_xy=court_xy, lunge=Lunge(40, 0.80, 0.4)))
            lunges = _near_lunges(load_tracks(doc))
            self.assertEqual(len(lunges), 1, court_xy)
            depths.append(float(lunges[0].data["depth"]))
        self.assertLess(abs(depths[0] - depths[1]), 0.03)

    def test_a_wrist_on_the_hip_is_not_a_swing(self):
        doc, _ = script(n=40, near=PlayerPlan(swing=Swing(24, 2.05, 170)))
        tracks = load_tracks(doc)
        player = tracks.players[Role.NEAR]
        before = _swings(detect_events(tracks), Role.NEAR)
        self.assertEqual(len(before), 1)
        peak = int(before[0].data["peak_frame"])
        start = int(before[0].data["start"])
        spd = both_hand_speeds(player, tracks)["right_wrist"].copy()
        player.xy[peak, JOINT_INDEX["right_wrist"]] = player.xy[peak, JOINT_INDEX["right_hip"]]
        self.assertTrue(_at_hip(player, "right_wrist", peak))
        self.assertGreaterEqual(_travel(player, "right_wrist", start, peak), MIN_TRAVEL)
        self.assertLess(
            float(np.linalg.norm(player.joint("right_wrist")[peak] - player.joint("right_hip")[peak]))
            / float(player.stature_px[peak]),
            HIP_GLUED + 1e-9,
        )
        foot = foot_speed_ratio(player, tracks)
        self.assertIsNone(_swing_from_peak(player, tracks, "right_wrist", spd, peak, foot, []))

    def test_an_extended_low_reach_is_not_dangling(self):
        doc, _ = script(n=20, near=PlayerPlan())
        player = load_tracks(doc).players[Role.NEAR]
        _place_arm(player, 8, bend=170.0, near_hip=False)
        self.assertFalse(_dangling(player, "right_wrist", 8))
        _place_arm(player, 8, bend=90.0, near_hip=True)
        self.assertTrue(_dangling(player, "right_wrist", 8))
        player.visible[8, JOINT_INDEX["right_elbow"]] = False
        self.assertFalse(_dangling(player, "right_wrist", 8))

    def test_a_running_low_reach_is_kept_and_a_dangle_is_not(self):
        kept = _low_reach_swing(bend=168.0, near_hip=False)
        self.assertEqual(len(kept), 1)
        dropped = _low_reach_swing(bend=90.0, near_hip=True)
        self.assertEqual(dropped, [])

    def test_a_measured_gap_is_not_an_isolated_spike(self):
        doc, _ = script(n=40, near=PlayerPlan(swing=Swing(24, 2.05, 170)))
        tracks = load_tracks(doc)
        player = tracks.players[Role.NEAR]
        before = _swings(detect_events(tracks), Role.NEAR)
        self.assertEqual(len(before), 1)
        peak = int(before[0].data["peak_frame"])
        wrist = JOINT_INDEX["right_wrist"]
        keep = {peak - 3, peak, peak + 3}
        for frame in range(player.n):
            if frame not in keep:
                player.visible[frame, wrist] = False
                player.measured[frame] = False
                player.stature_px[frame] = np.nan
        after = _swings(detect_events(tracks), Role.NEAR)
        self.assertEqual(len(after), 1)
        self.assertLessEqual(abs(int(after[0].data["peak_frame"]) - peak), 3)

    def test_an_unsupported_spike_does_not_erase_a_swing(self):
        doc, _ = script(n=40, near=PlayerPlan(swing=Swing(18, 2.05, 170)))
        tracks = load_tracks(doc)
        before = _swings(detect_events(tracks), Role.NEAR)
        self.assertEqual(len(before), 1)
        peak = int(before[0].data["peak_frame"])
        player = tracks.players[Role.NEAR]
        spike = peak + 6
        shift = np.array([1.6 * float(player.stature_px[spike]), 0.0])
        player.xy[spike:spike + 2, JOINT_INDEX["right_wrist"]] += shift
        after = _swings(detect_events(tracks), Role.NEAR)
        self.assertTrue(any(abs(int(event.data["peak_frame"]) - peak) <= 1 for event in after))

    def test_a_stronger_rejected_peak_does_not_erase_a_weaker_swing(self):
        tracks = _hip_neighbor_tracks()
        player = tracks.players[Role.NEAR]
        spd = both_hand_speeds(player, tracks)["right_wrist"]
        candidates = [
            int(peak) for peak in _peak_indices(spd)
            if _supported(spd, int(peak)) and player.joint_visible("right_wrist")[int(peak)]
        ]
        self.assertEqual(candidates, [16, 24])
        self.assertGreater(float(spd[24]), float(spd[16]))
        self.assertLess(float(tracks.t[24] - tracks.t[16]), SWING_GAP_S)
        self.assertFalse(_at_hip(player, "right_wrist", 16))
        self.assertTrue(_at_hip(player, "right_wrist", 24))
        foot = foot_speed_ratio(player, tracks)
        self.assertIsNotNone(_swing_from_peak(player, tracks, "right_wrist", spd, 16, foot, []))
        self.assertIsNone(_swing_from_peak(player, tracks, "right_wrist", spd, 24, foot, []))
        self.assertEqual(_keep_spaced(candidates, spd, tracks.t, SWING_GAP_S), [24])
        kept = _swings(detect_events(tracks), Role.NEAR)
        self.assertEqual([int(event.data["peak_frame"]) for event in kept], [16])
        self.assertEqual(int(kept[0].data["hand"]), 1)

    def test_invisible_travel_does_not_pass(self):
        doc, _ = script(n=40, near=PlayerPlan(swing=Swing(24, 2.05, 170)))
        tracks = load_tracks(doc)
        player = tracks.players[Role.NEAR]
        player.visible[8, JOINT_INDEX["right_wrist"]] = False
        self.assertEqual(_travel(player, "right_wrist", 8, 16), 0.0)
        self.assertLess(_travel(player, "right_wrist", 8, 16), MIN_TRAVEL)
        fresh = load_tracks(doc)
        kept = _swings(detect_events(fresh), Role.NEAR)
        self.assertEqual(len(kept), 1)
        peak = int(kept[0].data["peak_frame"])
        start = int(kept[0].data["start"])
        hidden = load_tracks(doc)
        wrist = JOINT_INDEX["right_wrist"]
        hidden.players[Role.NEAR].visible[:, wrist] = False
        hidden.players[Role.NEAR].visible[peak, wrist] = True
        self.assertEqual(_swings(detect_events(hidden), Role.NEAR), [])
        partial = load_tracks(doc)
        partial.players[Role.NEAR].visible[start, wrist] = False
        for event in _swings(detect_events(partial), Role.NEAR):
            seen = int(event.data["start"])
            body = partial.players[Role.NEAR]
            self.assertTrue(bool(body.joint_visible("right_wrist")[seen]))
            self.assertGreaterEqual(
                _travel(body, "right_wrist", seen, int(event.data["peak_frame"])),
                MIN_TRAVEL,
            )


def _player_pack(n):
    player = load_tracks(script(n=n, fps=30, near=PlayerPlan())[0]).players[Role.NEAR]
    return tuple(getattr(player, item.name) for item in fields(PlayerTrack))


def _rebuild_player(packed, drop=0.0):
    values = {item.name: value for item, value in zip(fields(PlayerTrack), packed)}
    if drop:
        xy = values["xy"].copy()
        xy[:, JOINT_INDEX["left_hip"], 1] += drop
        xy[:, JOINT_INDEX["right_hip"], 1] += drop
        values["xy"] = xy
    return PlayerTrack(**values)


def _standing(n):
    doc, _ = script(n=n, near=PlayerPlan())
    return load_tracks(doc)


def _near_lunges(tracks):
    return [event for event in detect_events(tracks) if event.kind == "lunge" and event.player == Role.NEAR]


def _drop(tracks, frames, fraction):
    player = tracks.players[Role.NEAR]
    for i in frames:
        dy = fraction * float(player.stature_px[i])
        player.xy[i, JOINT_INDEX["left_hip"], 1] += dy
        player.xy[i, JOINT_INDEX["right_hip"], 1] += dy


def _widen(tracks, frames, fraction):
    player = tracks.players[Role.NEAR]
    for i in frames:
        extra = fraction * float(player.stature_px[i])
        player.xy[i, JOINT_INDEX["left_ankle"], 0] -= extra / 2
        player.xy[i, JOINT_INDEX["right_ankle"], 0] += extra / 2


def _hide_knees(tracks, frames):
    _hide(tracks, frames, ("left_knee", "right_knee"))


def _hide(tracks, frames, names):
    player = tracks.players[Role.NEAR]
    for name in names:
        player.visible[list(frames), JOINT_INDEX[name]] = False


def _place_arm(player, frame, bend, near_hip):
    shoulder = player.xy[frame, JOINT_INDEX["right_shoulder"]].copy()
    hip = player.xy[frame, JOINT_INDEX["right_hip"]].copy()
    stature = float(player.stature_px[frame])
    elbow = shoulder + np.array([0.12 * stature, 0.20 * stature])
    if near_hip:
        wrist = hip + np.array([0.05 * stature, 0.05 * stature])
    else:
        wrist = shoulder + np.array([0.35 * stature, 0.45 * stature])
    if bend < 150.0:
        elbow = (shoulder + wrist) / 2 + np.array([0.12 * stature, 0.0])
    player.xy[frame, JOINT_INDEX["right_elbow"]] = elbow
    player.xy[frame, JOINT_INDEX["right_wrist"]] = wrist
    for name in ("right_shoulder", "right_elbow", "right_wrist", "right_hip"):
        player.visible[frame, JOINT_INDEX[name]] = True


def _blind_stance(tracks, frames):
    """Hide one ankle without changing hip height. Stance becomes unknown."""
    player = tracks.players[Role.NEAR]
    for i in frames:
        left = int(JOINT_INDEX["left_ankle"])
        right = int(JOINT_INDEX["right_ankle"])
        keep = left if player.xy[i, left, 1] >= player.xy[i, right, 1] else right
        drop = right if keep == left else left
        player.xy[i, drop] = player.xy[i, keep]
        player.visible[i, drop] = False


def _hip_neighbor_tracks():
    """Weaker eligible swing at 16, louder hip-glued peak at 24. Same chosen hand."""
    tracks = load_tracks(script(n=40, fps=30, near=PlayerPlan())[0])
    player = tracks.players[Role.NEAR]
    wrist_i = JOINT_INDEX["right_wrist"]
    hip_i = JOINT_INDEX["right_hip"]
    elbow_i = JOINT_INDEX["right_elbow"]
    shoulder_i = JOINT_INDEX["right_shoulder"]
    for frame in range(12, 28):
        stature = float(player.stature_px[frame])
        hip = player.xy[frame, hip_i].copy()
        shoulder = player.xy[frame, shoulder_i].copy()
        rest = player.xy[frame, wrist_i].copy()
        on_hip = frame >= 24
        if frame <= 14:
            wrist = rest
        elif frame == 15:
            wrist = rest + np.array([0.08 * stature, -0.04 * stature])
        elif frame == 16:
            wrist = rest + np.array([0.22 * stature, -0.16 * stature])
        elif frame == 17:
            wrist = rest + np.array([0.26 * stature, -0.18 * stature])
        elif frame <= 23:
            wrist = rest + np.array([0.28 * stature, -0.18 * stature])
        elif frame == 24:
            wrist = hip + np.array([0.04 * stature, 0.02 * stature])
        else:
            wrist = hip + np.array([0.09 * stature, 0.04 * stature])
        player.measured[frame] = True
        player.xy[frame, wrist_i] = wrist
        if on_hip:
            player.xy[frame, elbow_i] = (shoulder + wrist) / 2
        else:
            player.xy[frame, elbow_i] = shoulder + 0.55 * (wrist - shoulder)
        for index in (wrist_i, elbow_i, hip_i, shoulder_i):
            player.visible[frame, index] = True
    return tracks


def _low_reach_swing(bend, near_hip):
    doc, _ = script(n=36, near=PlayerPlan(path=Path(0, 34, (5.2, 8.5)), swing=Swing(22, 2.05, 170)))
    tracks = load_tracks(doc)
    player = tracks.players[Role.NEAR]
    peak = int(_swings(detect_events(tracks), Role.NEAR)[0].data["peak_frame"])
    wrist_i = JOINT_INDEX["right_wrist"]
    if near_hip:
        stature = float(player.stature_px[peak])
        delta = player.xy[:, wrist_i] - player.xy[peak, wrist_i]
        player.xy[:, wrist_i] = player.xy[peak, JOINT_INDEX["right_hip"]] + np.array([0.04 * stature, 0.04 * stature]) + delta
    else:
        player.xy[:, wrist_i, 1] += 0.45 * float(player.stature_px[peak])
    for frame in range(player.n):
        shoulder = player.xy[frame, JOINT_INDEX["right_shoulder"]]
        wrist = player.xy[frame, wrist_i]
        stature = float(player.stature_px[frame])
        if near_hip or bend < 150.0:
            elbow = (shoulder + wrist) / 2 + np.array([0.14 * stature, 0.0])
        else:
            elbow = shoulder + 0.62 * (wrist - shoulder)
        player.xy[frame, JOINT_INDEX["right_elbow"]] = elbow
    return _swings(detect_events(tracks), Role.NEAR)


if __name__ == "__main__":
    unittest.main()
