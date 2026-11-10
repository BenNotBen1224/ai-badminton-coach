import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from coach.schema import Role
from coach.tactics import (
    SegmentRun,
    _build_hit_rows,
    _click_candidates,
    _height_at_contact,
    _read_player_attribution,
    _reuse_pose_cache,
    _join_camera_segments,
    _make_rallies,
    audio_stream_listed,
    group_score_states,
    infer_against,
    infer_landing_zone,
    infer_shot_family,
    is_receiver_next_contact,
    _rally_spans,
    detect_same_angle_cuts,
    _serve_icon_name,
    _read_clicks,
    split_camera_intervals,
)


def test_score_groups_merge_repeated_score_overlay_and_infer_single_point():
    groups = group_score_states([
        {"k": 0, "game": 1, "start_s": 0.0, "end_s": 0.25, "score_han_wang": "0-0", "serve_icon": "W"},
        {"k": 1, "game": 1, "start_s": 0.25, "end_s": 4.0, "score_han_wang": "0-0", "serve_icon": "W"},
        {"k": 2, "game": 1, "start_s": 4.0, "end_s": 8.0, "score_han_wang": "0-1", "serve_icon": "H"},
    ])

    assert groups[0]["state_indices"] == [0, 1]
    assert groups[0]["start_s"] == 0.0
    assert groups[0]["end_s"] == 4.0
    assert groups[0]["winner"] == "Wang Zhiyi"
    assert groups[0]["winner_reason"] == "one_point_change"
    assert groups[1]["winner"] is None


def test_next_contact_zone_is_receiver_side_relative_depth_and_screen_thirds():
    assert infer_landing_zone(1.0, 7.5, "near")["label"] == "front-left"
    assert infer_landing_zone(4.8, 5.0, "far")["label"] == "front-right"
    assert infer_landing_zone(3.05, 11.2, "near")["label"] == "rear-center"
    assert infer_landing_zone(0.1, 7.5, "near")["label"] == "unknown"
    assert infer_landing_zone(3.05, 7.5, None)["label"] == "unknown"


def test_shot_family_rules_use_pose_height_and_next_contact_evidence():
    assert infer_shot_family(
        contact_height="overhead",
        hitter_depth="rear",
        receiver_zone={"depth": "rear"},
        time_to_next_s=0.8,
        is_serve=False,
        serve_supported=False,
    )["family"] == "clear"
    assert infer_shot_family(
        contact_height="underarm",
        hitter_depth="front",
        receiver_zone={"depth": "front"},
        time_to_next_s=0.7,
        is_serve=False,
        serve_supported=False,
    )["family"] == "net"
    assert infer_shot_family(
        contact_height="side",
        hitter_depth="mid",
        receiver_zone={"depth": "mid"},
        time_to_next_s=0.3,
        is_serve=False,
        serve_supported=False,
    )["family"] == "drive/push"
    assert infer_shot_family(
        contact_height=None,
        hitter_depth=None,
        receiver_zone={"depth": None},
        time_to_next_s=None,
        is_serve=False,
        serve_supported=False,
    )["family"] == "unknown"
    assert infer_shot_family(
        contact_height="underarm",
        hitter_depth="front",
        receiver_zone={"depth": "front"},
        time_to_next_s=0.3,
        is_serve=True,
        serve_supported=True,
    )["family"] == "serve"


def test_against_run_compares_direction_not_distance():
    assert infer_against([1.0, 0.0], [-2.0, 0.0])["against"] is True
    assert infer_against([1.0, 0.0], [0.0, 2.0])["against"] is False
    assert infer_against([0.0, 0.0], [2.0, 0.0])["against"] is None


def test_next_position_is_used_only_after_the_receiver_alternates_to_hit():
    assert is_receiver_next_contact("near", "far") is True
    assert is_receiver_next_contact("far", "near") is True
    assert is_receiver_next_contact("near", "near") is False
    assert is_receiver_next_contact(None, "far") is False


def test_long_camera_intervals_split_at_score_boundaries_without_exceeding_limit():
    chunks = split_camera_intervals(
        [{"start_s": 0.0, "end_s": 80.0}],
        [
            {"start_s": 30.0, "end_s": 40.0},
            {"start_s": 60.0, "end_s": 70.0},
        ],
        max_chunk_s=36.0,
    )

    assert chunks == [
        {"start_s": 0.0, "end_s": 30.0},
        {"start_s": 30.0, "end_s": 60.0},
        {"start_s": 60.0, "end_s": 80.0},
    ]


def test_audio_probe_counts_only_an_audio_codec_line():
    assert audio_stream_listed("") is False
    assert audio_stream_listed("video\n") is False
    assert audio_stream_listed("audio\n") is True
    assert audio_stream_listed("AUDIO\n") is False


def test_placeholder_score_does_not_name_china_open_players():
    groups = group_score_states([
        {"k": 0, "game": 0, "start_s": 0.0, "end_s": 26.0, "score_han_wang": "0-0", "serve_icon": None},
    ])

    assert len(groups) == 1
    assert groups[0]["winner"] is None
    assert groups[0]["winner_reason"] == "no_next_score"
    assert _serve_icon_name(groups[0]["serve_icon"]) is None


def test_score_interval_outside_approved_camera_is_not_selected_as_a_rally():
    group = {"start_s": 20.0, "end_s": 25.0}

    assert _rally_spans(group, [{"start_s": 0.0, "end_s": 10.0}]) == []
    assert _rally_spans({"start_s": 315.0, "end_s": 315.75},
                        [{"start_s": 315.7, "end_s": 334.2}]) == []


def _chunk_doc(start_s, cuts=(), frame_count=3):
    return {
        "schema_version": 3,
        "source": {
            "file": "synthetic.mp4",
            "original_file": "synthetic.mp4",
            "fps": 30.0,
            "width": 1280,
            "height": 720,
            "frame_count": frame_count,
            "reel_start_s": start_s,
            "reel_end_s": start_s + frame_count / 30.0,
            "players": {"near": {"hand": "right"}, "far": {"hand": "right"}},
        },
        "frames": [
            {"frame_index": i, "timestamp_seconds": i / 30.0, "players": []}
            for i in range(frame_count)
        ],
        "cuts": list(cuts),
    }


def test_repeated_score_interval_survives_adjacent_storage_chunks():
    states = [
        {"k": 0, "game": 1, "start_s": 0.0, "end_s": 0.3, "score_han_wang": "5-13", "serve_icon": "H"},
        {"k": 1, "game": 1, "start_s": 0.3, "end_s": 0.6, "score_han_wang": "5-13", "serve_icon": "H"},
    ]
    groups = group_score_states(states)
    docs = [_chunk_doc(0.0, frame_count=9), _chunk_doc(0.3, frame_count=9)]
    runs = [
        SegmentRun(1, 0.0, 0.3, Path("chunk-1.json"), (), 0.0, ()),
        SegmentRun(2, 0.3, 0.6, Path("chunk-2.json"), (), 0.0, ()),
    ]
    data = [(doc, None, [], [], 0.0) for doc in docs]
    events = [
        SimpleNamespace(
            frame=8, player=Role.NEAR, confidence=0.8, kind="swing",
            data={"start": 7, "end": 8, "peak_frame": 8, "peak_speed": 3.0},
        ),
        SimpleNamespace(
            frame=13, player=Role.FAR, confidence=0.8, kind="swing",
            data={"start": 12, "end": 13, "peak_frame": 13, "peak_speed": 3.0},
        ),
    ]

    with patch("coach.tactics.detect_events", return_value=events):
        joined_runs, joined_data = _join_camera_segments(
            [{"start_s": 0.0, "end_s": 0.6}], runs, data
        )
    rallies, hits, excluded = _make_rallies(
        groups, [{"start_s": 0.0, "end_s": 0.6}], joined_runs, joined_data
    )

    assert groups[0]["state_indices"] == [0, 1]
    assert joined_runs[0].start_s == 0.0
    assert joined_runs[0].end_s == 0.6
    assert len(hits) == 2
    assert [hit["time_s"] for hit in hits] == [round(8 / 30, 4), round(13 / 30, 4)]
    assert [hit["evidence_status"] for hit in hits] == [
        "provisional_pose_swing",
        "provisional_pose_swing",
    ]
    assert rallies[0]["selected_live_span_s"] == [0.0, round(17 / 30, 4)]
    assert rallies[0]["source_pose_segments"] == [1, 2]
    assert not excluded


def test_join_preserves_genuine_cuts_without_turning_chunk_boundaries_into_cuts():
    docs = [_chunk_doc(0.0, cuts=[2]), _chunk_doc(0.1, cuts=[1])]
    runs = [
        SegmentRun(1, 0.0, 0.1, Path("chunk-1.json"), (2,), 0.0, ()),
        SegmentRun(2, 0.1, 0.2, Path("chunk-2.json"), (1,), 0.0, ()),
    ]
    data = [(doc, None, [], [], 0.0) for doc in docs]

    joined_runs, _ = _join_camera_segments([{"start_s": 0.0, "end_s": 0.2}], runs, data)
    player = SimpleNamespace(
        measured=[True] * 6,
        foot_court=np.array([
            [0.0, 0.0], [0.1, 0.0], [0.2, 0.0],
            [3.0, 0.0], [3.1, 0.0], [3.2, 0.0],
        ]),
    )
    tracks = SimpleNamespace(n=6, players={"near": player, "far": player})

    assert joined_runs[0].cuts == (2, 4)
    assert detect_same_angle_cuts(tracks) == [3]
    assert sorted(set(joined_runs[0].cuts) | set(detect_same_angle_cuts(tracks))) == [2, 3, 4]


def test_pose_events_are_deduplicated_across_a_joined_chunk_boundary():
    candidates, _ = _click_candidates([
        {"time_s": 0.0667, "frame": 2, "role": "near", "confidence": 0.8, "kind": "swing", "data": {}},
        {"time_s": 0.1, "frame": 3, "role": "near", "confidence": 0.9, "kind": "swing", "data": {}},
    ], (), 30.0)

    assert len(candidates) == 1
    assert candidates[0]["frame"] == 2
    assert candidates[0]["role"] == "near"


def _swing_event(time_s, frame, role, start_s, end_s, peak_speed=4.0, shot_index=0):
    return {
        "time_s": time_s,
        "frame": frame,
        "role": role,
        "confidence": 0.8,
        "kind": "swing",
        "data": {"peak_speed": peak_speed},
        "swing_start_s": start_s,
        "swing_end_s": end_s,
        "shot_index": shot_index,
    }


def test_development_preparation_and_follow_through_peaks_form_one_phase():
    pose_events = [
        _swing_event(124.5167, 3735, "near", 124.4833, 124.55, 5.77),
        _swing_event(125.0167, 3750, "near", 124.85, 125.0167, 9.13),
    ]
    clicks = ({"time_s": 124.94, "rms": 10.0, "baseline_rms": 1.0, "ratio": 10.0},)

    candidates, _ = _click_candidates(pose_events, clicks, 30.0)

    assert len(candidates) == 1
    assert candidates[0]["contact_timing_status"] == "audio_supported_pose_phase"
    assert candidates[0]["contact_time_s"] == 124.94
    assert len(candidates[0]["pose_event"]["phase_events"]) == 2


def test_development_rescue_click_does_not_confirm_later_recovery_motion():
    pose_event = _swing_event(128.5167, 3855, "near", 128.45, 128.5167, 4.81)
    click = {"time_s": 128.24, "rms": 8.0, "baseline_rms": 1.0, "ratio": 8.0}

    candidates, _ = _click_candidates([pose_event], (click,), 30.0)

    assert len(candidates) == 1
    assert candidates[0]["contact_timing_status"] == "pose_phase_unresolved"
    assert candidates[0]["contact_time_s"] is None
    assert candidates[0]["click"] is None


def test_development_overhead_preparation_keeps_contact_phase_unknown_without_click():
    pose_event = _swing_event(132.6167, 3978, "near", 132.3833, 132.65, 3.46)

    candidates, _ = _click_candidates([pose_event], (), 30.0)

    assert len(candidates) == 1
    assert candidates[0]["contact_timing_status"] == "pose_phase_unresolved"
    assert candidates[0]["contact_time_s"] is None
    assert candidates[0]["pose_event_time_s"] == 132.6167


def test_click_is_ambiguous_when_two_players_have_overlapping_swing_phases():
    events = [
        _swing_event(1.0, 30, "near", 0.9, 1.05),
        _swing_event(1.0, 30, "far", 0.95, 1.1),
    ]
    click = {"time_s": 1.0, "rms": 10.0, "baseline_rms": 1.0, "ratio": 10.0}

    candidates, _ = _click_candidates(events, (click,), 30.0)

    assert len(candidates) == 2
    assert all(candidate["contact_time_s"] is None for candidate in candidates)
    assert all(candidate["click"] is None for candidate in candidates)


def test_attributed_hand_selects_matching_wrist_for_contact_height():
    joints = np.zeros((1, 17, 2), dtype=float)
    joints[0, 0, 1] = 100
    joints[0, 9, 1] = 50
    joints[0, 10, 1] = 150
    joints[0, 11, 1] = 200
    joints[0, 12, 1] = 200
    player = SimpleNamespace(hand="left", measured=np.array([True]), xy=joints, conf=np.ones((1, 17)))
    tracks = SimpleNamespace(players={"near": player}, height=1000)

    assert _height_at_contact(tracks, "near", 0)["label"] == "overhead"

    player.hand = "unknown"
    assert _height_at_contact(tracks, "near", 0)["label"] is None


def test_pose_only_observation_withholds_hitter_and_contact_specific_metrics():
    measured = SimpleNamespace(measured=np.array([True]))
    tracks = SimpleNamespace(
        n=1,
        fps=30.0,
        height=720,
        court=None,
        players={"near": measured, "far": measured},
    )
    candidate = {
        "role": "near",
        "frame": 0,
        "reel_time_s": 1.0,
        "pose_event_reel_time_s": 1.0,
        "contact_time_reel_s": None,
        "contact_timing_status": "pose_phase_unresolved",
        "time_s": 1.0,
        "swing_window_s": [0.9, 1.1],
        "source_frame_30fps": 30,
        "pose_event_source_frame_30fps": 30,
        "source_segment_number": 1,
        "confidence": 0.4,
        "source": ["pose_swing"],
        "pose_event": {},
        "click": None,
        "is_serve": False,
        "hitter_name": None,
    }

    with (
        patch("coach.tactics._nearest_measured", return_value={
            "xy_m": [3.0, 3.0], "frame": 0, "offset_frames": 0, "measured": True, "court_ok": True,
        }),
        patch("coach.tactics._movement_before", return_value={
            "vector_m": [0.0, 0.0], "speed_m_s": 0.0, "direction": None, "samples": 2,
        }),
        patch("coach.tactics._balance_state", return_value={"state": "unknown"}),
    ):
        row = _build_hit_rows(
            {"game": 1}, "G1-R001", 0, SegmentRun(1, 0.0, 2.0, Path("cache.json"), (), 0.0, ()),
            {}, tracks, [candidate], {}, 0.0, [], {},
        )[0]

    assert row["hitter"] is None
    assert row["pose_role_candidate"] == "near"
    assert row["contact_time_s"] is None
    assert row["contact_height"]["label"] is None
    assert row["shot_family"]["family"] == "unknown"
    assert row["positions_at_contact_m"] is None
    assert row["landing_zone"]["label"] == "unknown"


def test_player_attribution_requires_matching_source_and_explicit_evidence(tmp_path):
    attribution = tmp_path / "attribution.json"
    attribution.write_text(json.dumps({
        "source": {"file": "sources/china-open-2025-ws-final.webm"},
        "clips": [{
            "file": "04-china-open-ws-a.mp4",
            "players": {
                "near": {"hand": "right", "evidence": "visible racket in right hand"},
                "far": {"hand": "right", "evidence": "visible racket in right hand"},
            },
        }],
    }), encoding="utf-8")

    players, source = _read_player_attribution(Path("china-open-2025-ws-final.webm"), attribution)

    assert players["near"]["hand"] == "right"
    assert players["far"]["hand"] == "right"
    assert source["status"] == "matched_source_attribution"

    with pytest.raises(ValueError, match="does not match"):
        _read_player_attribution(Path("other-match.webm"), attribution)


def test_cached_pose_reanalysis_copies_files_and_replaces_unproven_hands(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    cached_segment = cache / "segment-01.json"
    cached_segment.write_text(json.dumps(_chunk_doc(0.0, frame_count=3)), encoding="utf-8")
    (cache / "index.json").write_text(json.dumps({
        "segments": [{
            "number": 1,
            "reel_start_s": 0.0,
            "reel_end_s": 0.1,
            "pose_path": str(cached_segment),
            "cuts_doc_frames": [],
        }],
    }), encoding="utf-8")
    hands = {"near": {"hand": "unknown"}, "far": {"hand": "unknown"}}

    with patch("coach.tactics._read_clicks", return_value=[]):
        runs, _ = _reuse_pose_cache(
            cache, Path("synthetic.mp4"), tmp_path / "output",
            [{"start_s": 0.0, "end_s": 0.1}], Path("ffmpeg.exe"), hands, {"status": "unavailable"},
        )

    reused = json.loads(runs[0].pose_path.read_text(encoding="utf-8"))
    original = json.loads(cached_segment.read_text(encoding="utf-8"))
    assert reused["source"]["players"]["near"]["hand"] == "unknown"
    assert original["source"]["players"]["near"]["hand"] == "right"
    assert runs[0].pose_path != cached_segment


def test_unmatched_audio_does_not_create_a_hit_or_hitter():
    clicks = ({"time_s": 0.5, "rms": 10.0, "baseline_rms": 1.0, "ratio": 10.0},)

    candidates, offset = _click_candidates([], clicks, 30.0)

    assert candidates == []
    assert offset == 0.0


def test_silent_video_returns_no_clicks_without_attempting_audio_decode():
    probe = subprocess.CompletedProcess(args=[], returncode=0, stdout="video\n", stderr="")
    with patch("coach.tactics.subprocess.run", return_value=probe) as run_command:
        with patch.object(Path, "is_file", return_value=True):
            clicks = _read_clicks(Path("ffmpeg.exe"), Path("silent.mp4"), 0.0, 1.0)

    assert clicks == []
    assert run_command.call_count == 1
