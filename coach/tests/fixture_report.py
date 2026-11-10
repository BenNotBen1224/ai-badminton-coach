"""Hand-made report for the chat tests: both players, all 16 findings, 900 frames at 30 fps.

The numbers are plausible, not measured. Nothing outside the tests may use them.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional

from coach.schema import (
    FINDING_CATALOG,
    SCHEMA_VERSION,
    Basis,
    Event,
    Exercise,
    Finding,
    Mark,
    Metric,
    Moment,
    Report,
    Role,
    Severity,
    Tier,
    Window,
    lunge_event,
    swing_event,
)

N_FRAMES = 900
FPS = 30.0

P, I, S, G = Severity.PRIORITY, Severity.IMPROVE, Severity.INFO, Severity.STRENGTH
PROJ, PLANE, TIME, COUNT = Basis.PROJECTED_2D, Basis.COURT_PLANE, Basis.TIMING, Basis.COUNT

# One row per finding:
# (id, severity, score, headline, explain, fix tags, focus joints, overlay series, metrics, moments)
# metric: (name, label, value, unit, basis, confidence, target, series, n_obs, best)
# moment: (key frame, label, kind, mark style)
NEAR = (
    ("technique.stay_lower", P, 0.92,
     "Hips stay high while you wait for the shot",
     "While you wait for the opponent's shot, your hips sit above the practice guide range. "
     "A lower ready stance lets you push off sooner in any direction.",
     ("lower_stance", "endurance"), ("left_hip", "right_hip", "left_knee", "right_knee"), ("hip_height_ratio",),
     (("hip_height_ratio", "Hip height while waiting", 0.53, "heights", PROJ, 0.82, (0.42, 0.48), "hip_height_ratio", 14, 0.47),),
     ((372, "Hips high while waiting", "issue", "height"), (624, "Hips high again", "issue", "height"),
      (510, "Your lowest ready stance", "best", "height"))),
    ("footwork.split_step", P, 0.81,
     "No split step seen at many of the opponent's hits",
     "Around many of the opponent's hits no small hop shows up, so you start moving from still feet. "
     "A low hop started just before they hit lets you push off just after.",
     ("split_step", "first_step"), ("left_ankle", "right_ankle"), (),
     (("split_step_share", "Opponent hits with a split step", 0.38, "share", TIME, 0.71, None, None, 13, None),
      ("split_step_landing_s", "Split-step landing after their hit", 0.06, "s", TIME, 0.55, None, None, 5, None)),
     ((310, "No split step as they hit", "issue", "feet"), (462, "Still feet as they hit", "issue", "feet"),
      (728, "A well-timed split step", "best", "feet"))),
    ("footwork.recovery", I, 0.66,
     "Slow return to base after your own shots",
     "After you hit, it takes you longer than the practice guide to get back near the middle of your half.",
     ("base_recovery", "lunge_recovery"), ("left_ankle", "right_ankle"), ("base_distance",),
     (("recovery_time_s", "Time back to base after your shot", 1.12, "s", TIME, 0.74, (0.6, 0.9), "base_distance", 11, 0.78),),
     ((552, "Slow way back to base", "issue", "path"),)),
    ("contact.arm_extension", I, 0.58,
     "Hitting arm is bent when you hit overhead",
     "At the hit, your elbow angle in the picture is below the practice guide range, so the arm is bent. "
     "This is a projected angle from one camera, so treat it as an estimate.",
     ("arm_extension", "shoulder_health"), ("right_shoulder", "right_elbow", "right_wrist"), (),
     (("elbow_angle_deg", "Elbow angle at the hit", 138, "deg", PROJ, 0.58, (155, 175), None, 9, 160),),
     ((208, "Bent arm at the hit", "issue", "angle"),)),
    ("technique.lunge_depth", I, 0.52,
     "Net lunges are shallow",
     "When you lunge to the net, your hips drop less than the practice guide range, so you reach the shuttle later.",
     ("lunge_depth", "hip_mobility"), ("left_hip", "right_hip", "left_knee", "right_knee"), (),
     (("lunge_depth_ratio", "Hip drop in net lunges", 0.17, "fraction", PROJ, 0.69, (0.22, 0.32), None, 6, 0.24),),
     ((650, "Shallow lunge at the net", "issue", "height"), (792, "Your deepest lunge", "best", "height"))),
    ("technique.knee_bend", S, 0.3,
     "Knees are fairly straight in the ready stance",
     "Your knees bend a little less than the practice guide range while you wait. This goes with the high hips.",
     ("lower_stance",), ("left_knee", "right_knee"), ("knee_angle_deg",),
     (("knee_angle_deg", "Knee angle in the ready stance", 151, "deg", PROJ, 0.6, (120, 145), "knee_angle_deg", 14, None),),
     ((372, "Straight knees while waiting", "issue", "angle"),)),
    ("technique.stance_width", G, 0.1,
     "Good stance width while you wait",
     "Your feet are a good distance apart in the ready stance, inside the practice guide range.",
     ("balance",), ("left_ankle", "right_ankle"), ("stance_width_ratio",),
     (("stance_width_ratio", "Feet apart in the ready stance", 0.34, "heights", PROJ, 0.8, (0.3, 0.4), "stance_width_ratio", 14, 0.36),),
     ((510, "Good stance width", "best", "width"),)),
    ("technique.trunk_lean", S, 0.2,
     "Upper body leans to the side at times",
     "In a couple of moments your upper body leans to the side more than the practice guide range.",
     ("core_stability",), ("left_shoulder", "right_shoulder", "left_hip", "right_hip"), ("trunk_lean_deg",),
     (("trunk_lean_deg", "Sideways lean of the upper body", 14, "deg", PROJ, 0.45, (0, 10), "trunk_lean_deg", 2, None),),
     ((266, "Upper body leans", "issue", "angle"),)),
    ("contact.reach_height", S, 0.25,
     "Overhead hits are a little low",
     "Your overhead hits are slightly below the practice guide range for contact height.",
     ("contact_height", "shoulder_health"), ("right_wrist", "right_shoulder"), (),
     (("contact_height_ratio", "Contact height on overheads", 1.18, "heights", PROJ, 0.62, (1.2, 1.35), None, 7, 1.27),),
     ((158, "Low overhead hit", "issue", "height"),)),
    ("contact.hand_speed", G, 0.1,
     "Fast, steady racket-hand speed",
     "Your racket hand moves fast at the swing peak, and the speed is steady from swing to swing.",
     ("core_rotation",), ("right_wrist",), ("hand_speed",),
     (("hand_speed_peak", "Racket-hand speed at the swing peak", 7.4, "stature/s", PROJ, 0.7, (6.0, 9.0), "hand_speed", 9, 8.6),),
     ((428, "Your fastest swing", "best", "path"),)),
    ("footwork.first_step", S, 0.3,
     "First move after their hit is on the slow side",
     "Your first move after the opponent hits is near the slow end of the practice guide range.",
     ("first_step", "split_step"), ("left_ankle", "right_ankle"), (),
     (("first_step_delay_s", "First move after their hit", 0.29, "s", TIME, 0.66, (0.15, 0.3), None, 12, 0.18),),
     ((462, "Late first move", "issue", "feet"),)),
    ("footwork.step_pattern", S, 0.2,
     "Step count before lunges looks normal",
     "You take a normal number of steps before most lunges.",
     ("lunge_depth", "lateral_movement"), ("left_ankle", "right_ankle"), (),
     (("steps_before_lunge", "Steps before a lunge", 3, "count", COUNT, 0.6, (1, 3), None, 6, None),),
     ()),
    ("footwork.coverage", S, 0.15,
     "You moved through about half of your court",
     "This describes where you moved in your half. It does not say whether that was right, because the shuttle is not tracked.",
     ("lateral_movement",), ("left_ankle", "right_ankle"), ("court_x", "court_y"),
     (("court_coverage_share", "Share of your half you moved through", 0.46, "share", PLANE, 0.77, None, None, 1, None),),
     ()),
    ("tactics.base_discipline", S, 0.2,
     "You are often away from base when the opponent hits",
     "When the opponent hits, you are often farther from the middle than the practice guide range. "
     "Some shots make that the right place to be, so this is a description, not a verdict.",
     ("base_recovery",), ("left_ankle", "right_ankle"), ("base_distance",),
     (("base_distance_m", "Distance from base when they hit", 0.9, "m", PLANE, 0.7, (0.3, 0.8), "base_distance", 12, 0.4),),
     ((300, "Away from base as they hit", "issue", "spot"),)),
    ("tactics.tempo", S, 0.1,
     "Steady rhythm between swings",
     "The time between swings in your rallies stays fairly steady.",
     ("endurance",), (), (),
     (("swing_gap_s", "Time between swings in rallies", 1.5, "s", TIME, 0.68, None, None, 10, None),),
     ()),
    ("tactics.initiative", S, 0.1,
     "You spend a bit more rally time nearer the net",
     "You spend a bit more of the rally nearer the net than the opponent. This is a position share only, "
     "not proof of attacking play, because the shuttle is not tracked.",
     ("first_step",), (), (),
     (("initiative_share", "Rally time nearer the net than them", 0.57, "share", PLANE, 0.66, None, None, 1, None),),
     ()),
)

FAR = (
    ("technique.stay_lower", G, 0.1,
     "You stay low and ready between shots",
     "Your hips stay low while you wait, inside the practice guide range.",
     ("lower_stance",), ("left_hip", "right_hip"), ("hip_height_ratio",),
     (("hip_height_ratio", "Hip height while waiting", 0.45, "heights", PROJ, 0.61, (0.42, 0.48), "hip_height_ratio", 12, 0.43),),
     ((330, "Low ready stance", "best", "height"),)),
    ("technique.knee_bend", G, 0.1,
     "Good knee bend in the ready stance",
     "Your knees stay bent while you wait, inside the practice guide range.",
     ("lower_stance",), ("left_knee", "right_knee"), ("knee_angle_deg",),
     (("knee_angle_deg", "Knee angle in the ready stance", 132, "deg", PROJ, 0.5, (120, 145), "knee_angle_deg", 12, None),),
     ()),
    ("technique.lunge_depth", S, 0.25,
     "Lunges are close to the practice guide depth",
     "Your lunges drop a little less than the practice guide range.",
     ("lunge_depth", "hip_mobility"), ("left_hip", "right_hip"), (),
     (("lunge_depth_ratio", "Hip drop in net lunges", 0.21, "fraction", PROJ, 0.48, (0.22, 0.32), None, 5, 0.26),),
     ((700, "Lunge a bit shallow", "issue", "height"),)),
    ("technique.stance_width", S, 0.2,
     "Feet a little wide in the ready stance",
     "Your feet are a little wider apart than the practice guide range while you wait.",
     ("balance",), ("left_ankle", "right_ankle"), ("stance_width_ratio",),
     (("stance_width_ratio", "Feet apart in the ready stance", 0.43, "heights", PROJ, 0.52, (0.3, 0.4), "stance_width_ratio", 12, None),),
     ((330, "Wide stance", "issue", "width"),)),
    ("technique.trunk_lean", I, 0.55,
     "Upper body leans to the side when you hit",
     "Your upper body leans to the side more than the practice guide range during swings.",
     ("core_stability", "core_rotation"), ("left_shoulder", "right_shoulder", "left_hip", "right_hip"), ("trunk_lean_deg",),
     (("trunk_lean_deg", "Sideways lean of the upper body", 17, "deg", PROJ, 0.57, (0, 10), "trunk_lean_deg", 8, 9),),
     ((240, "Leaning while hitting", "issue", "angle"), (410, "Upright swing", "best", "angle"))),
    ("contact.reach_height", I, 0.61,
     "Overhead hits are low",
     "Your overhead hits are below the practice guide range for contact height, so you hit the shuttle later.",
     ("contact_height", "jump_power"), ("right_wrist", "right_shoulder"), (),
     (("contact_height_ratio", "Contact height on overheads", 1.09, "heights", PROJ, 0.55, (1.2, 1.35), None, 8, 1.21),),
     ((180, "Low overhead hit", "issue", "height"),)),
    ("contact.arm_extension", S, 0.25,
     "Arm a little bent at the hit",
     "Your elbow angle at the hit is just under the practice guide range. This is a projected angle.",
     ("arm_extension",), ("right_elbow",), (),
     (("elbow_angle_deg", "Elbow angle at the hit", 152, "deg", PROJ, 0.44, (155, 175), None, 8, None),),
     ((182, "Slightly bent arm", "issue", "angle"),)),
    ("contact.hand_speed", S, 0.2,
     "Racket-hand speed is near the low end",
     "Your racket hand is near the low end of the practice guide range at the swing peak.",
     ("core_rotation",), ("right_wrist",), ("hand_speed",),
     (("hand_speed_peak", "Racket-hand speed at the swing peak", 6.1, "stature/s", PROJ, 0.5, (6.0, 9.0), "hand_speed", 8, None),),
     ()),
    ("footwork.split_step", S, 0.2,
     "Split step is hard to see from this distance",
     "The far player is small in the picture, so small hops are hard to measure. Treat this as a rough look.",
     ("split_step",), ("left_ankle", "right_ankle"), (),
     (("split_step_share", "Opponent hits with a split step", 0.55, "share", TIME, 0.35, None, None, 13, None),),
     ()),
    ("footwork.recovery", G, 0.1,
     "Quick return to base after your shots",
     "You get back near the middle quickly after you hit, inside the practice guide range.",
     ("base_recovery",), ("left_ankle", "right_ankle"), ("base_distance",),
     (("recovery_time_s", "Time back to base after your shot", 0.74, "s", TIME, 0.6, (0.6, 0.9), "base_distance", 10, 0.62),),
     ((560, "Fast way back to base", "best", "path"),)),
    ("footwork.first_step", P, 0.78,
     "Slow first move after the opponent hits",
     "After the opponent hits, your first move starts later than the practice guide range.",
     ("first_step", "split_step"), ("left_ankle", "right_ankle"), (),
     (("first_step_delay_s", "First move after their hit", 0.41, "s", TIME, 0.63, (0.15, 0.3), None, 12, 0.24),),
     ((350, "Late first move", "issue", "feet"), (610, "Late again", "issue", "feet"), (470, "Quick first move", "best", "feet"))),
    ("footwork.step_pattern", I, 0.5,
     "Many small steps before a lunge",
     "You take more steps than the practice guide range before you lunge, so you arrive later.",
     ("lunge_depth", "lateral_movement"), ("left_ankle", "right_ankle"), (),
     (("steps_before_lunge", "Steps before a lunge", 5, "count", COUNT, 0.5, (1, 3), None, 5, None),),
     ((690, "Extra small steps", "issue", "feet"),)),
    ("footwork.coverage", S, 0.15,
     "You moved through less than half of your court",
     "This describes where you moved in your half. It does not say whether that was right, because the shuttle is not tracked.",
     ("lateral_movement",), ("left_ankle", "right_ankle"), ("court_x", "court_y"),
     (("court_coverage_share", "Share of your half you moved through", 0.39, "share", PLANE, 0.6, None, None, 1, None),),
     ()),
    ("tactics.base_discipline", S, 0.2,
     "Often far from base when the opponent hits",
     "When the opponent hits, you are often farther from the middle than the practice guide range. "
     "Some shots make that the right place to be.",
     ("base_recovery",), ("left_ankle", "right_ankle"), ("base_distance",),
     (("base_distance_m", "Distance from base when they hit", 1.3, "m", PLANE, 0.62, (0.3, 0.8), "base_distance", 12, None),),
     ((350, "Far from base as they hit", "issue", "spot"),)),
    ("tactics.tempo", S, 0.1,
     "Steady rhythm between swings",
     "The time between swings in your rallies stays fairly steady.",
     ("endurance",), (), (),
     (("swing_gap_s", "Time between swings in rallies", 1.7, "s", TIME, 0.6, None, None, 10, None),),
     ()),
    ("tactics.initiative", S, 0.1,
     "You spend a bit more rally time farther back",
     "You spend more of the rally farther from the net than the opponent. This is a position share only.",
     ("first_step",), (), (),
     (("initiative_share", "Rally time nearer the net than them", 0.43, "share", PLANE, 0.6, None, None, 1, None),),
     ()),
)

# Rough pixel anchor of each player in a 1280 x 720 broadcast frame.
ANCHOR = {Role.NEAR: (640.0, 560.0), Role.FAR: (640.0, 230.0)}


def _marks(style: str, frame: int, player: Role, value: float) -> tuple[Mark, ...]:
    x, y = ANCHOR[player]
    if style == "height":
        return (
            Mark("line", frame, ((x - 45, y - 150), (x + 45, y - 150)), "hip height", value),
            Mark("band", frame, ((x - 55, y - 128), (x + 55, y - 90)), "practice guide", None, "target"),
            Mark("label", frame, ((x - 40, y - 175),), "Sit lower", None, "neutral"),
        )
    if style == "angle":
        return (Mark("angle", frame, ((x + 20, y - 230), (x + 50, y - 190), (x + 70, y - 240)), "elbow", value),)
    if style == "width":
        return (Mark("bracket", frame, ((x - 40, y), (x + 40, y)), "stance width", value),)
    if style == "path":
        return (Mark("trail", frame, tuple((x - 80 + 16 * k, y - 4 * k) for k in range(6)), "path", value),)
    if style == "spot":
        return (Mark("dot", frame, ((x - 70, y),), "", value), Mark("dot", frame, ((x, y - 20),), "base", None, "target"))
    return (Mark("dot", frame, ((x - 20, y),), "", value), Mark("dot", frame, ((x + 20, y),), "", value))


def _finding(player: Role, row: tuple, tier: Tier, values: Mapping[tuple[str, str], float]) -> tuple[Finding, tuple[Metric, ...]]:
    fid, severity, score, headline, explain, tags, joints, overlay, metric_rows, moment_rows = row
    metrics = tuple(
        Metric(name, label, values.get((player.value, name), value), unit, basis, conf, target, series, n_obs, best)
        for name, label, value, unit, basis, conf, target, series, n_obs, best in metric_rows
    )
    moments = tuple(
        Moment(player, key - 10, key, key + 10, label, kind, _marks(style, key, player, metrics[0].value))
        for key, label, kind, style in moment_rows
    )
    finding = Finding(
        fid, FINDING_CATALOG[fid][0], player, severity, headline, explain, metrics, moments, tier,
        joints, overlay, tags, score,
    )
    return finding, metrics


def _series(player: Role) -> dict[str, tuple[Optional[float], ...]]:
    lift = 0.0 if player == Role.NEAR else -0.06
    out: dict[str, list[Optional[float]]] = {
        "hip_height_ratio": [], "knee_angle_deg": [], "stance_width_ratio": [], "trunk_lean_deg": [],
        "hand_speed": [], "base_distance": [], "court_x": [], "court_y": [],
    }
    for i in range(N_FRAMES):
        gap = i % 97 == 0
        out["hip_height_ratio"].append(None if gap else 0.5 + lift + 0.04 * math.sin(i / 23))
        out["knee_angle_deg"].append(None if gap else 148 + 10 * math.sin(i / 29))
        out["stance_width_ratio"].append(0.35 + 0.05 * math.sin(i / 31))
        out["trunk_lean_deg"].append(6 + 5 * math.sin(i / 17))
        out["hand_speed"].append(max(0.0, 7 * math.sin(i / 9)))
        out["base_distance"].append(abs(1.1 * math.sin(i / 40)))
        out["court_x"].append(3.0 + 2.0 * math.sin(i / 50))
        out["court_y"].append((10.0 if player == Role.NEAR else 3.4) + 1.5 * math.cos(i / 45))
    return {name: tuple(values) for name, values in out.items()}


def _events() -> tuple[Event, ...]:
    events: list[Event] = []
    for k, frame in enumerate(range(60, N_FRAMES - 30, 50)):
        player = Role.NEAR if k % 2 == 0 else Role.FAR
        events.append(swing_event(player, frame, start=frame - 8, end=frame + 6, peak_frame=frame - 2,
                                  peak_speed=6.5 + (k % 3), overhead=k % 3 != 1))
    events.append(lunge_event(Role.NEAR, 640, depth=0.17, bottom_frame=650, end_frame=668))
    events.append(lunge_event(Role.NEAR, 782, depth=0.24, bottom_frame=792, end_frame=810))
    events.append(lunge_event(Role.FAR, 690, depth=0.21, bottom_frame=700, end_frame=716))
    events.append(Event("split_step", Role.NEAR, 726, 0.7))
    events.append(Event("split_step", Role.FAR, 466, 0.4))
    return tuple(sorted(events, key=lambda e: (e.frame, e.kind, e.player.value)))


def _build(tier: Tier, warnings: tuple[str, ...], values: Mapping[tuple[str, str], float]) -> Report:
    findings: list[Finding] = []
    metrics: dict[str, list[Metric]] = {}
    for player, rows in ((Role.NEAR, NEAR), (Role.FAR, FAR)):
        for row in rows:
            finding, found = _finding(player, row, tier, values)
            findings.append(finding)
            metrics.setdefault(player.value, []).extend(found)
    return Report(
        schema_version=SCHEMA_VERSION,
        source={"video": "fixture clip"},
        tier=tier,
        n_frames=N_FRAMES,
        fps=FPS,
        players=(Role.NEAR, Role.FAR),
        metrics={role: tuple(found) for role, found in metrics.items()},
        series={Role.NEAR.value: _series(Role.NEAR), Role.FAR.value: _series(Role.FAR)},
        events=_events(),
        findings=tuple(findings),
        windows=(Window(300, 330, "split step check"),),
        video_id="fixture",
        revision=1,
        warnings=warnings,
        timestamps=tuple(i / FPS for i in range(N_FRAMES)),
    )


def full_report() -> Report:
    return _build(Tier.REFINED, ("repeated_frames",), {})


def preview_report() -> Report:
    return _build(Tier.PREVIEW, ("repeated_frames", "preview_estimate"), {})


def earlier_report() -> Report:
    """An older session of the same players, for progress questions."""
    older = {
        ("near", "hip_height_ratio"): 0.56,
        ("near", "split_step_share"): 0.31,
        ("near", "recovery_time_s"): 1.25,
        ("near", "elbow_angle_deg"): 141,
        ("far", "first_step_delay_s"): 0.44,
        ("far", "contact_height_ratio"): 1.05,
    }
    return _build(Tier.REFINED, (), older)


def _stub(id: str, name: str, tags: tuple[str, ...], gear: tuple[str, ...], dose: str, level: int, why: str,
          easier: Optional[str] = None, harder: Optional[str] = None) -> Exercise:
    return Exercise(id=id, name=name, fix_tags=tags, equipment=gear, sets_reps=dose,
                    cues=(f"{name} cue one", f"{name} cue two", f"{name} cue three"),
                    mistakes=(f"{name} mistake one", f"{name} mistake two"),
                    safety=f"Stop {name.lower()} if it causes sharp pain.", difficulty=level, gif=f"gifs/{id}.gif",
                    duration_s=2.0, why=why, easier=easier, harder=harder)


def stub_library() -> dict[str, Exercise]:
    """Six exercises for tests that need a small, controlled library.

    low_squat's harder edge leads to jump_up, which trains jump power but not a low stance,
    like goblet_squat -> squat_jumps in the real library.
    """
    rows = (
        _stub("low_hold", "Low hold", ("lower_stance",), (), "3 x 20 seconds", 1, "Holds a low stance.", harder="low_squat"),
        _stub("low_squat", "Low squat", ("lower_stance",), ("dumbbell",), "3 x 8", 2, "Loads the low stance.",
              easier="low_hold", harder="jump_up"),
        _stub("jump_up", "Jump up", ("jump_power",), (), "3 x 5", 3, "Builds jump power.", easier="low_squat"),
        _stub("hop_drill", "Hop drill", ("split_step", "first_step"), (), "3 x 20 hops", 1, "Practises the small hop.",
              harder="rope_skip"),
        _stub("rope_skip", "Rope skip", ("split_step",), ("rope",), "3 x 45 seconds", 2, "Builds bouncy feet.",
              easier="hop_drill"),
        _stub("band_turn", "Band turn", ("core_rotation",), ("band",), "2 x 10 each side", 2, "Trains the trunk turn."),
    )
    return {e.id: e for e in rows}
