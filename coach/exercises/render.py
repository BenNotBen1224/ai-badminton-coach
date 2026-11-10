"""Original PIL drawings of the gym exercises. No downloaded images.

Each exercise is a side or front view, a few named poses, and a smooth loop.
Bone lengths and planted feet come from poses.py. This file only chooses the
poses, draws them, and writes the GIFs and contact sheets.
"""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from coach.exercises.poses import (
    FLOOR_Y,
    FOREARM,
    HEAD_R,
    HEIGHT,
    SHOULDER_W,
    TORSO,
    UPPER_ARM,
    WIDTH,
    Pt,
    chest_forward,
    max_joint_jump,
    seam_error,
    smoother,
    solve_key,
    to_px,
    validate_sequence,
)

ROOT = Path(__file__).resolve().parent
GIF_DIR = ROOT / "gifs"
SHEET_DIR = ROOT / "contact-sheets"
FRAMES = 36
# GIF delays are stored in centiseconds. A 67 ms delay truncates to 60,
# and merged frames then drift further, so the file no longer matches duration_s.
# The route still uses 90 frames. The delays are built to sum to the library time.
SIX_FRAMES = 90
SS = 3
SPLIT_HOP_CM = 4.5

BG = (245, 242, 234)
INK = (36, 44, 51)
FAR = (166, 172, 178)
WARM = (96, 54, 44)
ACCENT = (214, 98, 42)
SHOE = (42, 46, 52)
JOINT = (28, 96, 108)
FLOOR = (186, 178, 166)
WALL = (216, 212, 204)
WALL_EDGE = (164, 160, 152)
MAT = (198, 214, 204)
BAND = (36, 104, 176)
ROPE = (176, 132, 70)
IRON = (68, 72, 78)
TEXT = (32, 32, 32)
MUTED = (96, 90, 82)
COURT_FILL = (232, 236, 226)
COURT_LINE = (118, 138, 108)
GIF_NOTES = {
    "split_step_hops": "gym rebound, not court timing",
    "band_trunk_rotation": "chest turns, feet stay",
    "standing_trunk_turn": "chest turns, feet stay",
}


def _lg(toe, lift=0.0, air=0.0, knee=(1.0, 0.25), support=True):
    return {"toe": toe, "lift": lift, "air": air, "knee": knee, "support": support}


def _fk(sh, el, planted=False, support=False):
    return {"shoulder": sh, "elbow": el, "planted": planted, "support": support}


def _wr(x, y, planted=False, support=False, long=False, z=0.0):
    wrist = (x, y) if z == 0.0 else (x, y, z)
    row = {"shoulder": 22.0, "elbow": 35.0, "wrist": wrist, "planted": planted, "support": support}
    if long:
        row["long"] = True
    return row


def _gp(abduction, elbow):
    return {"shoulder": abduction, "elbow": elbow, "style": "goalpost", "planted": False, "support": False}


def _side(t, phase, pelvis, trunk, legs, arms, flight=0.0, accent=(), head="up"):
    return {
        "t": t,
        "phase": phase,
        "view": "side",
        "head": head,
        "pelvis": pelvis,
        "trunk": trunk,
        "flight": flight,
        "legs": legs,
        "arms": arms,
        "accent": accent,
    }


def _front(t, phase, pelvis, legs, arms, trunk=6.0, flight=0.0, accent=(), corner=None, twist=0.0, scale=1.0, yaw=0.0):
    row = {
        "t": t,
        "phase": phase,
        "view": "front",
        "head": "up",
        "pelvis": pelvis,
        "trunk": trunk,
        "flight": flight,
        "legs": legs,
        "arms": arms,
        "accent": accent,
        "shoulder_scale": scale,
        "twist_shift": twist,
        "yaw": yaw,
    }
    if corner:
        row["corner"] = corner
    return row


def _both_fk(sh, el):
    return {"l": _fk(sh, el), "r": _fk(sh + 6, el + 4)}


def _ready_arms(y, spread=11.0, drop=22.0):
    return {"l": _wr(-spread, y + drop), "r": _wr(spread, y + drop)}


def _feet(half, air_l=0.0, air_r=0.0, support_l=True, support_r=True, toe_l=None, toe_r=None):
    return {
        "l": _lg(toe_l if toe_l is not None else -half, air=air_l, knee=(-1.0, 0.3), support=support_l),
        "r": _lg(toe_r if toe_r is not None else half, air=air_r, knee=(1.0, 0.3), support=support_r),
    }


def _loop(keys):
    end = deepcopy(keys[0])
    end["t"] = 1.0
    return keys + [end]


def _motion(name, keys, camera=0.0, zoom=1.12, props=None, rope_turns=0.0, floor_y=FLOOR_Y, frames=None, azimuth=0.0):
    row = {
        "name": name,
        "keys": _loop(keys),
        "camera": camera,
        "zoom": zoom,
        "floor_y": floor_y,
        "props": props or {},
        "rope_turns": rope_turns,
        "azimuth": azimuth,
    }
    if frames:
        row["frames"] = frames
    return row


def _build_motions():
    wall_legs = {"l": _lg(42, knee=(1.0, 0.08)), "r": _lg(44.5, knee=(1.0, 0.08))}
    wall = _motion(
        "Wall sit",
        [
            _side(0.00, "sit", (-2, 60), 0, wall_legs, {"l": _wr(10, 64), "r": _wr(16, 62)}, accent=("l_knee", "r_knee")),
            _side(0.32, "hold", (-2, 47), 0, wall_legs, {"l": _wr(12, 58), "r": _wr(18, 56)}, accent=("l_knee", "r_knee", "l_hip", "r_hip")),
            _side(0.68, "hold", (-2, 47), 0, wall_legs, {"l": _wr(12, 58), "r": _wr(18, 56)}, accent=("l_knee", "r_knee", "l_hip", "r_hip")),
        ],
        camera=16,
        zoom=1.16,
        props={"wall": -8, "support_x": [-8]},
    )

    wide_feet = _feet(28)
    sit_high = _front(0.00, "ready", (0, 84), wide_feet, _ready_arms(84), accent=("l_hip", "r_hip"))
    sit_low = _front(0.34, "hold", (0, 76), wide_feet, _ready_arms(76, 16, 12), trunk=6, accent=("l_knee", "r_knee", "l_hip", "r_hip"))
    sit = _motion("Sit-low hold", [sit_high, sit_low, _front(0.66, "hold", (0, 76), wide_feet, _ready_arms(76, 16, 12), trunk=6, accent=("l_knee", "r_knee"))], zoom=1.02)

    squat_feet = {"l": _lg(38), "r": _lg(40)}
    squat_arms_up = {"l": _wr(22, 102), "r": _wr(28, 102)}
    squat_arms_down = {"l": _wr(6, 82), "r": _wr(12, 82)}
    squat_up = _side(0.00, "stand", (16, 86), 4, squat_feet, squat_arms_up, accent=("l_wrist",))
    squat_down = _side(0.36, "down", (-4, 54), 30, squat_feet, squat_arms_down, accent=("l_knee", "r_knee", "l_hip"))
    goblet = _motion(
        "Goblet squat",
        [squat_up, squat_down, _side(0.62, "up", (16, 86), 4, squat_feet, squat_arms_up)],
        camera=16,
        zoom=0.94,
        props={"dumbbell": True},
    )

    lunge_ready_legs = {"l": _lg(18, knee=(0.25, -1.0)), "r": _lg(24, knee=(1.0, 0.15))}
    lunge_step_legs = {"l": _lg(-8, lift=42, knee=(0.2, -1.0)), "r": _lg(40, knee=(1.0, 0.12))}
    lunge_deep_legs = {"l": _lg(-46, lift=65, knee=(-0.35, -1.0)), "r": _lg(50, knee=(1.0, 0.12))}
    lunge_ready = _side(0.00, "ready", (8, 84), 4, lunge_ready_legs, _both_fk(22, 28), accent=("r_hip",))
    lunge_step = _side(0.22, "step", (10, 70), 6, lunge_step_legs, _both_fk(46, 16), accent=("l_knee", "r_knee"))
    lunge_deep = _side(0.44, "reach", (12, 58), 8, lunge_deep_legs, _both_fk(62, 16), accent=("r_knee", "r_hip", "r_wrist"))
    forward = _motion(
        "Forward lunge with reach",
        [
            lunge_ready,
            lunge_step,
            lunge_deep,
            _side(0.62, "reach", (12, 58), 8, lunge_deep_legs, _both_fk(64, 16), accent=("r_knee", "r_wrist")),
            _side(0.82, "back", (10, 70), 6, lunge_step_legs, _both_fk(36, 20), accent=("l_knee",)),
        ],
        camera=2,
        zoom=0.82,
    )

    pause_ready_legs = {"l": _lg(18, knee=(0.25, -1.0)), "r": _lg(24, knee=(1.0, 0.15))}
    pause_step_legs = {"l": _lg(-8, lift=42, knee=(0.2, -1.0)), "r": _lg(40, knee=(1.0, 0.12))}
    pause_deep_legs = {"l": _lg(-46, lift=65, knee=(-0.35, -1.0)), "r": _lg(50, knee=(1.0, 0.12))}
    pause_ready = _side(0.00, "ready", (8, 84), 4, pause_ready_legs, _both_fk(24, 30), accent=("r_hip",))
    pause_step = _side(0.18, "step", (10, 70), 6, pause_step_legs, _both_fk(32, 24), accent=("l_knee",))
    pause_deep = _side(0.38, "pause", (12, 58), 8, pause_deep_legs, _both_fk(34, 26), accent=("r_knee", "r_hip"))
    pause = _motion(
        "Pause lunge and recover",
        [
            pause_ready,
            pause_step,
            pause_deep,
            _side(0.56, "pause", (12, 58), 8, pause_deep_legs, _both_fk(34, 26), accent=("r_knee",)),
            _side(0.76, "recover", (10, 72), 6, pause_step_legs, _both_fk(28, 28), accent=("r_hip",)),
        ],
        camera=2,
        zoom=0.82,
    )

    mid_feet = _feet(24)
    def _hip_arms(px, py):
        return {"l": _wr(px - 14, py + 6), "r": _wr(px + 14, py + 6)}

    lat_mid = _front(0.00, "middle", (0, 82), mid_feet, _hip_arms(0, 82), accent=("l_hip", "r_hip"))
    lat_r = _front(0.28, "right", (20, 76), _feet(16, toe_l=-14, toe_r=42), _hip_arms(20, 76), trunk=4, accent=("r_knee", "r_hip"))
    lat_l = _front(0.62, "left", (-20, 76), _feet(16, toe_l=-42, toe_r=14), _hip_arms(-20, 76), trunk=4, accent=("l_knee", "l_hip"))
    lateral = _motion(
        "Lateral lunge",
        [lat_mid, lat_r, _front(0.46, "middle", (0, 82), mid_feet, _hip_arms(0, 82)), lat_l],
        zoom=0.98,
    )

    # Gym split-step: a small rebound and a quick knee drop. Not a high jump,
    # and not an on-court timing cue. Feet stay under the body.
    hop_feet = _feet(18)
    split_keys = []
    hops = ((0.00, "ready", 82, 0.0), (0.08, "hop", 84, SPLIT_HOP_CM), (0.16, "land", 74, 0.0),
            (0.30, "hop", 84, SPLIT_HOP_CM), (0.38, "land", 74, 0.0),
            (0.52, "hop", 84, SPLIT_HOP_CM), (0.60, "land", 80, 0.0), (0.78, "ready", 82, 0.0))
    for t, phase, py, flight in hops:
        split_keys.append(_front(t, phase, (0, py), hop_feet, _ready_arms(py, 12, 18), flight=flight, accent=("l_knee", "r_knee")))
    split = _motion("Split-step hops", split_keys, zoom=1.36)

    rope_legs = {"l": _lg(14, lift=12), "r": _lg(16, lift=12)}
    rope_arms = _both_fk(10, 28)
    rope_keys = []
    for t, phase, flight in ((0.00, "hop", 10.0), (0.25, "land", 0.0), (0.50, "hop", 10.0), (0.75, "land", 0.0)):
        rope_keys.append(_side(t, phase, (0, 86), 2, rope_legs, rope_arms, flight=flight, accent=("l_ankle", "r_ankle")))
    rope = _motion("Jump rope", rope_keys, camera=2, zoom=1.08, props={"rope": True}, rope_turns=2.0)

    def _corner(t, phase, px, toe_l, toe_r, arm_l, arm_r, corner, py=76):
        return _front(
            t, phase, (px, py), _feet(14, toe_l=toe_l, toe_r=toe_r),
            {"l": _fk(*arm_l), "r": _fk(*arm_r)}, trunk=4,
            accent=("l_knee", "r_knee"), corner=corner,
        )

    def _base(t):
        return _front(
            t, "base", (0, 84), _feet(14),
            {"l": _fk(22, 30), "r": _fk(22, 30)},
            trunk=4, accent=("l_hip", "r_hip"), corner="base",
        )

    # The cue is corner-to-base recovery, so the loop is base, corner, base.
    # The last base matches the first so the seam closes. Duration is SIX_FRAMES.
    six = _motion(
        "Six-corner shadow",
        [
            _base(0.00),
            _corner(0.07, "front-left", -22, -46, -4, (26, 28), (26, 28), "front-left", py=72),
            _base(0.14),
            _corner(0.21, "front-right", 22, 4, 46, (26, 28), (26, 28), "front-right", py=72),
            _base(0.28),
            _corner(0.35, "side-right", 26, 4, 50, (26, 28), (26, 28), "side-right", py=72),
            _base(0.42),
            _corner(0.49, "back-right", 16, 0, 28, (26, 28), (26, 28), "back-right", py=80),
            _base(0.56),
            _corner(0.63, "back-left", -16, -28, 0, (26, 28), (26, 28), "back-left", py=80),
            _base(0.70),
            _corner(0.77, "side-left", -26, -50, -4, (26, 28), (26, 28), "side-left", py=72),
            _base(0.90),
        ],
        zoom=1.02,
        props={"court": True, "base_mark": True},
        frames=SIX_FRAMES,
    )

    def _skate_feet(stand, toe_stand, toe_free):
        # Free knee stays over its foot so the front-view shin stays near vertical.
        out_knee = (1.0, -1.0) if stand == "l" else (-1.0, -1.0)
        down = (0.0, -1.0)
        return {
            "l": _lg(toe_stand if stand == "l" else toe_free, air=0 if stand == "l" else 6, knee=down if stand == "l" else out_knee, support=stand == "l"),
            "r": _lg(toe_stand if stand == "r" else toe_free, air=0 if stand == "r" else 6, knee=down if stand == "r" else out_knee, support=stand == "r"),
        }

    def _skate_arms(px, py):
        return {"l": _wr(px - 14, py + 10), "r": _wr(px + 14, py + 10)}

    def _skate_land(t, phase, px, stand):
        # Foot sits under the hip, still within a hand's width of the pelvis.
        support_x = px + (-6 if stand == "l" else 6)
        free = support_x + (22 if stand == "l" else -22)
        return _front(
            t, phase, (px, 78), _skate_feet(stand, support_x, free), _skate_arms(px, 78),
            trunk=0, accent=(f"{stand}_knee",),
        )

    def _skate_air(t):
        return _front(
            t, "across", (0, 82),
            {"l": _lg(-8, air=5, knee=(-1.0, -1.0), support=False), "r": _lg(8, air=5, knee=(1.0, -1.0), support=False)},
            _skate_arms(0, 82), flight=8.0, trunk=0, accent=("l_hip", "r_hip"),
        )

    # The body travels and lands on the other foot. A 22 cm rise was a bounce, not a skater hop.
    skater = _motion(
        "Skater hops",
        [
            _skate_land(0.00, "left", -26, "l"),
            _skate_land(0.10, "left", -26, "l"),
            _skate_air(0.22),
            _skate_land(0.38, "right", 26, "r"),
            _skate_land(0.52, "right", 26, "r"),
            _skate_air(0.66),
            _skate_land(0.84, "left", -26, "l"),
        ],
        zoom=1.12,
        props={"base_mark": True},
    )

    calf_feet_down = {"l": _lg(16, lift=0), "r": _lg(18, lift=0)}
    calf_feet_up = {"l": _lg(16, lift=34), "r": _lg(18, lift=34)}
    calf_down = _side(0.00, "down", (12, 86), 0, calf_feet_down, _both_fk(12, 18), accent=("l_ankle", "r_ankle"))
    calf_up = _side(0.40, "up", (14, 94), 0, calf_feet_up, _both_fk(14, 16), accent=("l_ankle", "r_ankle"))
    calf = _motion("Calf raises", [calf_down, calf_up, _side(0.72, "down", (12, 86), 0, calf_feet_down, _both_fk(12, 18))], camera=8, zoom=1.08)

    sl_down_legs = {"l": _lg(6, air=8, lift=18, support=False, knee=(0.4, -1.0)), "r": _lg(14, lift=0)}
    sl_up_legs = {"l": _lg(6, air=10, lift=22, support=False, knee=(0.4, -1.0)), "r": _lg(14, lift=34)}
    sl_arms = {"l": _fk(18, 22), "r": _wr(36, 96, planted=True)}
    single = _motion(
        "Single-leg calf raise",
        [
            _side(0.00, "down", (4, 86), 0, sl_down_legs, sl_arms, accent=("r_ankle",)),
            _side(0.42, "up", (8, 94), 0, sl_up_legs, {"l": _fk(18, 22), "r": _wr(36, 100, planted=True)}, accent=("r_ankle",)),
            _side(0.74, "down", (4, 86), 0, sl_down_legs, sl_arms),
        ],
        camera=10,
        zoom=1.08,
        props={"wall": 46},
    )

    jump_feet = {"l": _lg(36), "r": _lg(38)}
    # Arms stay near the body so the hand does not swing in the same frames as the rise.
    jumps = _motion(
        "Squat jumps",
        [
            _side(0.00, "load", (-2, 54), 12, jump_feet, _both_fk(18, 26), accent=("l_knee", "r_knee")),
            _side(0.10, "drive", (0, 64), 10, jump_feet, _both_fk(32, 24), flight=2.0, accent=("l_knee", "r_hip")),
            _side(0.18, "rise", (4, 76), 6, jump_feet, _both_fk(46, 22), flight=12.0, accent=("l_ankle", "r_hip")),
            _side(0.26, "jump", (6, 84), 2, jump_feet, _both_fk(58, 20), flight=28.0, accent=("l_ankle", "r_hip")),
            _side(0.34, "hang", (6, 84), 2, jump_feet, _both_fk(54, 20), flight=32.0, accent=("l_ankle",)),
            _side(0.42, "down", (4, 74), 4, jump_feet, _both_fk(44, 22), flight=12.0, accent=("l_knee",)),
            _side(0.50, "land", (0, 60), 8, jump_feet, _both_fk(32, 24), accent=("l_knee", "r_knee")),
            _side(0.64, "stick", (6, 74), 4, jump_feet, _both_fk(22, 22)),
            _side(0.82, "load", (-2, 54), 12, jump_feet, _both_fk(18, 26)),
        ],
        camera=14,
        zoom=0.9,
    )

    plank = _motion(
        "Front plank",
        [{"t": 0.00, "phase": "set", "kind": "plank", "view": "side", "head": "follow", "toe": 8},
         {"t": 0.40, "phase": "hold", "kind": "plank", "view": "side", "head": "follow", "toe": 8}],
        camera=78,
        zoom=1.72,
        floor_y=292,
        props={"mat": True},
    )
    side_plank = _motion(
        "Side plank",
        [{"t": 0.00, "phase": "set", "kind": "plank", "view": "side", "head": "follow", "toe": 8, "side_plank": True},
         {"t": 0.40, "phase": "hold", "kind": "plank", "view": "side", "head": "follow", "toe": 8, "side_plank": True}],
        camera=86,
        zoom=1.55,
        floor_y=286,
        props={"mat": True},
    )

    # Hands stay fixed in the chest frame. Yaw turns the chest and the arms
    # together about the vertical axis. An x/y arm arc leaves the chest still.
    grip_y = 112.0
    shoulder_y = 84.0 + TORSO
    dy = shoulder_y - grip_y
    reach = UPPER_ARM + FOREARM
    grip_z = -math.sqrt(reach * reach - SHOULDER_W * SHOULDER_W - dy * dy)
    chest_hands = {"l": _wr(0.0, grip_y, long=True, z=grip_z), "r": _wr(0.0, grip_y, long=True, z=grip_z)}
    turn_feet = _feet(12)
    turn_mark = ("ribs",)

    def _turn_key(t, phase, yaw):
        return _front(t, phase, (0, 84), turn_feet, chest_hands, trunk=0, yaw=yaw, accent=turn_mark)

    turn = _motion(
        "Standing trunk turn",
        [
            _turn_key(0.00, "middle", 0),
            _turn_key(0.16, "middle", 0),
            _turn_key(0.42, "turn", 50),
            _turn_key(0.58, "turn", 50),
            _turn_key(0.84, "middle", 0),
        ],
        zoom=1.05,
        props={"shared_grip": True, "chest_turn": True},
    )
    band = _motion(
        "Band trunk rotation",
        [
            _turn_key(0.00, "set", -38),
            _turn_key(0.14, "set", -38),
            _turn_key(0.42, "turn", 44),
            _turn_key(0.58, "turn", 44),
            _turn_key(0.88, "set", -38),
        ],
        camera=-10,
        zoom=1.0,
        azimuth=26,
        props={"band_anchor": (-82, grip_y), "shared_grip": True, "chest_turn": True},
    )

    angel_feet = _feet(11)
    angel_low = _front(0.00, "down", (0, 84), angel_feet, {"l": _gp(42, 92), "r": _gp(42, 92)}, accent=("l_elbow", "r_elbow"))
    angel_up = _front(0.46, "up", (0, 84), angel_feet, {"l": _gp(152, 22), "r": _gp(152, 22)}, accent=("l_shoulder", "r_shoulder"))
    angels = _motion("Wall angels", [angel_low, angel_up, _front(0.78, "down", (0, 84), angel_feet, {"l": _gp(42, 92), "r": _gp(42, 92)})], zoom=1.04, props={"wall_panel": True})

    return {
        "wall_sit": wall,
        "sit_low_hold": sit,
        "goblet_squat": goblet,
        "forward_lunge_reach": forward,
        "pause_lunge_recover": pause,
        "lateral_lunge": lateral,
        "split_step_hops": split,
        "jump_rope": rope,
        "six_corner_shadow": six,
        "skater_hops": skater,
        "calf_raises": calf,
        "single_leg_calf_raise": single,
        "squat_jumps": jumps,
        "front_plank": plank,
        "side_plank": side_plank,
        "standing_trunk_turn": turn,
        "band_trunk_rotation": band,
        "wall_angels": angels,
    }


MOTIONS = _build_motions()


def _segment(keys, t):
    for index in range(len(keys) - 1):
        a, b = keys[index], keys[index + 1]
        if a["t"] <= t < b["t"] or index == len(keys) - 2:
            span = max(1e-6, b["t"] - a["t"])
            return a, b, max(0.0, min(1.0, (t - a["t"]) / span))
    return keys[-2], keys[-1], 1.0


def _lerp(a, b, u):
    return a + (b - a) * u


def _mix_prefer(a, b, u):
    if a[0] * b[0] + a[1] * b[1] < 0 and u < 0.5:
        return a
    if a[0] * b[0] + a[1] * b[1] < 0:
        return b
    return (_lerp(a[0], b[0], u), _lerp(a[1], b[1], u))


def _moving_contacts(a, b):
    if a.get("kind") == "plank":
        return []
    names = []
    if abs(a.get("flight", 0.0) - b.get("flight", 0.0)) > 0.25:
        names.extend(["l_toe", "l_heel", "r_toe", "r_heel", "l_wrist", "r_wrist"])
    for side in ("l", "r"):
        la, lb = a["legs"][side], b["legs"][side]
        if abs(la["toe"] - lb["toe"]) > 0.2 or abs(la["air"] - lb["air"]) > 0.25:
            names.extend([f"{side}_toe", f"{side}_heel"])
        elif abs(la["lift"] - lb["lift"]) > 0.6:
            names.append(f"{side}_heel")
        aa, ab = a["arms"][side], b["arms"][side]
        if bool(aa.get("planted")) != bool(ab.get("planted")):
            names.append(f"{side}_wrist")
        wa, wb = aa.get("wrist"), ab.get("wrist")
        if wa and wb and (abs(wa[0] - wb[0]) > 0.4 or abs(wa[1] - wb[1]) > 0.4):
            if aa.get("planted") or ab.get("planted"):
                names.append(f"{side}_wrist")
    return names


def _lerp_key(a, b, u):
    if a.get("kind") == "plank":
        out = deepcopy(a)
        out["phase"] = b["phase"] if u >= 0.5 else a["phase"]
        return out
    out = {
        "phase": b["phase"] if u >= 0.5 else a["phase"],
        "view": a["view"],
        "head": a.get("head", "up"),
        "pelvis": (_lerp(a["pelvis"][0], b["pelvis"][0], u), _lerp(a["pelvis"][1], b["pelvis"][1], u)),
        "trunk": _lerp(a.get("trunk", 0.0), b.get("trunk", 0.0), u),
        "flight": _lerp(a.get("flight", 0.0), b.get("flight", 0.0), u),
        "shoulder_scale": _lerp(a.get("shoulder_scale", 1.0), b.get("shoulder_scale", 1.0), u),
        "twist_shift": _lerp(a.get("twist_shift", 0.0), b.get("twist_shift", 0.0), u),
        "yaw": _lerp(a.get("yaw", 0.0), b.get("yaw", 0.0), u),
        "accent": b.get("accent", ()) if u >= 0.5 else a.get("accent", ()),
        "legs": {},
        "arms": {},
    }
    if a.get("corner") or b.get("corner"):
        out["corner"] = b.get("corner") if u >= 0.5 else a.get("corner")
    for side in ("l", "r"):
        la, lb = a["legs"][side], b["legs"][side]
        out["legs"][side] = {
            "toe": _lerp(la["toe"], lb["toe"], u),
            "lift": _lerp(la["lift"], lb["lift"], u),
            "air": _lerp(la["air"], lb["air"], u),
            "knee": _mix_prefer(la["knee"], lb["knee"], u),
            "support": la.get("support", True) and lb.get("support", True),
        }
        aa, ab = a["arms"][side], b["arms"][side]
        row = {
            "shoulder": _lerp(aa["shoulder"], ab["shoulder"], u),
            "elbow": _lerp(aa["elbow"], ab["elbow"], u),
            "planted": bool(aa.get("planted")) and bool(ab.get("planted")),
            "support": bool(aa.get("support")) and bool(ab.get("support")),
        }
        style = ab.get("style") or aa.get("style")
        if style:
            row["style"] = style
        if aa.get("wrist") and ab.get("wrist"):
            wa, wb = aa["wrist"], ab["wrist"]
            if len(wa) > 2 or len(wb) > 2:
                za = wa[2] if len(wa) > 2 else 0.0
                zb = wb[2] if len(wb) > 2 else 0.0
                row["wrist"] = (_lerp(wa[0], wb[0], u), _lerp(wa[1], wb[1], u), _lerp(za, zb, u))
            else:
                row["wrist"] = (_lerp(wa[0], wb[0], u), _lerp(wa[1], wb[1], u))
        if aa.get("long") and ab.get("long"):
            row["long"] = True
        out["arms"][side] = row
    return out


def frames_for(exercise_id: str, count: int | None = None) -> list:
    motion = MOTIONS[exercise_id]
    if count is None:
        count = motion.get("frames", FRAMES)
    keys = motion["keys"]
    frames = []
    for index in range(count):
        t = index / count
        a, b, raw = _segment(keys, t)
        key = _lerp_key(a, b, smoother(raw))
        key["camera"] = motion["camera"]
        key["zoom"] = motion["zoom"]
        key["floor_y"] = motion["floor_y"]
        key["azimuth"] = motion.get("azimuth", 0.0)
        key["_unplant"] = _moving_contacts(a, b)
        props = dict(motion["props"])
        if key.get("corner"):
            props["corner"] = key["corner"]
        key["props"] = props
        try:
            frames.append(solve_key(exercise_id, key, rope_t=t * motion["rope_turns"]))
        except ValueError as exc:
            raise ValueError(f"{exercise_id} t={t:.3f} {a.get('phase')}->{b.get('phase')}: {exc}") from exc
    return frames


def check(ids=None) -> int:
    failed = 0
    for exercise_id in ids or MOTIONS:
        try:
            frames = frames_for(exercise_id)
        except ValueError as exc:
            print(f"{exercise_id}\tUNREACHABLE\t{exc}")
            failed += 1
            continue
        issues = validate_sequence(frames)
        hops = max(frame.points["l_toe"].y for frame in frames)
        print(
            f"{exercise_id}\t{'OK' if not issues else 'FAIL'}\tseam={seam_error(frames):.2f}\t"
            f"jump={max_joint_jump(frames):.1f}\thop={hops:.1f}\tissues={len(issues)}"
        )
        for item in issues[:8]:
            print(f"  {item}")
        if issues:
            failed += 1
    return failed


def _font(size, bold=False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    try:
        return ImageFont.truetype(rf"C:\Windows\Fonts\{name}", size)
    except OSError:
        return ImageFont.load_default()


def _p(skel, pt, ox=0.0, oy=0.0):
    x, y = to_px(pt, skel.camera_x, skel.zoom, skel.floor_y, skel.azimuth)
    return ((x + ox) * SS, (y + oy) * SS)


def _cam_depth(skel, pt):
    """Larger is farther. At azimuth 0, forward (-z) is closer."""
    az = math.radians(skel.azimuth)
    return -pt.x * math.sin(az) + pt.z * math.cos(az)


def _bone(draw, a, b, width, fill):
    draw.line([a, b], fill=fill, width=width)
    r = max(1, width // 2)
    draw.ellipse([a[0] - r, a[1] - r, a[0] + r, a[1] + r], fill=fill)
    draw.ellipse([b[0] - r, b[1] - r, b[0] + r, b[1] + r], fill=fill)


def _poly_shoe(draw, skel, side, ox, fill):
    heel = skel.points[f"{side}_heel"]
    toe = skel.points[f"{side}_toe"]
    ankle = skel.points[f"{side}_ankle"]
    if skel.view == "front":
        cx = (heel.x + toe.x) / 2
        cy = min(heel.y, toe.y)
        left = _p(skel, Pt(cx - 6.5, cy + 5.5), ox)
        right = _p(skel, Pt(cx + 6.5, cy), ox)
        draw.ellipse([left[0], left[1], right[0], right[1]], fill=fill)
        return
    direction = Pt(toe.x - heel.x, toe.y - heel.y)
    length = math.hypot(direction.x, direction.y) or 1.0
    up = Pt(-direction.y / length * 6.0, direction.x / length * 6.0)
    if up.y < 0:
        up = Pt(-up.x, -up.y)
    pts = [
        _p(skel, heel, ox),
        _p(skel, toe, ox),
        _p(skel, Pt(toe.x + up.x, toe.y + up.y), ox),
        _p(skel, Pt(ankle.x + up.x * 0.3, ankle.y + up.y * 0.3), ox),
    ]
    draw.polygon(pts, fill=fill)


def _draw_limb(draw, skel, side, ox, fill, warm, arm_fill=None):
    p = skel.points
    accent = set(skel.accent)
    leg = warm if f"{side}_knee" in accent or f"{side}_hip" in accent else fill
    arm = warm if f"{side}_elbow" in accent or f"{side}_shoulder" in accent or f"{side}_wrist" in accent else (fill if arm_fill is None else arm_fill)
    _bone(draw, _p(skel, p[f"{side}_hip"], ox), _p(skel, p[f"{side}_knee"], ox), 12 * SS, leg)
    _bone(draw, _p(skel, p[f"{side}_knee"], ox), _p(skel, p[f"{side}_ankle"], ox), 8 * SS, leg)
    if skel.props.get("shared_grip"):
        _bone(draw, _p(skel, p[f"{side}_shoulder"], ox), _p(skel, p[f"{side}_wrist"], ox), 7 * SS, arm)
    else:
        _bone(draw, _p(skel, p[f"{side}_shoulder"], ox), _p(skel, p[f"{side}_elbow"], ox), 8 * SS, arm)
        _bone(draw, _p(skel, p[f"{side}_elbow"], ox), _p(skel, p[f"{side}_wrist"], ox), 6 * SS, arm)
    if not skel.props.get("shared_grip"):
        _bone(draw, _p(skel, p[f"{side}_wrist"], ox), _p(skel, p[f"{side}_hand"], ox), 5 * SS, arm)
    _poly_shoe(draw, skel, side, ox, SHOE)


def _draw_props(draw, skel, canvas):
    props = skel.props
    floor = skel.floor_y * SS
    if props.get("wall_panel"):
        draw.rectangle([36 * SS, 36 * SS, (WIDTH - 36) * SS, floor], fill=(232, 230, 224))
    if "wall" in props:
        x = _p(skel, Pt(props["wall"], 0))[0]
        draw.rectangle([x - 10 * SS, 28 * SS, x, floor], fill=WALL)
        draw.line([(x, 28 * SS), (x, floor)], fill=WALL_EDGE, width=2 * SS)
    if props.get("mat"):
        left = _p(skel, Pt(skel.points["l_toe"].x - 18, 0))[0]
        right = _p(skel, Pt(skel.points["r_hand"].x + 10, 0))[0]
        draw.rounded_rectangle([min(left, right), floor - 5 * SS, max(left, right), floor + 2 * SS], radius=4 * SS, fill=MAT)
    toes = [skel.points["l_toe"], skel.points["r_toe"]]
    mid = Pt(sum(t.x for t in toes) / 2, 0)
    shadow = _p(skel, mid)
    wide = 18 * SS if not skel.airborne else 10 * SS
    shade = (214, 208, 198) if not skel.airborne else (226, 222, 214)
    draw.ellipse([shadow[0] - wide, floor - 4 * SS, shadow[0] + wide, floor + 5 * SS], fill=shade)
    draw.line([(28 * SS, floor), ((WIDTH - 28) * SS, floor)], fill=FLOOR, width=2 * SS)
    if props.get("base_mark"):
        mark = _p(skel, Pt(0, 0))
        r = 5 * SS
        draw.ellipse([mark[0] - r, floor - r, mark[0] + r, floor + r], outline=ACCENT, width=2 * SS)


def _draw_shared_hands(draw, skel):
    if not skel.props.get("shared_grip") or skel.props.get("band_anchor"):
        return
    end = _p(skel, skel.points["l_wrist"])
    handle = 6 * SS
    draw.rounded_rectangle(
        [end[0] - handle, end[1] - handle, end[0] + handle, end[1] + handle],
        radius=3 * SS,
        outline=INK,
        width=3 * SS,
    )


def _draw_band(draw, skel):
    anchor = skel.props.get("band_anchor")
    if not anchor:
        return
    ax, ay = anchor
    post = _p(skel, Pt(ax, 0))
    hook = _p(skel, Pt(ax, ay))
    floor = skel.floor_y * SS
    draw.rectangle([post[0] - 8 * SS, hook[1] - 18 * SS, post[0] + 5 * SS, floor], fill=WALL)
    draw.line([(post[0] + 5 * SS, hook[1] - 18 * SS), (post[0] + 5 * SS, floor)], fill=WALL_EDGE, width=2 * SS)
    draw.ellipse([hook[0] - 5 * SS, hook[1] - 5 * SS, hook[0] + 5 * SS, hook[1] + 5 * SS], fill=IRON)
    grip = skel.points["l_wrist"]
    end = _p(skel, grip)
    draw.line([hook, end], fill=BAND, width=4 * SS)
    handle = 9 * SS
    draw.rounded_rectangle(
        [end[0] - handle, end[1] - handle, end[0] + handle, end[1] + handle],
        radius=3 * SS,
        outline=IRON,
        width=3 * SS,
    )


def _draw_dumbbell(draw, skel):
    if not skel.props.get("dumbbell"):
        return
    a = skel.points["l_wrist"]
    b = skel.points["r_wrist"]
    mid = Pt((a.x + b.x) / 2, (a.y + b.y) / 2)
    top = _p(skel, Pt(mid.x, mid.y + 8))
    bot = _p(skel, Pt(mid.x, mid.y - 8))
    _bone(draw, top, bot, 5 * SS, IRON)
    for pt in (top, bot):
        draw.ellipse([pt[0] - 7 * SS, pt[1] - 7 * SS, pt[0] + 7 * SS, pt[1] + 7 * SS], fill=IRON)


def _draw_rope(draw, skel):
    if not skel.props.get("rope"):
        return
    hand = skel.points["r_wrist"]
    toe = skel.points["r_toe"]
    heel = skel.points["r_heel"]
    head = skel.points["head"]
    # rope_t 0 is under the feet (the hop). 0.5 is over the head (the land).
    # An open arc, not a closed hoop: a hoop in side view never reads as passing under the shoe.
    phase = skel.rope_t % 1.0
    shapes = (
        [(hand.x + 1, hand.y - 6), (toe.x + 4, 8), (toe.x + 1, 1.6), (heel.x - 8, 1.6)],
        [(hand.x + 2, hand.y), (hand.x + 24, hand.y - 8), (hand.x + 30, 40), (hand.x + 18, 14)],
        [(hand.x + 2, hand.y + 4), (hand.x + 8, head.y + 4), (head.x + 2, head.y + 20), (head.x - 10, head.y + 6)],
        [(hand.x + 2, hand.y), (hand.x + 22, hand.y + 6), (hand.x + 28, 48), (hand.x + 16, 16)],
    )
    slot = phase * 4.0
    index = int(slot) % 4
    u = slot - int(slot)
    start, end = shapes[index], shapes[(index + 1) % 4]
    pts = []
    for a, b in zip(start, end):
        pts.append(_p(skel, Pt(a[0] + (b[0] - a[0]) * u, max(1.5, a[1] + (b[1] - a[1]) * u))))
    draw.line(pts, fill=ROPE, width=5 * SS, joint="curve")


def _draw_head(draw, skel, ox, fill):
    center = _p(skel, skel.points["head"], ox)
    radius = HEAD_R * 1.30 * skel.zoom * SS
    draw.ellipse([center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius], fill=fill)
    eye = (255, 248, 240)
    if skel.view == "front" and skel.props.get("chest_turn"):
        head = skel.points["head"]
        forward = chest_forward(skel.yaw)
        tip = _p(skel, Pt(head.x + forward.x * 22, head.y - 1, head.z + forward.z * 22), ox)
        dx, dy = tip[0] - center[0], tip[1] - center[1]
        span = math.hypot(dx, dy) or 1.0
        ux, uy = dx / span, dy / span
        px, py = -uy, ux
        for sign in (-1, 1):
            ex = center[0] + ux * radius * 0.18 + px * sign * radius * 0.28
            ey = center[1] + uy * radius * 0.18 + py * sign * radius * 0.28
            draw.ellipse([ex - 2 * SS, ey - 2 * SS, ex + 2 * SS, ey + 2 * SS], fill=eye)
        if span > radius * 0.28:
            draw.polygon(
                [
                    (center[0] + px * radius * 0.16, center[1] + py * radius * 0.16),
                    (center[0] - px * radius * 0.16, center[1] - py * radius * 0.16),
                    tip,
                ],
                fill=ACCENT,
            )
        return
    if skel.view == "front":
        for sign in (-1, 1):
            ex = center[0] + sign * 0.28 * radius
            ey = center[1] - radius * 0.08
            draw.ellipse([ex - 2 * SS, ey - 2 * SS, ex + 2 * SS, ey + 2 * SS], fill=eye)
        return
    facing = skel.points["head"].x - skel.points["neck"].x
    sign = 1 if facing >= 0 else -1
    ex = center[0] + sign * radius * 0.28
    ey = center[1] - radius * 0.05
    draw.ellipse([ex - 2.2 * SS, ey - 2.2 * SS, ex + 2.2 * SS, ey + 2.2 * SS], fill=eye)
    nose = (center[0] + sign * radius * 0.72, center[1] + radius * 0.12)
    draw.line([(center[0] + sign * radius * 0.35, ey), nose], fill=eye, width=2 * SS)


def draw_figure(skel) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH * SS, HEIGHT * SS), BG)
    draw = ImageDraw.Draw(canvas)
    _draw_props(draw, skel, canvas)
    rope_phase = skel.rope_t % 1.0
    rope_under = bool(skel.props.get("rope")) and (rope_phase < 0.18 or rope_phase > 0.82)
    if rope_under:
        _draw_rope(draw, skel)
    p = skel.points
    if skel.view == "side":
        if skel.props.get("shared_grip"):
            # One line for both arms. The usual far-arm offset would draw a second grip.
            _bone(draw, _p(skel, p["pelvis"]), _p(skel, p["shoulder_mid"]), 16 * SS, INK)
            _bone(draw, _p(skel, p["shoulder_mid"]), _p(skel, p["neck"]), 6 * SS, INK)
            _draw_head(draw, skel, 0, INK)
            _draw_limb(draw, skel, "l", 0, INK, WARM)
            _draw_limb(draw, skel, "r", 0, INK, WARM)
        else:
            _draw_limb(draw, skel, "l", -8, FAR, (186, 150, 132))
            _bone(draw, _p(skel, p["pelvis"]), _p(skel, p["shoulder_mid"]), 16 * SS, INK)
            _bone(draw, _p(skel, p["shoulder_mid"]), _p(skel, p["neck"]), 6 * SS, INK)
            _draw_limb(draw, skel, "r", 0, INK, WARM)
            _draw_head(draw, skel, 0, INK)
    else:
        shoulder_bar = [
            _p(skel, p["l_shoulder"]),
            _p(skel, p["r_shoulder"]),
        ]
        if skel.props.get("shared_grip"):
            # Hips stay wide. A turned chest narrows the shoulder line.
            if skel.props.get("chest_turn"):
                torso = [
                    _p(skel, p["l_hip"]),
                    _p(skel, p["l_shoulder"]),
                    _p(skel, p["r_shoulder"]),
                    _p(skel, p["r_hip"]),
                ]
                draw.polygon(torso, fill=(214, 206, 196), outline=(120, 108, 96))
                draw.line(torso + [torso[0]], fill=(120, 108, 96), width=3 * SS)
                draw.line([_p(skel, p["l_hip"]), _p(skel, p["r_hip"])], fill=INK, width=8 * SS)
                draw.line(shoulder_bar, fill=(120, 108, 96), width=5 * SS)
            _bone(draw, _p(skel, p["pelvis"]), _p(skel, p["shoulder_mid"]), 14 * SS, INK)
            if not skel.props.get("chest_turn"):
                draw.line(shoulder_bar, fill=WARM if "ribs" in skel.accent else INK, width=8 * SS)
            _bone(draw, _p(skel, p["shoulder_mid"]), _p(skel, p["neck"]), 6 * SS, INK)
            _draw_head(draw, skel, 0, INK)
            order = ("l", "r")
            if skel.props.get("chest_turn"):
                order = tuple(sorted(order, key=lambda side: _cam_depth(skel, skel.points[f"{side}_shoulder"]), reverse=True))
            for index, side in enumerate(order):
                arm_fill = FAR if skel.props.get("chest_turn") and index == 0 else None
                _draw_limb(draw, skel, side, 0, INK, WARM, arm_fill)
        else:
            _draw_limb(draw, skel, "l", 0, INK, WARM)
            _draw_limb(draw, skel, "r", 0, INK, WARM)
            _bone(draw, _p(skel, p["pelvis"]), _p(skel, p["shoulder_mid"]), 18 * SS, INK)
            draw.line(shoulder_bar, fill=WARM if "ribs" in skel.accent else INK, width=8 * SS)
            _bone(draw, _p(skel, p["shoulder_mid"]), _p(skel, p["neck"]), 6 * SS, INK)
            _draw_head(draw, skel, 0, INK)
    for side in ("l", "r"):
        joints = ("knee", "ankle", "shoulder") if skel.props.get("shared_grip") else ("knee", "ankle", "elbow", "shoulder")
        for joint in joints:
            name = f"{side}_{joint}"
            pt = _p(skel, p[name])
            r = 4 * SS if name in skel.accent else 2 * SS
            color = ACCENT if name in skel.accent else JOINT
            draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=color)
    if skel.props.get("rope") and not rope_under:
        _draw_rope(draw, skel)
    _draw_shared_hands(draw, skel)
    _draw_band(draw, skel)
    _draw_dumbbell(draw, skel)
    frame = canvas.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    pen = ImageDraw.Draw(frame)
    title = MOTIONS[skel.exercise_id]["name"]
    pen.text((14, 8), title, fill=TEXT, font=_font(18, bold=True))
    phase = skel.phase.replace("-", " ")
    pen.text((14, HEIGHT - 26), phase, fill=MUTED, font=_font(15))
    note = GIF_NOTES.get(skel.exercise_id)
    if note:
        pen.text((158, HEIGHT - 26), note, fill=MUTED, font=_font(13))
    if skel.props.get("court"):
        _draw_court(pen, skel.props.get("corner"))
    return frame


def _draw_court(pen, corner):
    left, top, right, bottom = 352, 36, 466, 188
    pen.rounded_rectangle([left, top, right, bottom], radius=8, fill=COURT_FILL, outline=COURT_LINE, width=2)
    pen.text((left + 8, top + 4), "net", fill=MUTED, font=_font(11))
    pen.text((left + 8, bottom - 16), "back", fill=MUTED, font=_font(11))
    spots = {
        "front-left": (0.28, 0.30),
        "front-right": (0.72, 0.30),
        "side-left": (0.28, 0.52),
        "side-right": (0.72, 0.52),
        "back-left": (0.28, 0.74),
        "back-right": (0.72, 0.74),
        "base": (0.50, 0.52),
    }
    xy = {}
    for name, (u, v) in spots.items():
        xy[name] = (left + 16 + u * (right - left - 32), top + 18 + v * (bottom - top - 36))
    base = xy["base"]
    for name, pt in xy.items():
        if name == "base":
            continue
        pen.line([base, pt], fill=(186, 196, 170), width=2)
    if corner in xy and corner != "base":
        pen.line([base, xy[corner]], fill=ACCENT, width=3)
    for name, pt in xy.items():
        r = 5 if name == corner else 3
        fill = ACCENT if name == corner else COURT_LINE
        pen.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=fill)


def contact_indices(count=FRAMES, slots=8):
    return [min(count - 1, round(i * count / slots)) for i in range(slots)]


def sheet_indices(exercise_id, frames):
    if exercise_id != "six_corner_shadow":
        return contact_indices(len(frames))
    # Phase labels flip halfway through a move. Sample the keys so the sheet
    # shows the corner and the base return, not the midpoint.
    count = len(frames)
    picks = []
    for key in MOTIONS[exercise_id]["keys"]:
        if key["t"] >= 0.999:
            continue
        index = min(range(count), key=lambda i: abs(i / count - key["t"]))
        if not picks or picks[-1] != index:
            picks.append(index)
    return picks


def loop_delays_ms(count: int, duration_s: float) -> list[int]:
    """Centisecond delays that sum to duration_s. GIF stores delay in 10 ms units."""
    target = int(round(duration_s * 100)) * 10
    if target <= 0 or count <= 0:
        raise ValueError(f"bad loop length {duration_s}s over {count} frames")
    base = (target // count // 10) * 10
    if base < 20:
        raise ValueError(f"frame delay {base}ms is too short for {count} frames")
    extra = (target - base * count) // 10
    delays = [base] * count
    for index in range(extra):
        delays[(index * count) // extra] += 10
    if sum(delays) != target:
        raise ValueError(f"delay sum {sum(delays)} != {target}")
    return delays


def save_gif(frames, path: Path, delays: list[int], shared=True) -> int:
    if shared:
        base = frames[0].quantize(colors=48, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        images = [base]
        for frame in frames[1:]:
            images.append(frame.quantize(palette=base, dither=Image.Dither.NONE))
    else:
        images = [frame.quantize(colors=48, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=delays,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return path.stat().st_size


def render_exercise(exercise_id: str, shared=True) -> dict:
    frames = frames_for(exercise_id)
    issues = validate_sequence(frames)
    if issues:
        raise RuntimeError(f"{exercise_id} failed validation: {issues[:4]}")
    drawn = [draw_figure(frame) for frame in frames]
    GIF_DIR.mkdir(exist_ok=True)
    SHEET_DIR.mkdir(exist_ok=True)
    gif_path = GIF_DIR / f"{exercise_id}.gif"
    from coach.exercises.library import load_library

    duration_s = load_library()[exercise_id].duration_s
    size = save_gif(drawn, gif_path, loop_delays_ms(len(drawn), duration_s), shared=shared)
    picks = [drawn[i] for i in sheet_indices(exercise_id, frames)]
    sheet = Image.new("RGB", (WIDTH * len(picks), HEIGHT), BG)
    for index, frame in enumerate(picks):
        sheet.paste(frame, (index * WIDTH, 0))
    sheet_path = SHEET_DIR / f"{exercise_id}.png"
    sheet.save(sheet_path, optimize=True)
    return {
        "id": exercise_id,
        "bytes": size,
        "seam": seam_error(frames),
        "jump": max_joint_jump(frames),
        "gif": gif_path,
        "sheet": sheet_path,
    }


def render_all(ids=None, shared=True) -> list[dict]:
    rows = []
    for exercise_id in ids or MOTIONS:
        row = render_exercise(exercise_id, shared=shared)
        rows.append(row)
        print(f"{exercise_id}\t{row['bytes']}\tseam={row['seam']:.2f}\tjump={row['jump']:.1f}")
    return rows


def main(argv):
    if "--check" in argv:
        ids = [item for item in argv if not item.startswith("--")]
        sys.exit(check(ids or None))
    ids = [item for item in argv if not item.startswith("--")]
    render_all(ids or None, shared="--flat" not in argv)


if __name__ == "__main__":
    main(sys.argv[1:])
