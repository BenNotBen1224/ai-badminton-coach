"""2D figure model for the exercise GIFs.

A pose is joint angles plus a root. Forward kinematics and a two-link leg
solver turn that into joint positions. Planted feet stay on fixed floor
anchors. Bone lengths come from the segment table, so they do not stretch.

Joint limits follow typical active range in Neumann, Kinesiology of the
Musculoskeletal System: knee flexion 0-140 (0 is straight, negative is
hyperextension), elbow 0-150, hip flexion 0-120 and extension to about 30,
shoulder flexion 0-180 and extension to about 60, ankle dorsiflexion to 20
and plantarflexion to 50, trunk bend well under a lumbar flexion of 60.
Numbers below are 2D and a few degrees wider so a normal deep squat passes,
but a backward knee does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

WIDTH = 480
HEIGHT = 360
PX_PER_CM = 1.30
FLOOR_Y = 318.0
ORIGIN_X = 240.0

THIGH = 41.0
SHANK = 39.0
UPPER_ARM = 29.0
FOREARM = 25.0
HAND = 8.0
TORSO = 48.0
NECK = 8.0
HEAD_R = 11.0
HIP_W = 9.0
SHOULDER_W = 18.0
FOOT = 20.0
ANKLE_DROP = 7.0
ANKLE_BACK = 15.0  # ankle is this far behind the toe when the foot is flat

# Drawn segment lengths that must stay within 0.5%.
SEGMENTS = (
    ("pelvis", "ribs", TORSO * 0.55),
    ("ribs", "shoulder_mid", TORSO * 0.45),
    ("shoulder_mid", "neck", NECK),
    ("l_hip", "l_knee", THIGH),
    ("l_knee", "l_ankle", SHANK),
    ("r_hip", "r_knee", THIGH),
    ("r_knee", "r_ankle", SHANK),
    ("l_shoulder", "l_elbow", UPPER_ARM),
    ("l_elbow", "l_wrist", FOREARM),
    ("l_wrist", "l_hand", HAND),
    ("r_shoulder", "r_elbow", UPPER_ARM),
    ("r_elbow", "r_wrist", FOREARM),
    ("r_wrist", "r_hand", HAND),
)

# name -> (low, high) in degrees. Knee and elbow: 0 straight, negative = hyperextension.
ANGLE_LIMITS = {
    "knee": (-0.5, 142.0),
    "elbow": (-8.0, 152.0),
    "hip": (-32.0, 128.0),
    # Front view cannot see flexion. This is abduction: about 45 in Neumann, a few
    # degrees wider so a wide squat passes. A split still fails.
    "hip_abduction": (-12.0, 50.0),
    "shoulder": (-55.0, 185.0),
    "ankle": (-52.0, 28.0),
    "spine": (-28.0, 48.0),
}

# A quick balance check, not a force model. COM may sit this far outside the feet.
SUPPORT_TOLERANCE_CM = 6.0
SLIDE_LIMIT_PX = 1.0
SEAM_LIMIT_PX = 2.0
FLOOR_EPS_CM = 0.05


@dataclass
class Pt:
    x: float
    y: float
    z: float = 0.0

    def __iter__(self):
        yield self.x
        yield self.y


def add(a: Pt, b: Pt) -> Pt:
    return Pt(a.x + b.x, a.y + b.y)


def sub(a: Pt, b: Pt) -> Pt:
    return Pt(a.x - b.x, a.y - b.y)


def mul(a: Pt, s: float) -> Pt:
    return Pt(a.x * s, a.y * s)


def dot(a: Pt, b: Pt) -> float:
    return a.x * b.x + a.y * b.y


def cross(a: Pt, b: Pt) -> float:
    return a.x * b.y - a.y * b.x


def norm(a: Pt) -> float:
    return math.hypot(a.x, a.y)


def unit(a: Pt) -> Pt:
    d = norm(a)
    if d < 1e-8:
        return Pt(1.0, 0.0)
    return Pt(a.x / d, a.y / d)


def dist(a: Pt, b: Pt) -> float:
    return norm(sub(b, a))


def dist3(a: Pt, b: Pt) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _line_distance(point: Pt, a: Pt, b: Pt) -> float:
    dx, dy = b.x - a.x, b.y - a.y
    span = math.hypot(dx, dy) or 1.0
    return abs((point.x - a.x) * dy - (point.y - a.y) * dx) / span


# A far planted foot is a trail leg. Draw it straight instead of bowing the knee.
LONG_LEG_DX_CM = 22.0
KNEE_OUTSIDE_CM = 8.0


def _knee_at_x(hip: Pt, ankle: Pt, kx: float) -> Pt | None:
    """Knee at a chosen frontal x, with the leftover length in depth."""
    dxh = kx - hip.x
    dxa = kx - ankle.x
    denom = 2.0 * (ankle.y - hip.y)
    if abs(denom) < 1e-3:
        return None
    numer = (
        THIGH * THIGH - SHANK * SHANK
        - dxh * dxh + dxa * dxa
        + (ankle.y - hip.y) * (hip.y + ankle.y)
    )
    ky = numer / denom
    if not (ankle.y + 1.5 < ky < hip.y - 1.5):
        return None
    z2 = THIGH * THIGH - dxh * dxh - (ky - hip.y) ** 2
    if z2 < -0.05:
        return None
    knee = Pt(kx, ky, math.sqrt(max(0.0, z2)))
    if abs(dist3(hip, knee) - THIGH) > 0.3 or abs(dist3(knee, ankle) - SHANK) > 0.3:
        return None
    return knee


def _elbow_on_line(shoulder: Pt, wrist: Pt) -> Pt | None:
    """A nearly straight reach. Leftover length goes into depth so the arm does not kink."""
    dx = wrist.x - shoulder.x
    dy = wrist.y - shoulder.y
    reach2 = dx * dx + dy * dy
    reach = math.sqrt(reach2)
    if reach < 8.0 or reach > UPPER_ARM + FOREARM - 0.15:
        return None
    t = ((UPPER_ARM * UPPER_ARM - FOREARM * FOREARM) / reach2 + 1.0) / 2.0
    if not (0.2 <= t <= 0.8):
        return None
    z2 = UPPER_ARM * UPPER_ARM - (t * reach) * (t * reach)
    if z2 < -0.05:
        return None
    return Pt(shoulder.x + dx * t, shoulder.y + dy * t, math.sqrt(max(0.0, z2)))


def _elbow_straight3(shoulder: Pt, wrist: Pt) -> Pt | None:
    """Elbow on the 3D shoulder-wrist line when that reach is a straight arm."""
    dx = wrist.x - shoulder.x
    dy = wrist.y - shoulder.y
    dz = wrist.z - shoulder.z
    reach = math.sqrt(dx * dx + dy * dy + dz * dz)
    full = UPPER_ARM + FOREARM
    if abs(reach - full) > 0.6:
        return None
    t = UPPER_ARM / reach
    return Pt(shoulder.x + dx * t, shoulder.y + dy * t, shoulder.z + dz * t)


def _knee_on_line(hip: Pt, ankle: Pt) -> Pt:
    """Project a straight leg. Extra bone length goes into unseen depth."""
    dx = ankle.x - hip.x
    dy = ankle.y - hip.y
    reach2 = dx * dx + dy * dy
    if reach2 < 4.0:
        return _intersect(hip, THIGH, ankle, SHANK, Pt(0.0, 1.0))
    t = ( (THIGH * THIGH - SHANK * SHANK) / reach2 + 1.0 ) / 2.0
    if not (0.12 <= t <= 0.88):
        return _intersect(hip, THIGH, ankle, SHANK, Pt(dx, 1.0))
    planar = t * math.sqrt(reach2)
    z2 = THIGH * THIGH - planar * planar
    if z2 < -0.05:
        return _intersect(hip, THIGH, ankle, SHANK, Pt(dx, 1.0))
    return Pt(hip.x + dx * t, hip.y + dy * t, math.sqrt(max(0.0, z2)))


def _leg_should_look_straight(hip: Pt, ankle: Pt, leg: dict) -> bool:
    dx = abs(ankle.x - hip.x)
    dy = hip.y - ankle.y
    reach = math.hypot(dx, dy)
    full = THIGH + SHANK
    planted = bool(leg.get("support", True)) and float(leg.get("air", 0.0)) < 0.4
    if planted and dx >= LONG_LEG_DX_CM and reach >= 0.85 * full:
        return True
    if dy >= 70.0 and dx <= 16.0 and reach >= 0.90 * full:
        return True
    return False


def _front_knee(hip: Pt, ankle: Pt, sign: float, leg: dict) -> Pt:
    """Front view hides knee flexion in depth. Do not spend that length as a sideways bow."""
    if _leg_should_look_straight(hip, ankle, leg):
        return _knee_on_line(hip, ankle)
    best: tuple[float, Pt] | None = None
    candidates = [ankle.x + sign * off for off in (4.0, 2.5, 1.0, 0.0, -2.0)]
    for frac in (0.62, 0.50, 0.38):
        candidates.append(hip.x + (ankle.x - hip.x) * frac)
    for kx in candidates:
        outside = sign * (kx - ankle.x)
        if outside > KNEE_OUTSIDE_CM or outside < 0.8:
            continue
        knee = _knee_at_x(hip, ankle, kx)
        if knee is None:
            continue
        drop = hip.y - knee.y
        if drop < 1.0:
            continue
        abduction = math.degrees(math.atan2((knee.x - hip.x) * sign, drop))
        if abduction > 46.0 or abduction < -8.0:
            continue
        # Low knee reads as a bend. Stay near the foot.
        score = knee.y + abs(outside - 2.5) * 0.35
        if best is None or score < best[0]:
            best = (score, knee)
    if best is not None:
        return best[1]
    return _knee_on_line(hip, ankle)


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def smoother(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def to_px(
    pt: Pt,
    camera_x: float = 0.0,
    zoom: float = 1.0,
    floor_y: float | None = None,
    azimuth: float = 0.0,
) -> tuple[float, float]:
    """Project to the screen. Azimuth 0 is a front view and ignores depth."""
    fy = FLOOR_Y if floor_y is None else floor_y
    scale = PX_PER_CM * zoom
    az = math.radians(azimuth)
    x = pt.x * math.cos(az) + pt.z * math.sin(az)
    return (
        ORIGIN_X + (x - camera_x) * scale,
        fy - pt.y * scale,
    )


def yaw_y(pt: Pt, ox: float, oz: float, deg: float) -> Pt:
    """Rotate about the vertical axis through (ox, oz).

    Positive yaw turns a forward point (negative z) toward +x.
    """
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    x = pt.x - ox
    z = pt.z - oz
    return Pt(ox + x * c - z * s, pt.y, oz + x * s + z * c)


def chest_forward(yaw: float) -> Pt:
    """Unit chest-forward in world space. Yaw 0 faces the camera, along -z."""
    a = math.radians(yaw)
    return Pt(math.sin(a), 0.0, -math.cos(a))


def _intersect(p1: Pt, r1: float, p2: Pt, r2: float, prefer: Pt) -> Pt:
    d = dist(p1, p2)
    reach = r1 + r2
    gap = abs(r1 - r2)
    if d > reach + 0.4 or d < gap - 0.4:
        raise ValueError(f"unreachable {d:.1f}cm not in [{gap:.1f}, {reach:.1f}]")
    d = min(max(d, gap + 1e-4), reach - 1e-4)
    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h = math.sqrt(max(0.0, r1 * r1 - a * a))
    direction = unit(sub(p2, p1))
    mid = add(p1, mul(direction, a))
    perp = Pt(-direction.y, direction.x)
    k1 = add(mid, mul(perp, h))
    k2 = add(mid, mul(perp, -h))
    if dot(sub(k1, mid), prefer) >= dot(sub(k2, mid), prefer):
        return k1
    return k2


def _rotate_cw(p: Pt, deg: float) -> Pt:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return Pt(p.x * c + p.y * s, -p.x * s + p.y * c)


def place_foot(toe_x: float, lift: float, air: float, view: str = "side") -> dict[str, Pt]:
    """Side view: foot points +x and `lift` raises the heel around the toe.

    Front view: `toe_x` is the foot center. The shoe is a rigid shape under the ankle.
    """
    if view == "front":
        ground = air
        return {
            "toe": Pt(toe_x + 4.0, ground),
            "heel": Pt(toe_x - 4.0, ground),
            "ankle": Pt(toe_x, ground + ANKLE_DROP),
        }
    toe = Pt(toe_x, air)
    heel = add(toe, _rotate_cw(Pt(-FOOT, 0.0), lift))
    ankle = add(toe, _rotate_cw(Pt(-ANKLE_BACK, ANKLE_DROP), lift))
    return {"toe": toe, "heel": heel, "ankle": ankle}


def _arm_fk(shoulder: Pt, shoulder_deg: float, elbow_deg: float, forward: float) -> tuple[Pt, Pt, Pt]:
    """0 degrees hangs down. Positive shoulder angle swings toward `forward` (+1 = +x)."""
    s = math.radians(shoulder_deg)
    direction = Pt(forward * math.sin(s), -math.cos(s))
    elbow = add(shoulder, mul(direction, UPPER_ARM))
    # Positive elbow flexion folds the forearm toward +y from a hanging arm (a curl),
    # which is a CCW rotation when forward is +x.
    bend = math.radians(elbow_deg * forward)
    c, sn = math.cos(bend), math.sin(bend)
    folded = Pt(direction.x * c - direction.y * sn, direction.x * sn + direction.y * c)
    wrist = add(elbow, mul(folded, FOREARM))
    hand = add(wrist, mul(unit(folded), HAND))
    return elbow, wrist, hand


def _goalpost_arm(shoulder: Pt, abduction: float, elbow_flex: float, outward: float) -> tuple[Pt, Pt, Pt]:
    """Front-view arm. Abduction 0 is down, 90 is out to the side, 180 is up."""
    a = math.radians(abduction)
    upper = Pt(outward * math.sin(a), -math.cos(a))
    elbow = add(shoulder, mul(upper, UPPER_ARM))
    f = math.radians(abduction + elbow_flex)
    fore = Pt(outward * math.sin(f), -math.cos(f))
    wrist = add(elbow, mul(unit(fore), FOREARM))
    hand = add(wrist, mul(unit(fore), HAND))
    return elbow, wrist, hand


@dataclass
class Skeleton:
    exercise_id: str
    phase: str
    view: str
    points: dict[str, Pt]
    planted: frozenset[str]
    support: frozenset[str]
    airborne: bool
    camera_x: float
    zoom: float
    floor_y: float
    knee_prefer: dict[str, Pt]
    arm_forward: dict[str, float]
    elbow_prefer: dict[str, Pt]
    accent: tuple[str, ...] = ()
    accent_side: str = "both"
    props: dict = field(default_factory=dict)
    rope_t: float = 0.0
    yaw: float = 0.0
    azimuth: float = 0.0


def _solve_limbs(key: dict, flight: float) -> tuple[dict[str, Pt], dict[str, Pt], dict[str, float]]:
    view = key["view"]
    pelvis = Pt(*key["pelvis"])
    trunk = math.radians(key["trunk"])
    chest = Pt(math.sin(trunk), math.cos(trunk))
    shoulder_mid = add(pelvis, mul(chest, TORSO))
    ribs = add(pelvis, mul(chest, TORSO * 0.55))
    if key.get("head") == "follow":
        neck = add(shoulder_mid, mul(chest, NECK))
        head = add(neck, mul(chest, HEAD_R + 1.0))
    else:
        neck = Pt(shoulder_mid.x, shoulder_mid.y + NECK)
        head = Pt(neck.x, neck.y + HEAD_R + 1.0)

    points: dict[str, Pt] = {
        "pelvis": pelvis,
        "ribs": ribs,
        "shoulder_mid": shoulder_mid,
        "neck": neck,
        "head": head,
    }
    knee_prefer: dict[str, Pt] = {}
    arm_forward: dict[str, float] = {}

    for side, sign in (("l", -1.0), ("r", 1.0)):
        leg = key["legs"][side]
        foot = place_foot(leg["toe"], leg["lift"], leg["air"], view)
        hip = pelvis if view == "side" else Pt(pelvis.x + sign * HIP_W, pelvis.y)
        prefer = Pt(*leg["knee"])
        if view == "front":
            knee = _front_knee(hip, foot["ankle"], sign, leg)
        else:
            knee = _intersect(hip, THIGH, foot["ankle"], SHANK, prefer)
        points[f"{side}_hip"] = hip
        points[f"{side}_knee"] = knee
        points[f"{side}_ankle"] = foot["ankle"]
        points[f"{side}_heel"] = foot["heel"]
        points[f"{side}_toe"] = foot["toe"]
        knee_prefer[side] = prefer

        arm = key["arms"][side]
        if view == "side":
            shoulder = Pt(shoulder_mid.x, shoulder_mid.y)
            forward = 1.0
        else:
            outward = sign
            shoulder = Pt(shoulder_mid.x + outward * SHOULDER_W * key.get("shoulder_scale", 1.0), shoulder_mid.y)
            shoulder = Pt(shoulder.x + key.get("twist_shift", 0.0), shoulder.y)
            forward = outward
        if arm.get("wrist"):
            raw = arm["wrist"]
            wrist = Pt(raw[0], raw[1], raw[2] if len(raw) > 2 else 0.0)
            elbow = _elbow_straight3(shoulder, wrist) if arm.get("long") else None
            if elbow is None and arm.get("long"):
                elbow = _elbow_on_line(shoulder, wrist)
            if elbow is None:
                bend_prefer = Pt(forward, 0.4)
                elbow = _intersect(shoulder, UPPER_ARM, wrist, FOREARM, bend_prefer)
            dx = wrist.x - elbow.x
            dy = wrist.y - elbow.y
            dz = wrist.z - elbow.z
            span = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
            hand = Pt(
                wrist.x + dx / span * HAND,
                wrist.y + dy / span * HAND,
                wrist.z + dz / span * HAND,
            )
        elif arm.get("style") == "goalpost":
            elbow, wrist, hand = _goalpost_arm(shoulder, arm["shoulder"], arm["elbow"], forward)
        else:
            swing = forward if view == "front" else 1.0
            elbow, wrist, hand = _arm_fk(shoulder, arm["shoulder"], arm["elbow"], swing)
        points[f"{side}_shoulder"] = shoulder
        points[f"{side}_elbow"] = elbow
        points[f"{side}_wrist"] = wrist
        points[f"{side}_hand"] = hand
        arm_forward[side] = forward if view == "front" else 1.0

    if flight:
        for name, pt in list(points.items()):
            points[name] = Pt(pt.x, pt.y + flight, pt.z)
    return points, knee_prefer, arm_forward


def planted_contacts(key: dict, flight: float) -> frozenset[str]:
    if flight > 0.4:
        return frozenset()
    names = []
    for side in ("l", "r"):
        leg = key["legs"][side]
        if leg["air"] > 0.4:
            continue
        names.append(f"{side}_toe")
        if leg["lift"] < 1.0:
            names.append(f"{side}_heel")
    for side in ("l", "r"):
        if key["arms"][side].get("planted"):
            names.append(f"{side}_wrist")
    return frozenset(names)


def support_contacts(key: dict, flight: float) -> frozenset[str]:
    """Weight-bearing contacts. A light wall touch is planted but not support."""
    if flight > 0.4:
        return frozenset()
    names = []
    for side in ("l", "r"):
        leg = key["legs"][side]
        if leg["air"] > 0.4 or not leg.get("support", True):
            continue
        names.append(f"{side}_toe")
        if leg["lift"] < 1.0:
            names.append(f"{side}_heel")
    for side in ("l", "r"):
        arm = key["arms"][side]
        if arm.get("support"):
            names.append(f"{side}_wrist")
    return frozenset(names)


def _solve_plank(key: dict) -> tuple[dict[str, Pt], dict[str, Pt], dict[str, float], set[str]]:
    """Straight forearm plank or side plank. Toe and elbow stay on the floor."""
    toe_x = float(key["toe"])
    foot = place_foot(toe_x, float(key.get("lift", 58.0)), 0.0, "side")
    ankle = foot["ankle"]
    shoulder_y = UPPER_ARM + 1.2
    # Almost straight. A 3 cm shortfall bows the knee about 11 cm off the body line.
    leg_reach = THIGH + SHANK - 0.15
    chain = leg_reach + TORSO
    dy = shoulder_y - ankle.y
    dx = math.sqrt(max(1.0, chain * chain - dy * dy))
    direction = unit(Pt(dx, dy))
    hip = add(ankle, mul(direction, leg_reach))
    shoulder_mid = add(hip, mul(direction, TORSO))
    ribs = add(hip, mul(direction, TORSO * 0.55))
    neck = add(shoulder_mid, mul(direction, NECK))
    head = add(neck, mul(direction, HEAD_R + 0.6))
    knee = _intersect(hip, THIGH, ankle, SHANK, Pt(0.0, 1.0))
    points: dict[str, Pt] = {
        "pelvis": hip,
        "ribs": ribs,
        "shoulder_mid": shoulder_mid,
        "neck": neck,
        "head": head,
    }
    for side in ("l", "r"):
        points[f"{side}_hip"] = hip
        points[f"{side}_knee"] = knee
        points[f"{side}_ankle"] = ankle
        points[f"{side}_heel"] = foot["heel"]
        points[f"{side}_toe"] = foot["toe"]
    elbow = Pt(shoulder_mid.x, shoulder_mid.y - UPPER_ARM)
    wrist = Pt(elbow.x + FOREARM, elbow.y)
    hand = Pt(wrist.x + HAND, wrist.y)
    if key.get("side_plank"):
        points["r_shoulder"] = shoulder_mid
        points["r_elbow"] = elbow
        points["r_wrist"] = wrist
        points["r_hand"] = hand
        top = _arm_fk(shoulder_mid, float(key.get("top_arm", 172.0)), 8.0, 1.0)
        points["l_shoulder"] = shoulder_mid
        points["l_elbow"], points["l_wrist"], points["l_hand"] = top
        planted_arms = {"r"}
    else:
        for side in ("l", "r"):
            points[f"{side}_shoulder"] = shoulder_mid
            points[f"{side}_elbow"] = elbow
            points[f"{side}_wrist"] = wrist
            points[f"{side}_hand"] = hand
        planted_arms = {"l", "r"}
    return points, {"l": Pt(0.0, 1.0), "r": Pt(0.0, 1.0)}, {"l": 1.0, "r": 1.0}, planted_arms


def _elbow_side(points: dict[str, Pt], side: str, forward: float) -> Pt:
    """Solved elbow offset. A later flip across the shoulder-wrist line is hyperextension."""
    offset = sub(points[f"{side}_elbow"], points[f"{side}_shoulder"])
    if norm(offset) < 0.5:
        return Pt(forward, 0.4)
    return offset


def chest_yaw(skel: Skeleton) -> float:
    """Measured shoulder-line yaw in degrees. 0 faces the camera. Not the pose flag."""
    left = skel.points["l_shoulder"]
    right = skel.points["r_shoulder"]
    return math.degrees(math.atan2(right.z - left.z, right.x - left.x))


_YAW_POINTS = (
    "ribs", "shoulder_mid", "neck", "head",
    "l_shoulder", "l_elbow", "l_wrist", "l_hand",
    "r_shoulder", "r_elbow", "r_wrist", "r_hand",
)


def _apply_chest_yaw(points: dict[str, Pt], yaw: float, origin: Pt) -> None:
    if abs(yaw) < 0.05:
        return
    for name in _YAW_POINTS:
        points[name] = yaw_y(points[name], origin.x, origin.z, yaw)


def solve_key(exercise_id: str, key: dict, rope_t: float = 0.0) -> Skeleton:
    flight = float(key.get("flight") or 0.0)
    yaw = float(key.get("yaw") or 0.0)
    azimuth = float(key.get("azimuth") or 0.0)
    if key.get("kind") == "plank":
        points, knee_prefer, arm_forward, planted_arms = _solve_plank(key)
        planted = {f"{side}_toe" for side in ("l", "r")}
        planted.update(f"{side}_wrist" for side in planted_arms)
        support = frozenset(planted)
        airborne = False
    else:
        points, knee_prefer, arm_forward = _solve_limbs(key, flight)
        _apply_chest_yaw(points, yaw, points["pelvis"])
        if key.get("props", {}).get("shared_grip"):
            grip = points["l_wrist"]
            # Both hands occupy one grip, 8 cm along chest-forward.
            # Do not leave them at the pre-yaw wrist depth.
            forward = chest_forward(yaw)
            hand = Pt(grip.x + forward.x * HAND, grip.y + forward.y * HAND, grip.z + forward.z * HAND)
            points["l_hand"] = hand
            points["r_hand"] = hand
            points["r_wrist"] = Pt(grip.x, grip.y, grip.z)
        planted = set(planted_contacts(key, flight))
        support = set(support_contacts(key, flight))
        airborne = flight > 0.4 or not support
        for name in key.get("_unplant") or ():
            planted.discard(name)
    return Skeleton(
        exercise_id=exercise_id,
        phase=key["phase"],
        view=key.get("view", "side"),
        points=points,
        planted=frozenset(planted),
        support=frozenset(support),
        airborne=airborne,
        camera_x=float(key.get("camera", 0.0)),
        zoom=float(key.get("zoom", 1.0)),
        floor_y=float(key.get("floor_y", FLOOR_Y)),
        knee_prefer=knee_prefer,
        arm_forward=arm_forward,
        elbow_prefer={side: _elbow_side(points, side, arm_forward[side]) for side in ("l", "r")},
        accent=tuple(key.get("accent", ())),
        accent_side=key.get("accent_side", "both"),
        props=dict(key.get("props", {})),
        rope_t=rope_t,
        yaw=yaw,
        azimuth=azimuth,
    )


def _angle_at(a: Pt, b: Pt, c: Pt) -> float:
    u = unit(sub(a, b))
    v = unit(sub(c, b))
    cos = max(-1.0, min(1.0, dot(u, v)))
    return math.degrees(math.acos(cos))


def _signed_from_down(origin: Pt, reference_up: Pt, target: Pt) -> float:
    down = unit(mul(reference_up, -1.0))
    arm = unit(sub(target, origin))
    return math.degrees(math.atan2(cross(down, arm), dot(down, arm)))


def joint_angles(skel: Skeleton) -> dict[str, float]:
    p = skel.points
    out: dict[str, float] = {}
    trunk = unit(sub(p["shoulder_mid"], p["pelvis"]))
    out["spine"] = 180.0 - _angle_at(p["pelvis"], p["ribs"], p["shoulder_mid"])
    if cross(sub(p["shoulder_mid"], p["pelvis"]), sub(p["ribs"], p["pelvis"])) < 0:
        out["spine"] = -out["spine"]
    for side in ("l", "r"):
        hip, knee, ankle = p[f"{side}_hip"], p[f"{side}_knee"], p[f"{side}_ankle"]
        heel, toe = p[f"{side}_heel"], p[f"{side}_toe"]
        flex = 180.0 - _angle_at(hip, knee, ankle)
        line = unit(sub(ankle, hip))
        rel = sub(knee, hip)
        prefer = skel.knee_prefer[side]
        if cross(line, prefer) * cross(line, rel) < 0 and flex > 2.0:
            flex = -flex
        out[f"{side}_knee"] = flex
        hip_ang = _signed_from_down(hip, trunk, knee)
        if skel.view == "front":
            outward = -1.0 if side == "l" else 1.0
            hip_ang *= outward
        out[f"{side}_hip"] = hip_ang
        shoulder = p[f"{side}_shoulder"]
        elbow = p[f"{side}_elbow"]
        wrist = p[f"{side}_wrist"]
        # A legal forward reach, once yawed, looks like extension in world x/y.
        # Measure the arm in the chest frame, where the reach was posed.
        if abs(skel.yaw) > 0.05:
            ox, oz = p["pelvis"].x, p["pelvis"].z
            shoulder = yaw_y(shoulder, ox, oz, -skel.yaw)
            elbow = yaw_y(elbow, ox, oz, -skel.yaw)
            wrist = yaw_y(wrist, ox, oz, -skel.yaw)
        eflex = 180.0 - _angle_at(shoulder, elbow, wrist)
        # A fixed forward axis mislabels a normal curl. Hyperextension is the elbow
        # flipped across the shoulder-wrist line from the side the solver chose.
        line = unit(sub(wrist, shoulder))
        prefer = skel.elbow_prefer[side]
        if cross(line, prefer) * cross(line, sub(elbow, shoulder)) < -1e-4 and eflex > 8.0:
            eflex = -eflex
        out[f"{side}_elbow"] = eflex
        sflex = _signed_from_down(shoulder, trunk, elbow)
        forward = skel.arm_forward[side]
        if skel.view == "front" and forward < 0:
            sflex = -sflex
        # Side plank: the free arm points at the ceiling. With a horizontal trunk
        # that 2D angle is signed as extension. It is abduction, so flip the sign.
        if skel.exercise_id == "side_plank" and elbow.y > shoulder.y + 4.0 and sflex < 0.0:
            sflex = -sflex
        out[f"{side}_shoulder"] = sflex
        foot_dir = unit(sub(toe, heel))
        shank_dir = unit(sub(knee, ankle))
        interior = math.degrees(math.acos(max(-1.0, min(1.0, dot(shank_dir, foot_dir)))))
        out[f"{side}_ankle"] = 90.0 - interior
    return out


def com(skel: Skeleton) -> Pt:
    p = skel.points
    parts = [
        ("head", 0.081, p["head"]),
        ("torso", 0.497, add(p["pelvis"], mul(sub(p["shoulder_mid"], p["pelvis"]), 0.45))),
    ]
    for side, mass_arm, mass_fore, mass_hand, mass_thigh, mass_shank, mass_foot in (
        ("l", 0.028, 0.016, 0.006, 0.100, 0.047, 0.014),
        ("r", 0.028, 0.016, 0.006, 0.100, 0.047, 0.014),
    ):
        parts.append((side, mass_arm, add(p[f"{side}_shoulder"], mul(sub(p[f"{side}_elbow"], p[f"{side}_shoulder"]), 0.5))))
        parts.append((side, mass_fore, add(p[f"{side}_elbow"], mul(sub(p[f"{side}_wrist"], p[f"{side}_elbow"]), 0.5))))
        parts.append((side, mass_hand, p[f"{side}_hand"]))
        parts.append((side, mass_thigh, add(p[f"{side}_hip"], mul(sub(p[f"{side}_knee"], p[f"{side}_hip"]), 0.45))))
        parts.append((side, mass_shank, add(p[f"{side}_knee"], mul(sub(p[f"{side}_ankle"], p[f"{side}_knee"]), 0.45))))
        parts.append((side, mass_foot, add(p[f"{side}_heel"], mul(sub(p[f"{side}_toe"], p[f"{side}_heel"]), 0.45))))
    mass = sum(m for _, m, _ in parts)
    return Pt(sum(m * pt.x for _, m, pt in parts) / mass, sum(m * pt.y for _, m, pt in parts) / mass)


def support_interval(skel: Skeleton) -> tuple[float, float] | None:
    xs = [skel.points[name].x for name in skel.support if name in skel.points]
    xs.extend(float(x) for x in (skel.props.get("support_x") or ()))
    if not xs:
        return None
    lo, hi = min(xs), max(xs)
    if hi - lo < 10.0:
        mid = (lo + hi) / 2.0
        return mid - 6.0, mid + 6.0
    return lo, hi


def _limit_hit(name: str, value: float) -> str | None:
    low, high = ANGLE_LIMITS[name]
    if value < low or value > high:
        return f"{name} {value:.1f} outside {low:.0f}..{high:.0f}"
    return None


def _front_leg_issues(skel: Skeleton) -> list[str]:
    """A planar angle check can pass a knee that is drawn far outside its foot."""
    if skel.view != "front":
        return []
    issues = []
    for side, sign in (("l", -1.0), ("r", 1.0)):
        knee = skel.points[f"{side}_knee"]
        ankle = skel.points[f"{side}_ankle"]
        hip = skel.points[f"{side}_hip"]
        outside = sign * (knee.x - ankle.x)
        if outside > KNEE_OUTSIDE_CM:
            issues.append(f"{side} knee {outside:.1f}cm outside the foot")
        dx = abs(ankle.x - hip.x)
        if f"{side}_toe" in skel.planted and dx >= LONG_LEG_DX_CM and _line_distance(knee, hip, ankle) > 4.0:
            issues.append(f"{side} long leg stays bent")
    return issues


def validate_frame(skel: Skeleton) -> list[str]:
    issues = []
    for a, b, length in SEGMENTS:
        actual = dist3(skel.points[a], skel.points[b])
        if length <= 0:
            continue
        err = abs(actual - length) / length
        if err > 0.005:
            issues.append(f"bone {a}-{b} off by {err:.2%} ({actual:.2f} vs {length:.2f})")
    angles = joint_angles(skel)
    for side in ("l", "r"):
        for joint in ("knee", "elbow", "hip", "shoulder", "ankle"):
            limit_name = "hip_abduction" if joint == "hip" and skel.view == "front" else joint
            hit = _limit_hit(limit_name, angles[f"{side}_{joint}"])
            if hit:
                issues.append(f"{side} {hit}")
    hit = _limit_hit("spine", angles["spine"])
    if hit:
        issues.append(hit)
    issues.extend(_front_leg_issues(skel))
    for name, pt in skel.points.items():
        if pt.y < -FLOOR_EPS_CM:
            issues.append(f"{name} below floor by {-pt.y:.2f}cm")
    if not skel.airborne:
        span = support_interval(skel)
        if span is not None:
            cx = com(skel).x
            if cx < span[0] - SUPPORT_TOLERANCE_CM or cx > span[1] + SUPPORT_TOLERANCE_CM:
                issues.append(
                    f"com x {cx:.1f} outside support {span[0]:.1f}..{span[1]:.1f}"
                )
    for name, pt in skel.points.items():
        if name in ("ribs",):
            continue
        x, y = to_px(pt, skel.camera_x, skel.zoom, skel.floor_y, skel.azimuth)
        if not (12 <= x <= WIDTH - 12 and 30 <= y <= HEIGHT - 8):
            issues.append(f"{name} off screen at ({x:.0f}, {y:.0f})")
    return issues


def _contact_px(skel: Skeleton, name: str) -> tuple[float, float]:
    return to_px(skel.points[name], skel.camera_x, skel.zoom, skel.floor_y, skel.azimuth)


_FOOT_PAIRS = (
    ("l_ankle", "l_toe"), ("l_ankle", "l_heel"), ("l_heel", "l_toe"),
    ("r_ankle", "r_toe"), ("r_ankle", "r_heel"), ("r_heel", "r_toe"),
)


def validate_sequence(frames: list[Skeleton]) -> list[str]:
    issues = []
    if len(frames) < 2:
        return ["need at least 2 frames"]
    for index, skel in enumerate(frames):
        for item in validate_frame(skel):
            issues.append(f"f{index} {skel.phase}: {item}")
    for index in range(len(frames)):
        a = frames[index]
        b = frames[(index + 1) % len(frames)]
        shared = a.planted & b.planted
        for name in sorted(shared):
            ax, ay = _contact_px(a, name)
            bx, by = _contact_px(b, name)
            delta = math.hypot(ax - bx, ay - by)
            if delta > SLIDE_LIMIT_PX:
                issues.append(f"planted {name} slides {delta:.2f}px from f{index}")
    base = frames[0]
    for pair in _FOOT_PAIRS:
        expected = dist(base.points[pair[0]], base.points[pair[1]])
        if expected < 1.0:
            continue
        for index, skel in enumerate(frames):
            actual = dist(skel.points[pair[0]], skel.points[pair[1]])
            err = abs(actual - expected) / expected
            if err > 0.005:
                issues.append(f"f{index} foot {pair[0]}-{pair[1]} stretched {err:.2%}")
    seam = 0.0
    for name in frames[0].points:
        if name == "ribs":
            continue
        ax, ay = _contact_px(frames[0], name)
        bx, by = _contact_px(frames[-1], name)
        seam = max(seam, math.hypot(ax - bx, ay - by))
    if seam > SEAM_LIMIT_PX:
        issues.append(f"seam {seam:.2f}px")
    return issues


def max_joint_jump(frames: list[Skeleton]) -> float:
    worst = 0.0
    for index in range(len(frames) - 1):
        for name in frames[index].points:
            ax, ay = _contact_px(frames[index], name)
            bx, by = _contact_px(frames[index + 1], name)
            worst = max(worst, math.hypot(ax - bx, ay - by))
    return worst


def seam_error(frames: list[Skeleton]) -> float:
    worst = 0.0
    for name in frames[0].points:
        ax, ay = _contact_px(frames[0], name)
        bx, by = _contact_px(frames[-1], name)
        worst = max(worst, math.hypot(ax - bx, ay - by))
    return worst
