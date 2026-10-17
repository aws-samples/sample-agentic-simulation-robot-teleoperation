"""Geometry and unit conversions that do not need the simulator.

Everything here is plain numpy so it can be unit-tested on a laptop. The server wraps these with
tensors from Isaac Lab; the agent uses them to reason about poses it receives over the wire.

Conventions
-----------
* Quaternions are ``(w, x, y, z)``, the Isaac Lab / USD convention.
* Positions are metres in the world frame. The robot base sits at the world origin with the
  workshop's 90° yaw, so "+x" is straight ahead of the robot (towards the vials at x≈0.23).
* Joint vectors follow :data:`simteleop_common.protocol.JOINT_NAMES`.

Object dimensions come from the workshop USDs (``Vial_opaque.usda`` and ``Vial_rack_simple.usda``
extents, metres):

* vial: cylinder Ø 34 mm, 116 mm long along its local +z (−0.017 … 0.0995); the vials spawn lying on
  their side (rotated 90° about y), so their axis is horizontal.
* rack: 120 × 120 mm footprint, body top at local z = 0.073, four slot entries ``top_01..04`` at
  local z = 0.10.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .protocol import JOINT_NAMES, USD_JOINT_RANGE_DEG

VIAL_RADIUS_M = 0.017
VIAL_LENGTH_M = 0.1165
#: Along the vial's local z, from its origin: the body spans −0.017 … 0.0995, so the centre of the
#: cylinder is ~0.041 m up the axis from the prim origin.
VIAL_AXIS_CENTRE_OFFSET_M = 0.041
RACK_FOOTPRINT_M = 0.12
RACK_TOP_Z_M = 0.073
RACK_SLOT_ENTRY_Z_M = 0.10

# --------------------------------------------------------------------------- units


def rad_to_lerobot(q_rad: Sequence[float]) -> list[float]:
    """Radians → LeRobot normalised units (−100..100 for the arm, 0..100 for the gripper).

    The exact inverse of the workshop's ``get_mapped_actions_vectorized``, so agent-recorded actions
    are on the same scale as leader-arm recordings.
    """
    out = []
    for name, q in zip(JOINT_NAMES, q_rad):
        lo, hi = USD_JOINT_RANGE_DEG[name]
        n = (math.degrees(q) - lo) / (hi - lo)  # 0..1
        out.append(n * 100.0 if name == "Jaw" else n * 200.0 - 100.0)
    return out


def lerobot_to_rad(v: Sequence[float]) -> list[float]:
    out = []
    for name, x in zip(JOINT_NAMES, v):
        lo, hi = USD_JOINT_RANGE_DEG[name]
        n = x / 100.0 if name == "Jaw" else (x + 100.0) / 200.0
        out.append(math.radians(lo + n * (hi - lo)))
    return out


def joint_limits_rad() -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([math.radians(USD_JOINT_RANGE_DEG[n][0]) for n in JOINT_NAMES])
    hi = np.array([math.radians(USD_JOINT_RANGE_DEG[n][1]) for n in JOINT_NAMES])
    return lo, hi


def clamp_joints(q: Sequence[float]) -> np.ndarray:
    lo, hi = joint_limits_rad()
    return np.clip(np.asarray(q, dtype=float), lo, hi)


# --------------------------------------------------------------------------- quaternions


def quat_normalize(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("zero quaternion")
    return q / n


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return quat_normalize([w, x, y, z])


def quat_mul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_slerp(q0: Sequence[float], q1: Sequence[float], t: float) -> np.ndarray:
    """Spherical interpolation, shortest arc, t in [0, 1]."""
    a = quat_normalize(q0)
    b = quat_normalize(q1)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        return quat_normalize(a + t * (b - a))
    th = math.acos(d)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def quat_rotate(q: Sequence[float], v: Sequence[float]) -> np.ndarray:
    return quat_to_matrix(q) @ np.asarray(v, dtype=float)


def quat_from_axis_angle(axis: Sequence[float], angle_rad: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    s = math.sin(angle_rad / 2)
    return np.array([math.cos(angle_rad / 2), *(a * s)])


def compose(p_a: Sequence[float], q_a: Sequence[float], p_b: Sequence[float], q_b: Sequence[float]):
    """Pose ``b`` expressed in frame ``a`` → world: ``T_a ∘ T_b``."""
    p = np.asarray(p_a, dtype=float) + quat_rotate(q_a, p_b)
    return p, quat_mul(q_a, q_b)


def angle_between(u: Sequence[float], v: Sequence[float]) -> float:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.acos(max(-1.0, min(1.0, c)))


# --------------------------------------------------------------------------- scene geometry


def vial_axis_world(q_vial: Sequence[float]) -> np.ndarray:
    """Unit vector of the vial's cylinder axis (local +z) in the world."""
    return quat_rotate(q_vial, [0.0, 0.0, 1.0])


def vial_centre_world(p_vial: Sequence[float], q_vial: Sequence[float]) -> np.ndarray:
    """Centre of the vial's cylinder (the prim origin is near one end)."""
    return np.asarray(p_vial, dtype=float) + VIAL_AXIS_CENTRE_OFFSET_M * vial_axis_world(q_vial)


#: cos(15°). The workshop's ``vial_placed_on_rack`` accepts |axis·z| > 0.7 (45°), which passed vials
#: leaning on the rim of the hole at 33-37° (measured 2026-10-07, five "successful" episodes, all with
#: the centre 3.5 cm above a seated vial's). A vial standing in a slot measures < 2° from vertical.
VERTICAL_COS_MIN = 0.966
#: A vial counts as seated when its LOWER END is at least this far below the slot entry: the hole is
#: 10 cm deep (a vial on the floor has its bottom at entry - 0.10) and 4 cm of wall around a 3.4 cm
#: vial in a 4.5 cm hole limits its tilt to ~15 deg. A vial leaning on the rim has its bottom 5-6 cm
#: down too, so the tilt test is what rejects it; one lying across the rack has it at the entry.
SEATED_BELOW_ENTRY_M = 0.04


def vial_tilt_deg(q_vial: Sequence[float]) -> float:
    """Angle between the vial's axis and the vertical, 0 = standing, 90 = lying."""
    return math.degrees(math.acos(min(1.0, abs(float(vial_axis_world(q_vial)[2])))))


def vial_is_vertical(q_vial: Sequence[float], threshold: float = VERTICAL_COS_MIN) -> bool:
    """Standing within 15° of vertical (stricter than the workshop's 45°, see VERTICAL_COS_MIN)."""
    return abs(float(vial_axis_world(q_vial)[2])) > threshold


def vial_bottom_world(centre_world: Sequence[float], q_vial: Sequence[float]) -> np.ndarray:
    """The lower end of the vial's cylinder."""
    a = vial_axis_world(q_vial)
    if a[2] > 0:
        a = -a
    return np.asarray(centre_world, dtype=float) + (VIAL_LENGTH_M / 2) * a


def vial_is_seated(centre_world: Sequence[float], q_vial: Sequence[float], p_rack: Sequence[float], q_rack: Sequence[float]) -> bool:
    """The vial's lower end is inside the rack footprint and at least SEATED_BELOW_ENTRY_M below the
    slot entry: it is down in a hole, not resting on the rim or lying across the rack."""
    return point_in_rack(vial_bottom_world(centre_world, q_vial), p_rack, q_rack, z_max=RACK_SLOT_ENTRY_Z_M - SEATED_BELOW_ENTRY_M)


def rack_slots_world(
    p_rack: Sequence[float], q_rack: Sequence[float], slots_local: Iterable[Sequence[float]]
) -> list[np.ndarray]:
    """World positions of the rack's slot entries given their rack-local positions."""
    return [compose(p_rack, q_rack, s, [1, 0, 0, 0])[0] for s in slots_local]


def point_in_rack(
    p: Sequence[float], p_rack: Sequence[float], q_rack: Sequence[float], z_max: float = RACK_SLOT_ENTRY_Z_M
) -> bool:
    """Workshop's placement test: rack-local x,y within the footprint and z below the slot entry."""
    r = quat_to_matrix(q_rack)
    local = r.T @ (np.asarray(p, dtype=float) - np.asarray(p_rack, dtype=float))
    return bool(0.0 <= local[0] <= RACK_FOOTPRINT_M and 0.0 <= local[1] <= RACK_FOOTPRINT_M and local[2] <= z_max)


def nearest_free_slot(
    slots_world: Sequence[np.ndarray], vial_positions: Sequence[Sequence[float]], radius_m: float = 0.02
) -> int | None:
    """Index of the first slot with no vial centre within ``radius_m`` horizontally, else None."""
    for i, s in enumerate(slots_world):
        occupied = any(np.linalg.norm(np.asarray(v[:2]) - s[:2]) < radius_m for v in vial_positions)
        if not occupied:
            return i
    return None


# --------------------------------------------------------------------------- tool orientation
#
# Gripper frame of the workshop SO-101 asset, measured from its USD joints (2026-09-29):
#   * tool axis: -z (the fixed finger reaches z = -0.104; Wrist_Roll turns the gripper about z);
#   * jaw pivot axis: y (the Jaw joint frame is rotated 90 deg about x, so its axis is the gripper's
#     -y); the jaw therefore swings in the gripper's x-z plane and the fingers CLOSE ALONG x;
#   * the bulky side (servo + wrist camera mount) extends 8 cm along +y.
# To grasp a cylinder the pivot (y) must be parallel to the cylinder axis; to hold it upright the
# pivot must be vertical. The arm has 5 DOF: the tool axis can only pitch within the vertical plane
# through the base, and roll about itself freely -- so the pivot direction is realised as the
# projection of the wanted direction onto the plane perpendicular to the tool axis.


def tool_orientation(approach_dir: Sequence[float], radial_dir: Sequence[float], tool_axis: str = "-z", pivot_pref: Sequence[float] | None = None) -> np.ndarray:
    """Quaternion of a gripper frame whose tool axis points along ``approach_dir`` and whose jaw
    pivot axis (gripper y) is as parallel as possible to ``pivot_pref``.

    ``radial_dir`` (base → target, horizontal) is the default pivot preference: the bulky +y side of
    the gripper then points away from the base. ``tool_axis`` names which gripper-local axis the
    fingers point along ("-z" on the workshop asset).
    """
    a = np.asarray(approach_dir, dtype=float)
    a = a / np.linalg.norm(a)
    r = np.asarray(radial_dir, dtype=float).copy()
    r[2] = 0.0
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    r = r / np.linalg.norm(r)
    pref = np.asarray(pivot_pref if pivot_pref is not None else r, dtype=float)
    y = pref - np.dot(pref, a) * a  # component of the preference perpendicular to the tool axis
    if np.linalg.norm(y) < 1e-6:  # preference parallel to the approach: fall back to the radial
        y = r - np.dot(r, a) * a
        if np.linalg.norm(y) < 1e-6:
            y = np.cross(a, [0.0, 0.0, 1.0])
    y = y / np.linalg.norm(y)
    sign = -1.0 if tool_axis.startswith("-") else 1.0
    axis = tool_axis.strip("+-")
    if axis != "z":  # other assets: build a generic right-handed frame with the tool axis as given
        cols = {"x": None, "y": None, "z": None}
        cols[axis] = sign * a
        rem = [k for k in ("x", "y", "z") if k != axis]
        cols[rem[0]] = y
        cols[rem[1]] = np.cross(cols[axis], y) if rem == ["y", "z"] or rem == ["x", "z"] else np.cross(y, cols[axis])
        m = np.column_stack([cols["x"], cols["y"], cols["z"]])
        if np.linalg.det(m) < 0:
            cols[rem[1]] = -cols[rem[1]]
            m = np.column_stack([cols["x"], cols["y"], cols["z"]])
        return matrix_to_quat(m)
    z = sign * a
    x = np.cross(y, z)
    return matrix_to_quat(np.column_stack([x, y, z]))


def top_down_grasp_pose(p_vial: Sequence[float], q_vial: Sequence[float], hover_m: float, tool_axis: str = "-z", tilt_deg: float = 0.0, along_m: float = 0.0):
    """TCP pose above a lying vial: tool axis down over the cylinder centre (shifted ``along_m``
    towards the cap end, local +z), optionally tilted back towards the robot base by ``tilt_deg``
    (the only tilt a 5-DOF arm can realise), jaw pivot along the vial's axis so the fingers close
    across the cylinder, bulky side away from the base."""
    c = vial_centre_world(p_vial, q_vial) + along_m * vial_axis_world(q_vial)
    target = c + np.array([0.0, 0.0, hover_m])
    radial = np.array([c[0], c[1], 0.0])
    radial = radial / (np.linalg.norm(radial) or 1.0)
    t = math.radians(tilt_deg)
    approach = np.array([0.0, 0.0, -math.cos(t)]) + math.sin(t) * radial  # down, leaning away from the base
    axis = vial_axis_world(q_vial)
    if np.dot(axis, radial) < 0:
        axis = -axis  # +y (bulk) away from the base
    q = tool_orientation(approach, c, tool_axis, pivot_pref=axis)
    return target, q


def quat_from_two_vectors(u: Sequence[float], v: Sequence[float]) -> np.ndarray:
    """Minimal rotation taking direction ``u`` onto direction ``v``."""
    a = np.asarray(u, dtype=float)
    b = np.asarray(v, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d > 1 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if d < -1 + 1e-9:  # opposite: rotate 180 deg about any perpendicular axis
        axis = np.cross(a, [1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.cross(a, [0.0, 1.0, 0.0])
        return quat_from_axis_angle(axis, math.pi)
    axis = np.cross(a, b)
    return quat_from_axis_angle(axis, math.acos(max(-1.0, min(1.0, d))))


def upright_hold_orientation(
    q_gripper_now: Sequence[float],
    held_axis_world: Sequence[float],
    down: bool = True,
    prefer_tool_dir: Sequence[float] | None = None,
    tool_axis: str = "-z",
    keep_end: bool = False,
) -> np.ndarray:
    """Gripper orientation that turns a held object's axis vertical by a MINIMAL rotation from the
    current pose.

    After a top-down grasp the object's axis and the tool axis both lie (nearly) in the vertical
    plane through the base, so the minimal rotation is a pure pitch -- exactly what a 5-DOF arm can
    do. Constructing the target orientation any other way (e.g. choosing which way the tool should
    lean) can demand the mirror configuration, which needs a 180 deg roll; measured 2026-09-29: the
    IK ended with 180 deg orientation error and the wrist at its limit.

    The object is symmetric, so there are two candidates: turn one end up, or the other. Without
    ``prefer_tool_dir`` the end already nearer the goal is used (the shorter turn). That choice is
    arbitrary for a lying object (its vertical component is ~0 and its sign is noise) and picked the
    unreachable branch in 2 of 8 trials (tool pointing back-down, Wrist_Pitch at its limit; measured
    2026-10-07). With ``prefer_tool_dir`` the candidate whose resulting tool axis points most along
    that direction wins -- pass the direction the arm can realise (radial-down, wrist above the TCP).
    With ``keep_end`` the object is treated as oriented: its +axis goes to the goal (``down=False``
    puts the +axis, the vial's cap, up) and there is only one candidate.
    """
    a = np.asarray(held_axis_world, dtype=float)
    a = a / np.linalg.norm(a)
    goal = np.array([0.0, 0.0, -1.0 if down else 1.0])
    cands = []
    for end in (a, -a):
        delta = quat_from_two_vectors(end, goal)
        cands.append(quat_normalize(quat_mul(delta, q_gripper_now)))
    if keep_end:
        return cands[0]
    if prefer_tool_dir is None:
        return cands[0] if np.dot(a, goal) >= 0 else cands[1]
    sign = -1.0 if tool_axis.startswith("-") else 1.0
    local = np.array({"x": [1.0, 0, 0], "y": [0, 1.0, 0], "z": [0, 0, 1.0]}[tool_axis.strip("+-")]) * sign
    pref = np.asarray(prefer_tool_dir, dtype=float)
    return max(cands, key=lambda q: float(np.dot(quat_rotate(q, local), pref)))


def slot_place_pose(slot_world: Sequence[float], clearance_m: float, tool_axis: str = "-z"):
    """TCP pose that holds a vial upright above a slot.

    The vial was grasped at its centre with its axis along the jaw pivot (gripper y). Pointing the
    tool horizontally away from the base and the pivot straight up stands the vial vertical; the TCP
    is then ``VIAL_LENGTH/2 + clearance`` above the slot entry.
    """
    s = np.asarray(slot_world, dtype=float)
    target = s + np.array([0.0, 0.0, VIAL_LENGTH_M / 2 + clearance_m])
    radial = np.array([s[0], s[1], 0.0])
    q = tool_orientation(radial, radial, tool_axis, pivot_pref=[0.0, 0.0, 1.0])
    return target, q
