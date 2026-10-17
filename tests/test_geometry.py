import math

import numpy as np
import pytest

from simteleop_common import geometry as g
from simteleop_common.protocol import JOINT_NAMES


def test_lerobot_roundtrip_and_scale():
    q = [-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465]
    v = g.rad_to_lerobot(q)
    assert np.allclose(g.lerobot_to_rad(v), q)
    # Arm joints live in -100..100, gripper in 0..100 (LeRobot convention).
    assert all(-100 <= x <= 100 for x in v[:5])
    assert 0 <= v[5] <= 100
    # Mid-range of each arm joint maps to 0.
    mid = [math.radians(sum(g.USD_JOINT_RANGE_DEG[n]) / 2) for n in JOINT_NAMES]
    assert np.allclose(g.rad_to_lerobot(mid)[:5], 0.0)


def test_quaternion_helpers_consistent():
    q = g.quat_normalize([0.7, 0.1, -0.2, 0.3])
    assert np.allclose(g.matrix_to_quat(g.quat_to_matrix(q)), q)
    r = g.quat_from_axis_angle([0, 0, 1], math.pi / 2)
    assert np.allclose(g.quat_rotate(r, [1, 0, 0]), [0, 1, 0], atol=1e-9)
    assert np.allclose(g.quat_rotate(g.quat_mul(r, r), [1, 0, 0]), [-1, 0, 0], atol=1e-9)


def test_vertical_is_strict_and_seated_needs_depth():
    # 10 deg off vertical stands; 20 deg does not (the workshop accepted up to 45 deg).
    q10 = g.quat_from_axis_angle([1, 0, 0], math.radians(10))
    q20 = g.quat_from_axis_angle([1, 0, 0], math.radians(20))
    assert g.vial_is_vertical(q10) and not g.vial_is_vertical(q20)
    assert np.isclose(g.vial_tilt_deg(q20), 20)
    # Rack at the origin, unrotated. Standing on the hole floor (bottom 10 cm below the entry): seated.
    rack_p, rack_q = [0, 0, 0], [1, 0, 0, 0]
    up = [1, 0, 0, 0]
    floor_centre = [0.03, 0.03, g.RACK_SLOT_ENTRY_Z_M - 0.10 + g.VIAL_LENGTH_M / 2]
    assert g.vial_is_seated(floor_centre, up, rack_p, rack_q)
    # Vertical but only 2.5 cm into the hole (jammed): not seated. 6 cm in: seated.
    assert not g.vial_is_seated([0.03, 0.03, g.RACK_SLOT_ENTRY_Z_M - 0.025 + g.VIAL_LENGTH_M / 2], up, rack_p, rack_q)
    assert g.vial_is_seated([0.03, 0.03, g.RACK_SLOT_ENTRY_Z_M - 0.06 + g.VIAL_LENGTH_M / 2], up, rack_p, rack_q)
    # Leaning on the rim at 35 deg with the centre just below the entry (the 2026-09-29 "successes"):
    # the bottom is deep enough, so it is the tilt that must reject it.
    lean = g.quat_from_axis_angle([1, 0, 0], math.radians(35))
    lean_centre = [0.03, 0.03, g.RACK_SLOT_ENTRY_Z_M - 0.007]
    assert g.vial_is_seated(lean_centre, lean, rack_p, rack_q) and not g.vial_is_vertical(lean)
    # Lying across the rack top: bottom at the entry height, not seated.
    lying = g.quat_from_axis_angle([0, 1, 0], math.pi / 2)
    assert not g.vial_is_seated([0.03, 0.03, g.RACK_SLOT_ENTRY_Z_M + 0.017], lying, rack_p, rack_q)


def test_vial_lying_and_vertical():
    lying = g.quat_from_axis_angle([0, 1, 0], math.pi / 2)  # the workshop spawn rotation
    assert not g.vial_is_vertical(lying)
    assert np.allclose(g.vial_axis_world(lying), [1, 0, 0], atol=1e-9)
    assert g.vial_is_vertical([1, 0, 0, 0])
    c = g.vial_centre_world([0.23, -0.08, 0.05], lying)
    assert np.allclose(c, [0.23 + g.VIAL_AXIS_CENTRE_OFFSET_M, -0.08, 0.05])


def test_rack_slots_and_membership():
    p_rack, q_rack = [0.18, 0.08, 0.06], g.quat_from_axis_angle([0, 0, 1], 0.3)
    slots_local = [[0.03, 0.03, 0.1], [0.09, 0.09, 0.1]]
    slots = g.rack_slots_world(p_rack, q_rack, slots_local)
    assert len(slots) == 2
    # A slot entry itself is at z_max: inside; 2 cm above is outside.
    assert g.point_in_rack(slots[0], p_rack, q_rack)
    assert not g.point_in_rack(slots[0] + np.array([0, 0, 0.02]), p_rack, q_rack)
    assert not g.point_in_rack([0.0, 0.0, 0.05], p_rack, q_rack)
    assert g.nearest_free_slot(slots, [slots[0]]) == 1
    assert g.nearest_free_slot(slots, [slots[0], slots[1]]) is None


def test_tool_orientation_axes():
    q = g.tool_orientation([0, 0, -1], [0.2, -0.1, 0], "-z", pivot_pref=[1, 0, 0])
    m = g.quat_to_matrix(q)
    assert np.isclose(np.linalg.det(m), 1.0)
    assert np.allclose(-m[:, 2], [0, 0, -1], atol=1e-9)  # tool axis (-z) points down
    assert np.allclose(m[:, 1], [1, 0, 0], atol=1e-9)  # jaw pivot (y) along the preference
    assert np.allclose(m[:, 0], np.cross(m[:, 1], m[:, 2]), atol=1e-9)  # right-handed


def test_grasp_pose_aligns_pivot_with_vial_and_bulk_away_from_base():
    lying = g.quat_from_axis_angle([0, 1, 0], math.pi / 2)  # axis -> +x
    t, q = g.top_down_grasp_pose([0.23, -0.08, 0.05], lying, 0.08, tilt_deg=20)
    m = g.quat_to_matrix(q)
    tool = -m[:, 2]
    assert np.isclose(t[2], 0.13)
    assert tool[2] < -0.9 and tool[0] > 0  # down, leaning away from the base
    assert abs(np.dot(m[:, 1], [1, 0, 0])) > 0.93  # pivot within ~20 deg of the vial axis
    assert np.dot(m[:, 1], [0.23, -0.08, 0]) > 0  # bulk (+y) away from the base
    assert abs(np.dot(m[:, 0], [0, 1, 0])) > 0.9  # fingers close across the vial (along y)


def test_place_pose_holds_vial_vertical():
    t2, q2 = g.slot_place_pose([0.2, 0.1, 0.16], 0.02)
    m = g.quat_to_matrix(q2)
    tool = -m[:, 2]
    assert np.isclose(t2[2], 0.16 + g.VIAL_LENGTH_M / 2 + 0.02)
    assert np.isclose(tool[2], 0.0, atol=1e-9) and np.dot(tool[:2], [0.2, 0.1]) > 0  # horizontal, away
    assert np.allclose(m[:, 1], [0, 0, 1], atol=1e-9)  # pivot (vial axis) vertical


def test_clamp_joints():
    lo, hi = g.joint_limits_rad()
    q = g.clamp_joints([10, -10, 0, 0, 0, 10])
    assert np.allclose(q[[0, 1, 5]], [hi[0], lo[1], hi[5]])


def _grasp_pose():
    """Gripper after the 35 deg tilted top-down grasp: tool forward-down 35 deg, pivot forward-up."""
    return g.tool_orientation([math.sin(math.radians(35)), 0, -math.cos(math.radians(35))], [1, 0, 0], "-z", pivot_pref=[1, 0, 0])


#: The vial in the gripper frame after that grasp: 31 deg off the jaw pivot towards the fingertips
#: (measured [0.04, 0.85, -0.52] on 2026-09-29 and 2026-10-07). The vial stays horizontal on the mat;
#: it is the tilted gripper that is off.
VIAL_IN_GRIPPER = np.array([0.0, math.cos(math.radians(31)), -math.sin(math.radians(31))])
FORWARD_DOWN = [math.cos(math.radians(35)), 0, -math.sin(math.radians(35))]


def test_upright_hold_orientation_is_a_pure_pitch_and_stands_the_vial_up():
    q_now = _grasp_pose()
    axis_w = g.quat_rotate(q_now, VIAL_IN_GRIPPER)
    assert abs(axis_w[2]) < 0.08  # horizontal, as grasped from the mat
    for end in (axis_w, -axis_w):  # the sign of a ~0 vertical component is noise
        q = g.upright_hold_orientation(q_now, end, down=True, prefer_tool_dir=FORWARD_DOWN)
        m = g.quat_to_matrix(q)
        assert np.isclose(abs((m @ VIAL_IN_GRIPPER)[2]), 1.0, atol=1e-9)  # vial vertical
        tool = -m[:, 2]
        assert abs(tool[1]) < 1e-9  # tool stays in the radial plane: a pure pitch
        # Unrolled, the two branches are forward-up (+31 deg) and back-down; the preference picks forward.
        assert tool[0] > 0 and np.isclose(math.degrees(math.atan2(tool[2], tool[0])), 31, atol=1e-6)
    # Without a preference the choice follows the sign of the vertical component. For a vial lying
    # flat that sign is noise, and one of the two gives the back-down tool the arm cannot reach.
    flat = np.array([axis_w[0], axis_w[1], 0.0])
    tools = [(-g.quat_to_matrix(g.upright_hold_orientation(q_now, flat + [0, 0, e], down=False))[:, 2])[0] for e in (1e-4, -1e-4)]
    assert min(tools) < 0 < max(tools)


def test_keep_end_puts_the_cap_up_and_grasp_along_shifts_towards_the_cap():
    q_now = _grasp_pose()
    # Vial hanging cap towards the gripper body (after the off-centre grasp): cap-up needs tool-down.
    body_lean = np.array([0.0, math.cos(math.radians(20)), math.sin(math.radians(20))])
    q = g.upright_hold_orientation(q_now, g.quat_rotate(q_now, body_lean), down=False, keep_end=True)
    m = g.quat_to_matrix(q)
    assert np.allclose(m @ body_lean, [0, 0, 1], atol=1e-9)  # the CAP end is up, not just some end
    assert (-m[:, 2])[2] < 0  # and the tool points down
    # Cap towards the fingertips (centre grasp): cap-up forces tool-up.
    q2 = g.upright_hold_orientation(q_now, g.quat_rotate(q_now, VIAL_IN_GRIPPER), down=False, keep_end=True)
    assert (-g.quat_to_matrix(q2)[:, 2])[2] > 0
    lying = g.quat_from_axis_angle([0, 1, 0], math.pi / 2)  # cap (+z local) along +x
    t0, _ = g.top_down_grasp_pose([0.23, -0.08, 0.05], lying, 0.0)
    t1, _ = g.top_down_grasp_pose([0.23, -0.08, 0.05], lying, 0.0, along_m=0.025)
    assert np.allclose(t1 - t0, [0.025, 0, 0], atol=1e-9)


def test_rolled_hold_hangs_the_vial_with_the_tool_forward_down():
    # A 180 deg roll about the tool axis mirrors the vial's 31 deg offset: it then hangs 24 deg from
    # vertical and the stand-up needs only a small pitch, ending with the tool forward-DOWN -- the
    # wrist above the fingertips, which the arm can hold over the near slots.
    q_now = _grasp_pose()
    q_rolled = g.quat_normalize(g.quat_mul(q_now, g.quat_from_axis_angle([0, 0, 1], math.pi)))
    axis_w = g.quat_rotate(q_rolled, VIAL_IN_GRIPPER)
    assert 20 < math.degrees(math.acos(abs(axis_w[2]))) < 30
    q = g.upright_hold_orientation(q_rolled, axis_w, down=True, prefer_tool_dir=FORWARD_DOWN)
    m = g.quat_to_matrix(q)
    assert np.isclose(abs((m @ VIAL_IN_GRIPPER)[2]), 1.0, atol=1e-9)
    tool = -m[:, 2]
    assert abs(tool[1]) < 1e-9 and tool[0] > 0 and tool[2] < 0
    assert -40 < math.degrees(math.atan2(tool[2], tool[0])) < -25
    # The rotation applied is just the vial's remaining angle from vertical.
    d = g.quat_mul(q, [q_rolled[0], -q_rolled[1], -q_rolled[2], -q_rolled[3]])
    assert 20 < math.degrees(2 * math.acos(abs(d[0]))) < 30


def test_quat_from_two_vectors():
    q = g.quat_from_two_vectors([1, 0, 0], [0, 0, -1])
    assert np.allclose(g.quat_rotate(q, [1, 0, 0]), [0, 0, -1], atol=1e-9)
    assert np.allclose(g.quat_from_two_vectors([0, 1, 0], [0, 2, 0]), [1, 0, 0, 0])


def test_slerp_endpoints_and_midpoint():
    q0 = np.array([1.0, 0, 0, 0])
    q1 = g.quat_from_axis_angle([0, 0, 1], math.pi / 2)
    assert np.allclose(g.quat_slerp(q0, q1, 0), q0)
    assert np.allclose(g.quat_slerp(q0, q1, 1), q1)
    mid = g.quat_slerp(q0, q1, 0.5)
    assert np.allclose(g.quat_rotate(mid, [1, 0, 0]), [math.cos(math.pi / 4), math.sin(math.pi / 4), 0])
    # Shortest arc: the negated target gives the same path.
    assert np.allclose(np.abs(g.quat_slerp(q0, -q1, 0.5)), np.abs(mid))
