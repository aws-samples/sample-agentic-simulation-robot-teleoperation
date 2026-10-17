"""Task primitives for the vials-to-rack scene, with verdicts measured from ground truth.

These are the "verified skills" the agent is allowed to call. Each returns a dict with a boolean
``success`` decided from the simulator's object poses and contact sensor after the motion, plus the
measurements it was decided on. The agent can retry, choose another vial or slot, or give up — it
cannot loosen the criteria.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from simteleop_common import geometry as geo

from . import motion as mo
from .scene_state import TOOL_AXIS, SceneReader

# The workshop's initial joint state: arm folded back, jaw slightly closed.
HOME_RAD = [-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465]
# A raised, forward-looking pose from which the wrist camera sees the mat; used before a pick.
READY_RAD = [0.0, -0.3, 0.6, 1.1, 0.0, mo.JAW_OPEN_RAD]

HOVER_M = 0.08
#: Approach tilted from vertical towards the robot base: a pure top-down pose at the far vials needs
#: the wrist at its limit (measured 2026-09-29: 5.7 cm short at x=0.31 with Wrist_Pitch at 94 deg).
GRASP_TILT_DEG = float(__import__("os").environ.get("SIMTELEOP_GRASP_TILT_DEG", "35"))  # 15/30 deg fell 4-5 cm short at z=0.05; 45 reached
#: TCP height above the vial centre at the grasp. The fixed finger reaches 1.4 cm past the TCP, so
#: at 0 the finger is in the mat (measured: descend stalls at TCP z=0.049); at +0.015 the fingertip is
#: at the cylinder centre height.
GRASP_DEPTH_M = 0.015
#: 12 cm put the far vials (x~0.28) at the edge of reach, where the IK let the tool rotate and the vial
#: slid out (rose 5-9 cm of 12). 8 cm clears the mat; the place primitive rises further on its own.
LIFT_M = 0.08
MIN_LIFT_FOR_SUCCESS_M = 0.05
#: Pinch the vial this far from its centre towards the cap (0: at the centre). Measured 2026-10-07:
#: an off-centre pinch does NOT make the vial swing to vertical -- the jaw's friction holds it at
#: whatever angle it was grasped (tilt in hand 88-90 deg after the lift, 8/8).
GRASP_ALONG_M = float(__import__("os").environ.get("SIMTELEOP_GRASP_ALONG_M", "0.0"))
#: After closing, with the vial still on the mat, the gripper is rotated about the pinch point from
#: the forward grasp tilt (35 deg, needed for reach) down to this forward tilt. The mat keeps the vial
#: horizontal while the jaw turns around it (measured 2026-10-07: in-hand offset 31 -> 3 deg), so the
#: vial ends up nearly along the jaw pivot and standing it cap-up needs the tool only slightly UP --
#: a posture the arm holds anywhere over the rack (P+E+W ~ 0; from the 35 deg grasp cap-up needed the
#: tool up 31 deg and Wrist_Pitch pinned at -95 deg in 5/8 places). Turning further, to a backward
#: tilt, ran the arm into Wrist_Pitch +95 and the lift stalled; 0-5 deg is the arm's limit at r=0.25.
REGRIP_TILT_DEG = float(__import__("os").environ.get("SIMTELEOP_REGRIP_TILT_DEG", "12"))


def home(reader: SceneReader) -> mo.Motion:
    r = yield from mo.move_joints(reader, HOME_RAD, duration_s=2.5)
    return {"success": r["reached"], **r}


def ready(reader: SceneReader) -> mo.Motion:
    r = yield from mo.move_joints(reader, READY_RAD, duration_s=2.0)
    return {"success": r["reached"], **r}


def pick_vial(reader: SceneReader, ik: mo.IKSolver, vial: str, hover_m: float = HOVER_M, lift_m: float = LIFT_M, depth_m: float | None = None, tilt_deg: float | None = None, along_m: float | None = None) -> mo.Motion:
    """Top-down grasp of a lying vial ``along_m`` towards the cap from its centre, then lift and let
    it settle in the jaw.

    Success: the jaw stalled on the vial (contact sensor sees it) AND the vial's centre rose at least
    ``MIN_LIFT_FOR_SUCCESS_M`` AND it is still in contact after the lift.
    """
    if vial not in reader.vials:
        return {"success": False, "reason": f"unknown vial {vial!r}; known: {sorted(reader.vials)}"}
    v0 = reader.object_state(vial)
    if v0.placed:
        return {"success": False, "reason": f"{vial} is already standing in the rack"}
    # A vial lying ON the rack (a failed place) is picked like any other; only a standing one is refused.
    z0 = v0.centre[2]
    depth = GRASP_DEPTH_M if depth_m is None else depth_m
    tilt = GRASP_TILT_DEG if tilt_deg is None else tilt_deg
    along = GRASP_ALONG_M if along_m is None else along_m
    log: dict[str, Any] = {"vial": vial, "start_centre": v0.centre, "depth_m": depth, "tilt_deg": tilt, "along_m": along}

    # 1. open, 2. hover above the centre, 3. descend, 4. close, 5. lift.
    log["open"] = yield from mo.gripper(reader, mo.JAW_OPEN_RAD)
    hover_p, hover_q = geo.top_down_grasp_pose(v0.position, v0.quaternion, hover_m, TOOL_AXIS, tilt_deg=tilt, along_m=along)
    log["hover"] = yield from mo.move_ee(reader, ik, hover_p, hover_q, timeout_s=5.0)
    if log["hover"]["position_error_m"] > 0.04:  # a waypoint; the far vials (x≈0.31) end 2-3 cm short with the wrist at its limit
        return {"success": False, "reason": "could not reach the hover pose", **log}
    v1 = reader.object_state(vial)  # re-read: the approach may have nudged it
    grasp_p, grasp_q = geo.top_down_grasp_pose(v1.position, v1.quaternion, depth, TOOL_AXIS, tilt_deg=tilt, along_m=along)
    log["descend"] = yield from mo.move_ee(reader, ik, grasp_p, grasp_q, timeout_s=4.0, pos_tol_m=0.004)
    log["close"] = yield from mo.gripper(reader, mo.JAW_CLOSED_RAD, duration_s=1.0, settle_s=0.6)
    if vial not in log["close"]["contact_vials"]:
        # Nothing between the fingers: open again so a retry starts clean.
        yield from mo.gripper(reader, mo.JAW_OPEN_RAD)
        return {"success": False, "reason": "jaw closed without touching the vial", **log}
    if REGRIP_TILT_DEG < tilt:
        # Turn the jaw around the vial on the mat (see REGRIP_TILT_DEG). Same TCP, new orientation.
        v1b = reader.object_state(vial)
        _, regrip_q = geo.top_down_grasp_pose(v1b.position, v1b.quaternion, depth, TOOL_AXIS, tilt_deg=REGRIP_TILT_DEG, along_m=along)
        log["regrip"] = yield from mo.move_ee(reader, ik, grasp_p, regrip_q, timeout_s=4.0, pos_tol_m=0.006, axis_tol_deg=8.0, path_speed_mps=0.05)
        yield from mo.hold(reader, int(0.3 * mo.STEP_HZ))
        v1c = reader.object_state(vial)
        _, ee_q = reader.ee_pose_w()
        log["axis_in_gripper_after_regrip"] = [round(float(x), 3) for x in geo.quat_to_matrix([float(x) for x in ee_q.detach().cpu().tolist()]).T @ np.asarray(v1c.axis_world)]
        if vial not in reader.contact_vials():
            yield from mo.gripper(reader, mo.JAW_OPEN_RAD)
            return {"success": False, "reason": "lost the vial while turning the jaw around it", **log}
        # The arm is near its reach limit in this posture (the lift stalled at 1-5 of 8 cm, 7/8); so
        # lift just clear of the mat like this, then finish the lift turning back to the comfortable
        # grasp tilt -- off the mat the vial keeps its new angle in the jaw.
        log["lift_clear"] = yield from mo.move_ee(reader, ik, np.asarray(grasp_p) + np.array([0.0, 0.0, 0.015]), regrip_q, timeout_s=2.5, pos_tol_m=0.006, axis_tol_deg=10.0, path_speed_mps=0.05)
    lift_p = np.asarray(grasp_p) + np.array([0.0, 0.0, lift_m])
    log["lift"] = yield from mo.move_ee(reader, ik, lift_p, grasp_q, timeout_s=4.0, pos_tol_m=0.01)
    yield from mo.hold(reader, int(0.5 * mo.STEP_HZ))
    v2 = reader.object_state(vial)
    _, ee_q = reader.ee_pose_w()
    log["axis_in_gripper"] = [round(float(x), 3) for x in geo.quat_to_matrix([float(x) for x in ee_q.detach().cpu().tolist()]).T @ np.asarray(v2.axis_world)]
    log["tilt_in_hand_deg"] = v2.tilt_deg
    rose = v2.centre[2] - z0
    still_held = vial in reader.contact_vials()
    # A vial that rose much less than the hand did is slipping in the fingers (measured: 6-7 cm on a
    # 12 cm lift, then lost during the carry); require it to have followed most of the lift.
    followed = rose > 0.7 * lift_m
    return {
        "success": bool(still_held and rose > MIN_LIFT_FOR_SUCCESS_M and followed),
        "reason": None if (still_held and followed) else ("vial slipped: rose %.0f of %.0f cm" % (rose * 100, lift_m * 100) if still_held else "vial lost during the lift"),
        "rose_m": float(rose),
        "still_in_contact": still_held,
        "vial_axis_world": v2.axis_world,
        **log,
    }


#: Release only when the held vial is this close to vertical and this close to the slot axis. The
#: hole is 4.5 cm for a 3.4 cm vial (5.5 mm clearance per side); a vertical vial released within 1 cm
#: slides in, a tilted one lands on the rim and stays there (measured 2026-10-07: 33-37 deg). With
#: 6 cm of the vial inside the hole at the release, the wall limits the tilt to ~10 deg anyway; the
#: gate equals the verdict's 15 deg (a 10 deg gate refused two of eight at 11 deg).
RELEASE_MAX_TILT_DEG = float(__import__("os").environ.get("SIMTELEOP_RELEASE_MAX_TILT_DEG", "15"))
RELEASE_MAX_OFFSET_M = 0.010


def _radial_pitched(tcp_xy, pitch_deg: float) -> np.ndarray:
    """Tool direction forward (away from the base) and pitched ``pitch_deg`` below the horizontal
    (negative: above)."""
    r = np.array([float(tcp_xy[0]), float(tcp_xy[1]), 0.0])
    r = r / (np.linalg.norm(r) or 1.0)
    t = math.radians(pitch_deg)
    return math.cos(t) * r + np.array([0.0, 0.0, -math.sin(t)])


def _hold_orientation(reader: SceneReader, held: str, cap_up: bool) -> tuple[np.ndarray, list[float]]:
    """Orientation that holds the vial vertical with its cap (local +z) up or down.

    Cap-up is a single rotation (the cap axis to +z); which way the tool ends up pointing follows
    from how the vial sits in the jaw -- cap leaning towards the gripper body gives tool-down (the
    posture the arm can hold over the rack), towards the fingertips gives tool-up. The off-centre
    grasp makes the vial hang cap-up, which gives the former. Cap-down (after the wrist roll) picks
    the branch whose tool points forward-down.
    """
    v = reader.object_state(held)
    tcp, ee_q = reader.tcp_pose_w()
    ee_q = [float(x) for x in ee_q]
    if cap_up:
        q = geo.upright_hold_orientation(ee_q, v.axis_world, down=False, keep_end=True)
    else:
        q = geo.upright_hold_orientation(ee_q, v.axis_world, down=True, prefer_tool_dir=_radial_pitched(tcp[:2], 35.0), tool_axis=TOOL_AXIS)
    return q, [round(float(x), 3) for x in geo.quat_to_matrix(ee_q).T @ np.asarray(v.axis_world)]


def place_in_rack(reader: SceneReader, ik: mo.IKSolver, slot: int | None = None, clearance_m: float = 0.02, cap_up: bool = True) -> mo.Motion:
    """Stand the held vial up (cap up by default), carry it over a rack slot, lower it in, release
    and retreat.

    Success: after release the vial stands in the rack -- within 15 deg of vertical AND seated (its
    lower end 4 cm or more down the hole) -- and the jaw is no longer touching it.

    Posture: how the vial sits in the jaw decides which way the tool must point to hold it vertical
    cap-up -- cap leaning towards the fingertips needs the tool UP by that angle (wrist below the
    fingers; from the 35 deg grasp that was 31 deg and Wrist_Pitch pinned at -95 over the near slots),
    cap along the pivot needs the tool horizontal. pick_vial's regrip on the mat leaves the vial 5-15
    deg off the pivot, so the hold is tool-up 5-15 deg, which the arm holds anywhere over the rack
    (6/7 standing, 2026-10-07). ``cap_up=False`` is the older cap-down hold: a 180 deg wrist roll and
    the tool forward-down; it stands the vial reliably but inverted, and the cap wedges in the hole.
    """
    held = None
    for n in reader.vials:
        st = reader.object_state(n)
        if n in reader.contact_vials() and st.centre[2] > 0.055:
            held = n
            break
    if held is None:
        return {"success": False, "reason": "no vial is held (contact + lifted)"}
    snap = reader.snapshot(0)
    if not snap.rack_slots_world:
        return {"success": False, "reason": "no rack slots known"}
    if slot is None:
        free = [i for i, f in enumerate(snap.rack_slots_free) if f]
        if not free:
            return {"success": False, "reason": "no free slot"}
        # Nearest free slot to the robot base is the easiest reach.
        slot = min(free, key=lambda i: np.linalg.norm(snap.rack_slots_world[i][:2]))
    if not (0 <= slot < len(snap.rack_slots_world)):
        return {"success": False, "reason": f"slot {slot} out of range 0..{len(snap.rack_slots_world) - 1}"}
    slot_w = np.asarray(snap.rack_slots_world[slot])
    log: dict[str, Any] = {"vial": held, "slot": slot, "slot_world": slot_w.tolist()}
    hang = geo.VIAL_LENGTH_M / 2  # the vial's lower end is half a length below the TCP when it hangs vertical

    def vial_now():
        v = reader.object_state(held)
        return v, float(np.linalg.norm(np.asarray(v.centre[:2]) - slot_w[:2]))

    def in_hand(tag: str) -> None:
        # Where the vial sits relative to the fingertips (gripper frame, metres) and its axis there:
        # the diagnostics that showed the vial sliding 4 cm along its axis during the roll (2026-10-07).
        v = reader.object_state(held)
        tcp, ee_q = reader.tcp_pose_w()
        rot = geo.quat_to_matrix([float(x) for x in ee_q]).T
        log[f"hand_{tag}"] = {"centre_rel_tcp": [round(float(x), 3) for x in rot @ (np.asarray(v.centre) - tcp)],
                             "axis": [round(float(x), 2) for x in rot @ np.asarray(v.axis_world)], "tilt_deg": v.tilt_deg}

    in_hand("start")

    if not cap_up:
        # Cap-down hold: roll the wrist 180 deg about the tool axis (joint space, within Wrist_Roll's
        # +-160 deg) with the vial lifted; it swings its ends around the fingertips, nothing else moves.
        # The cap then hangs 24 deg from vertical and a small pitch finishes it, tool forward-down.
        lo, hi = geo.joint_limits_rad()
        q_t = reader.last_targets.detach().cpu().numpy().copy()
        roll_goal = next((q_t[4] + d for d in (math.pi, -math.pi) if lo[4] <= q_t[4] + d <= hi[4]), None)
        if roll_goal is None:
            return {"success": False, "reason": f"wrist roll {math.degrees(q_t[4]):.0f} deg: no 180 deg turn within limits; still held", **log}
        q_t[4] = roll_goal
        log["roll"] = yield from mo.move_joints(reader, q_t, duration_s=4.0)
        v, _ = vial_now()
        log["axis_after_roll"] = [round(float(x), 3) for x in v.axis_world]
        in_hand("roll")
        if held not in reader.contact_vials() or v.centre[2] < 0.055:
            return {"success": False, "reason": "vial slipped out during the wrist roll", **log}

    # 1. rise in place with the vial turning the rest of the way to vertical, 2. pan to the slot's
    #    azimuth, 3. carry in-plane, 4. lower until the vial's bottom is 2 cm into the slot, 5. release
    #    only if the vial is vertical and over the slot, 6. retreat.
    log["cap_up"] = cap_up
    place_q, log["axis_in_gripper"] = _hold_orientation(reader, held, cap_up)
    log["place_tool_axis"] = [round(float(x), 2) for x in geo.quat_rotate(place_q, mo._tool_local_axis())]
    carry_p = slot_w + np.array([0.0, 0.0, hang + clearance_m + 0.02])  # vial bottom 4 cm above the slot entry
    tcp_now, _ = reader.tcp_pose_w()
    rise_p = np.array([tcp_now[0], tcp_now[1], carry_p[2]])
    log["rise"] = yield from mo.move_ee(reader, ik, rise_p, place_q, timeout_s=6.0, pos_tol_m=0.015, axis_tol_deg=10.0)
    in_hand("rise")
    # Lateral carry = pan the base to the slot's azimuth (joint space, keeps radius and height), then
    # an in-plane servo to the slot. A single IK servo across 17 cm folded the arm into a corner.
    log["pan"] = yield from mo.pan_to_azimuth(reader, slot_w[:2])
    # The pan rotated the arm's plane: re-derive the hold orientation in the new plane (the vial is
    # already vertical, so this is "keep the current orientation" there).
    place_q, _ = _hold_orientation(reader, held, cap_up)
    log["carry"] = yield from mo.move_ee(reader, ik, carry_p, place_q, timeout_s=7.0, pos_tol_m=0.01, axis_tol_deg=10.0)
    in_hand("carry")
    v, _ = vial_now()
    log["tilt_before_lower_deg"] = v.tilt_deg
    if v.tilt_deg > 2 * RELEASE_MAX_TILT_DEG:
        return {"success": False, "reason": f"vial {v.tilt_deg:.0f} deg from vertical over the slot (axis {np.round(v.axis_world, 2).tolist()}); still held", **log}
    # The vial's centre is not exactly at the TCP (the grasp is a centimetre or so off), so aim with
    # the VIAL: measure its centre from ground truth and shift the TCP by the residual, twice at most.
    aim_p = carry_p.copy()
    for k in range(2):
        v, off = vial_now()
        log[f"aim_{k}_offset_m"] = off
        if off < 0.005:
            break
        aim_p = aim_p + np.array([slot_w[0] - v.centre[0], slot_w[1] - v.centre[1], 0.0])
        log[f"aim_{k}"] = yield from mo.move_ee(reader, ik, aim_p, place_q, timeout_s=4.0, pos_tol_m=0.004, axis_tol_deg=8.0)
    # Descend in two stages. A single 7 cm descent drifted 2-3 cm sideways at this reach and the
    # vial's bottom caught the rim, which rotated the vial 20-30 deg in the jaw before the release gate
    # could see it (measured 2026-10-07, in-hand axis [0, .86, -.5] -> [0, 1, 0]). So: stop with the
    # bottom 1.5 cm above the entry, re-aim on the vial's measured centre, then insert 2.5 cm slowly
    # while aiming again. Targets are derived from the VIAL's measured centre, not the TCP geometry.
    def descend_vial_to(bottom_z: float, label: str, speed: float, tol: float) -> mo.Motion:
        nonlocal aim_p
        v, _ = vial_now()
        dz = (bottom_z + hang) - v.centre[2]
        aim_p = aim_p + np.array([0.0, 0.0, dz])
        log[label] = yield from mo.move_ee(reader, ik, aim_p, place_q, timeout_s=5.0, pos_tol_m=tol, axis_tol_deg=8.0, path_speed_mps=speed)
        for k in range(3):
            v, off = vial_now()
            log[f"{label}_aim_{k}_offset_m"] = off
            if off < 0.004:
                break
            aim_p = aim_p + np.array([slot_w[0] - v.centre[0], slot_w[1] - v.centre[1], 0.0])
            log[f"{label}_aim_{k}"] = yield from mo.move_ee(reader, ik, aim_p, place_q, timeout_s=3.0, pos_tol_m=0.003, axis_tol_deg=6.0, path_speed_mps=0.04)

    yield from descend_vial_to(slot_w[2] + 0.015, "lower", 0.08, 0.005)
    v, off = vial_now()
    log["tilt_above_entry_deg"] = v.tilt_deg
    if v.tilt_deg > RELEASE_MAX_TILT_DEG or off > RELEASE_MAX_OFFSET_M:
        yield from mo.move_ee(reader, ik, carry_p, place_q, timeout_s=4.0, pos_tol_m=0.015, axis_tol_deg=20.0)
        why = f"{off * 100:.1f} cm off the slot" if off > RELEASE_MAX_OFFSET_M else f"{v.tilt_deg:.0f} deg from vertical"
        return {"success": False, "reason": f"vial {why} above the slot (arm reach); still held -- try a nearer slot", **log}
    # Insert depth. Cap-up (tool forward-up, gripper body 4.6 cm BELOW the fingertips): 2.5 cm, so the
    # body stays above the rack top; a vertical vial released there drops to the slot floor (measured
    # 2026-10-07: centre 0.098, tilt 2.4 deg). Cap-down (body above the fingertips): 6 cm, because the
    # cap wedges in the hole and the vial stops where it is let go (2.5 cm left it at 9-20 deg).
    yield from descend_vial_to(slot_w[2] - (0.025 if cap_up else 0.06), "insert", 0.03, 0.004)
    lower_p = aim_p
    in_hand("lower")
    v, off = vial_now()
    log["vial_xy_offset_from_slot_m"] = off
    log["tilt_at_release_deg"] = v.tilt_deg
    if off > RELEASE_MAX_OFFSET_M or v.tilt_deg > RELEASE_MAX_TILT_DEG:
        # Releasing here lands the vial on the rim or the mat. Lift clear, keep it, report.
        yield from mo.move_ee(reader, ik, carry_p, place_q, timeout_s=4.0, pos_tol_m=0.015, axis_tol_deg=20.0)
        why = f"{off * 100:.1f} cm off the slot" if off > RELEASE_MAX_OFFSET_M else f"{v.tilt_deg:.0f} deg from vertical"
        return {"success": False, "reason": f"vial {why} after lowering (arm reach); still held -- try a nearer slot", **log}
    log["release"] = yield from mo.gripper(reader, mo.JAW_OPEN_RAD, duration_s=0.8)
    yield from mo.hold(reader, int(0.8 * mo.STEP_HZ))
    retreat_p = lower_p + np.array([0.0, 0.0, 0.08])
    log["retreat"] = yield from mo.move_ee(reader, ik, retreat_p, place_q, timeout_s=3.0, pos_tol_m=0.015, axis_tol_deg=30.0)
    yield from mo.hold(reader, int(0.7 * mo.STEP_HZ))
    v, _ = vial_now()
    return {
        "success": bool(v.placed and held not in reader.contact_vials()),
        "placed": v.placed,
        "in_rack": v.in_rack,
        "seated": v.seated,
        "vertical": v.vertical,
        "tilt_deg": v.tilt_deg,
        "vial_centre": v.centre,
        "vial_speed_mps": v.speed_mps,
        **log,
    }


def describe_home_pitch(reader: SceneReader) -> str:
    a = reader.tool_axis_world()
    return f"{math.degrees(math.atan2(-a[2], math.hypot(a[0], a[1]))):+.0f}° below horizontal"
