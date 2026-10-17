"""Motions as generators that the server drives one simulator step at a time.

Every motion is ``Generator[targets, None, verdict]``: each ``yield`` hands the server the six joint
targets (radians) to apply this step, and the generator's return value is a dict describing what
was measured at the end — never what was planned. The server steps the env, records a frame if a
recording is on, and resumes the generator.

Design rules carried over from the OpenArm work:

* a gripper stall is the only evidence of contact; a commanded closure is not;
* an IK move reports the residual error, and the caller decides whether that is good enough;
* nothing here throws on a missed target — a failed motion is data.
"""
from __future__ import annotations

import math
from typing import Any, Generator

import numpy as np
import torch

from simteleop_common import geometry as geo
from simteleop_common.protocol import JOINT_NAMES

from .scene_state import SceneReader

Motion = Generator[torch.Tensor, None, dict[str, Any]]

STEP_HZ = 60.0  # env steps per second: sim dt 1/120 with decimation 2 (workshop config)
JAW_OPEN_RAD = math.radians(60.0)
JAW_CLOSED_RAD = math.radians(-8.0)


def _t(reader: SceneReader, values) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float32), device=reader.env.device)


def hold(reader: SceneReader, steps: int) -> Motion:
    """Keep the current targets for ``steps`` steps (settling)."""
    targets = reader.last_targets.clone()
    for _ in range(steps):
        yield targets
    return {"held_steps": steps}


def move_joints(reader: SceneReader, goal_rad, duration_s: float = 2.0, settle_s: float = 0.4, tol_rad: float = 0.03) -> Motion:
    """Linear interpolation of all six joints from the current targets to ``goal_rad``."""
    goal = geo.clamp_joints(goal_rad)
    start = reader.last_targets.detach().cpu().numpy()
    n = max(1, int(duration_s * STEP_HZ))
    for i in range(1, n + 1):
        a = i / n
        yield _t(reader, start + a * (goal - start))
    for _ in range(int(settle_s * STEP_HZ)):
        yield _t(reader, goal)
    q = reader.joints().detach().cpu().numpy()
    err = np.abs(q - goal)
    return {
        "reached": bool(err[:5].max() < tol_rad),
        "max_arm_error_rad": float(err[:5].max()),
        "jaw_error_rad": float(err[5]),
        "joints_rad": [float(x) for x in q],
    }


def gripper(reader: SceneReader, opening_rad: float, duration_s: float = 0.8, settle_s: float = 0.5) -> Motion:
    """Drive only the jaw. The verdict says whether it stalled (contact) short of the target."""
    goal = float(np.clip(opening_rad, geo.joint_limits_rad()[0][5], geo.joint_limits_rad()[1][5]))
    targets = reader.last_targets.clone()
    start = float(targets[5])
    n = max(1, int(duration_s * STEP_HZ))
    for i in range(1, n + 1):
        targets[5] = start + (goal - start) * i / n
        yield targets.clone()
    history = []
    for _ in range(int(settle_s * STEP_HZ)):
        yield targets.clone()
        history.append(float(reader.joints()[5]))
    final = history[-1] if history else float(reader.joints()[5])
    stalled = abs(final - goal) > math.radians(2.0) and (len(history) < 2 or abs(history[-1] - history[-2]) < math.radians(0.2))
    if stalled and goal < final:
        # Closing on an object: hold a light squeeze (4 deg past the stall) rather than the full
        # closed target. The jaw actuator's full squeeze reacts through the soft wrist joints
        # (Wrist_Roll stiffness 7, Wrist_Pitch 12) and deflects the tool; measured 2026-09-29: the
        # lift after a hard squeeze folded the arm (27 deg orientation error, 4 cm position error).
        targets[5] = final - math.radians(4.0)
        for _ in range(int(0.2 * STEP_HZ)):
            yield targets.clone()
    return {
        "target_rad": goal,
        "jaw_rad": final,
        "stalled": bool(stalled),
        "contact_vials": reader.contact_vials(),
    }


class IKSolver:
    """Damped least-squares differential IK on the five arm joints, world frame, position-weighted.

    Own implementation rather than Isaac Lab's ``DifferentialIKController`` because the SO-101 has
    5 DOF: the tool's out-of-plane orientation component is not controllable, and an equally weighted
    6-D pose error lets the solver trade position for an unreachable 2 deg of orientation until the
    arm wanders into a folded corner (measured 2026-09-29: 12 cm short with lift -74, elbow 87).
    Position error is in metres; orientation error (axis-angle, rad) is scaled by ``rot_weight`` so a
    radian counts like ``rot_weight`` metres.
    """

    def __init__(self, reader: SceneReader, rot_weight: float = 0.15, damping: float = 0.02):
        self.reader = reader
        self.robot = reader.robot
        self.arm_ids = reader.joint_ids[:5]
        # For a fixed-base articulation the Jacobian rows start at the first link after the root.
        self.jacobi_idx = reader.ee_idx - 1
        self.rot_weight = rot_weight
        self.damping = damping

    def step(self, target_pos_w: np.ndarray, target_quat_w: np.ndarray, rot_weight: float | None = None) -> np.ndarray:
        """Desired arm joint positions (rad, numpy) for one step towards the world-frame pose."""
        rw = self.rot_weight if rot_weight is None else rot_weight
        ee_p, ee_q = self.reader.ee_pose_w()
        ee_p = ee_p.detach().cpu().numpy().astype(float)
        ee_q = geo.quat_normalize(ee_q.detach().cpu().numpy().astype(float))
        jac = self.robot.root_physx_view.get_jacobians()[0, self.jacobi_idx, :, :][:, self.arm_ids].detach().cpu().numpy().astype(float)  # (6, 5)
        pos_err = np.asarray(target_pos_w, dtype=float) - ee_p
        # Orientation error as a rotation vector: q_err = q_target * conj(q_now).
        q_err = geo.quat_mul(target_quat_w, [ee_q[0], -ee_q[1], -ee_q[2], -ee_q[3]])
        if q_err[0] < 0:
            q_err = -q_err
        s_ = float(np.linalg.norm(q_err[1:]))
        rot_err = np.zeros(3) if s_ < 1e-9 else q_err[1:] / s_ * (2.0 * math.atan2(s_, float(q_err[0])))
        w = np.array([1.0, 1.0, 1.0, rw, rw, rw])
        e = np.concatenate([pos_err, rot_err]) * w
        jw = jac * w[:, None]
        dq = jw.T @ np.linalg.solve(jw @ jw.T + (self.damping ** 2) * np.eye(6), e)
        q_now = self.robot.data.joint_pos[0, self.arm_ids].detach().cpu().numpy().astype(float)
        return q_now + dq


def move_ee(
    reader: SceneReader,
    ik: IKSolver,
    target_pos_w,
    target_quat_w,
    timeout_s: float = 4.0,
    pos_tol_m: float = 0.006,
    axis_tol_deg: float = 12.0,
    max_step_rad: float = 0.03,
    path_speed_mps: float = 0.12,
    rot_weight: float | None = None,
) -> Motion:
    """Servo the gripper to a world pose with differential IK, one small joint step per sim step.

    Ends when the position residual is under ``pos_tol_m`` and the tool axis within
    ``axis_tol_deg`` of the target's, or at the timeout. The verdict carries both residuals; with a
    5-DOF arm the orientation is the one that gives, and the caller can decide if it matters.
    """
    from .scene_state import TCP_OFFSET_LOCAL

    # The caller specifies where the fingertips (TCP) should be; IK works on the gripper body frame.
    tcp_target_w = np.asarray(target_pos_w, dtype=float)
    target_quat_w = geo.quat_normalize(target_quat_w)
    target_pos_w = tcp_target_w - geo.quat_rotate(target_quat_w, TCP_OFFSET_LOCAL)
    target_axis = geo.quat_rotate(target_quat_w, _tool_local_axis())
    lo, hi = geo.joint_limits_rad()
    steps = int(timeout_s * STEP_HZ)
    converged = False
    pos_err = axis_err = float("nan")
    stall_count = 0
    # The commanded target moves from the current TCP to the goal at path_speed_mps, so the TCP
    # follows a straight line. Fed the final goal directly, DLS IK curves the path -- measured
    # 2026-09-29: a 7 cm descent drifted 4.7 cm towards the base and the finger hit the mat early.
    start_tcp, start_q = reader.tcp_pose_w()
    total = float(np.linalg.norm(tcp_target_w - start_tcp))
    # The orientation is interpolated along the same fraction (slerp), at no less than ~90 deg/s: a
    # 90 deg reorientation handed to DLS IK in one go went to the wrong branch (180 deg error, wrist
    # at its limit; measured 2026-09-29).
    rot_deg = math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(geo.quat_normalize(start_q), target_quat_w))))))
    per_step = path_speed_mps / STEP_HZ
    rot_steps = max(1, int(rot_deg / 90.0 * STEP_HZ))
    move_steps = max(1, int(total / per_step)) if total > 1e-6 else 1
    n_path = max(rot_steps, move_steps)
    for i in range(steps):
        frac = min(1.0, (i + 1) / n_path)
        cmd_tcp = start_tcp + frac * (tcp_target_w - start_tcp)
        cmd_q = geo.quat_slerp(start_q, target_quat_w, frac)
        cmd_pos = cmd_tcp - geo.quat_rotate(cmd_q, TCP_OFFSET_LOCAL)
        q_des = ik.step(cmd_pos, cmd_q, rot_weight)
        q_now = reader.joints().detach().cpu().numpy()
        delta = q_des - q_now[:5]
        biggest = float(np.abs(delta).max())
        if biggest > max_step_rad:  # scale the whole step: per-joint clipping bends the direction
            delta = delta * (max_step_rad / biggest)
        targets = reader.last_targets.clone()
        targets[:5] = _t(reader, np.clip(q_now[:5] + delta, lo[:5], hi[:5]))
        yield targets
        tcp_p, _ = reader.tcp_pose_w()
        pos_err = float(np.linalg.norm(tcp_p - tcp_target_w))
        axis_err = math.degrees(geo.angle_between(reader.tool_axis_world(), target_axis))
        if pos_err < pos_tol_m and axis_err < axis_tol_deg:
            converged = True
            break
        stall_count = stall_count + 1 if (frac >= 1.0 and float(np.abs(delta).max()) < 1e-4) else 0
        if stall_count > int(0.5 * STEP_HZ):  # IK has nothing left to give
            break
    for _ in range(int(0.3 * STEP_HZ)):
        yield reader.last_targets.clone()
    tcp_p, ee_q_now = reader.tcp_pose_w()
    # Full orientation error (includes the roll about the tool axis, which axis_error_deg ignores).
    q_err = geo.quat_mul(target_quat_w, [ee_q_now[0], -ee_q_now[1], -ee_q_now[2], -ee_q_now[3]])
    orient_err_deg = math.degrees(2 * math.acos(min(1.0, abs(float(q_err[0])))))
    return {
        "converged": converged,
        "position_error_m": float(np.linalg.norm(tcp_p - tcp_target_w)),
        "axis_error_deg": float(math.degrees(geo.angle_between(reader.tool_axis_world(), target_axis))),
        "orientation_error_deg": orient_err_deg,
        "joints_deg": [round(math.degrees(float(x))) for x in reader.joints().detach().cpu().tolist()],
        "tcp_position": [float(x) for x in tcp_p],
        "steps": i + 1,
    }


def _tool_local_axis() -> list[float]:
    from .scene_state import TOOL_AXIS

    sign = -1.0 if TOOL_AXIS.startswith("-") else 1.0
    return [sign * c for c in {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[TOOL_AXIS.strip("+-")]]


def pan_to_azimuth(reader: SceneReader, target_xy, duration_s: float = 1.5) -> Motion:
    """Rotate only the base joint so the TCP's azimuth matches ``target_xy``'s.

    All other joints are pitch joints with parallel axes, so a pan move keeps the TCP's radius and
    height: the cheap, always-reachable half of a lateral carry. The sign of Rotation vs azimuth is
    measured rather than assumed (one trial move, then a correction), because it depends on the
    asset's joint convention (this one: azimuth ≈ -Rotation).
    """
    tx, ty = float(target_xy[0]), float(target_xy[1])
    goal_az = math.atan2(ty, tx)

    def az_now() -> float:
        p, _ = reader.tcp_pose_w()
        return math.atan2(float(p[1]), float(p[0]))

    def wrap(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    err0 = wrap(goal_az - az_now())
    q = reader.last_targets.detach().cpu().numpy().copy()
    pan0 = q[0]
    q[0] = pan0 - err0  # assume azimuth ≈ -Rotation
    yield from move_joints(reader, q, duration_s=duration_s, settle_s=0.3)
    err1 = wrap(goal_az - az_now())
    if abs(err1) > abs(err0) * 0.5 and abs(err0) > math.radians(3):  # wrong sign: go the other way
        q[0] = pan0 + err0
        yield from move_joints(reader, q, duration_s=duration_s, settle_s=0.3)
        err1 = wrap(goal_az - az_now())
    if abs(err1) > math.radians(1.5):  # one proportional correction with the measured gain
        gain = (err1 - err0) / (q[0] - pan0) if abs(q[0] - pan0) > 1e-6 else -1.0
        q[0] = q[0] - err1 / (gain if abs(gain) > 0.2 else -1.0)
        yield from move_joints(reader, q, duration_s=0.8, settle_s=0.3)
        err1 = wrap(goal_az - az_now())
    return {"azimuth_error_deg": math.degrees(err1), "pan_rad": float(reader.joints()[0])}


def chain(*motions: Motion) -> Motion:
    """Run motions back to back; return the list of their verdicts."""
    results = []
    for m in motions:
        r = yield from m
        results.append(r)
    return {"steps": results}
