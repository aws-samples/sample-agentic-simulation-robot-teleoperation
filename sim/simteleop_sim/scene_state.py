"""Ground truth from the running Isaac Lab environment, as plain Python for the wire.

The agent never sees a tensor. Everything it reasons about — joint angles, the gripper pose, where
each vial and the rack are, whether the jaw is touching a vial — comes through :func:`snapshot`,
which reads the simulator's own state, not a camera. Cameras are for the agent's eyes; verdicts
come from here.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from simteleop_common import geometry as geo
from simteleop_common.protocol import JOINT_NAMES

VIALS = ("vial_1", "vial_2", "vial_3")
RACK = "rack_left"
EE_BODY = "gripper"
JAW_BODY = "jaw"
CONTACT_SENSOR = "contact_grasp"
#: Which gripper-local axis the fingers point along. "-z" is what the camera mount in the workshop's
#: task config implies (the wrist camera at +y/−z looks along −y−z at the fingertips); the server
#: prints the measured world direction at start-up so a wrong guess is visible in the log and can be
#: overridden with SIMTELEOP_TOOL_AXIS.
TOOL_AXIS = os.environ.get("SIMTELEOP_TOOL_AXIS", "-z")
#: Tool centre point (between the fingertips) in the gripper body frame, metres. From the link mesh
#: extents reported by the `bodies` op on the workshop asset (2026-09-29): the fixed finger reaches
#: z = -0.104 along the tool axis; the jaw pivots at (0.020, 0.019, -0.023) and is 0.082 long, so its
#: closed tip is near (0.02, 0.02, -0.10). Override with SIMTELEOP_TCP_OFFSET="x,y,z".
TCP_OFFSET_LOCAL = np.array([float(v) for v in os.environ.get("SIMTELEOP_TCP_OFFSET", "0.01,0.01,-0.09").split(",")])


@dataclass
class ObjectState:
    name: str
    position: list[float]
    quaternion: list[float]
    axis_world: list[float]
    centre: list[float]
    vertical: bool
    in_rack: bool
    speed_mps: float
    tilt_deg: float = 90.0
    #: Standing on the slot floor (centre well below the entry), not leaning on the rim.
    seated: bool = False
    #: The placement verdict: vertical AND seated.
    placed: bool = False


@dataclass
class Snapshot:
    step: int
    sim_time_s: float
    joints_rad: list[float]
    joints_lerobot: list[float]
    joint_targets_rad: list[float]
    ee_position: list[float]
    ee_quaternion: list[float]
    tcp_position: list[float]
    tool_axis_world: list[float]
    jaw_opening_rad: float
    contact_vials: list[str]
    grasped_vial: str | None
    vials: list[ObjectState]
    rack_position: list[float]
    rack_quaternion: list[float]
    rack_slots_world: list[list[float]]
    rack_slots_free: list[bool]
    recording: dict[str, Any] = field(default_factory=dict)
    task: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _f(t: torch.Tensor) -> list[float]:
    return [float(x) for x in t.detach().cpu().flatten().tolist()]


class SceneReader:
    """Resolves entity handles once and produces snapshots on demand."""

    def __init__(self, env, task_id: str):
        self.env = env.unwrapped
        self.scene = self.env.scene
        self.task_id = task_id
        self.robot = self.scene["robot"]
        self.joint_ids = [self.robot.joint_names.index(n) for n in JOINT_NAMES]
        self.ee_idx = self.robot.body_names.index(EE_BODY)
        self.vials = {n: self.scene[n] for n in VIALS if n in self.scene.keys()}
        self.rack = self.scene[RACK] if RACK in self.scene.keys() else None
        self.contact = self.scene[CONTACT_SENSOR] if CONTACT_SENSOR in self.scene.keys() else None
        self.slots_local = self._read_slots_local()
        self.last_targets = torch.zeros(len(JOINT_NAMES), device=self.env.device)
        self.recording_status: dict[str, Any] = {"active": False, "frames": 0, "episodes_saved": 0}

    # ------------------------------------------------------------------ rack slots
    def _read_slots_local(self) -> list[list[float]]:
        """Slot entry positions in the rack frame, read from the USD like the workshop's reset does."""
        if self.rack is None:
            return []
        try:
            from isaacsim.core.prims import XFormPrim

            view = XFormPrim(prim_paths_expr=f"{self.rack.cfg.prim_path}/Body1/Mesh/top_*".replace("{ENV_REGEX_NS}", "/World/envs/env_.*"))  # regex form is what XFormPrim wants
            pos, _ = view.get_local_poses()
            slots = [[float(v) for v in p] for p in pos.cpu().numpy()] if hasattr(pos, "cpu") else [[float(v) for v in p] for p in np.asarray(pos)]
            # One env only: the view lists slots for env_0 once each.
            return slots[: max(1, len(slots))]
        except Exception as exc:  # noqa: BLE001 -- fall back to authored layout, but say so
            print(f"[scene_state] could not read rack slots from USD ({exc}); using the authored 2x2 layout")
            return [[0.03, 0.03, geo.RACK_SLOT_ENTRY_Z_M], [0.09, 0.03, geo.RACK_SLOT_ENTRY_Z_M],
                    [0.03, 0.09, geo.RACK_SLOT_ENTRY_Z_M], [0.09, 0.09, geo.RACK_SLOT_ENTRY_Z_M]]

    # ------------------------------------------------------------------ pieces
    def joints(self) -> torch.Tensor:
        return self.robot.data.joint_pos[0, self.joint_ids]

    def ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.robot.data.body_pos_w[0, self.ee_idx], self.robot.data.body_quat_w[0, self.ee_idx]

    def root_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.robot.data.root_pos_w[0], self.robot.data.root_quat_w[0]

    def tool_axis_world(self) -> np.ndarray:
        _, q = self.ee_pose_w()
        sign = -1.0 if TOOL_AXIS.startswith("-") else 1.0
        local = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[TOOL_AXIS.strip("+-")]
        return sign * geo.quat_rotate(_f(q), local)

    def contact_vials(self, force_threshold_n: float = 2.0) -> list[str]:
        """Vials the jaw is pressing on, from the workshop's filtered contact sensor."""
        if self.contact is None:
            return []
        fm = self.contact.data.force_matrix_w  # (envs, bodies, filters, 3)
        if fm is None:
            return []
        forces = torch.linalg.norm(fm[0], dim=-1)  # (bodies, filters)
        per_filter = forces.max(dim=0).values
        names = list(self.vials.keys())
        return [names[i] for i in range(min(len(names), per_filter.shape[0])) if float(per_filter[i]) > force_threshold_n]

    def object_state(self, name: str) -> ObjectState:
        obj = self.vials[name] if name in self.vials else self.rack
        p = _f(obj.data.root_pos_w[0])
        q = _f(obj.data.root_quat_w[0])
        v = float(torch.linalg.norm(obj.data.root_lin_vel_w[0]))
        in_rack = seated = False
        centre = geo.vial_centre_world(p, q)
        if self.rack is not None and name in self.vials:
            rp, rq = _f(self.rack.data.root_pos_w[0]), _f(self.rack.data.root_quat_w[0])
            # Judged on the vial's LOWER END: a vial standing in a hole has its centre 1 cm above the
            # slot entry (the workshop's centre-based test said "not in the rack" for a placed vial).
            in_rack = geo.point_in_rack(geo.vial_bottom_world(centre, q), rp, rq)
            seated = geo.vial_is_seated(centre, q, rp, rq)
        vertical = geo.vial_is_vertical(q)
        return ObjectState(
            name=name,
            position=p,
            quaternion=q,
            axis_world=[float(x) for x in geo.vial_axis_world(q)],
            centre=[float(x) for x in centre],
            vertical=vertical,
            in_rack=in_rack,
            speed_mps=v,
            tilt_deg=round(geo.vial_tilt_deg(q), 1),
            seated=seated,
            placed=bool(vertical and seated),
        )

    # ------------------------------------------------------------------ snapshot
    def snapshot(self, step: int) -> Snapshot:
        q = self.joints()
        ee_p, ee_q = self.ee_pose_w()
        vials = [self.object_state(n) for n in self.vials]
        contact = self.contact_vials()
        grasped = None
        # Grasped = in contact and lifted clear of the mat (the workshop uses 55 mm for the same test).
        for v in vials:
            if v.name in contact and v.centre[2] > 0.055:
                grasped = v.name
                break
        if self.rack is not None:
            rp, rq = _f(self.rack.data.root_pos_w[0]), _f(self.rack.data.root_quat_w[0])
            slots = geo.rack_slots_world(rp, rq, self.slots_local)
            free = [geo.nearest_free_slot([s], [v.centre for v in vials]) is not None for s in slots]
        else:
            rp, rq, slots, free = [0, 0, 0], [1, 0, 0, 0], [], []
        jl = _f(q)
        return Snapshot(
            step=step,
            sim_time_s=float(step * self.env.step_dt),
            joints_rad=jl,
            joints_lerobot=geo.rad_to_lerobot(jl),
            joint_targets_rad=_f(self.last_targets),
            ee_position=_f(ee_p),
            ee_quaternion=_f(ee_q),
            tcp_position=[float(x) for x in self.tcp_pose_w()[0]],
            tool_axis_world=[float(x) for x in self.tool_axis_world()],
            jaw_opening_rad=float(q[-1]),
            contact_vials=contact,
            grasped_vial=grasped,
            vials=vials,
            rack_position=rp,
            rack_quaternion=rq,
            rack_slots_world=[[float(x) for x in s] for s in slots],
            rack_slots_free=free,
            recording=dict(self.recording_status),
            task=self.task_id,
        )

    def bodies(self) -> dict[str, dict[str, list[float]]]:
        """World pose of every robot link (for measuring frames such as the fingertip offset)."""
        out = {}
        for i, name in enumerate(self.robot.body_names):
            out[name] = {"position": _f(self.robot.data.body_pos_w[0, i]), "quaternion": _f(self.robot.data.body_quat_w[0, i])}
        p, q = self.ee_pose_w()
        out["_tcp"] = {"position": [float(x) for x in (np.asarray(_f(p)) + geo.quat_rotate(_f(q), TCP_OFFSET_LOCAL))], "quaternion": _f(q)}
        out["_mesh_extents_local"] = self.link_mesh_extents(("gripper", "jaw"))
        out["_joints"] = self.joint_frames()
        return out

    def joint_frames(self) -> dict[str, dict]:
        """Every USD joint under the robot: bodies, axis token and the joint frame in body0."""
        try:
            import omni.usd
            from pxr import UsdPhysics

            stage = omni.usd.get_context().get_stage()
            root = self.robot.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")
            out = {}
            for prim in stage.Traverse():
                if not str(prim.GetPath()).startswith(root) or not prim.IsA(UsdPhysics.Joint):
                    continue
                j = UsdPhysics.Joint(prim)
                b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
                b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
                axis = prim.GetAttribute("physics:axis").Get()
                q0 = j.GetLocalRot0Attr().Get()
                p0 = j.GetLocalPos0Attr().Get()
                q1 = j.GetLocalRot1Attr().Get()
                out[prim.GetName()] = {
                    "type": prim.GetTypeName(), "body0": b0[-1].split("/")[-1] if b0 else None, "body1": b1[-1].split("/")[-1] if b1 else None,
                    "axis": str(axis), "localPos0": [float(x) for x in p0] if p0 is not None else None,
                    "localRot0_wxyz": [float(q0.GetReal()), *[float(x) for x in q0.GetImaginary()]] if q0 is not None else None,
                    "localRot1_wxyz": [float(q1.GetReal()), *[float(x) for x in q1.GetImaginary()]] if q1 is not None else None,
                }
            return out
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}

    def link_mesh_extents(self, names) -> dict[str, dict[str, list[float]]]:
        """Axis-aligned bounds of each link's render meshes in the link's own frame, from the USD.

        Independent of the simulation state, so it is the right thing to derive the fingertip (TCP)
        offset from: the tips are at the far end of the gripper/jaw extents along the tool axis.
        """
        try:
            import omni.usd
            from pxr import Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            # cfg.prim_path is already the regex form (/World/envs/env_.*/Robot); we want env_0.
            root = self.robot.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            out = {}
            for n in names:
                prim = stage.GetPrimAtPath(f"{root}/{n}")
                if not prim.IsValid():
                    out[n] = {"error": f"no prim {root}/{n}"}
                    continue
                # Untransformed = in the link's own frame (what a TCP offset needs); local = in the parent's.
                r = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
                rl = cache.ComputeLocalBound(prim).ComputeAlignedRange()
                out[n] = {"min": [float(x) for x in r.GetMin()], "max": [float(x) for x in r.GetMax()],
                          "parent_frame_min": [float(x) for x in rl.GetMin()], "parent_frame_max": [float(x) for x in rl.GetMax()]}
            return out
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}

    def tcp_pose_w(self) -> tuple[np.ndarray, np.ndarray]:
        p, q = self.ee_pose_w()
        q = np.asarray(_f(q))
        return np.asarray(_f(p)) + geo.quat_rotate(q, TCP_OFFSET_LOCAL), q

    def describe_tool_axis(self) -> str:
        a = self.tool_axis_world()
        pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, -float(a[2])))))
        return f"tool axis {TOOL_AXIS} points {np.round(a, 3).tolist()} in the world ({pitch_deg:+.0f}° below horizontal)"
