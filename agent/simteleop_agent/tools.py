"""Tools the agent may call. Every tool goes through the env server; none moves the arm outside it.

Tool returns are compact text with the numbers the agent needs to decide the next step. ``look``
returns an image block so the model can see the scene; ``describe_scene`` gives the same scene from
ground truth, which is what verdicts are made from.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import numpy as np
from strands import tool

from simteleop_common import geometry as geo

from .client import EnvClient

_client: EnvClient | None = None
CALL_LOG: list[dict[str, Any]] = []


def client() -> EnvClient:
    global _client
    if _client is None:
        _client = EnvClient(os.environ.get("SIMTELEOP_ENV_HOST", "127.0.0.1"), int(os.environ.get("SIMTELEOP_ENV_PORT", "7010")))
    return _client


def _log(tool_name: str, args: dict[str, Any], result: Any) -> None:
    CALL_LOG.append({"t": time.time(), "tool": tool_name, "args": args, "result": _brief(result)})


def _brief(result: Any) -> Any:
    if isinstance(result, dict):
        return {k: _brief(v) for k, v in result.items() if k not in ("state", "jpeg_b64")}
    if isinstance(result, list) and len(result) > 12:
        return result[:12] + ["..."]
    if isinstance(result, float):
        return round(result, 4)
    return result


def _fmt(v: list[float], nd: int = 3) -> str:
    return "[" + ", ".join(f"{x:.{nd}f}" for x in v) + "]"


def scene_text(s: dict[str, Any]) -> str:
    lines = [
        f"step {s['step']} (t={s['sim_time_s']:.1f}s)",
        "joints rad " + _fmt(s["joints_rad"]) + "  (LeRobot units " + _fmt(s["joints_lerobot"], 0) + ")",
        f"gripper at {_fmt(s['ee_position'])}, tool axis {_fmt(s['tool_axis_world'], 2)}, jaw {math.degrees(s['jaw_opening_rad']):.0f}°",
        f"jaw contact: {s['contact_vials'] or 'none'}; held vial: {s['grasped_vial'] or 'none'}",
    ]
    for v in s["vials"]:
        if v.get("placed"):
            where = "STANDING IN RACK"
        elif v["in_rack"]:
            where = "on the rack, NOT placed (leaning on the rim or lying)"
        else:
            where = "vertical" if v["vertical"] else "lying"
        lines.append(f"  {v['name']}: centre {_fmt(v['centre'])} {where}, tilt {v.get('tilt_deg', 0):.0f}° from vertical, speed {v['speed_mps']:.3f} m/s")
    lines.append(f"rack at {_fmt(s['rack_position'])}; slots (world): " + "; ".join(f"{i}:{_fmt(p)}{'' if f else ' (occupied)'}" for i, (p, f) in enumerate(zip(s["rack_slots_world"], s["rack_slots_free"]))))
    rec = s.get("recording") or {}
    lines.append(f"recording: {'ON, %d frames' % rec.get('frames', 0) if rec.get('active') else 'off'}; episodes saved {rec.get('episodes_saved', 0)}")
    return "\n".join(lines)


def verdict_text(r: dict[str, Any]) -> str:
    keep = {k: v for k, v in r.items() if k != "state"}
    return json.dumps(_brief(keep), default=str)


# --------------------------------------------------------------------------- perception


@tool
def describe_scene() -> str:
    """Ground-truth description of the scene: joint angles, gripper pose, every vial's position and
    orientation, whether one is held, the rack's slots and which are free, and the recording status.
    Use this before deciding what to pick and after every motion to verify the effect."""
    s = client().state()
    _log("describe_scene", {}, {"vials": [(v["name"], v["in_rack"]) for v in s["vials"]]})
    return scene_text(s)


@tool
def look(camera: str = "external_D455") -> dict:
    """Take a picture from a camera. `external_D455` is the fixed overview camera on the light box;
    `ego` is the camera on the gripper. Use it to check what describe_scene reports (for example
    that a vial is really standing in the rack and not leaning on the rim); if the picture and the
    numbers disagree, say so in your summary."""
    jpeg, meta = client().frame(camera)
    _log("look", {"camera": camera}, meta)
    return {"status": "success", "content": [{"text": f"{camera} camera, {meta['width']}x{meta['height']}, step {meta['step']}"}, {"image": {"format": "jpeg", "source": {"bytes": jpeg}}}]}


@tool
def measure_vial(vial: str) -> str:
    """Precise pose of one vial from ground truth: centre (m), axis direction, whether it is
    vertical and whether it is inside the rack. Cheaper than describe_scene when you only need one."""
    s = client().state()
    for v in s["vials"]:
        if v["name"] == vial:
            _log("measure_vial", {"vial": vial}, v)
            return json.dumps(_brief(v))
    return f"unknown vial {vial!r}; known: {[v['name'] for v in s['vials']]}"


# --------------------------------------------------------------------------- verified primitives


@tool
def pick_vial(vial: str) -> str:
    """Pick up a lying vial with a verified top-down grasp: open, hover, descend, close, lift 12 cm.
    Returns a measured verdict: success only if the jaw is touching the vial AND it rose more than
    5 cm. On failure the reason says which stage failed (hover unreachable, jaw closed on nothing,
    vial slipped). One call takes 15-30 s of simulated motion."""
    r = client().pick_vial(vial)
    _log("pick_vial", {"vial": vial}, r)
    return verdict_text(r)


@tool
def place_in_rack(slot: int | None = None) -> dict:
    """Stand the held vial up and lower it into a rack slot (nearest free slot if none given), then
    release and retreat. Success only if the vial ends standing in the rack: within 15° of vertical
    AND seated on the slot floor (not leaning on the rim), out of the jaw. The primitive refuses to
    release a vial that is tilted or more than 1 cm off the slot ("still held" in the reason).
    Requires a held vial (pick_vial succeeded). Returns the verdict and an overview picture of the
    result, so you can see what the measurement describes."""
    r = client().place_in_rack(slot)
    _log("place_in_rack", {"slot": slot}, r)
    content: list[dict] = [{"text": verdict_text(r)}]
    try:
        jpeg, meta = client().frame("external_D455")
        content.append({"text": f"[overview camera after the place, step {meta['step']}]"})
        content.append({"image": {"format": "jpeg", "source": {"bytes": jpeg}}})
    except Exception as exc:  # the verdict is what matters; a missing picture must not fail the tool
        content.append({"text": f"(no picture: {exc})"})
    return {"status": "success", "content": content}


@tool
def home() -> str:
    """Move the arm to its rest pose (the workshop's initial joint state). Use at the start of an
    episode and after a failed attempt so the next one starts from a known place."""
    r = client().home()
    _log("home", {}, r)
    return verdict_text(r)


# --------------------------------------------------------------------------- low-level moves (teleoperation)


@tool
def move_gripper_to(x: float, y: float, z: float, approach: str = "down") -> str:
    """Servo the gripper to a world position (metres) with the tool pointing `down` (top-down grasp)
    or `forward` (horizontal, away from the robot; how a vial is held upright). Returns the residual
    position error and tool-axis error — check them; a 5-DOF arm cannot reach every pose exactly.
    The robot base is at the origin; the mat and vials are around x≈0.2, |y|<0.2, z≈0.05."""
    approach_vec = [0.0, 0.0, -1.0] if approach == "down" else [x, y, 0.0]
    r = client().move_ee([x, y, z], approach=approach_vec, timeout_s=5.0)
    _log("move_gripper_to", {"x": x, "y": y, "z": z, "approach": approach}, r)
    return verdict_text(r)


@tool
def jog_joints(shoulder_pan: float | None = None, shoulder_lift: float | None = None, elbow_flex: float | None = None, wrist_flex: float | None = None, wrist_roll: float | None = None) -> str:
    """Move individual arm joints to absolute angles in DEGREES (omitted joints keep their target).
    Ranges: pan ±110, lift ±100, elbow −100..90, wrist_flex ±95, wrist_roll ±160. This is the
    joint-space teleoperation path; prefer move_gripper_to or the primitives when they apply."""
    s = client().state()
    q = list(s["joint_targets_rad"])
    for i, v in enumerate([shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]):
        if v is not None:
            q[i] = math.radians(v)
    r = client().move_joints(q, duration_s=1.5)
    _log("jog_joints", {"deg": [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]}, r)
    return verdict_text(r)


@tool
def gripper(command: str) -> str:
    """`open` or `close` the jaw. The verdict reports whether it stalled on something and which
    vial the contact sensor sees; a closed jaw touching nothing means the grasp missed."""
    r = client().gripper(command=command)
    _log("gripper", {"command": command}, r)
    return verdict_text(r)


# --------------------------------------------------------------------------- episode control


@tool
def reset_scene(seed: int | None = None) -> str:
    """Re-randomise the scene (vial and rack positions, lighting) and return the arm to its start.
    Cancels any recording in progress. Use to start a fresh attempt after a failure."""
    r = client().reset(seed)
    _log("reset_scene", {"seed": seed}, {"seed": seed})
    return "scene reset\n" + scene_text(r["state"])


@tool
def recording(action: str, success: bool | None = None, note: str = "") -> str:
    """Control episode recording: `start` before the first motion of an episode, `stop` after the
    verdict (pass success=true/false and a short note — they go into the episode manifest), `cancel`
    to discard, `status` to check. Only stop with success=true when place_in_rack reported success."""
    kw: dict[str, Any] = {}
    if action == "stop":
        kw["verdict"] = {"success": bool(success), "note": note, "tool_calls": len(CALL_LOG)}
    r = client().record(action, **kw)
    _log("recording", {"action": action, "success": success}, r)
    return json.dumps(_brief(r))


PERCEPTION_TOOLS = [describe_scene, look, measure_vial]
PRIMITIVE_TOOLS = [pick_vial, place_in_rack, home]
TELEOP_TOOLS = [move_gripper_to, jog_joints, gripper]
EPISODE_TOOLS = [reset_scene, recording]
ALL_TOOLS = [*PERCEPTION_TOOLS, *PRIMITIVE_TOOLS, *TELEOP_TOOLS, *EPISODE_TOOLS]
