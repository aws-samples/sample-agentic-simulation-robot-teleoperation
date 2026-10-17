---
name: simteleop-add-primitive
description: Add or change a verified manipulation primitive in the sim-teleop platform (a motion generator in sim/simteleop_sim/primitives.py with a ground-truth verdict), wire it through the env-server protocol, the Python client and a Strands tool, and measure it over seeded layouts before letting the agent use it. Use when asked for a new skill the agent can call (e.g. "pour", "regrasp", "push the rack"), to change what a primitive does, or to add a knob to pick_vial/place_in_rack.
---

# Add or change a primitive

A primitive is a Python generator that yields joint targets once per sim step and returns a dict
whose `success` is decided from the simulator's ground truth. The agent calls it as a tool and can
retry or choose something else; it cannot change the criterion. Keep that contract.

## 1. Design the verdict first

Write down, before any motion code: what measurable fact means success (object pose, contact,
displacement), with numbers and tolerances, and which failure reasons the caller should be able to
act on (`"still held"` → try another slot; `"jaw closed without touching"` → re-read the pose and
retry once). Use `SceneReader` facts only: `object_state(name)` (centre, axis, tilt_deg, in_rack,
seated, placed, speed), `contact_vials()`, `tcp_pose_w()`, `joints()`. If the fact you need is not
exposed, add it to `scene_state.py` (and `ObjectState`/`Snapshot`) — never infer it from the plan.

## 2. Write the motion in `sim/simteleop_sim/primitives.py`

Pattern (see `pick_vial`, `place_in_rack`):

```python
def my_primitive(reader: SceneReader, ik: mo.IKSolver, arg: str) -> mo.Motion:
    """One sentence. Success: <the measurable criterion>."""
    v0 = reader.object_state(arg)
    log: dict[str, Any] = {"arg": arg, "start": v0.centre}
    log["stage1"] = yield from mo.move_ee(reader, ik, target_p, target_q, timeout_s=4.0, pos_tol_m=0.006, axis_tol_deg=10.0)
    if log["stage1"]["position_error_m"] > 0.03:
        return {"success": False, "reason": "could not reach <where>", **log}
    log["grip"] = yield from mo.gripper(reader, mo.JAW_CLOSED_RAD, duration_s=1.0, settle_s=0.6)
    yield from mo.hold(reader, int(0.5 * mo.STEP_HZ))          # let physics settle before measuring
    v1 = reader.object_state(arg)
    return {"success": bool(<criterion on v1>), "reason": None if ok else "<what failed, with the number>", **log}
```

Building blocks in `motion.py`: `move_joints` (joint-space interpolation), `move_ee` (differential
IK servo of the TCP to a world pose along a straight path, with slerp; returns `converged`,
`position_error_m`, `axis_error_deg`, `joints_deg`), `gripper` (open/close with stall detection),
`pan_to_azimuth` (base yaw only), `hold`. Geometry helpers in `common/simteleop_common/geometry.py`
(poses are quaternions `(w,x,y,z)`; tool axis local `-z`; TCP offset `scene_state.TCP_OFFSET_LOCAL`).

Rules learned the hard way (all measured; the numbers are in the code comments next to the constants):

- Command the TCP along a straight line (`move_ee` does); feeding DLS IK a far goal curves the path.
- A 5-DOF arm cannot realise arbitrary orientations: the tool stays in the arm's radial plane. Derive
  target orientations from the current pose by a pure pitch (`geo.upright_hold_orientation`) or with
  `geo.tool_orientation(approach, radial, …)`, and re-derive after any base pan.
- Lateral carries: pan the base (joint space), then servo in-plane. One IK servo across 17 cm folded
  the arm into a corner.
- The jaw's full squeeze bends the wrist (actuator stiffness 4 vs 7): `mo.gripper` holds 4° past the
  stall; keep it.
- Friction holds a grasped object at the angle it was grasped; gravity does not re-align it. If you
  need a different in-hand angle, use a constraint (the mat, a wall) as `pick_vial`'s regrip does.
- Wrist_Pitch is ±95°: for the SO-101 the tool pitch below horizontal ≈ Pitch + Elbow + Wrist_Pitch − 10°
  (degrees). A hold that needs the tool pointing UP by α forces Pitch + Elbow ≤ 85° − α and the arm
  cannot also reach low, near targets. Prefer tool-down or horizontal holds.
- Settle (`mo.hold`) before measuring; the contact sensor and poses lag a step or two.
- Log every stage into the returned dict (`log[...] = yield from …`); the agent's trace and the
  manifests carry it, and it is how you will debug it later. Put the vial-in-gripper pose in the log
  when a grasp is involved (`in_hand()` in `place_in_rack`).

## 3. Wire it through

1. `common/simteleop_common/protocol.py`: add the op name to `OPS` (the preflight checks the server
   speaks every op the agent uses).
2. `sim/simteleop_sim/env_server.py`: add it to `MOTION_OPS` and to the dispatch
   (`if op == "my_primitive": return prim.my_primitive(r, ik, req["arg"])`). Optional args:
   `req.get("knob")`.
3. `agent/simteleop_agent/client.py`: a thin method (`def my_primitive(self, arg): return self.call("my_primitive", timeout_s=240, arg=arg)`).
4. `agent/simteleop_agent/tools.py`: a `@tool` whose docstring tells the model *when* to use it and
   *what the verdict means*; `_log(...)` the call; return `verdict_text(r)` (or a content list with an
   image block like `place_in_rack` if the agent should see the result). Add it to the right list at
   the end of the file (`PRIMITIVE_TOOLS`…).
5. `agent/simteleop_agent/prompts.py`: one line on when to prefer it and what to do on each failure
   reason.
6. `docs/protocol.md` row; `docs/agent_teleoperation.md` row.

## 4. Measure before the agent uses it

```bash
scripts/push-code                                   # sync + restart the env server
```

Probe over seeded layouts from the box's venv (write it to `/tmp` on the box via `scripts/remote`,
or adapt `skills/simteleop-debug-manipulation/probe.py`):

```python
from simteleop_agent.client import EnvClient
c = EnvClient(); ok = 0
for seed in range(1001, 1009):
    c.reset(seed=seed); c.wait_idle(); c.home(); c.wait_idle()
    r = c.call("my_primitive", timeout_s=240, arg="vial_2"); c.wait_idle()
    ok += bool(r["success"]); print(seed, r["success"], r.get("reason"), {k: r[k] for k in r if k.endswith("_deg") or k.endswith("_m")})
print(f"{ok}/8")
```

Then run it detached (`scripts/remote --detach probe …`, `--wait probe`), look at a camera frame of
at least one success and one failure, and only then run agent episodes. Record the rate, the seeds,
and the failure modes in the primitive's docstring/constants and in the commit message. Add a unit
test in `tests/test_geometry.py` for any new geometric rule.

## Checklist

- [ ] verdict from ground truth, with numbers; failure reasons the agent can act on
- [ ] stage logs in the returned dict; `hold` before measuring
- [ ] op in `protocol.OPS`, `env_server` dispatch, client method, tool, prompt line, docs
- [ ] `python -m pytest -q` green
- [ ] probe k/8 with seeds, one frame per outcome inspected, numbers in the constants' comments and the commit message
