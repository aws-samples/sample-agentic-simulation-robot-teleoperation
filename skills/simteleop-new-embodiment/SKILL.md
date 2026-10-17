---
name: simteleop-new-embodiment
description: Port the sim-teleop platform to another robot, gripper, scene or task (a different Isaac Lab gym task, another arm, other objects) while keeping the env server, protocol, agent, recorder and infrastructure. Use when asked to "use robot X", "add task Y", "change the objects", "support another upstream repo", or when a scene loads but describe_scene shows no objects, the tool axis is wrong, or pick_vial cannot reach anything after a port.
---

# Port to a new embodiment or scene

What is robot/scene-specific (≈ 600 lines) and what is not: `scene_state.py` (how to read the
scene), `primitives.py` (how to manipulate), the joint tables in `protocol.py`, the geometry
constants in `geometry.py`, and the bootstrap's upstream checkout/image. Everything else — the TCP
protocol, `motion.py` (joint interpolation, differential IK, jaw, pan), `env_server.py`, the
recorder wrapper, the Strands agent, the CDK stack — is agnostic and should not need edits.

## 1. Upstream and image

- Pin the upstream repository and commit in `infrastructure/cdk.json` (`workshopRepo`,
  `workshopCommit`) and `VERSIONS.md`. The bootstrap (`50_workshop_checkout.sh`) clones it; the sim
  image (`60_sim_image.sh`) builds the upstream Dockerfile and layers `sim/` + `common/` on top.
  Do not vendor upstream files (AGENTS.md invariant 3).
- The gym task id goes to `simteleop-up --task …` / the server's `--task`. It must be registered by
  the upstream package inside the container; check with
  `docker exec simteleop_sim python -c "import gymnasium as gym, <upstream>; print([k for k in gym.registry if 'Teleop' in k])"`.
- The recorder: `recording.py` imports the upstream `LeRobotRecorder`; if the new upstream has none,
  write the dataset with LeRobot directly behind the same `EpisodeRecorder` interface
  (`start/stop/cancel/push/status`). Keep features = what a human teleop session of that upstream
  would produce.

## 2. Joint tables — `common/simteleop_common/protocol.py`

`JOINT_NAMES` (order = the env's action vector), `USD_JOINT_RANGE_DEG` (from the robot USD: the IK
clips to these and the debug skill reads limits from them), and the LeRobot unit mapping tables
used by `geometry.rad_to_lerobot`/`lerobot_to_rad` (copied from the upstream's own mapping so
datasets match).
`tests/test_protocol.py` and `test_geometry.py::test_lerobot_roundtrip_and_scale` must be updated
and pass.

## 3. Reading the scene — `sim/simteleop_sim/scene_state.py`

`SceneReader` is the only place that touches Isaac Lab objects. Replace:

- object handles (`self.vials`, `self.rack`): the env's scene entities by name; `object_state()` must
  produce centre, axis, tilt, speed and the task's membership facts (`in_rack`/`seated`/`placed` or
  their analogues) from poses, not from the plan.
- contact: `contact_vials()` filters the contact sensor's per-filter forces; the sensor must be
  configured in the task (`activate_contact_sensors=True`, filter prim paths = the objects).
- fixture geometry: slots are read from the USD at start (`Body1/Mesh/top_*`); do the same for your
  fixture, with a fallback and a log line.
- gripper geometry: `TOOL_AXIS` (which local axis of the end-effector body points out of the
  fingers), `TCP_OFFSET_LOCAL` (body frame → fingertips). Measure them with the `bodies` op / the USD
  joint frames and link extents; the env-server start-up line prints the tool axis in the world at
  home — fingers should point roughly forward-down. Env overrides `SIMTELEOP_TOOL_AXIS`,
  `SIMTELEOP_TCP_OFFSET` exist so you can try before editing.
- IK: `motion.IKSolver` uses the env's Jacobian (world frame — PhysX gives world-frame Jacobians;
  the SO-101 base is yawed 90°, see the comment) with position weighting (`rot_weight` 0.15) for a
  5-DOF arm. A 6/7-DOF arm can raise `rot_weight` and drop the pitch-only orientation construction.

## 4. Geometry constants and primitives

`geometry.py`: object dimensions, fixture dimensions, placement criteria (`VERTICAL_COS_MIN`,
`SEATED_BELOW_ENTRY_M`, footprint test), grasp/hold pose constructors. `primitives.py`: rewrite
`pick_*`/`place_*` for the task following `simteleop-add-primitive` — verdict first, measured
constants, stage logs. Keep `home`. Reuse the hard-won patterns: straight-line TCP paths, pan then
in-plane carry, re-derive orientations after a pan, settle before measuring, gate the release on
the object's measured pose, constraints (mat) to change the in-hand angle since gravity will not.

## 5. Agent side

`tools.py`: rename/replace the primitive tools and `describe_scene`'s text (`scene_text`) so the
model gets the task's facts in words (`STANDING IN RACK`, `lying`, occupancy…); `prompts.py`: the
task description, what to pick first, what each failure reason means; `__main__.py`:
`ground_truth_verdict` for the new task. `tests/test_tools_text.py` covers the text.

## 6. Verify

1. `python -m pytest -q` (protocol, geometry, tool text).
2. `scripts/push-code`; `scripts/remote simteleop-up --task <id>`; `simteleop-status` shows the
   objects and fixture; the start-up tool-axis line makes sense.
3. Low-level sanity from the venv: `home`, `move_ee` to a point 10 cm above an object with
   `approach="down"` → `position_error_m` < 1 cm; `gripper("close")` on nothing → no contact.
4. Probe each primitive over 8 seeds (`simteleop-debug-manipulation`); frames of outcomes.
5. One agent episode; a dataset episode whose manifest verdict matches the frame.
6. Update `README.md` (robot row, task row) and `docs/operating.md` (tools, knobs), `docs/protocol.md` (snapshot fields),
   `VERSIONS.md`; put the measurements in the constants' comments and the commit message.
