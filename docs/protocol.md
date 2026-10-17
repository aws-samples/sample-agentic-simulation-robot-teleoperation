# Env server protocol

TCP, `127.0.0.1:7010`, one JSON object per line each way. Try it: `echo '{"op":"ping"}' | nc localhost 7010`.

| op | args | returns | blocks |
|---|---|---|---|
| `ping` | — | protocol version, task, step, `busy`, tool-axis description | no |
| `state` | — | full snapshot (below) | no |
| `bodies` | — | the robot's body names, poses and the joint frames (used to measure the tool axis and TCP offset for a new gripper) | no |
| `frame` | `camera` (`external_D455` \| `ego`), `quality` | `jpeg_b64`, width, height, step | no |
| `record` | `action`: `start` \| `stop` (+`verdict`) \| `cancel` \| `note` (+`event`) \| `status` | recorder status (+ manifest on stop) | no |
| `reset` | `seed` | new snapshot | yes |
| `home` | — | `reached`, joint errors | yes |
| `move_joints` | `joints_rad[6]`, `duration_s` | `reached`, `max_arm_error_rad` | yes |
| `move_ee` | `position[3]`, `quaternion[4]` or `approach[3]`, `timeout_s` | `converged`, `position_error_m`, `axis_error_deg` | yes |
| `gripper` | `command` `open`/`close` or `opening_rad` | `jaw_rad`, `stalled`, `contact_vials` | yes |
| `pick_vial` | `vial`; optional `tilt_deg` (grasp tilt, default 35), `depth_m`, `along_m` (pinch offset towards the cap) | `success`, `rose_m`, `still_in_contact`, `axis_in_gripper` (vial axis in the gripper frame after the lift), stage logs (`hover`, `descend`, `close`, `regrip`, `lift_clear`, `lift`) | yes |
| `place_in_rack` | `slot` (optional, default nearest free); `cap_up` (default true; false = the older 180° wrist-roll hold, vial ends cap-down) | `success`, `placed`, `seated`, `vertical`, `tilt_deg`, `in_rack`, `vial_centre`, `tilt_at_release_deg`, `vial_xy_offset_from_slot_m`, stage logs (`rise`, `pan`, `carry`, `aim_*`, `lower`, `insert`, `release`, `retreat`, `hand_*` = vial pose in the gripper frame at each stage) | yes |
| `shutdown` | — | — | no |

Blocking ops reply when the motion's generator returns its verdict; a second blocking op while one
runs is refused with `error: busy`. Every verdict also carries `state` (a snapshot after the motion)
and `wall_seconds`.

Snapshot fields: `step`, `sim_time_s`, `joints_rad`, `joints_lerobot`, `joint_targets_rad`,
`ee_position`, `ee_quaternion` (w,x,y,z), `tool_axis_world`, `jaw_opening_rad`, `contact_vials`,
`grasped_vial`, `vials[]` (`name`, `position`, `quaternion`, `axis_world` (cap direction, local +z),
`centre`, `tilt_deg` (from vertical), `vertical` (tilt < 15°), `in_rack` (lower end inside the rack
footprint below the slot entry), `seated` (lower end ≥ 4 cm down a hole), `placed` (= vertical and
seated: the placement verdict), `speed_mps`), `rack_position`, `rack_quaternion`,
`rack_slots_world[]`, `rack_slots_free[]`, `recording`, `task`.

Every motion stage result has `converged`, `position_error_m`, `axis_error_deg`, `orientation_error_deg`,
`joints_deg` and `tcp_position`; a stage that ended at a joint limit shows it in `joints_deg`
(SO-101 ranges in `common/simteleop_common/protocol.py`).
