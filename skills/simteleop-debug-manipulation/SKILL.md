---
name: simteleop-debug-manipulation
description: Diagnose why a sim-teleop primitive fails or why a verdict disagrees with what the camera shows — run the seeded scripted probe, read stage logs (IK convergence, joint limits, in-hand object pose), pull camera frames, and locate the cause (grasp geometry, hold posture, reach, verdict criterion, stale code on the box). Use when pick_vial or place_in_rack fail repeatedly, an episode is "success" but the object is visibly wrong, the agent loops on a tool, rates drop after a change, or a number needs tuning.
---

# Debug a manipulation primitive

Rule one: numbers and pictures must agree. Every diagnosis here ends with a camera frame of the
state the numbers describe. Rule two: change one thing, re-measure over the same seeds, write the
result into the code comment and the commit message.

## 1. Is the box running your code?

```bash
scripts/remote simteleop-status
scripts/remote 'docker inspect simteleop_sim --format "{{range .Mounts}}{{.Destination}}\n{{end}}" | grep -c simteleop_sim'   # 1 = dev mount active
scripts/remote 'cd /opt/simteleop/code && find sim common -name "*.py" ! -path "*__pycache__*" | sort | xargs cat | md5sum'
cat $(git ls-files sim common | grep '\.py$' | sort) | md5sum                                      # same hash?
```

Different hash or no mount → `scripts/push-code`. (Symptom of stale baked code: `reset` raising
`Inplace update to inference tensor`.)

## 2. Run the probe

`probe.py` next to this file: pick the nearest vial, place in the nearest free slot, print one JSON
line per seed with the stage facts, save a frame per seed.

```bash
B=$(scripts/remote --outputs | awk '/ExchangeBucketName/{print $2}')
aws s3 cp skills/simteleop-debug-manipulation/probe.py s3://$B/tmp/probe.py
scripts/remote --detach probe bash -c "aws s3 cp s3://$B/tmp/probe.py /tmp/probe.py && /home/ubuntu/simteleop/agent-venv/bin/python /tmp/probe.py 1001 1002 1003 1004 1005 1006 1007 1008 && aws s3 sync /tmp/probe/ s3://$B/tmp/probe/"
scripts/remote --wait probe 20                      # ~2.5 min per seed; the last line is "placed k/n (picks n/8)"
aws s3 sync s3://$B/tmp/probe/ outputs/probe/ && ls outputs/probe
```

Each line: `success`, `reason`, `pick.axis_in_gripper` (vial axis in the gripper frame after the
lift: `[~0, ~1, 0]` = along the jaw pivot; a large third component = leaning towards the fingertips
(−) or the body (+)), `tool_axis` at the hold, `tilt_above`/`tilt_release` (vial tilt before
lowering and at release), `final` (tilt, centre z, seated, placed), `wrist_min` (lowest Wrist_Pitch
seen; −95/+95 = pinned), and per stage `[converged, pos_err_m, axis_err_deg, joints_deg]`.

## 3. Read the logs — known signatures

| Signature | Meaning | Where to look |
|---|---|---|
| a stage has `converged: false`, large `axis_err`, and a joint in `joints_deg` sits at its limit (Wrist_Pitch ±95, Elbow −100/90, Pitch ±100) | the posture is unreachable, not "too far": the orientation demand conflicts with the position. SO-101 rule: tool pitch below horizontal ≈ Pitch + Elbow + Wrist_Pitch − 10°. A tool pointing UP by α needs Pitch + Elbow ≤ 85 − α, and low/near targets need Pitch + Elbow large → conflict | the hold orientation (`_hold_orientation`, `upright_hold_orientation`), the grasp tilt / regrip that sets the in-hand angle |
| `axis_in_gripper` third component ≈ −0.5 after the pick | the vial leans 30° towards the fingertips (the 35° grasp tilt). Cap-up then needs tool-up 31° → the signature above. The regrip (`REGRIP_TILT_DEG`) should have brought it to ±0.25 | `pick_vial` regrip stage: did `regrip` converge? did `lift_clear` lift? |
| `in_hand` axis changes between `carry` and `lower` | the vial rotated in the jaw during the descent — its bottom caught the rim. Descend in two stages aimed on the vial's measured centre (as `place_in_rack` does), slow the insert | `descend_vial_to`, `aim_*_offset_m` |
| `success: true` but the frame shows the object leaning / in the wrong place | a verdict criterion is too loose or measures the wrong thing. Compare `final.cz` with a known-good value (a vial standing on the slot floor: centre 0.098 m; on the rim: 0.13–0.15) | `geometry.vial_is_vertical` (15°), `vial_is_seated` (lower end ≥ 4 cm down), `point_in_rack` |
| pick `rose 0–3 of 8 cm`, `still_in_contact: true` | the lift stalled at the reach limit while gripping (not a slip): the lift posture is unreachable; lift in a reachable orientation | `lift_clear`/`lift` joints_deg at limits |
| `jaw closed without touching the vial` | pose read before the approach nudged it, or the depth is wrong (finger in the mat stalls at TCP z 0.049) | `GRASP_DEPTH_M`, re-read before descending |
| the vial slips during a wrist roll or carry | the light squeeze (4° past stall) is the trade-off against bending the wrist; slow the motion rather than squeezing harder | `mo.gripper`, `move_joints` duration |
| `place_tool_axis` z < 0 while the hold should be tool-up (or vice versa) | the stand-up rotation picked the other end of a symmetric object (sign of a near-zero component) | `upright_hold_orientation(prefer_tool_dir=…, keep_end=…)` |
| agent episode fails while the probe succeeds | the agent chose a far slot / another vial, or a tool result confused it: read `runs/<ep>/episode_NNN.json` calls in order, check the prompt | `prompts.py`, tool docstrings |

## 4. Look at it

Frames from the probe (`outputs/probe/<seed>.jpg`) or ad hoc:

```bash
scripts/remote /home/ubuntu/simteleop/agent-venv/bin/python -c "from simteleop_agent.client import EnvClient;c=EnvClient();open('/tmp/f.jpg','wb').write(c.frame('external_D455')[0]);open('/tmp/e.jpg','wb').write(c.frame('ego')[0])"
scripts/remote aws s3 cp /tmp/f.jpg s3://$B/tmp/f.jpg && aws s3 cp s3://$B/tmp/f.jpg outputs/
```

Stills from a recorded episode (the box has no ffmpeg on the host; the container does; write outside
the dataset tree):
`scripts/remote bash -c 'mkdir -p /home/ubuntu/simteleop/stills && docker exec simteleop_sim ffmpeg -v error -y -ss 30 -i /workspace/datasets/<ds>/videos/observation.images.external_D455/chunk-000/file-00N.mp4 -frames:v 1 -update 1 /workspace/datasets/../stills/t30.jpg'`
(`/workspace/datasets` is `/home/ubuntu/simteleop/datasets` on the host, so `/workspace/datasets/../stills`
is `/home/ubuntu/simteleop/stills`; copy it out via the bucket as above).

## 5. Fix, re-measure, record

- Change one constant or one stage. Push with `scripts/push-code`. Re-run the probe on the same seeds.
- Report `k/n` before and after, with the failure modes that remain.
- Write the measurement into the constant's comment (date, what was tried, numbers) and into
  the commit message. Add a geometry unit test if a rule changed. Do not loosen a verdict to raise a
  rate.

## Reference values (SO-101 workshop scene)

Vial 11.65 cm × Ø3.4 cm, cap = local +z; rack slots Ø4.5 cm, 10 cm deep, entry z 0.141 world (rack
top); standing vial centre z 0.098 (sticks out 1.65 cm); vial centre on the mat z 0.05–0.057;
TCP offset (0.01, 0.01, −0.09) in the gripper frame; tool axis −z; jaw pivot y; Wrist_Roll ±160°,
Wrist_Pitch ±95°, Elbow −100…90°, Pitch ±100°, Rotation ±110°.
