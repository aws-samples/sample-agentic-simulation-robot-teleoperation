# Agent teleoperation

The agent is the operator. It gets the same information a human at a leader arm would have (the two
camera views) plus what a human would not (ground-truth poses), and drives the arm through tools.

## Tools

| Tool | Kind | Verdict |
|---|---|---|
| `describe_scene` | perception (ground truth) | — |
| `look(camera)` | perception (image) | — |
| `measure_vial(vial)` | perception | — |
| `pick_vial(vial)` | verified primitive | contact on the vial, centre rose > 5 cm and followed ≥ 70% of the lift |
| `place_in_rack(slot?)` | verified primitive | vial within 15° of vertical AND its lower end ≥ 4 cm down a slot (seated), not in the jaw; returns an overview frame with the verdict. Refuses to release a vial tilted > 15° or > 1 cm off the slot ("still held") |
| `home()` | primitive | joints within 0.03 rad |
| `move_gripper_to(x,y,z,approach)` | low-level (differential IK) | residual position / axis error |
| `jog_joints(...)` | low-level | joint error |
| `gripper(open/close)` | low-level | stall + contact |
| `reset_scene(seed?)`, `recording(start/stop/cancel/status)` | episode control | — |

## How the primitives move (what the agent is relying on)

`pick_vial`: open, hover 8 cm above the vial with the tool tilted 35° forward (reach), descend, close
until the jaw stalls (light squeeze), then — with the vial still on the mat — turn the jaw around it
to a 12° tilt so the vial sits nearly along the jaw pivot, lift 1.5 cm, finish the lift turning back
to 35°. `place_in_rack`: stand the vial up cap-up (tool slightly up), pan the base to the slot,
carry in-plane, descend in two stages aiming on the vial's measured centre, insert 2.5 cm, gate on
tilt and centring, release, retreat. The measurements behind every number are in the comments next
to the constants in `sim/simteleop_sim/primitives.py`.

## Running episodes

`simteleop-agent --episodes N --out DIR` runs N episodes unattended: reset with seed `1000+k`, fresh
agent context, task string, then an **independent ground-truth verdict** (a vial that was on the mat
now stands in the rack — within 15° of vertical and seated — and nothing is held) regardless of what the agent claimed; if the agent forgot
to stop the recording the runner stops it with that verdict. Per-episode JSON (tool calls, final
text, verdict) plus `summary.json` land in DIR; the dataset accumulates under
`/home/ubuntu/simteleop/datasets/<name>/` with a manifest per episode in `manifests/`.

Failed episodes are recorded too (the manifest says `success: false`); filter on the manifest when
building a training set. A budget: expect an episode to take 3–8 minutes wall clock with Astra.

## Model

`SIMTELEOP_AGENT_MODEL` (default `global.openai.gpt-6-astra`). Anthropic models work without the image
hoisting wrapper; the wrapper is selected automatically from the model id.
