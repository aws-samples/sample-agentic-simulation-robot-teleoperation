# AGENTS.md — working in this repository with a coding agent

This file is for the coding agent (Kiro, Claude Code, Codex, Cursor…) that a user points at this
repository. It says what the project is, what must not be broken, how to verify a change, and where
the step-by-step skills live. Read it before editing; follow the skills in [`skills/`](./skills/README.md)
for the common jobs.

## What this is

A platform that records robot-learning datasets in simulation with an LLM agent as the operator.
One GPU EC2 box runs Isaac Lab with NVIDIA's SO-101 "vials to rack" scene; an **env server** inside
the sim container exposes ground truth and verified manipulation primitives over a JSON-lines TCP
protocol; a **Strands agent** (Bedrock, GPT-6 Astra by default) calls them as tools; the workshop's
**LeRobot recorder** writes the dataset. Users deploy with CDK and operate the box from their
workstation through `scripts/remote` (SSM). `README.md` is the front page and `docs/operating.md` the user guide; the measurements that
shaped the primitives live next to the constants they chose (code comments) and in commit messages.

## Map

| Path | What | Runs where | Tests |
|---|---|---|---|
| `common/simteleop_common/protocol.py` | ops, snapshot schema, joint names/ranges, LeRobot unit mapping | both sides | `tests/test_protocol.py` |
| `common/simteleop_common/solution.py` + `solution.json` | AWS Solution identity: stack `Description` and SDK `User-Agent` suffix; the JSON is the only place the id/version live | both sides | `tests/test_solution.py` |
| `common/simteleop_common/geometry.py` | quaternions (w,x,y,z), vial/rack geometry, grasp and hold poses, placement criteria | both sides | `tests/test_geometry.py` |
| `sim/simteleop_sim/env_server.py` | TCP server, request dispatch, 60 Hz step loop, recorder hooks | sim container (Isaac Lab python 3.11) | on the box |
| `sim/simteleop_sim/scene_state.py` | ground-truth reader (vials, rack slots from the USD, contact sensor, TCP), **robot-specific** | sim container | on the box |
| `sim/simteleop_sim/motion.py` | motions as generators yielding joint targets: joint interpolation, differential IK servo, jaw, base pan | sim container | on the box |
| `sim/simteleop_sim/primitives.py` | `home`, `pick_vial`, `place_in_rack` with measured verdicts and stage logs, **robot-specific** | sim container | on the box |
| `sim/simteleop_sim/recording.py` | wraps the workshop `LeRobotRecorder`; manifests | sim container | on the box |
| `agent/simteleop_agent/tools.py` | Strands tools (perception, primitives, low-level, episode control), scene text | host venv (python 3.12+) | `tests/test_tools_text.py` |
| `agent/simteleop_agent/__main__.py` | CLI, episode runner, independent ground-truth verdict | host venv | on the box |
| `agent/simteleop_agent/prompts.py`, `model_compat.py`, `preflight.py`, `client.py` | system prompt, Astra image hoisting, `--check`, protocol client | host venv | — |
| `infrastructure/` | CDK stack, first-boot bootstrap steps, `simteleop-*` operator commands | workstation / box | `cdk synth` |
| `scripts/` | `remote`, `push-code`, `pull-results` | workstation | manual |

## Invariants — do not break these

1. **Ground truth decides success; the agent cannot loosen it.** Verdicts are computed in
   `primitives.py`/`scene_state.py` from simulator state (contact force, vial centre, tilt, lower-end
   depth). The runner (`__main__.py: ground_truth_verdict`) re-judges every episode independently of
   the agent's claim. Never make a verdict depend on the agent's report, and never relax a criterion
   to make a run pass — tighten the primitive instead. The current placement criterion is
   `placed = tilt < 15° AND lower end ≥ 4 cm down a slot` (`geometry.VERTICAL_COS_MIN`,
   `SEATED_BELOW_ENTRY_M`); it was tightened on 2026-10-07 after the workshop's 45° criterion passed
   vials leaning on the rim.
2. **Measure, then change.** Every numeric constant in the sim code carries a comment with the
   measurement that chose it (date, what was tried, what happened). Keep that habit: when you change
   a number, run the scripted probe (skill `simteleop-debug-manipulation`) and record the before/after
   in the comment and in the commit message. No "should work" tuning.
3. **Nothing from the upstream workshop is copied.** `VERSIONS.md` pins its commit; the bootstrap
   clones it; the sim image is its Dockerfile plus our layer. Robot asset, tasks, recorder are used as
   they are. If you need something from it, import it or pin a new commit — do not vendor files.
4. **The dataset format is the workshop's.** Actions are commanded joint targets, states measured
   joints, both in LeRobot units via `protocol.py`'s copy of `SO101_USD_MAPPING`. Do not add or rename
   features; downstream training scripts are the workshop's.
5. **Robot-specific code stays in `scene_state.py`, `primitives.py` and the joint tables in
   `protocol.py`.** `motion.py`, `env_server.py`, `recording.py`, the agent and the protocol are
   embodiment-agnostic; keep them that way (see skill `simteleop-new-embodiment`).
6. **Conventions.** Quaternions `(w, x, y, z)`; positions in metres, world frame, robot base at the
   origin, +x forward, +z up; joint vectors in `protocol.JOINT_NAMES` order; angles in radians in
   code and degrees in logs/docs; the gripper's tool axis is local `-z`, the jaw pivot local `y`,
   the TCP `(0.01, 0.01, -0.09)` in the gripper frame. The vial's cap is its local `+z` end.
7. **Motions are generators** that `yield` joint-target tensors once per sim step and `return` a
   dict verdict; the server steps the sim on the main thread. Never call the env or `torch` from
   another thread; never block inside a motion.
8. **Security posture.** No inbound rules by default, SSM for shell, env server bound to
   `127.0.0.1`, instance role scoped to its bucket + Bedrock invoke. Do not add ports, public
   buckets or broader IAM without saying so to the user.
9. **Datasets and credentials are not committed.** `.gitignore` covers `datasets/`, `outputs/`,
   `workshop/`, `*.tgz`, `*.log`, venvs. Secrets never go in code or docs.
10. **Adding an AWS SDK call?** Build the client with `simteleop_common.solution.get_client()` /
    `get_resource()` (or pass `solution_config()` as the botocore `Config`), never bare `boto3.client`.
    A bare client works and is simply never attributed, so nothing fails — which is why it is easy to
    miss. The id and version are written once, in `solution.json`; `tests/test_solution.py` sweeps the
    tree for a second copy.

## How to verify a change

| Change | Verification |
|---|---|
| `common/`, `agent/` text or logic | `python -m pytest -q` from the repo root (no simulator needed; `tests/conftest.py` shims `strands`). Add a test when you add a geometric rule or a tool text. |
| `sim/` | `scripts/push-code` (restarts the env server in dev mode), then the scripted probe over seeds 1001–1008 (skill `simteleop-debug-manipulation`) and read the stage logs; look at a camera frame of the result — numbers and pictures must agree. |
| Agent behaviour (`prompts.py`, tools) | `scripts/remote --detach ep simteleop-agent --episodes 3 --out /home/ubuntu/simteleop/runs/ep`, then `--wait ep`; read `runs/ep/episode_*.json` tool traces and the ground-truth verdicts. |
| `infrastructure/` | `cd infrastructure && cdk synth`; a real `cdk deploy` only with the user's go-ahead (it creates billable resources). |
| Operator commands (`infrastructure/assets/bin`) | `scripts/push-code` installs them; run the changed command on the box. |

Report rates as `k/n` with the seeds used; wall-clock per episode; what was not verified.

## Typical workflows (each has a skill)

- Operate: bring the stack up, run agent episodes, publish, pull results, stop the instance →
  `skills/simteleop-operate/SKILL.md`.
- Add or change a manipulation primitive and expose it as an agent tool →
  `skills/simteleop-add-primitive/SKILL.md`.
- Diagnose a primitive that fails or a verdict that disagrees with the picture →
  `skills/simteleop-debug-manipulation/SKILL.md`.
- Port the platform to another robot or scene → `skills/simteleop-new-embodiment/SKILL.md`.

## Things that look wrong but are deliberate

- `simteleop-up` mounts `/opt/simteleop/code` over the sim package baked into the image (dev mode is
  the default). The image is rebuilt only for workshop-pin or Dockerfile changes.
- `place_in_rack` has a `cap_up=False` mode (180° wrist roll) that stands the vial inverted. It is
  kept as the reachability fallback and for comparison, not as a default.
- The agent's `place_in_rack` tool returns an image block; `model_compat.py` moves it out of the
  tool result for GPT-6 Astra, which rejects images there. Anthropic models take it as is.
- `pick_vial` turns the jaw around the vial on the mat (regrip) before lifting. It looks like a
  wasted motion; it is what makes the cap-up hold reachable (31° → 4–16° in-hand offset).
- Episodes that fail are recorded too, with `success: false` in the manifest. The consumer filters.

## Cost and safety reminders for the agent

- The GPU instance costs money while running: after a session, `scripts/remote --stop` unless the
  user says otherwise. Never `cdk destroy` or delete S3 data without explicit instruction.
- Long jobs on the box must be detached (`scripts/remote --detach`); plain SSM commands are killed
  after their timeout and take the job with them.
- One env server per box; `simteleop-down` before `simteleop-up`. A blocking op while another runs is
  refused with `busy`.
