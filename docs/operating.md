# Operating the platform

Everything runs on the GPU instance. From your workstation you drive it over AWS Systems Manager
(SSM) with three scripts; on the instance, a handful of `simteleop-*` commands wrap the container,
the environment server and the agent. This page is the reference for both, for the agent's command
line and knobs, for the scripted (no-agent) path, and for what the platform writes.

## Contents

- [From your workstation](#from-your-workstation)
- [On the instance](#on-the-instance)
- [Running the agent](#running-the-agent)
- [Scripted control (no agent)](#scripted-control-no-agent)
- [Outputs](#outputs)
- [Verifying a deployment](#verifying-a-deployment)

## From your workstation

The scripts in `scripts/` need the AWS CLI, credentials for the account and `AWS_REGION` (or the
CLI's default region). They resolve the instance and bucket from the stack's outputs;
`SIMTELEOP_STACK` selects a stack with a custom name, and `SIMTELEOP_INSTANCE` / `SIMTELEOP_BUCKET`
target an instance that the stack did not create.

| Command | Does |
|---|---|
| `scripts/remote <cmd ...>` | run a command on the instance as root, wait, print the output (10 min limit); use `scripts/remote bash -c '…'` for pipes or several commands |
| `scripts/remote --detach <name> <cmd ...>` | start a long job as `ubuntu` in a transient systemd unit, logging to `/var/log/simteleop-<name>.log` |
| `scripts/remote --log <name>` / `--wait <name>` | tail the job's log / block until it ends |
| `scripts/remote --start` / `--stop` / `--outputs` | instance power; the stack's outputs |
| `scripts/push-code` | upload this checkout's `common/ sim/ agent/ docs/` and the operator commands to the bucket and the instance, reinstall the agent package, restart the environment server |
| `scripts/pull-results [dataset] [run-prefix*]` | list the bucket, or download a dataset and run logs to `./outputs/` |

Interactive shell: `aws ssm start-session --target <InstanceId>` (the `SsmSessionCommand` output).
Desktop: the `DcvUrl` output, user `ubuntu` (set a password once with `sudo passwd ubuntu`), then
`simteleop-up --gui` to put the Isaac Sim window on it.

## On the instance

The commands live in `/usr/local/bin` (source: `infrastructure/assets/bin`) and read
`/etc/simteleop.env`, which the stack writes.

| Command | Does |
|---|---|
| `simteleop-up [--gui] [--no-record] [--task T] [--dataset NAME] [--image-code]` | start the simulator container and the environment server, headless by default; records to `datasets/<NAME>` (default `agent_vials_to_rack`); `--gui` shows the Isaac Sim window on the DCV desktop; the code is mounted from `/opt/simteleop/code` unless `--image-code` |
| `simteleop-status` | container state, environment server step and busy flag, tool axis, vials and slots, recording status |
| `simteleop-agent [--check] ["task"] [--episodes N --out DIR] [--model ID] [--wall S]` | the agent (next section) |
| `simteleop-publish <dataset> [runs-dir]` | sync the dataset (and run logs) to the stack's bucket |
| `simteleop-down` | stop the container (waits for the recorder's write queue) |
| `simteleop-build-image`, `simteleop-fix-display` | rebuild the simulator image (`FORCE=1`), repair the DCV X authority |

The first `simteleop-up` after a boot takes a few minutes (shader compilation); later ones about a
minute.

## Running the agent

`simteleop-agent` runs the Strands agent on the instance with the instance role's Bedrock access.

```bash
simteleop-agent --check                                   # preflight: dependencies, environment server, one model call
simteleop-agent "put a vial in the rack"                  # one task; transcript on stdout, no verdict file
simteleop-agent --episodes 5 --out runs/ep1               # unattended batch, recorded, ground-truth verdicts
simteleop-agent --episodes 5 --model eu.anthropic.claude-sonnet-4-5-20250929-v1:0   # another Bedrock model
```

An episode (`--episodes`): the runner resets the scene with seed `1000 + k` (the first episode is
seed 1000), builds a fresh agent with no memory of earlier episodes, gives it the task, lets it work
for up to `--wall` seconds (default 900, or `SIMTELEOP_WALL_S`), then judges from ground truth whether
a vial that was on the mat now stands in the rack and nothing is held — regardless of what the agent
reported. If the agent forgot to stop the recording, the runner stops it with that verdict. Per
episode, `--out` receives a JSON file with every tool call (arguments and results), the model's final
text and the verdict, plus a `summary.json` for the batch.

What the agent can do (`agent/simteleop_agent/tools.py`; verdict criteria and motion details in
[`agent_teleoperation.md`](./agent_teleoperation.md)):

| Tool | Kind |
|---|---|
| `describe_scene()`, `measure_vial(v)` | ground truth: joints, gripper, every vial's centre, tilt and standing-in-rack flag, slots and occupancy, recording status |
| `look(camera)` | a JPEG from `external_D455` (overview) or `ego` (gripper) |
| `pick_vial(v)`, `place_in_rack(slot?)`, `home()` | verified primitives with measured verdicts; `place_in_rack` also returns an overview frame |
| `move_gripper_to(x,y,z,approach)`, `jog_joints(...)`, `gripper(open/close)` | low-level teleoperation |
| `reset_scene(seed?)`, `recording(start/stop/cancel/status)` | episode control |

Knobs (environment variables on the instance; `/etc/simteleop.env` is sourced by the operator
commands):

| Variable | Default | Effect |
|---|---|---|
| `SIMTELEOP_AGENT_MODEL` | `global.openai.gpt-6-astra` (from the `agentModelId` context key) | Bedrock model id; Anthropic ids skip the image-hoisting wrapper |
| `SIMTELEOP_WALL_S` | 900 | per-episode wall-clock budget |
| `SIMTELEOP_GRASP_TILT_DEG` | 35 | grasp approach tilt (reach at the far vials) |
| `SIMTELEOP_REGRIP_TILT_DEG` | 12 | tilt the jaw is turned to around the vial on the mat before lifting (what makes the cap-up hold reachable) |
| `SIMTELEOP_RELEASE_MAX_TILT_DEG` | 15 | the place primitive refuses to let go of a vial tilted more than this |
| `SIMTELEOP_TOOL_AXIS`, `SIMTELEOP_TCP_OFFSET` | `-z`, `0.01,0.01,-0.09` | gripper geometry (another embodiment) |
| `SIMTELEOP_ENV_HOST`, `SIMTELEOP_ENV_PORT` | `127.0.0.1`, `7010` | where the agent finds the environment server |

The system prompt is `agent/simteleop_agent/prompts.py`. The task the agent is given is the
positional argument (default `DEFAULT_TASK` in `__main__.py`: "Pick up a vial and stand it in a free
slot of the rack. Record the episode."). The dataset's task label is a different, fixed string set by
the environment server (`--task-name`, "Pick up the vial and place it in the rack"), so every episode
of a dataset carries the same label whatever the agent was told. To change what the agent may do,
edit the tool lists at the end of `tools.py` (`PERCEPTION_TOOLS`, `PRIMITIVE_TOOLS`, `TELEOP_TOOLS`,
`EPISODE_TOOLS`); `build_agent` takes `ALL_TOOLS` by default.

### Agent or script?

Both paths drive the same environment server and record the same dataset; they differ in who
decides.

| | Agent (`simteleop-agent`) | Scripted (`EnvClient`, `probe.py`) |
|---|---|---|
| Decides which vial, which slot, when to retry or give up | the model, from `describe_scene`, pictures and the verdicts it gets back | your script, deterministically (for example nearest vial to nearest free slot) |
| Motions | the same verified primitives and low-level tools; the agent cannot change their criteria | the same primitives, called directly |
| Verdict | measured server-side; the runner re-judges each episode from ground truth | measured server-side; you read it |
| Good for | collecting data with an operator that handles the unexpected (declined grasp, slot occupied, vial rolled) and leaves a reasoning trace; testing a model as a teleoperator | measuring the primitives (success rates over seeds), regression tests, bulk collection when the layout is known |
| Time per episode | about 3 minutes (most of it model latency) | about 75 s (pick and place), no model |
| Variability | the model may look, re-check, retry, pick a far slot — useful behaviour, but not repeatable turn by turn | identical given the seed |
| Needs | Bedrock access on the instance role, `--check` green | nothing beyond the environment server |
| Records | the dataset plus `runs/<name>/episode_NNN.json` (every tool call, the model's text, the verdict) | the dataset plus whatever your script prints |

Rule of thumb: tune and measure with the scripted path (`skills/simteleop-debug-manipulation`);
collect with the agent once the primitives' rates are known, so the agent's choices are the only
variable.

## Scripted control (no agent)

The environment server speaks one JSON object per line over TCP `127.0.0.1:7010`
([`protocol.md`](./protocol.md)); `agent/simteleop_agent/client.py` wraps it. From the instance's
agent virtualenv (`/home/ubuntu/simteleop/agent-venv/bin/python`):

```python
from simteleop_agent.client import EnvClient
c = EnvClient()
c.reset(seed=1001); c.wait_idle(); c.home(); c.wait_idle()
s = c.state()                                   # vials, slots, joints, recording ...
c.record("start")
p = c.pick_vial("vial_2"); c.wait_idle()        # {"success": true, "rose_m": 0.078, ...}
q = c.place_in_rack(); c.wait_idle()            # {"success": true, "tilt_deg": 2.4, "seated": true, ...}
c.record("stop", verdict={"success": q["success"], "note": "scripted"})
jpeg, meta = c.frame("external_D455")           # a picture of the result
```

Every blocking operation returns the full log of its stages (IK convergence, joint angles, in-hand
vial axis), which is how the primitives were tuned; the constants in `sim/simteleop_sim/primitives.py`
carry the measurements that chose them.

## Outputs

On the instance under `/home/ubuntu/simteleop/`:

- `datasets/<name>/` — the LeRobot v3 dataset (`meta/info.json`, `data/chunk-000/*.parquet`,
  `videos/observation.images.*/chunk-000/*.mp4`) plus `manifests/episode_NNNNNN.json`: frames, wall
  time, the ground-truth verdict and every primitive call with its full result. All episodes are
  recorded, failed ones too; filter on the manifest's `verdict.success` when building a training set.
- `runs/<name>/episode_NNN.json`, `summary.json` — the agent's side: tool calls, final text, verdict.

Numbering: run files count from 0 per run (`episode_000.json`, `"episode": 0`, log line
`episode 1/1`); dataset episodes and manifests count from the dataset's total, so the first episode
of a second run on the same dataset is `episode_000001.json`. The run's `wall_seconds` includes the
agent's thinking before `recording('start')` and after `stop`; the manifest's covers the recording.

`simteleop-publish` syncs both to `s3://<ExchangeBucketName>/datasets/<name>/` and
`runs/<name>-<stamp>/`; `scripts/pull-results` brings them to `./outputs/`. The videos are AV1 (the
workshop's encoder); `ffmpeg` and VLC play them, many OpenCV builds do not.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset       # a LeRobot with v3.0 dataset support (the workshop pins one)
ds = LeRobotDataset("local/agent_vials_to_rack", root="outputs/datasets/agent_vials_to_rack")
```

## Verifying a deployment

| Step | Pass criterion |
|---|---|
| `scripts/remote grep -c "the platform is ready" /var/log/simteleop-bootstrap.log` | `1` (the line is followed by a usage banner, so `tail` does not show it) |
| `scripts/remote simteleop-up` | "ready after N s"; `simteleop-status` lists 3 vials and 4 slots (all free after a fresh start) |
| `scripts/remote simteleop-agent --check` | three OK lines; the Bedrock line names the model and answers `OK` |
| `scripts/remote --detach ep1 simteleop-agent --episodes 1 --out /home/ubuntu/simteleop/runs/ep1`, then `--wait ep1` | `RESULT: 1/1 episodes succeeded by ground truth`; the manifest's place verdict has `tilt_deg` < 15 and `seated: true` |
| `scripts/remote simteleop-publish agent_vials_to_rack` | `UPLOADED s3://…/datasets/agent_vials_to_rack/` |
| `python -m pytest -q` at the repository root | unit tests pass (protocol, geometry, tool text, solution identity; no simulator needed) |
