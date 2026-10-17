---
name: simteleop-operate
description: Operate the simulated teleoperation platform from a workstation over SSM — start the GPU box, bring the Isaac Lab env server up, run GPT-Astra agent episodes (or scripted ones), publish the LeRobot dataset to S3, pull results, stop the instance. Use when asked to run, record, batch, publish or stop anything on the platform, or when a remote command returns nothing, a job seems stuck, or the env server reports "busy".
---

# Operate the platform

Everything runs on the EC2 box; you drive it from the repository root with `scripts/remote`, which
resolves the instance from the CDK stack outputs (`SimTeleopPlatform`, or `SIMTELEOP_STACK`) and
runs commands over SSM. You need AWS credentials for the account and `AWS_REGION`.

## 0. Preconditions

```bash
scripts/remote --outputs                 # InstanceId, DcvUrl, ExchangeBucketName, SsmSessionCommand
aws ec2 describe-instances --instance-ids <InstanceId> --query 'Reservations[].Instances[].State.Name' --output text
```

- `stopped` → `scripts/remote --start` (waits until SSM is online, ~1 min).
- First deployment: `scripts/remote grep -c "the platform is ready" /var/log/simteleop-bootstrap.log`
  must print `1` (the line sits above a usage banner, so `tail` does not show it). First boot takes
  16–60 min, the sim image build most of it.
- No stack (a box set up by hand): export `SIMTELEOP_INSTANCE=i-…` and `SIMTELEOP_BUCKET=…`.

## 1. Bring the stack up

```bash
scripts/remote simteleop-status          # already up? "env server: task ... step N, busy False", 3 vials, slots -> go to step 2
scripts/remote simteleop-up              # otherwise: headless, recording to datasets/agent_vials_to_rack
```

A running env server can be reused as it is: the dataset stays open and the next episode appends to
it (`recording.episodes_saved` tells you how many it holds; a vial left in the rack from an earlier
episode shows as a slot that is not free — `reset` at the start of each episode clears it). Only
`simteleop-down` + `simteleop-up` when you need another `--dataset`/`--task` or the server is unhealthy.

`simteleop-up` returns when the env server answers (≈40 s warm, up to 10 min after a reboot while
shaders compile). Options: `--gui` (Isaac Sim window on the DCV desktop; slower), `--no-record`,
`--dataset NAME` (a fresh dataset directory; the recorder cannot reopen an episode-less one),
`--task Lerobot-So101-Teleop-Vials-To-Rack-DR` (domain-randomised variant). If the code on the box
is older than your checkout, `scripts/push-code` first (it restarts the server).

## 2. Run episodes

Agent episodes are long (3–5 min each with Astra); always detach:

```bash
scripts/remote simteleop-agent --check                                   # 3 OK lines; Bedrock answers OK
scripts/remote --detach ep1 simteleop-agent --episodes 5 --out /home/ubuntu/simteleop/runs/ep1
scripts/remote --log ep1                                                 # progress; "=== episode 1/5 (seed 1000) ===" per episode (seed 1000 + k, k from 0)
scripts/remote --wait ep1                                                # blocks; ends with "RESULT: k/n episodes succeeded by ground truth"
```

Options: `--model <bedrock id>` (or `SIMTELEOP_AGENT_MODEL`), `--wall 600` per-episode budget,
a task string as positional argument (default: `DEFAULT_TASK` in `agent/simteleop_agent/__main__.py`;
the dataset's task label is the env server's fixed `--task-name`, not this string).
`scripts/remote` builds one argv: for pipes or several commands use `scripts/remote bash -c '…'`. One interactive task without recording a verdict file:
`scripts/remote simteleop-agent "put a vial in the rack"` (fits in the 10-min SSM limit for one task).

Scripted episodes (no model, for measuring the primitives) — see `simteleop-debug-manipulation`.

## 3. Read the results

```bash
scripts/remote cat /home/ubuntu/simteleop/runs/ep1/summary.json
scripts/remote bash -c 'M=$(ls /home/ubuntu/simteleop/datasets/agent_vials_to_rack/manifests/*.json | tail -1); python3 -c "import json;m=json.load(open(\"$M\"));print(m[\"verdict\"]);print([(e[\"op\"], e[\"verdict\"].get(\"success\"), e[\"verdict\"].get(\"tilt_deg\")) for e in m[\"events\"] if \"verdict\" in e])"'
```

- `runs/<name>/episode_NNN.json`: the agent's tool calls (arguments and results), final text,
  `ground_truth` verdict, `seed`. Numbered from 0 per run; `wall_seconds` includes the model's
  thinking outside the recording window.
- `datasets/<name>/manifests/episode_NNNNNN.json`: frames, wall seconds (recording window only),
  verdict, every primitive call with its stage logs. Numbered from the dataset's total, so a second
  run's first episode is `episode_000001.json`.
- A placement is real only if the place verdict has `placed: true` (`tilt_deg` < 15, `seated: true`).
  Look at a frame when in doubt: `look`-equivalent from the box's venv —
  `scripts/remote /home/ubuntu/simteleop/agent-venv/bin/python -c "from simteleop_agent.client import EnvClient;open('/tmp/f.jpg','wb').write(EnvClient().frame('external_D455')[0])"`
  then, with `B=$(scripts/remote --outputs | awk '/ExchangeBucketName/{print $2}')`:
  `scripts/remote aws s3 cp /tmp/f.jpg s3://$B/tmp/f.jpg && aws s3 cp s3://$B/tmp/f.jpg outputs/`.

## 4. Publish and pull

```bash
scripts/remote simteleop-publish agent_vials_to_rack /home/ubuntu/simteleop/runs/ep1   # dataset + run logs -> s3://<bucket>/datasets/, runs/
scripts/pull-results                                     # list
scripts/pull-results agent_vials_to_rack ep1             # -> ./outputs/datasets/..., ./outputs/runs/...
```

The recorder encodes video asynchronously after each episode (~1 min per 3000 frames); if the last
`file-NNN.mp4` is missing, wait and publish again. Videos are AV1: `ffmpeg`/VLC play them.

## 5. Stop

```bash
scripts/remote simteleop-down            # waits for the recorder queue, stops the container
scripts/remote --stop                    # the GPU instance is billed while running; data and Elastic IP stay
```

## Failure modes

| Symptom | Cause / fix |
|---|---|
| `scripts/remote` prints "stack not found" | wrong region/account or custom stack name: `AWS_REGION`, `SIMTELEOP_STACK`, or `SIMTELEOP_INSTANCE` |
| SSM command returns no output, `Status: Failed`, exit 1 with nothing | the box's root disk is full (`scripts/remote df -h /`) — each failed episode leaves ~100 MB; or the command exceeded the 10-min limit — detach it |
| `simteleop-status`: "not running" right after `simteleop-up` | `scripts/remote docker logs --tail 50 simteleop_sim`; a `RuntimeError: Inplace update to inference tensor` on reset means the baked image code is running — use `simteleop-up` without `--image-code` (dev mode) |
| `error: busy` from the env server | a blocking op is in flight (another agent or probe): wait, or `scripts/remote pgrep -af simteleop_agent` |
| agent `--check` fails on Bedrock with an SCP deny | the account blocks Bedrock for EC2; `-c enableAgentBedrock` is set but an org policy wins. Everything but the agent still works. |
| episodes all `DECLINED`/`still held` at the same stage | a primitive regression: switch to `simteleop-debug-manipulation` before running more |
| detached job vanished | `scripts/remote systemctl status simteleop-<name>`; the unit is transient and `--collect`ed when done; its log stays in `/var/log/simteleop-<name>.log` |
