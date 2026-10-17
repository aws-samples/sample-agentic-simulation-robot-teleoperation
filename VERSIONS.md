# Pinned versions

| Component | Version | Where pinned |
|---|---|---|
| Sim-to-Real SO-101 workshop | commit `ce807d99724cb65671abec01f908a2fcb4a6eab7` (2026-07-20, "Expose calibration scripts in teleop container") | `cdk.json` → `/etc/simteleop.env` → `50_workshop_checkout.sh` |
| Isaac Lab base image | `nvcr.io/nvidia/isaac-lab:2.3.2` | the workshop's `docker/sim/Dockerfile` (not ours) |
| LeRobot | commit `e670ac5daf9b76` | same Dockerfile |
| Python in the sim container | Isaac Lab's (`/workspace/isaaclab/_isaac_sim/python.sh`, 3.11) | — |
| strands-agents | 1.57.1 | `agent/pyproject.toml` |
| boto3 | ≥ 1.40 | `agent/pyproject.toml` |
| aws-cdk-lib | 2.180.0 | `infrastructure/requirements.txt` |
| Default model | `global.openai.gpt-6-astra` (Bedrock inference profile) | `cdk.json` `agentModelId`; runtime `SIMTELEOP_AGENT_MODEL` |
| AMI | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) **20260623** (NVIDIA driver 580.159.04) — later builds carry the 595 driver, on which Isaac Sim 5.1 crashes at startup (measured 2026-10-08) | `app.py` `amiNameFilter`; override `-c amiNameFilter=`/`-c amiId=` |
| Instance | `g7e.4xlarge` (RTX PRO 6000 Blackwell) — the workshop lists Blackwell and Ada as tested; `g6e.4xlarge` (L40S) verified 2026-10-08 in ap-northeast-2 | `cdk.json`, `-c instanceType=` |

Moving the workshop pin: change `workshopCommit`, redeploy or edit `/etc/simteleop.env`, re-run
`sudo bash /opt/simteleop/code/bootstrap/50_workshop_checkout.sh` and `FORCE=1 sudo simteleop-build-image`.
Check that `LeRobotSO101Interface.SO101_USD_MAPPING` and the task/camera names in
`tasks/task_env_cfg.py` are unchanged — `common/simteleop_common/protocol.py` copies the former and
`sim/simteleop_sim/scene_state.py` relies on the latter.
