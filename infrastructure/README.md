# Infrastructure

One CDK stack (`infra/simteleop_stack.py`): VPC with public subnets in every AZ (no NAT), an encrypted S3
exchange bucket, an instance role (SSM, the bucket, the DCV licence bucket, Bedrock invoke), an
operator role (SSM on this instance, start/stop, the bucket), a security group with no ingress, and
one GPU instance whose user data pulls this repository's code from the bucket and runs the bootstrap.

## Deploy

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
cdk bootstrap                       # once per account/region
cdk deploy -c dcvAllowedCidr=$(curl -s https://checkip.amazonaws.com)/32 \
           -c operatorPrincipalArns=arn:aws:iam::<account>:role/<the role you operate from>
```

`operatorPrincipalArns` is required: the operator role is root on the box over SSM, so its trust
policy names the principals (comma-separated or a JSON list). `-c allowAccountRootOperator=true`
trusts every principal in the account instead — only for an account nobody else uses.

Context knobs (`-c key=value` or `cdk.json`): `instanceType` (default `g7e.4xlarge`; `g6e.4xlarge`
for Ada), `availabilityZone` (retry a capacity shortage), `rootVolumeGb` (400), `amiId`,
`amiNameFilter` (pinned to the 20260623 DLAMI build, the last with the NVIDIA 580 driver; see
`VERSIONS.md`), `dcvAllowedCidr`, `operatorPrincipalArns`, `workshopCommit`, `agentModelId`,
`enableAgentBedrock`, `retainDataOnDestroy`, `allowAccountRootOperator`.

The instance role is prefix-scoped on the bucket: read on `code/` (what user data executes at boot),
read/write on `datasets/`, `runs/`, `tmp/`; the bucket is versioned. `scripts/push-code` uploads code
with the caller's own credentials, not the instance's.

Typical timings on a `g6e.4xlarge`: `cdk deploy` under 5 minutes, first boot 16–60 minutes to "the
platform is ready" (the simulator image build dominates), the first `simteleop-up` about 3 minutes
(shader compilation), an agent episode about 3 minutes.

## First boot

`assets/bootstrap/00_bootstrap.sh` runs numbered, idempotent steps and logs to
`/var/log/simteleop-bootstrap.log`:

| Step | Does | Time |
|---|---|---|
| 10_system | packages, desktop for DCV, limits, `/home/ubuntu/simteleop` | 3 min |
| 20_dcv, 25_dcv_permissions | NICE DCV server, virtual session `simteleop`, xauth readable for the container, boot-time recreation unit | 2 min |
| 30_docker_gpu | Docker + NVIDIA container toolkit, GPU passthrough test | 1 min |
| 40_operator_commands | `simteleop-*` on PATH | — |
| 50_workshop_checkout | clone the workshop at the pinned commit | 1 min (300 MB) |
| 60_sim_image | `docker build` the workshop's sim image + our layer (public NGC base) | 30–40 min |
| 70_agent_venv | Strands + boto3 + our agent package in `/home/ubuntu/simteleop/agent-venv` | 1 min |

Re-run after fixing a step: `sudo bash /opt/simteleop/code/bootstrap/00_bootstrap.sh`. Rebuild the image:
`FORCE=1 sudo simteleop-build-image`.

## Updating the code on a running box

From the repository root, with credentials for the account: `scripts/push-code`. It syncs `common/
sim/ agent/ docs/` and `infrastructure/assets/{bin,bootstrap}` to `s3://<bucket>/code/` (the same
layout the first boot pulls), syncs them onto the box, reinstalls the agent package into its venv,
refreshes the `simteleop-*` commands and restarts the env server. The sim container mounts
`/opt/simteleop/code` over the image's copy (`simteleop-up` default; `--image-code` to run the baked
copy), so sim-side changes need no image rebuild. Rebuild the image only when the workshop pin or the
sim Dockerfile layer changes: `FORCE=1 sudo simteleop-build-image`. `cdk deploy` uploads the same
code as part of the stack (the `Code*` bucket deployments).

## Cost

The instance is billed while running; `scripts/remote --stop` (or `aws ec2 stop-instances`) when idle —
the Elastic IP and the data are kept. The instance is the expensive part; the bucket and roles are
negligible. Prices and measured usage: [`docs/costs.md`](../docs/costs.md).
