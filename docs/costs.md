# Costs

Two things cost money: the GPU instance, billed per hour while it runs, and the Amazon Bedrock model
calls the agent makes. Prices below are on-demand list prices at the time of writing; check the
[EC2](https://aws.amazon.com/ec2/pricing/on-demand/) and
[Bedrock](https://aws.amazon.com/bedrock/pricing/) pricing pages for your region.

## GPU instance (the main cost)

| Instance | GPU | Approximate on-demand price |
|---|---|---|
| `g6e.4xlarge` | 1 × L40S, 48 GB | ≈ US$3–4 per hour |
| `g7e.4xlarge` (default) | 1 × RTX PRO 6000 Blackwell, 96 GB | ≈ US$5–8 per hour |

The instance is billed whether or not anything runs on it. Left on for a day, a `g6e.4xlarge` costs
US$70–100 — more than a hundred agent episodes. **Stop it when you are done** (`scripts/remote --stop`);
the Elastic IP and the data stay, and `scripts/remote --start` brings it back in about a minute. The
first boot adds about 45 minutes of instance time for the simulator image build; the 400 GB root
volume costs about US$30 per month while the instance exists, stopped or not.

## Bedrock (the agent's model calls)

GPT-6 Astra on Bedrock lists at about US$11 per million input tokens and US$55 per million output
tokens. Measured over a 100-episode run (`AWS/Bedrock` CloudWatch metrics for the batch window): 1,268
model invocations, 204,791 input tokens and 46,036 output tokens — **about 2.0 k input and 460 output
tokens, US$0.05 per episode; US$4.78 for the 100**. Treat that as a lower bound: the system prompt and
tool descriptions alone are about 1.6 k tokens and accompany every request, so the reported input count
is not the whole context; if the full context is billed on every turn an episode is closer to 22 k input
tokens, about US$0.31. Cost Explorer or your bill is the authoritative figure. Other Bedrock models
(`--model`) have their own prices; a cheaper model changes this line, not the instance line.

## Per episode, all in

| | Instance time | Model | Total |
|---|---|---|---|
| Agent episode (about 3 min) | ≈ US$0.18 | US$0.05–0.31 | **≈ US$0.23–0.49** |
| Scripted episode (about 75 s) | ≈ US$0.07 | — | **≈ US$0.07** |

## Worked example: a 100-episode dataset with the agent

| Item | Basis | Cost |
|---|---|---|
| Instance time while collecting | measured: 5.1 h for 100 episodes, plus ~15 min of `simteleop-up` and publish, on a `g6e.4xlarge` at ≈ US$3.5/h | ≈ US$19 |
| Bedrock | measured US$4.78 (lower bound, see above); up to ≈ US$31 if the full context is billed | US$5–31 |
| First boot (once per deployment) | ~45 min of instance time for the image build | ≈ US$3 |
| S3 | ~5 GB of videos and parquet for 100 episodes | < US$1 per month |
| **Total** | | **≈ US$27–54**, if the instance is stopped when the run ends |

That run produced 98 usable episodes (185,393 frames). The same dataset collected with the scripted
path is about US$7 of instance time. The figure that changes the picture is idle time: leaving the
instance on overnight after the run adds US$30–40 to the 100-episode total, and a forgotten week adds
about US$600. Budget for the instance-hours you keep the instance up, not for the episodes.
