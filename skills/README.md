# Skills for coding agents

Step-by-step procedures in the `SKILL.md` format (YAML front matter with `name` and `description`,
then the procedure). Point your coding agent at this folder (Kiro: add it as a skills directory;
Claude Code / Codex: reference the file); `AGENTS.md` at the repository root is the contract they
all assume.

| Skill | Use when |
|---|---|
| [`simteleop-operate`](./simteleop-operate/SKILL.md) | start/stop the box, bring the env server up, run agent or scripted episodes, publish and pull datasets, recover stuck remote jobs |
| [`simteleop-add-primitive`](./simteleop-add-primitive/SKILL.md) | add a new manipulation skill the agent can call, or change what `pick_vial`/`place_in_rack` do; wiring through protocol → server → client → tool → prompt, and the measurement before release |
| [`simteleop-debug-manipulation`](./simteleop-debug-manipulation/SKILL.md) (+ `probe.py`) | a primitive fails, a verdict disagrees with the picture, rates dropped; reading stage logs, joint limits, in-hand pose, camera frames |
| [`simteleop-new-embodiment`](./simteleop-new-embodiment/SKILL.md) | another robot, gripper, scene or upstream repository; what is robot-specific and the porting checklist |

Conventions the skills rely on: `scripts/remote` for every command on the box, `scripts/push-code`
after editing `sim/`/`common/`/`agent/`, seeds 1001–1008 for probes, `k/n` with seeds when reporting
a rate, and a camera frame for every claimed outcome.
