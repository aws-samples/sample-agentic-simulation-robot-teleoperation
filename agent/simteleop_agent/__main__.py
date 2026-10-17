"""Command line for the agent operator (simulated teleoperation platform).

    python -m simteleop_agent --check                       # preflight
    python -m simteleop_agent "put a vial in the rack"     # one task, transcript to stdout
    python -m simteleop_agent                              # interactive (Ctrl-D to exit)
    python -m simteleop_agent --episodes 10 --out /data/runs/batch1   # unattended batch, recorded

Environment: SIMTELEOP_AGENT_MODEL (Bedrock model id; default GPT-6 Astra), AWS_REGION, SIMTELEOP_ENV_HOST,
SIMTELEOP_ENV_PORT, SIMTELEOP_WALL_S (per-episode wall clock budget, default 900).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import tools as T
from .prompts import SYSTEM_PROMPT

DEFAULT_MODEL = os.environ.get("SIMTELEOP_AGENT_MODEL", "global.openai.gpt-6-astra")
DEFAULT_TASK = "Pick up a vial and stand it in a free slot of the rack. Record the episode."


def build_agent(model_id: str, tool_set: list | None = None):
    from strands import Agent
    from strands.models import BedrockModel

    from simteleop_common.solution import solution_config

    from .model_compat import ToolImageHoistingBedrockModel, needs_hoisting

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "eu-central-1"
    cls = ToolImageHoistingBedrockModel if needs_hoisting(model_id) else BedrockModel
    # boto_client_config carries the solution User-Agent suffix; Strands appends its own token to it.
    kwargs = {"model_id": model_id, "region_name": region, "max_tokens": 4096, "boto_client_config": solution_config()}
    if not needs_hoisting(model_id):
        # GPT-6 Astra on Bedrock rejects the temperature field outright (ValidationException,
        # 2026-09-29); Anthropic models accept it.
        kwargs["temperature"] = 0.2
    model = cls(**kwargs)
    return Agent(model=model, system_prompt=SYSTEM_PROMPT, tools=tool_set or T.ALL_TOOLS)


def run_task(agent, task: str, wall_s: float) -> dict:
    t0 = time.time()
    T.CALL_LOG.clear()
    result = agent(task)
    text = str(result)
    return {"task": task, "wall_seconds": round(time.time() - t0, 1), "tool_calls": len(T.CALL_LOG), "final_text": text, "calls": [dict(c, t=round(c["t"] - t0, 1)) for c in T.CALL_LOG], "over_budget": time.time() - t0 > wall_s}


def ground_truth_verdict(state_before: dict, state_after: dict) -> dict:
    """Independent of the agent's claim: a vial that was not in the rack before now stands in it
    (within 15 deg of vertical and seated on the slot floor, not leaning on the rim) and is not held."""
    before = {v["name"]: v for v in state_before["vials"]}
    placed = [v["name"] for v in state_after["vials"] if v.get("placed", v["in_rack"] and v["vertical"]) and not before[v["name"]]["in_rack"]]
    return {"success": bool(placed) and state_after["grasped_vial"] is None, "placed": placed, "in_rack_after": [v["name"] for v in state_after["vials"] if v["in_rack"]]}


def run_episodes(model_id: str, n: int, out: Path, task: str, wall_s: float) -> int:
    out.mkdir(parents=True, exist_ok=True)
    c = T.client()
    summary = []
    successes = 0
    for k in range(n):
        seed = 1000 + k
        print(f"\n=== episode {k + 1}/{n} (seed {seed}) ===", flush=True)
        c.reset(seed)
        before = c.state()
        agent = build_agent(model_id)  # fresh context per episode: no carry-over of stale beliefs
        try:
            r = run_task(agent, task, wall_s)
        except Exception as exc:  # noqa: BLE001 -- a model/transport error is a failed episode, not a crash
            r = {"task": task, "error": f"{type(exc).__name__}: {exc}", "tool_calls": len(T.CALL_LOG), "calls": T.CALL_LOG[:]}
        rec = c.record("status")
        if rec.get("active"):  # the agent forgot to stop: stop it with the measured verdict
            after = c.state()
            gt = ground_truth_verdict(before, after)
            c.record("stop", verdict={**gt, "note": "auto-stopped by the runner"})
        after = c.state()
        gt = ground_truth_verdict(before, after)
        successes += int(gt["success"])
        r.update({"seed": seed, "ground_truth": gt, "recording": c.record("status")})
        (out / f"episode_{k:03d}.json").write_text(json.dumps(r, indent=1, default=str))
        summary.append({"episode": k, "seed": seed, "success": gt["success"], "tool_calls": r.get("tool_calls"), "wall_seconds": r.get("wall_seconds"), "error": r.get("error")})
        print(f"--- ground truth: {'SUCCESS' if gt['success'] else 'FAIL'} {gt['placed']} | {r.get('tool_calls')} calls, {r.get('wall_seconds')} s", flush=True)
        c.home()
    (out / "summary.json").write_text(json.dumps({"model": model_id, "task": task, "episodes": n, "successes": successes, "per_episode": summary}, indent=1))
    print(f"\nRESULT: {successes}/{n} episodes succeeded by ground truth; details in {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="agent operator for the simulated teleoperation platform")
    ap.add_argument("task", nargs="?", default=None)
    ap.add_argument("--check", action="store_true", help="preflight and exit")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--episodes", type=int, default=0, help="run N recorded episodes unattended")
    ap.add_argument("--out", default="runs/latest", help="where episode JSONs go for --episodes")
    ap.add_argument("--wall", type=float, default=float(os.environ.get("SIMTELEOP_WALL_S", "900")))
    a = ap.parse_args()

    if a.check:
        from .preflight import run

        return run(a.model)
    if a.episodes:
        return run_episodes(a.model, a.episodes, Path(a.out), a.task or DEFAULT_TASK, a.wall)
    agent = build_agent(a.model)
    if a.task:
        r = run_task(agent, a.task, a.wall)
        print(f"\n[{r['tool_calls']} tool calls, {r['wall_seconds']} s]")
        return 0
    print("Agent operator. Type a task; Ctrl-D to exit.")
    try:
        while True:
            line = input("\ntask> ").strip()
            if line:
                agent(line)
    except EOFError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
