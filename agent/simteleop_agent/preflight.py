"""Preflight: which dependency is missing, in one screen, before an episode is attempted."""
from __future__ import annotations

import importlib
import os
import sys
from typing import Callable

from .client import EnvClient


def _check(name: str, fn: Callable[[], str]) -> bool:
    try:
        detail = fn()
        print(f"  OK    {name:<18} {detail}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  {name:<18} {type(exc).__name__}: {exc}")
        return False


def check_python_deps() -> str:
    mods = ["strands", "boto3", "numpy", "simteleop_common"]
    versions = []
    for m in mods:
        mod = importlib.import_module(m)
        versions.append(f"{m} {getattr(mod, '__version__', '')}".strip())
    return ", ".join(versions)


def check_env_server() -> str:
    c = EnvClient(os.environ.get("SIMTELEOP_ENV_HOST", "127.0.0.1"), int(os.environ.get("SIMTELEOP_ENV_PORT", "7010")), connect_timeout_s=3)
    p = c.ping()
    s = c.state()
    c.close()
    rec = s.get("recording") or {}
    return f"task {p['task']}, step {p['step']}, {len(s['vials'])} vials, {len(s['rack_slots_world'])} slots, recording {'enabled' if 'active' in rec else 'DISABLED (no --dataset-root)'}; {p['tool_axis']}"


def check_bedrock(model_id: str) -> str:
    from simteleop_common.solution import get_client

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "eu-central-1"
    rt = get_client("bedrock-runtime", region_name=region)
    r = rt.converse(modelId=model_id, messages=[{"role": "user", "content": [{"text": "Reply with the single word OK."}]}], inferenceConfig={"maxTokens": 16})  # Astra requires >= 16
    text = " ".join(b.get("text", "") for b in r["output"]["message"]["content"]).strip()
    return f"{model_id} in {region} answered {text!r}"


def run(model_id: str) -> int:
    print("simteleop-agent preflight (embodiment: SO-101 workshop scene)")
    results = [
        _check("python deps", check_python_deps),
        _check("env server", check_env_server),
        _check("bedrock model", lambda: check_bedrock(model_id)),
    ]
    ok = all(results)
    print("  all checks passed" if ok else "  fix the FAIL lines above; the agent will not run reliably until they are green")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run(os.environ.get("SIMTELEOP_AGENT_MODEL", "global.openai.gpt-6-astra")))
