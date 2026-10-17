"""Scripted pick-and-place probe: the measurement behind every primitive change.

Run on the box from the agent venv against a running env server:

    /home/ubuntu/simteleop/agent-venv/bin/python probe.py 1001 1002 1003 1004 1005 1006 1007 1008

Per seed: reset, home, pick the vial nearest to the base, place it in the nearest free slot, print
one JSON line with the stage facts, and save /tmp/probe/<seed>.jpg from the overview camera.
Options: PROBE_CAP_UP=0 (cap-down hold), PROBE_SLOT=far (farthest free slot), PROBE_TILT=<deg> (grasp tilt).
"""
import json
import os
import sys

import numpy as np

from simteleop_agent.client import EnvClient


def stage(r: dict, k: str):
    s = r.get(k) or {}
    return [s.get("converged"), round(s.get("position_error_m", -1), 3), round(s.get("axis_error_deg", -1), 1), s.get("joints_deg")]


def main(seeds: list[int]) -> int:
    c = EnvClient()
    cap_up = os.environ.get("PROBE_CAP_UP", "1") != "0"
    pick_kw = {"tilt_deg": float(os.environ["PROBE_TILT"])} if os.environ.get("PROBE_TILT") else {}
    placed = picks = 0
    for seed in seeds:
        c.reset(seed=seed); c.wait_idle(); c.home(); c.wait_idle()
        s = c.state()
        vial = min([v for v in s["vials"] if not v["in_rack"]], key=lambda v: np.hypot(*v["centre"][:2]))
        p = c.call("pick_vial", timeout_s=180, vial=vial["name"], **pick_kw); c.wait_idle()
        if not p.get("success"):
            print(json.dumps({"seed": seed, "pick": False, "reason": p.get("reason"), "rose_m": p.get("rose_m"), "regrip": stage(p, "regrip"), "lift": stage(p, "lift")}))
            continue
        picks += 1
        free = [i for i, f in enumerate(s["rack_slots_free"]) if f]
        pick_slot = max if os.environ.get("PROBE_SLOT") == "far" else min
        slot = pick_slot(free, key=lambda i: np.hypot(*s["rack_slots_world"][i][:2]))
        q = c.call("place_in_rack", timeout_s=300, slot=slot, cap_up=cap_up); c.wait_idle()
        v = [x for x in c.state()["vials"] if x["name"] == vial["name"]][0]
        placed += bool(v["placed"])
        print(json.dumps({
            "seed": seed, "vial": vial["name"], "slot": slot, "slot_r": round(float(np.hypot(*s["rack_slots_world"][slot][:2])), 3),
            "success": q.get("success"), "reason": (q.get("reason") or "")[:80],
            "pick": {"axis_in_gripper": p.get("axis_in_gripper"), "regrip": stage(p, "regrip")},
            "tool_axis": q.get("place_tool_axis"), "tilt_above": q.get("tilt_above_entry_deg"), "tilt_release": q.get("tilt_at_release_deg"),
            "off_release_m": q.get("vial_xy_offset_from_slot_m"),
            "final": {"tilt": v["tilt_deg"], "cz": round(v["centre"][2], 3), "axis_z": round(v["axis_world"][2], 2), "seated": v["seated"], "placed": v["placed"]},
            "wrist_min": min([(q[k].get("joints_deg") or [0] * 5)[3] for k in ("rise", "carry", "lower", "insert") if q.get(k)] or [0]),
            **{k: stage(q, k) for k in ("rise", "carry", "lower", "insert") if q.get(k)},
        }))
        os.makedirs("/tmp/probe", exist_ok=True)
        jpg, _ = c.frame("external_D455")
        open(f"/tmp/probe/{seed}.jpg", "wb").write(jpg)
    print(f"placed {placed}/{picks} (picks {picks}/{len(seeds)})")
    return 0


if __name__ == "__main__":
    sys.exit(main([int(x) for x in sys.argv[1:]] or list(range(1001, 1009))))
