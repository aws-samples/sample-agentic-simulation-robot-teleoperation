"""Env server: runs a workshop task in Isaac Lab and exposes it over a JSON line protocol.

Runs INSIDE the workshop sim container with Isaac Lab's Python::

    python -m simteleop_sim.env_server --task Lerobot-So101-Teleop-Vials-To-Rack --headless \
        --dataset-root /workspace/datasets/agent_vials --repo-id local/so101_agent_vials

Threading model: Isaac Sim must be driven from the main thread, so the socket listener runs in a
thread and only enqueues requests. The main loop steps the simulator at a fixed rate; between steps
it answers quick requests (state, frame, record) immediately and advances at most one long-running
motion (move/pick/place), replying when the motion's generator returns its verdict. A second motion
request while one is running is refused rather than queued, so the agent always knows which call is
moving the arm.
"""
from __future__ import annotations

import argparse
import base64
import io
import queue
import socket
import threading
import time
import traceback
from typing import Any

from simteleop_common import protocol as pr

# --------------------------------------------------------------------------- CLI first (AppLauncher)
parser = argparse.ArgumentParser(description="SO-101 env server for agent teleoperation")
parser.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack")
parser.add_argument("--port", type=int, default=pr.DEFAULT_PORT)
parser.add_argument("--host", default=pr.DEFAULT_HOST)
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--dataset-root", default=None, help="record LeRobot episodes here (omit to disable recording)")
parser.add_argument("--repo-id", default="local/so101_agent_vials_to_rack")
parser.add_argument("--task-name", default="Pick up the vial and place it in the rack")
parser.add_argument("--max-episode-minutes", type=float, default=6.0)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- everything else
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import sim_to_real_so101.tasks  # noqa: F401, E402

from simteleop_sim import motion as mo  # noqa: E402
from simteleop_sim import primitives as prim  # noqa: E402
from simteleop_sim.recording import EpisodeRecorder  # noqa: E402
from simteleop_sim.scene_state import SceneReader  # noqa: E402

QUICK_OPS = {"ping", "state", "bodies", "frame", "record", "shutdown"}
MOTION_OPS = {"move_joints", "move_ee", "gripper", "home", "pick_vial", "place_in_rack", "reset"}


class Listener(threading.Thread):
    """Accepts connections and enqueues (conn, request) pairs for the main thread."""

    def __init__(self, host: str, port: int, inbox: "queue.Queue[tuple[pr.LineSocket, dict]]"):
        super().__init__(daemon=True, name="listener")
        self.inbox = inbox
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen(8)
        print(f"[env_server] listening on {host}:{port}", flush=True)

    def run(self) -> None:
        while True:
            conn, _ = self.srv.accept()
            threading.Thread(target=self._serve, args=(pr.LineSocket(conn),), daemon=True).start()

    def _serve(self, ls: pr.LineSocket) -> None:
        try:
            while True:
                req = ls.recv()
                if req is None:
                    break
                self.inbox.put((ls, req))
        except pr.ProtocolError as exc:
            ls.send(pr.err("protocol", str(exc)))
        except OSError:
            pass
        finally:
            ls.close()


class Server:
    def __init__(self, env, task_id: str, recorder: EpisodeRecorder | None):
        self.env = env
        self.reader = SceneReader(env, task_id)
        self.ik = mo.IKSolver(self.reader)
        self.recorder = recorder
        self.step_count = 0
        self.obs = None
        self.motion: tuple[pr.LineSocket, Any, mo.Motion, str, float] | None = None
        self.inbox: "queue.Queue[tuple[pr.LineSocket, dict]]" = queue.Queue()
        self.stop = False
        self.frame_every = max(1, int(round(mo.STEP_HZ / (recorder.fps if recorder else 30))))

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if self.recorder and self.recorder.active:
            self.recorder.cancel()
        with torch.inference_mode():
            self.obs, _ = self.env.reset(seed=seed)
        # Targets follow the actual joints after a reset so the first motion starts from reality. Built
        # as a fresh tensor: robot data are inference tensors and the motions update targets in place.
        self.reader.last_targets = torch.tensor(self.reader.joints().detach().cpu().tolist(), dtype=torch.float32, device=self.env.unwrapped.device)
        for _ in range(int(0.5 * mo.STEP_HZ)):
            self.step(self.reader.last_targets)
        return {"seed": seed, "state": self.reader.snapshot(self.step_count).to_json()}

    def step(self, targets: torch.Tensor) -> None:
        self.reader.last_targets = targets.detach().clone()
        actions = targets.reshape(1, -1)
        with torch.inference_mode():
            self.obs, _, _, _, _ = self.env.step(actions)
        self.step_count += 1
        if self.recorder and self.recorder.active and self.step_count % self.frame_every == 0:
            self.recorder.push(self.obs, self.reader.last_targets, self.reader.joints())
        if self.recorder:
            self.reader.recording_status = self.recorder.status()

    # ------------------------------------------------------------------ requests
    def handle_quick(self, ls: pr.LineSocket, req: dict) -> None:
        op = req["op"]
        rid = req.get("id")
        try:
            if op == "ping":
                ls.send(pr.ok({"protocol": pr.PROTOCOL_VERSION, "task": self.reader.task_id, "step": self.step_count, "busy": self.motion is not None, "tool_axis": self.reader.describe_tool_axis()}, rid))
            elif op == "state":
                ls.send(pr.ok(self.reader.snapshot(self.step_count).to_json(), rid))
            elif op == "bodies":
                ls.send(pr.ok(self.reader.bodies(), rid))
            elif op == "frame":
                ls.send(pr.ok(self.frame(req.get("camera", "external_D455"), int(req.get("quality", 80))), rid))
            elif op == "record":
                ls.send(pr.ok(self.record(req), rid))
            elif op == "shutdown":
                ls.send(pr.ok({"bye": True}, rid))
                self.stop = True
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            ls.send(pr.err(type(exc).__name__, str(exc), rid))

    def frame(self, camera: str, quality: int) -> dict[str, Any]:
        from PIL import Image

        key = f"rgb_{camera}"
        visual = self.obs.get("visual", {}) if self.obs else {}
        if key not in visual:
            raise KeyError(f"no camera {camera!r}; available: {[k[4:] for k in visual if k.startswith('rgb_')]}")
        arr = visual[key][0].detach().cpu().numpy()
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr[..., :3]).save(buf, format="JPEG", quality=quality)
        return {"camera": camera, "width": int(arr.shape[1]), "height": int(arr.shape[0]), "jpeg_b64": base64.b64encode(buf.getvalue()).decode("ascii"), "step": self.step_count}

    def record(self, req: dict) -> dict[str, Any]:
        if self.recorder is None:
            raise RuntimeError("recording disabled: start the server with --dataset-root")
        action = req.get("action", "status")
        if action == "start":
            return self.recorder.start()
        if action == "stop":
            return self.recorder.stop(verdict=req.get("verdict"))
        if action == "cancel":
            return self.recorder.cancel()
        if action == "note":
            self.recorder.note(req.get("event", {}))
            return self.recorder.status()
        return self.recorder.status()

    def start_motion(self, ls: pr.LineSocket, req: dict) -> None:
        op = req["op"]
        rid = req.get("id")
        if self.motion is not None:
            ls.send(pr.err("busy", f"a {self.motion[3]} motion is running; wait for it or call it again later", rid))
            return
        try:
            if op == "reset":
                ls.send(pr.ok(self.reset(req.get("seed")), rid))
                return
            gen = self.make_motion(op, req)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            ls.send(pr.err(type(exc).__name__, str(exc), rid))
            return
        if self.recorder:
            self.recorder.note({"op": op, "args": {k: v for k, v in req.items() if k not in ("op", "id")}})
        self.motion = (ls, rid, gen, op, time.time())

    def make_motion(self, op: str, req: dict) -> mo.Motion:
        r, ik = self.reader, self.ik
        if op == "move_joints":
            return mo.move_joints(r, req["joints_rad"], float(req.get("duration_s", 2.0)))
        if op == "move_ee":
            quat = req.get("quaternion")
            if quat is None:
                from simteleop_common import geometry as geo
                from simteleop_sim.scene_state import TOOL_AXIS

                approach = req.get("approach", [0.0, 0.0, -1.0])
                quat = geo.tool_orientation(approach, req["position"], TOOL_AXIS).tolist()
            return mo.move_ee(r, ik, req["position"], quat, float(req.get("timeout_s", 4.0)), rot_weight=req.get("rot_weight"))
        if op == "gripper":
            opening = {"open": mo.JAW_OPEN_RAD, "close": mo.JAW_CLOSED_RAD}.get(req.get("command"), req.get("opening_rad"))
            if opening is None:
                raise ValueError("gripper needs command=open|close or opening_rad")
            return mo.gripper(r, float(opening))
        if op == "home":
            return prim.home(r)
        if op == "pick_vial":
            return prim.pick_vial(r, ik, req["vial"], depth_m=req.get("depth_m"), tilt_deg=req.get("tilt_deg"), along_m=req.get("along_m"))
        if op == "place_in_rack":
            return prim.place_in_rack(r, ik, req.get("slot"), cap_up=bool(req.get("cap_up", True)))
        raise pr.ProtocolError(f"not a motion op: {op}")

    def advance_motion(self) -> None:
        """One step of the running motion; reply and clear it when the generator returns."""
        ls, rid, gen, op, t0 = self.motion
        try:
            targets = next(gen)
            self.step(targets)
        except StopIteration as done:
            verdict = done.value if isinstance(done.value, dict) else {"result": done.value}
            verdict["wall_seconds"] = round(time.time() - t0, 2)
            verdict["state"] = self.reader.snapshot(self.step_count).to_json()
            if self.recorder:
                self.recorder.note({"op": op, "verdict": {k: v for k, v in verdict.items() if k != "state"}})
            ls.send(pr.ok(verdict, rid))
            self.motion = None
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            ls.send(pr.err(type(exc).__name__, str(exc), rid))
            self.motion = None

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        Listener(args.host, args.port, self.inbox).start()
        self.reset(args.seed)
        print(f"[env_server] ready: {self.reader.describe_tool_axis()}", flush=True)
        period = 1.0 / mo.STEP_HZ
        last_report = time.time()
        while simulation_app.is_running() and not self.stop:
            t = time.time()
            # Drain quick requests; start a motion if one is asked for and none runs.
            try:
                while True:
                    ls, req = self.inbox.get_nowait()
                    try:
                        op = pr.validate_request(req)
                    except pr.ProtocolError as exc:
                        ls.send(pr.err("protocol", str(exc), req.get("id")))
                        continue
                    if op in QUICK_OPS:
                        self.handle_quick(ls, req)
                    else:
                        self.start_motion(ls, req)
            except queue.Empty:
                pass
            if self.motion is not None:
                self.advance_motion()
            else:
                self.step(self.reader.last_targets)  # idle: hold position, keep rendering
            if time.time() - last_report > 30:
                last_report = time.time()
                print(f"[env_server] step={self.step_count} recording={self.reader.recording_status}", flush=True)
            dt = time.time() - t
            if dt < period:
                time.sleep(period - dt)
        if self.recorder:
            self.recorder.wait_for_writes()
        self.env.close()


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=not args.disable_fabric)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    cameras = {}
    for name in env.unwrapped.scene.keys():
        if name.startswith("camera_"):
            cfg = getattr(env.unwrapped.scene.cfg, name)
            cameras[name[len("camera_"):]] = {"height": cfg.height, "width": cfg.width}
    print(f"[env_server] cameras: {cameras}", flush=True)
    recorder = None
    if args.dataset_root:
        recorder = EpisodeRecorder(env, cameras, args.repo_id, args.dataset_root, args.task_name, fps=30, max_minutes=args.max_episode_minutes)
    Server(env, args.task, recorder).run()
    simulation_app.close()


if __name__ == "__main__":
    main()
