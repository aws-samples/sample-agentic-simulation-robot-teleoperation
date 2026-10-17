"""Episode recording through the workshop's own ``LeRobotRecorder``.

Using the workshop recorder unchanged is the point: a dataset recorded by the agent has the same
features, scaling (LeRobot −100..100 units) and camera names as one recorded with a leader arm, so
the workshop's ``lerobot_eval`` and training recipes apply without conversion.

What is added on top is a JSON manifest per episode (``<root>/manifests/episode_<n>.json``) with the
task string, the agent's tool calls and the measured verdict — the recorder itself has no notion of
success.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from types import SimpleNamespace
from typing import Any

import torch

from simteleop_common import geometry as geo


def _episodes_on_disk(root: str) -> int:
    info = os.path.join(root, "meta", "info.json")
    if not os.path.exists(info):
        return 0
    try:
        with open(info) as f:
            return int(json.load(f).get("total_episodes", 0))
    except (OSError, ValueError):
        return 0


class EpisodeRecorder:
    def __init__(self, env, cameras: dict[str, dict[str, int]], repo_id: str, root: str, task_name: str, fps: int = 30, max_minutes: float = 6.0):
        from sim_to_real_so101.utils.lerobot_recorder import LeRobotRecorder

        self.fps = fps
        self.cameras = cameras
        self.root = root
        self.repo_id = repo_id
        self.task_name = task_name
        self.device = env.unwrapped.device
        self.rec = LeRobotRecorder(task_name=task_name, repo_id=repo_id, dataset_root=root, fps=fps, device=self.device, cameras=cameras)
        # The workshop buffer holds two minutes; agent episodes are longer. (The attribute is spelled
        # this way upstream.)
        self.rec.capcity = int(max_minutes * 60 * fps)
        # LeRobotDataset.create writes meta/info.json immediately, but a dataset with zero episodes has
        # no episode metadata and LeRobotDataset() then falls back to the Hub (401 for a local repo id).
        # A resume is only possible once one episode was saved, so clear an episode-less shell.
        if os.path.isdir(root) and _episodes_on_disk(root) == 0:
            print(f"[recording] {root} exists with no saved episodes; starting fresh (lerobot cannot reopen an episode-less dataset offline)", flush=True)
            shutil.rmtree(root)
        # Manifests are numbered like the dataset's episodes: continue from what is already on disk
        # (a restart of the server used to start at 0 again and overwrote the first manifest).
        self.first_index = _episodes_on_disk(root)
        self.rec.init_dataset()
        self.active = False
        self.frames = 0
        self.episodes_saved = 0
        self.started_at = 0.0
        self.events: list[dict[str, Any]] = []
        os.makedirs(os.path.join(root, "manifests"), exist_ok=True)

    # ------------------------------------------------------------------ control
    def start(self) -> dict[str, Any]:
        if self.active:
            return self.status()
        self.active = True
        self.frames = 0
        self.events = []
        self.started_at = time.time()
        return self.status()

    def stop(self, verdict: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.active:
            return self.status()
        self.active = False
        n = self.frames
        if n == 0:
            return self.status()
        # Synchronous call into the recorder's own handler (it only checks the event name), which
        # copies the buffers and queues the episode for the writer thread.
        self.rec.save_episode(SimpleNamespace(event_name=self.rec.STOP_RECORDING_EVENT))
        self.episodes_saved += 1
        idx = self.first_index + self.episodes_saved - 1
        manifest = {
            "episode_index_hint": idx,
            "repo_id": self.repo_id,
            "task": self.task_name,
            "fps": self.fps,
            "frames": n,
            "wall_seconds": round(time.time() - self.started_at, 1),
            "verdict": verdict,
            "events": self.events,
        }
        with open(os.path.join(self.root, "manifests", f"episode_{idx:06d}.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        return {**self.status(), "saved_frames": n, "manifest": manifest}

    def cancel(self) -> dict[str, Any]:
        if self.active:
            self.rec.cancel_recording(SimpleNamespace(event_name=self.rec.CANCEL_RECORDING_EVENT))
        self.active = False
        self.frames = 0
        return self.status()

    def note(self, event: dict[str, Any]) -> None:
        if self.active:
            self.events.append({"t": round(time.time() - self.started_at, 2), **event})

    def status(self) -> dict[str, Any]:
        return {"active": self.active, "frames": self.frames, "episodes_saved": self.episodes_saved, "pending_writes": self.rec.episode_queue.qsize()}

    # ------------------------------------------------------------------ frames
    def push(self, obs: dict, joint_targets_rad: torch.Tensor, joint_pos_rad: torch.Tensor) -> None:
        """Push one frame: action = commanded targets, state = measured joints, both in LeRobot units."""
        if not self.active:
            return
        visual = obs.get("visual")
        if visual is None:
            return
        action = torch.tensor(geo.rad_to_lerobot(joint_targets_rad.detach().cpu().tolist()), dtype=torch.float32, device=self.device)
        state = torch.tensor(geo.rad_to_lerobot(joint_pos_rad.detach().cpu().tolist()), dtype=torch.float32, device=self.device)
        rgb = {cam: visual[f"rgb_{cam}"][0] for cam in self.cameras}
        depth = {cam: visual[f"depth_{cam}"][0] for cam in self.cameras if f"depth_{cam}" in visual}
        seg = {cam: visual[f"instance_id_seg_{cam}"][0][..., :3] for cam in self.cameras if f"instance_id_seg_{cam}" in visual}
        self.rec.push_frame_to_buffer(action, state, rgb, depth, seg)
        self.frames += 1

    def wait_for_writes(self, timeout_s: float = 600.0) -> bool:
        t0 = time.time()
        while self.rec.episode_queue.qsize() > 0 and time.time() - t0 < timeout_s:
            time.sleep(1.0)
        return self.rec.episode_queue.qsize() == 0
