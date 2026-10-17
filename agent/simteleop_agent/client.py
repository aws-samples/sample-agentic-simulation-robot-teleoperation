"""Client for the env server, used by the agent's tools and by the episode runner."""
from __future__ import annotations

import base64
import socket
import time
from typing import Any

from simteleop_common import protocol as pr


class EnvClient:
    def __init__(self, host: str = pr.DEFAULT_HOST, port: int = pr.DEFAULT_PORT, connect_timeout_s: float = 5.0):
        self.host, self.port = host, port
        s = socket.create_connection((host, port), timeout=connect_timeout_s)
        self.ls = pr.LineSocket(s)
        self._id = 0

    def close(self) -> None:
        self.ls.close()

    def call(self, op: str, timeout_s: float = 30.0, **args: Any) -> dict[str, Any]:
        """Send one request and wait up to ``timeout_s`` for its reply. Wire arguments go in ``args``;
        an op that itself takes a ``timeout_s`` (move_ee) passes it as ``args["timeout_s"]`` via
        :meth:`call_args`."""
        return self.call_args(op, args, timeout_s)

    def call_args(self, op: str, args: dict[str, Any], timeout_s: float = 30.0) -> dict[str, Any]:
        self._id += 1
        rid = self._id
        self.ls.send({"op": op, "id": rid, **args})
        while True:
            resp = self.ls.recv(timeout_s)
            if resp is None:
                raise ConnectionError("env server closed the connection")
            if resp.get("id") not in (rid, None):
                continue  # not ours (should not happen on a private connection)
            if resp.get("ok"):
                return resp["result"]
            raise pr.RemoteError(resp.get("error", "error"), resp.get("message", ""))

    # ------------------------------------------------------------------ quick
    def ping(self) -> dict[str, Any]:
        return self.call("ping", timeout_s=10)

    def state(self) -> dict[str, Any]:
        return self.call("state", timeout_s=10)

    def frame(self, camera: str = "external_D455", quality: int = 80) -> tuple[bytes, dict[str, Any]]:
        r = self.call("frame", timeout_s=20, camera=camera, quality=quality)
        return base64.b64decode(r.pop("jpeg_b64")), r

    def record(self, action: str, **kw: Any) -> dict[str, Any]:
        return self.call("record", timeout_s=60, action=action, **kw)

    # ------------------------------------------------------------------ motions (block until the verdict)
    def reset(self, seed: int | None = None) -> dict[str, Any]:
        return self.call("reset", timeout_s=120, seed=seed)

    def home(self) -> dict[str, Any]:
        return self.call("home", timeout_s=60)

    def move_joints(self, joints_rad: list[float], duration_s: float = 2.0) -> dict[str, Any]:
        return self.call("move_joints", timeout_s=60, joints_rad=joints_rad, duration_s=duration_s)

    def move_ee(self, position: list[float], approach: list[float] | None = None, quaternion: list[float] | None = None, timeout_s: float = 4.0) -> dict[str, Any]:
        args: dict[str, Any] = {"position": position, "timeout_s": timeout_s}
        if quaternion is not None:
            args["quaternion"] = quaternion
        elif approach is not None:
            args["approach"] = approach
        return self.call_args("move_ee", args, timeout_s=timeout_s + 30)

    def gripper(self, command: str | None = None, opening_rad: float | None = None) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if command:
            args["command"] = command
        if opening_rad is not None:
            args["opening_rad"] = opening_rad
        return self.call("gripper", timeout_s=30, **args)

    def pick_vial(self, vial: str) -> dict[str, Any]:
        return self.call("pick_vial", timeout_s=120, vial=vial)

    def place_in_rack(self, slot: int | None = None) -> dict[str, Any]:
        args = {"slot": slot} if slot is not None else {}
        return self.call("place_in_rack", timeout_s=120, **args)

    def wait_idle(self, timeout_s: float = 120.0) -> None:
        t0 = time.time()
        while self.ping().get("busy") and time.time() - t0 < timeout_s:
            time.sleep(0.5)
