"""Wire protocol between the env server (inside the sim container) and the agent (on the host).

One JSON object per line, UTF-8, over a plain TCP socket bound to localhost. Both sides are Python;
the framing is deliberately trivial so it can be driven from ``nc`` when something goes wrong::

    echo '{"op": "ping"}' | nc localhost 7010

Requests::

    {"op": "<name>", "id": <optional client id>, ...args}

Responses::

    {"ok": true,  "id": ..., "result": {...}}
    {"ok": false, "id": ..., "error": "<class>", "message": "<text>"}

Large binary payloads (camera frames) are base64 strings inside the JSON. A frame is ~60 kB as
JPEG, well under anything that needs chunking.
"""
from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7010

#: Joint order of the SO-101 articulation in the workshop USD, and of every joint vector on the wire.
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
ARM_JOINT_NAMES = JOINT_NAMES[:5]
JAW_JOINT = "Jaw"

#: LeRobot names for the same joints, in the same order (what the recorded dataset uses).
LEROBOT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

#: Joint range in degrees that the workshop maps LeRobot's normalised -100..100 (gripper 0..100)
#: onto. Copied from ``LeRobotSO101Interface.SO101_USD_MAPPING`` so a dataset recorded by the agent
#: has exactly the same action/state scaling as one recorded with a leader arm.
USD_JOINT_RANGE_DEG = {
    "Rotation": (-110.0, 110.0),
    "Pitch": (-100.0, 100.0),
    "Elbow": (-100.0, 90.0),
    "Wrist_Pitch": (-95.0, 95.0),
    "Wrist_Roll": (-160.0, 160.0),
    "Jaw": (-10.0, 100.0),
}

#: Operations the server accepts. Kept as data so the agent's preflight can check the server speaks
#: the same version.
PROTOCOL_VERSION = 1
OPS = (
    "ping",
    "state",
    "bodies",
    "frame",
    "reset",
    "move_joints",
    "move_ee",
    "gripper",
    "home",
    "pick_vial",
    "place_in_rack",
    "record",
    "shutdown",
)


class ProtocolError(Exception):
    """The peer sent something we cannot parse or that names an unknown op."""


class RemoteError(Exception):
    """The server executed the request and reported a failure."""

    def __init__(self, error: str, message: str):
        super().__init__(f"{error}: {message}")
        self.error = error
        self.message = message


def encode(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def decode(line: bytes) -> dict[str, Any]:
    try:
        obj = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"not a JSON line: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError("a request must be a JSON object")
    return obj


def ok(result: dict[str, Any], req_id: Any = None) -> dict[str, Any]:
    return {"ok": True, "id": req_id, "result": result}


def err(error: str, message: str, req_id: Any = None) -> dict[str, Any]:
    return {"ok": False, "id": req_id, "error": error, "message": message}


def validate_request(req: dict[str, Any]) -> str:
    op = req.get("op")
    if not isinstance(op, str):
        raise ProtocolError("missing 'op'")
    if op not in OPS:
        raise ProtocolError(f"unknown op {op!r}; known: {', '.join(OPS)}")
    return op


@dataclass
class LineSocket:
    """Blocking line-oriented wrapper around a socket; one instance per connection."""

    sock: socket.socket
    _buf: bytes = b""

    def send(self, obj: dict[str, Any]) -> None:
        self.sock.sendall(encode(obj))

    def recv(self, timeout_s: float | None = None) -> dict[str, Any] | None:
        """Return the next decoded line, or None when the peer closed the connection."""
        self.sock.settimeout(timeout_s)
        while b"\n" not in self._buf:
            chunk = self.sock.recv(1 << 16)
            if not chunk:
                return None
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return decode(line)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
