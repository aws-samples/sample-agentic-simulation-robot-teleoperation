import socket
import threading

import pytest

from simteleop_common import protocol as pr


def test_encode_decode_roundtrip():
    obj = {"op": "state", "id": 3, "x": [1.5, 2]}
    assert pr.decode(pr.encode(obj).rstrip(b"\n")) == obj


def test_decode_rejects_garbage():
    with pytest.raises(pr.ProtocolError):
        pr.decode(b"not json")
    with pytest.raises(pr.ProtocolError):
        pr.decode(b"[1,2]")


def test_validate_request():
    assert pr.validate_request({"op": "ping"}) == "ping"
    with pytest.raises(pr.ProtocolError):
        pr.validate_request({})
    with pytest.raises(pr.ProtocolError):
        pr.validate_request({"op": "fly"})


def test_ok_err_shapes():
    assert pr.ok({"a": 1}, 7) == {"ok": True, "id": 7, "result": {"a": 1}}
    e = pr.err("busy", "wait", 8)
    assert e["ok"] is False and e["error"] == "busy"


def test_line_socket_framing_across_chunks():
    a, b = socket.socketpair()
    ls_a, ls_b = pr.LineSocket(a), pr.LineSocket(b)
    payload = pr.encode({"op": "ping", "id": 1}) + pr.encode({"op": "state", "id": 2})
    # Deliver in awkward chunks.
    def send():
        for i in range(0, len(payload), 7):
            b.sendall(payload[i : i + 7])
        b.close()
    threading.Thread(target=send).start()
    assert ls_a.recv(2)["id"] == 1
    assert ls_a.recv(2)["id"] == 2
    assert ls_a.recv(2) is None  # peer closed
    ls_a.close()
    ls_b.close()


def test_client_call_against_fake_server():
    from simteleop_agent.client import EnvClient

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def serve():
        conn, _ = srv.accept()
        ls = pr.LineSocket(conn)
        req = ls.recv(5)
        assert req["op"] == "ping"
        ls.send(pr.ok({"protocol": 1, "task": "t", "step": 0, "busy": False, "tool_axis": "-z"}, req["id"]))
        req = ls.recv(5)
        ls.send(pr.err("busy", "running", req["id"]))
        ls.close()

    threading.Thread(target=serve).start()
    c = EnvClient("127.0.0.1", port)
    assert c.ping()["task"] == "t"
    with pytest.raises(pr.RemoteError) as ei:
        c.home()
    assert ei.value.error == "busy"
    c.close()
    srv.close()
