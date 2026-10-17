from simteleop_agent.tools import scene_text, verdict_text


def sample_state():
    return {
        "step": 10, "sim_time_s": 0.2, "joints_rad": [0] * 6, "joints_lerobot": [0, 0, 0, 0, 0, 50],
        "ee_position": [0.1, 0.0, 0.2], "tool_axis_world": [0, 0, -1], "jaw_opening_rad": 0.5,
        "contact_vials": [], "grasped_vial": None,
        "vials": [{"name": "vial_1", "centre": [0.27, -0.08, 0.05], "axis_world": [1, 0, 0], "vertical": False, "in_rack": False, "speed_mps": 0.0},
                  {"name": "vial_2", "centre": [0.2, 0.1, 0.09], "axis_world": [0, 0, 1], "vertical": True, "in_rack": True, "speed_mps": 0.0, "tilt_deg": 1.0, "seated": True, "placed": True}],
        "rack_position": [0.18, 0.08, 0.06], "rack_slots_world": [[0.2, 0.1, 0.16], [0.26, 0.1, 0.16]], "rack_slots_free": [False, True],
        "recording": {"active": True, "frames": 120, "episodes_saved": 2},
    }


def test_scene_text_mentions_everything_the_agent_needs():
    t = scene_text(sample_state())
    assert "vial_1" in t and "lying" in t
    assert "vial_2" in t and "STANDING IN RACK" in t
    assert "(occupied)" in t and "1:[0.260" in t
    assert "recording: ON, 120 frames" in t


def test_verdict_text_drops_state_blob():
    t = verdict_text({"success": True, "rose_m": 0.123456, "state": {"huge": 1}})
    assert "state" not in t and "0.1235" in t
