# Troubleshooting

- **`simteleop-up` waits forever / "container exited".** `docker logs simteleop_sim`. The first start compiles
  shaders and can take 10 minutes; a missing `--gpus` or the wrong driver shows up as an RTX error
  in the first 100 lines. The workshop base needs `ACCEPT_EULA=Y` (set by `simteleop-up`).
- **Env server ready but `pick_vial` always "could not reach the hover pose".** The tool-axis guess
  is wrong for this asset. Read the start-up line `tool axis -z points [...]` with the arm at home:
  the fingers should point roughly forward-down. Try `SIMTELEOP_TOOL_AXIS=+z`, `-y`, `+x` and restart.
- **Jaw closes, contact sensor sees nothing, vial visibly between the fingers.** The contact sensor
  filters on `Vial_1..3` prim paths and needs `activate_contact_sensors=True` (the vials-to-rack
  task uses `S0101_CONTACT_GRASP_CFG`, which sets it). The `-Base` and `-Task` debug envs have no
  vials and no sensor; use a `Vials-To-Rack*` task.
- **`describe_scene` shows the rack slots at the authored 2×2 layout.** The USD slot prims
  (`Body1/Mesh/top_*`) could not be read; the log says why. Placement still works, less precisely.
- **Astra: `ValidationException ... image field`.** The image-hoisting wrapper was not selected;
  check `SIMTELEOP_AGENT_MODEL` contains no "anthropic" typo and that `model_compat.py` is installed.
- **`simteleop-agent --check` fails on Bedrock with an SCP deny.** The account blocks Bedrock for EC2;
  everything else works. Needs `bedrock:InvokeModel*` allowed for the inference profile.
- **No Isaac Sim window in DCV.** `simteleop-fix-display`, then `simteleop-up --gui`. The session is named
  `simteleop` (`DCV_SESSION` in `/etc/simteleop.env`); `dcv list-sessions` must show it.
- **Recording: `meta/info.json` missing after episodes.** The recorder writes asynchronously and the
  encoder runs after each `stop`; `simteleop-down` waits for the queue. `docker logs simteleop_sim | grep
  'Episode .* saved'` counts finished episodes.
- **Disk.** Each 30 fps episode with two 640×480 cameras is ~50–150 MB encoded; the Isaac Lab image is
  ~25 GB. 400 GB root leaves room for ~1000 episodes.
