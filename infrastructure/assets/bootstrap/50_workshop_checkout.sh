#!/usr/bin/env bash
# Clone the NVIDIA Sim-to-Real SO-101 workshop at the pinned commit. The scene, the robot asset, the
# task definitions and the LeRobot recorder all come from there; this repository copies none of them.
set -xeuo pipefail
# shellcheck disable=SC1091
. /etc/simteleop.env
DEST=/home/ubuntu/simteleop/workshop
if [ ! -d "$DEST/.git" ]; then
    sudo -u ubuntu git clone "$SIMTELEOP_WORKSHOP_REPO" "$DEST"
fi
cd "$DEST"
sudo -u ubuntu git fetch -q origin
sudo -u ubuntu git checkout -q "$SIMTELEOP_WORKSHOP_COMMIT"
# The HDRIs and USDs are plain files in the repo (no LFS), but be safe if that changes upstream.
sudo -u ubuntu git lfs pull 2>/dev/null || true
echo "workshop at $(sudo -u ubuntu git rev-parse HEAD) ($(sudo -u ubuntu git log -1 --format=%cd))"
test -f source/sim_to_real_so101/assets/usd/SO-ARM101-USD.usd
test -f docker/sim/Dockerfile
install -d -o ubuntu -g ubuntu "$DEST/datasets" "$DEST/outputs"
