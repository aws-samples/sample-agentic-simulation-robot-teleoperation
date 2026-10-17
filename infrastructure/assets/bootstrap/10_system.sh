#!/usr/bin/env bash
# OS packages and a desktop for the DCV session. On the Deep Learning Base AMI the driver, CUDA and
# Docker are already present.
set -xeuo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends ca-certificates curl wget unzip jq vim less htop tmux git git-lfs \
    build-essential python3-pip python3-venv ffmpeg
apt-get install -y --no-install-recommends ubuntu-desktop-minimal gdm3 mesa-utils xorg xauth x11-utils
cat > /etc/security/limits.d/99-simteleop.conf <<'LIM'
*  soft  nofile  65535
*  hard  nofile  65535
LIM
install -d -o ubuntu -g ubuntu /home/ubuntu/simteleop /home/ubuntu/simteleop/datasets /home/ubuntu/simteleop/runs
nvidia-smi || echo "!! nvidia-smi failed -- GPU driver missing; DCV and Isaac Sim will not work"
