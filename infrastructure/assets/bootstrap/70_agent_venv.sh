#!/usr/bin/env bash
# The agent runs on the HOST (not in the sim container): Strands + boto3 in a venv, reasoning through
# the instance role. Python 3.10+ is required by Strands; Ubuntu 22.04 ships 3.10.
set -xeuo pipefail
VENV=/home/ubuntu/simteleop/agent-venv
CODE=/opt/simteleop/code
# Not every AMI ships the venv module (seen: Python 3.14 without python3-venv).
python3 -m venv --help >/dev/null 2>&1 && python3 -c "import ensurepip" 2>/dev/null || { export DEBIAN_FRONTEND=noninteractive; apt-get update -y; apt-get install -y python3-venv python3-pip; }
# A venv created while ensurepip was missing has a python but no pip: rebuild it.
if [ ! -x "$VENV/bin/pip" ]; then rm -rf "$VENV"; sudo -u ubuntu python3 -m venv "$VENV"; fi
sudo -u ubuntu "$VENV/bin/pip" install -q --upgrade pip
# setuptools writes *.egg-info into the source tree, and /opt/simteleop/code is root-owned: install
# from a copy the user owns.
SRC=$(sudo -u ubuntu mktemp -d)
cp -r "$CODE/common" "$CODE/agent" "$SRC/" && chown -R ubuntu:ubuntu "$SRC"
sudo -u ubuntu "$VENV/bin/pip" install -q "$SRC/common" "$SRC/agent"
rm -rf "$SRC"
sudo -u ubuntu "$VENV/bin/python" -c "import boto3, simteleop_agent, importlib.metadata as m; print('agent venv ok, strands-agents', m.version('strands-agents'))"
