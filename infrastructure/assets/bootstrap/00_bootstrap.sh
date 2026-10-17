#!/usr/bin/env bash
# First-boot orchestrator, called from EC2 user data after s3://$SIMTELEOP_BUCKET/code/ was synced to
# /opt/simteleop/code. Re-runnable by hand after fixing a step:
#   sudo bash /opt/simteleop/code/bootstrap/00_bootstrap.sh
# Each step is idempotent; a finished step leaves a marker in /var/lib/simteleop-bootstrap.
set -xeuo pipefail
# shellcheck disable=SC1091
. /etc/simteleop.env
export SIMTELEOP_BUCKET AWS_DEFAULT_REGION AWS_REGION

HERE=$(cd "$(dirname "$0")" && pwd)
MARK=/var/lib/simteleop-bootstrap
mkdir -p "$MARK"

step () {
    local name=$1
    if [ -f "$MARK/$name.done" ]; then echo "== $name already done, skipping"; return 0; fi
    echo "== $name"
    if bash "$HERE/$name.sh"; then touch "$MARK/$name.done"; else echo "!! $name FAILED -- see /var/log/simteleop-bootstrap.log" >&2; return 1; fi
}

step 10_system
step 20_dcv
step 30_docker_gpu
step 40_operator_commands
step 50_workshop_checkout
# The image build pulls ~20 GB from nvcr.io and takes 30-40 minutes. Not fatal: the box is useful
# (DCV, docker, commands) even if this has to be retried with `simteleop-build-image`.
echo "== 60_sim_image"
if bash "$HERE/60_sim_image.sh"; then touch "$MARK/60_sim_image.done"; else echo "!! 60_sim_image did not complete -- run: sudo simteleop-build-image" >&2; fi
step 70_agent_venv

echo
echo "================================================================"
if [ -f "$MARK/60_sim_image.done" ]; then
    echo " bootstrap complete -- the platform is ready"
    echo "   simteleop-up            start the simulator + env server (add --gui for the DCV window)"
    echo "   simteleop-status        is it alive?"
    echo "   simteleop-agent --check preflight; then: simteleop-agent --episodes 5"
else
    echo " bootstrap complete EXCEPT the simulator image; run: sudo simteleop-build-image"
fi
echo "================================================================"
