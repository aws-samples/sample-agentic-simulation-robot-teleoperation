#!/usr/bin/env bash
# Operator commands on PATH: a fixed vocabulary instead of remembered docker invocations.
set -xeuo pipefail
for f in /opt/simteleop/code/bin/*; do [ -f "$f" ] || continue; install -m 0755 "$f" "/usr/local/bin/$(basename "$f")"; done
install -m 0755 /opt/simteleop/code/bootstrap/25_dcv_permissions.sh /usr/local/bin/simteleop-fix-display
install -m 0755 /opt/simteleop/code/bootstrap/60_sim_image.sh /usr/local/bin/simteleop-build-image
ls -l /usr/local/bin/simteleop-*
