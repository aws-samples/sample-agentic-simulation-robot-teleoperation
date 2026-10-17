#!/usr/bin/env bash
# Make the DCV session's X authority reachable by the container. Without it Isaac Sim runs perfectly
# with NO window and only warns 'GLFW initialization failed' once at startup. Re-runnable; installed
# as `simteleop-fix-display`.
set -xeuo pipefail
DCV_SESSION=${DCV_SESSION:-simteleop}
RUNDIR=/run/user/1000
XAUTH=$RUNDIR/dcv/$DCV_SESSION.xauth
for _i in $(seq 1 "${XAUTH_WAIT_S:-60}"); do [ -e "$XAUTH" ] && break; sleep 1; done
if [ ! -e "$XAUTH" ]; then echo "!! $XAUTH absent; sessions:" >&2; dcv list-sessions >&2 || true; exit 1; fi
chmod o+x "$RUNDIR"; chmod g+x "$RUNDIR"; chmod o+rx "$RUNDIR/dcv"; chmod g+rx "$RUNDIR/dcv"
chmod g+r "$XAUTH"; chmod o+r "$XAUTH"
ls -l "$XAUTH"
for sock in /tmp/.X11-unix/X*; do [ -S "$sock" ] && echo "X display available: :${sock##*/X}"; done
