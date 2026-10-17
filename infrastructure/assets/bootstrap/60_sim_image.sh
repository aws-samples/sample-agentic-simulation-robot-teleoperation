#!/usr/bin/env bash
# Build the simulator image: the workshop's own docker/sim/Dockerfile (Isaac Lab 2.3.2 + LeRobot at
# its pinned commit) with our env-server package layered on top. The base image is public on NGC, so
# no credentials are needed. First build pulls ~20 GB; allow 30-40 minutes. Installed as
# `simteleop-build-image` for retries.
set -xeuo pipefail
# shellcheck disable=SC1091
. /etc/simteleop.env
WS=/home/ubuntu/simteleop/workshop
CODE=/opt/simteleop/code
TAG=${SIMTELEOP_SIM_IMAGE:-simteleop-sim:workshop-${SIMTELEOP_WORKSHOP_COMMIT:0:8}}
if docker image inspect "$TAG" >/dev/null 2>&1 && [ "${FORCE:-0}" != "1" ]; then
    echo "image $TAG present; FORCE=1 to rebuild"; exit 0
fi
cd "$WS"
# Upstream drift: the workshop Dockerfile downloads an FFmpeg build from BtbN's rolling "latest"
# release, and the n7.1 asset it names was removed (404 on 2026-09-29). Substitute a build that
# exists; override with SIMTELEOP_FFMPEG_BUILD if that one disappears too (see
# https://github.com/BtbN/FFmpeg-Builds/releases/tag/latest). A no-op if upstream fixes the name.
FFBUILD=${SIMTELEOP_FFMPEG_BUILD:-ffmpeg-n8.1-latest-linux64-lgpl-shared-8.1}
sed "s#ffmpeg-n7.1-latest-linux64-lgpl-shared-7.1#${FFBUILD}#" docker/sim/Dockerfile > /tmp/Dockerfile.simteleop-base
docker build -t simteleop-workshop-base -f /tmp/Dockerfile.simteleop-base .
# Layer: our packages + Pillow. Installed into Isaac Lab's Python, non-editable, with the same
# constraints trick the workshop uses so nothing in the base gets upgraded.
BUILD=$(mktemp -d)
cp -r "$CODE/common" "$CODE/sim" "$BUILD/"
cat > "$BUILD/Dockerfile" <<'DF'
FROM simteleop-workshop-base
ENV PYTHON=/workspace/isaaclab/_isaac_sim/python.sh
COPY common /opt/simteleop/common
COPY sim /opt/simteleop/sim
# pyarrow is pulled unpinned by the workshop's lerobot install; 26.0.0 (2026-10) requires NumPy 2 while
# Isaac Lab pins numpy 1.26.0, and the env server then dies on `import datasets` (measured 2026-10-09
# on a fresh build; 21.0.0 imports fine). Pin it here until the workshop pins it upstream.
RUN printf '%s\n' "numpy==1.26.0" "packaging==23.0" "torch==2.7.0" "pyarrow==21.0.0" > /tmp/c.txt && \
    $PYTHON -m pip install -c /tmp/c.txt --no-deps /opt/simteleop/common /opt/simteleop/sim && \
    $PYTHON -m pip install -c /tmp/c.txt "Pillow>=10" "pyarrow==21.0.0"
DF
docker build -t "$TAG" "$BUILD"
rm -rf "$BUILD"
grep -q '^SIMTELEOP_SIM_IMAGE=' /etc/simteleop.env || echo "SIMTELEOP_SIM_IMAGE=$TAG" >> /etc/simteleop.env
docker image ls "$TAG"
