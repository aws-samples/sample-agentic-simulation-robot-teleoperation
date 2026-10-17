#!/usr/bin/env bash
# NICE DCV server plus a virtual session owned by ubuntu. Two details make the difference between a
# working GUI and a healthy process with no window: the session must be owned by the user that runs
# the container, and its xauth cookie must be usable inside the container (see 25_dcv_permissions.sh).
set -xeuo pipefail
export DEBIAN_FRONTEND=noninteractive
DCV_SESSION=${DCV_SESSION:-simteleop}
if ! command -v dcv >/dev/null 2>&1; then
    cd /tmp
    wget -q https://d1uj6qtbmh3dt5.cloudfront.net/NICE-GPG-KEY && gpg --import NICE-GPG-KEY
    wget -q https://d1uj6qtbmh3dt5.cloudfront.net/nice-dcv-ubuntu2204-x86_64.tgz
    tar -xzf nice-dcv-ubuntu2204-x86_64.tgz && cd nice-dcv-*-ubuntu2204-x86_64
    apt-get install -y ./nice-dcv-server_*.deb ./nice-dcv-web-viewer_*.deb
    apt-get install -y ./nice-xdcv_*.deb || true
    apt-get install -y ./nice-dcv-gl_*.deb || true
    usermod -aG video dcv || true
fi
systemctl stop gdm3 2>/dev/null || true
systemctl disable gdm3 2>/dev/null || true
systemctl enable --now dcvserver
if ! dcv list-sessions | grep -q "$DCV_SESSION"; then
    dcv create-session --owner ubuntu --type virtual --name "SimTeleop" "$DCV_SESSION"
fi
for _i in $(seq 1 60); do dcv list-sessions 2>/dev/null | grep -q "$DCV_SESSION" && break; sleep 1; done
# Recreate the session on every boot: a stop/start otherwise leaves DCV with no session.
cat > /etc/systemd/system/simteleop-dcv-session.service <<UNIT
[Unit]
Description=Create the SO-101 DCV virtual session
After=dcvserver.service
Requires=dcvserver.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c '/usr/bin/dcv list-sessions | grep -q ${DCV_SESSION} || /usr/bin/dcv create-session --owner ubuntu --type virtual --name "SimTeleop" ${DCV_SESSION}'
ExecStartPost=/bin/bash -c 'DCV_SESSION=${DCV_SESSION} bash /opt/simteleop/code/bootstrap/25_dcv_permissions.sh || true'

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable simteleop-dcv-session.service
echo "DCV_SESSION=$DCV_SESSION" >> /etc/simteleop.env
bash /opt/simteleop/code/bootstrap/25_dcv_permissions.sh || echo "!! xauth permissions not set yet -- run simteleop-fix-display once DCV settles"
dcv list-sessions
echo "DCV ready. Set a password for the viewer login:  sudo passwd ubuntu"
