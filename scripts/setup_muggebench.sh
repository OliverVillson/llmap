#!/usr/bin/env bash
# MuggeBench's prerequisites on a GPU VM, after scripts/setup_vm.sh has mounted the NVMe:
# Docker with its images on the NVMe (the root disk is 14 GB), Bun for the runner
# (mugge/bench/run.ts), and the sandbox images the bench projects check in
# (mugge/sandbox/build.sh). Safe to re-run.
#   NVME=/mnt/nvme bash scripts/setup_muggebench.sh [image ...]   (default: c node python web dotnet)
set -euo pipefail
NVME=${NVME:-/mnt/nvme}
BUN_VERSION=${BUN_VERSION:-1.3.14}  # the node sandbox image's
REPO=$(cd "$(dirname "$0")/.." && pwd)
images=("$@")
[ ${#images[@]} -eq 0 ] && images=(c node python web dotnet)
apt_install() { sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$@" >/dev/null; }

if ! command -v docker >/dev/null; then
  sudo apt-get update -qq
  apt_install docker.io
fi
# Images on the NVMe: data-root there, and the classic image store, since Docker's
# containerd store (the default on new installs) keeps every image under
# /var/lib/containerd on the root disk whatever data-root says.
sudo mkdir -p /etc/docker
conf=$(sudo python3 - "$NVME/docker" <<'EOF'
import json, sys
from pathlib import Path
p = Path("/etc/docker/daemon.json")
conf = json.loads(p.read_text() or "{}") if p.exists() else {}
want = {**conf, "data-root": sys.argv[1], "features": {**conf.get("features", {}), "containerd-snapshotter": False}}
if want != conf:  # merged into what is there (an NVIDIA runtime, say)
    print(json.dumps(want, indent=2))
EOF
)
if [ -n "$conf" ]; then
  # what the old stores hold is on the root disk and is rebuilt on the NVMe: drop it first,
  # since a full root disk cannot even take the new daemon.json
  sudo systemctl stop docker docker.socket containerd 2>/dev/null || true
  sudo rm -rf /var/lib/docker /var/lib/containerd
  echo "$conf" | sudo tee /etc/docker/daemon.json >/dev/null
  sudo systemctl start containerd
fi
sudo systemctl enable --now docker
for i in $(seq 30); do sudo docker info >/dev/null 2>&1 && break; sleep 2; done
sudo usermod -aG docker "$(id -un)"  # for later logins
sudo chmod 666 /var/run/docker.sock  # for this one, without a new login
root=$(docker info --format '{{.DockerRootDir}}')
store=$(docker info --format '{{json .DriverStatus}}')
if [ "$root" != "$NVME/docker" ] || [[ "$store" == *containerd.snapshotter* ]]; then
  echo "Docker keeps its images at $root ($store), not on the NVMe at $NVME/docker"; exit 1
fi
df -h / "$NVME" | sed 's/^/  /'

if [ "$("$HOME/.bun/bin/bun" --version 2>/dev/null)" != "$BUN_VERSION" ]; then
  command -v unzip >/dev/null || apt_install unzip
  curl -fsSL https://bun.sh/install | bash -s "bun-v$BUN_VERSION"
fi
"$HOME/.bun/bin/bun" --version

# Base images, from Google's Docker Hub mirror when Docker Hub refuses (it rate-limits
# anonymous pulls per IP).
for base in debian:bookworm-slim node:22-bookworm-slim python:3.12-slim-bookworm; do
  docker image inspect "$base" >/dev/null 2>&1 && continue
  docker pull -q "$base" || { docker pull -q "mirror.gcr.io/library/$base" && docker tag "mirror.gcr.io/library/$base" "$base"; }
done
cd "$REPO/mugge"
ENGINE=docker bash sandbox/build.sh "${images[@]}"
echo MUGGEBENCH_SETUP_DONE
