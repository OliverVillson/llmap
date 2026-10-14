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
sudo mkdir -p /etc/docker
changed=$(sudo python3 - "$NVME/docker" <<'EOF'
import json, sys
from pathlib import Path
p = Path("/etc/docker/daemon.json")
conf = json.loads(p.read_text() or "{}") if p.exists() else {}
if conf.get("data-root") != sys.argv[1]:
    conf["data-root"] = sys.argv[1]  # merged into what is there (an NVIDIA runtime, say)
    p.write_text(json.dumps(conf, indent=2) + "\n")
    print("changed")
EOF
)
sudo systemctl enable --now docker
if [ "$changed" = changed ]; then sudo systemctl restart docker; fi
for i in $(seq 30); do sudo docker info >/dev/null 2>&1 && break; sleep 2; done
sudo usermod -aG docker "$(id -un)"  # for later logins
sudo chmod 666 /var/run/docker.sock  # for this one, without a new login
docker info >/dev/null

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
