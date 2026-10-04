#!/usr/bin/env bash
# One-time setup of a Mugge project VM image (Ubuntu 24.04, e.g. an evroc L40S or B200 VM).
# Installs rootless podman (+ gVisor when available), bun, the mugge engine, vLLM in its own venv,
# and builds the sandbox images. Snapshot the disk afterwards so a reopen is "start VM, mmap weights".
#   MUGGE_SRC=~/llmap/mugge vm/setup.sh
set -euo pipefail
MUGGE_SRC="${MUGGE_SRC:-$(cd "$(dirname "$0")/.." && pwd)}"

sudo apt-get update
sudo apt-get install -y --no-install-recommends podman uidmap slirp4netns git curl rsync unzip python3-venv build-essential

# gVisor as an extra OCI runtime (MUGGE_OCI_RUNTIME=runsc). Optional; skipped if the repo is unreachable.
if ! command -v runsc >/dev/null; then
  ( curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" \
       | sudo tee /etc/apt/sources.list.d/gvisor.list >/dev/null \
    && sudo apt-get update && sudo apt-get install -y runsc ) || echo "gVisor not installed; sandboxes use the default runtime"
fi

# bun + the engine
command -v bun >/dev/null || curl -fsSL https://bun.sh/install | bash
export PATH="$HOME/.bun/bin:$PATH"
(cd "$MUGGE_SRC" && bun install && bun link)
mkdir -p "$HOME/.bun/bin" && ln -sf "$MUGGE_SRC/src/cli.ts" "$HOME/.bun/bin/mugge"

# vLLM in its own venv (only on GPU VMs)
if command -v nvidia-smi >/dev/null; then
  python3 -m venv "$HOME/venvs/vllm"
  "$HOME/venvs/vllm/bin/pip" install -U pip vllm
fi

# Toolchain images, kept on the persistent disk
ENGINE=podman "$MUGGE_SRC/sandbox/build.sh"
echo "done: start inference with vm/serve-vllm.sh, then mugge engine serve --project ~/mugge/<name>"
