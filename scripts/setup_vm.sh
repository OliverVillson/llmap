#!/usr/bin/env bash
# One-time setup for an evroc GPU VM, B200 or L40S (Ubuntu 24.04).
#   NVME=/mnt/nvme bash scripts/setup_vm.sh
#   SKIP_EXTRAS=1 NVME=/mnt/nvme bash scripts/setup_vm.sh   # only what Event Horizon uses
# Creates two venvs (vLLM and training, which pin different torch versions),
# builds llama.cpp with CUDA at LLAMA_CPP_SHA, fetches the code suites and downloads
# the teacher and student weights to the local NVMe. Safe to re-run. Local NVMe is
# wiped if the VM is stopped.
# SKIP_EXTRAS=1 leaves out Node/TypeScript, llm-compressor and Aider Polyglot, and
# fetches only the livecodebench, codecontests and apps suites: ~15-20 minutes less
# per boot, which a B200 pays for by the minute.
# SKIP_SUITES=1 fetches no code suite: the caller restores them from the bucket once the
# NVMe is mounted (scripts/event_horizon.py, after the first session's fetch).
# TEACHER_REVISION: the revision the teacher is downloaded at (default: the hub's main).
# Pins: vLLM, and torch, transformers and accelerate in the training venv. UV_EXCLUDE_NEWER
# (an RFC 3339 time, read by every uv call) resolves the rest as PyPI stood then, and
# CONSTRAINTS (a dir of <venv>.txt, an earlier setup's $NVME/venvs) installs each venv at
# exactly those versions; both keep the venvs the same from one VM to the next. What
# everything resolved to, and the llama.cpp commit, go to $NVME/versions.txt, so a reading
# can say what produced it, and each venv's freeze to $NVME/venvs/<venv>.txt.
set -euo pipefail

NVME=${NVME:-/mnt/nvme}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEACHER=${TEACHER:-Qwen/Qwen3-30B-A3B-Instruct-2507}
STUDENT=${STUDENT-google/gemma-4-E4B-it}  # STUDENT= (empty) skips it
TEACHER_REVISION=${TEACHER_REVISION:-}
SKIP_EXTRAS=${SKIP_EXTRAS:-}
SKIP_SUITES=${SKIP_SUITES:-}
CONSTRAINTS=${CONSTRAINTS:-}
if [ -n "${UV_EXCLUDE_NEWER:-}" ]; then export UV_EXCLUDE_NEWER; fi
# The llama.cpp commit everything GGUF comes from: llama-quantize's formats and type
# rules, convert_hf_to_gguf.py and gguf-py's dequantization all change between commits,
# and the shipped file must be reproducible.
LLAMA_CPP_SHA=${LLAMA_CPP_SHA:-af50f2f248b475d9760f8e67503d01c0f194f5f4}
LLAMA_CPP_REPO=${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp}

say() { printf '\n==> %s\n' "$*"; }

# llama_cpp_checkout DIR: DIR holds llama.cpp at exactly LLAMA_CPP_SHA, fetched on its
# own (--depth 1). A checkout already there at another commit is moved to the pin when
# it has no local changes to tracked files, and refused when it has.
llama_cpp_checkout() {
  local dir=$1 head
  [[ $LLAMA_CPP_SHA =~ ^[0-9a-f]{40}$ ]] || { echo "LLAMA_CPP_SHA must be a full 40-character commit, not '$LLAMA_CPP_SHA'"; return 1; }
  if [ ! -e "$dir/.git" ]; then
    if [ -e "$dir" ] && [ -n "$(ls -A "$dir")" ]; then
      echo "$dir is there but is not a git checkout; remove it and re-run"; return 1
    fi
    git init -q "$dir"
  fi
  head=$(git -C "$dir" rev-parse -q --verify HEAD || true)
  if [ "$head" != "$LLAMA_CPP_SHA" ]; then
    if [ -n "$head" ] && [ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]; then
      echo "$dir is at $head with local changes, not at LLAMA_CPP_SHA $LLAMA_CPP_SHA; remove it and re-run"; return 1
    fi
    git -C "$dir" fetch -q --depth 1 "$LLAMA_CPP_REPO" "$LLAMA_CPP_SHA"
    git -C "$dir" checkout -q --detach FETCH_HEAD
  fi
  head=$(git -C "$dir" rev-parse HEAD)
  [ "$head" = "$LLAMA_CPP_SHA" ] || { echo "$dir is at $head, not at LLAMA_CPP_SHA $LLAMA_CPP_SHA"; return 1; }
  echo "llama.cpp at $head"
}

# missing_suites DIR SUITE...: the suites DIR has no finished copy of. A suite counts as
# fetched once scripts/fetch_codebench.py has written its meta file, which it does last;
# a suite restored from the bucket counts too.
missing_suites() {
  local dir=$1 s
  shift
  for s in "$@"; do
    case $s in
      multipl-e) [ -f "$dir/multipl-e-py.jsonl" ] || echo "$s" ;;
      general-calib) [ -f "$dir/general-calib.jsonl" ] || echo "$s" ;;
      *) [ -f "$dir/$s.meta.json" ] || echo "$s" ;;
    esac
  done
}

# constraints_for VENV: uv's constraint arguments (one per line) for VENV when CONSTRAINTS
# holds an earlier setup's freeze of it, else nothing.
constraints_for() {
  if [ -n "$CONSTRAINTS" ] && [ -f "$CONSTRAINTS/$1.txt" ]; then printf -- '-c\n%s\n' "$CONSTRAINTS/$1.txt"; fi
}

# pinned_lines: a `uv pip freeze` as constraints: only name==version lines (not the repo's
# editable install or gguf from the llama.cpp checkout), names normalized, sorted, so two
# VMs that installed the same versions write the same file.
pinned_lines() {
  { grep -E '^[A-Za-z0-9][A-Za-z0-9._-]*==' || true; } | awk -F'==' '{n=tolower($1); gsub(/[._]/, "-", n); print n "==" $2}' \
    | LC_ALL=C sort
}

# tests/test_fetch_codebench.py sources this file for the functions above only.
if [ -n "${SETUP_VM_LIB:-}" ]; then return 0 2>/dev/null || exit 0; fi

# evroc's GPU image installs the NVIDIA driver on first boot and then reboots.
command -v cloud-init >/dev/null && sudo cloud-init status --wait >/dev/null || true

say "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

say "Local NVMe at $NVME"
if ! mountpoint -q "$NVME"; then
  # The local disk arrives blank. Only a disk with no filesystem or partitions is formatted.
  DISK=$(lsblk -dbno NAME,SIZE,TYPE | awk '$3=="disk" && $2>1e12 {print "/dev/"$1}' | while read -r d; do
    [ -z "$(sudo blkid -o value -s TYPE "$d")" ] && [ "$(lsblk -no NAME "$d" | wc -l)" = 1 ] && echo "$d"; done | head -1)
  [ -n "$DISK" ] || { echo "No blank disk over 1 TB to mount at $NVME"; exit 1; }
  sudo mkfs.ext4 -q -L nvme -E nodiscard,lazy_itable_init=1,lazy_journal_init=1 "$DISK"
  sudo mkdir -p "$NVME"
  grep -q "LABEL=nvme" /etc/fstab || echo "LABEL=nvme $NVME ext4 defaults,noatime,nofail 0 2" | sudo tee -a /etc/fstab
  sudo mount "$NVME"
fi
sudo chown "$(id -u):$(id -g)" "$NVME"
df -h "$NVME"
mkdir -p "$NVME/models" "$NVME/jobs"
export HF_HOME="$NVME/hf-cache"
# The root disk is 14 GB, too small for the torch and vLLM wheels in uv's cache.
export UV_CACHE_DIR="$NVME/uv-cache"

say "System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential cmake git git-lfs python3-venv python3-dev rsync curl
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

if grep -q DOWNLOADS_DONE "$NVME/download.log" 2>/dev/null; then
  say "Weights already downloaded"
elif pgrep -f "hf download" >/dev/null; then
  say "Weights download already running (log: $NVME/download.log)"
else
say "Downloading weights in the background (log: $NVME/download.log)"
(
  uv tool run --from huggingface_hub hf download "$TEACHER" ${TEACHER_REVISION:+--revision "$TEACHER_REVISION"} \
    --local-dir "$NVME/models/${TEACHER##*/}"
  if [ -n "$STUDENT" ]; then
    uv tool run --from huggingface_hub hf download "$STUDENT" --local-dir "$NVME/models/${STUDENT##*/}"
  fi
  echo DOWNLOADS_DONE
) > "$NVME/download.log" 2>&1 &
fi

mapfile -t VLLM_C < <(constraints_for venv-vllm)
mapfile -t TRAIN_C < <(constraints_for venv-train)
say "vLLM venv (data stage)${UV_EXCLUDE_NEWER:+, PyPI as of $UV_EXCLUDE_NEWER}${VLLM_C[1]:+, constraints $CONSTRAINTS}"
uv venv -q --python 3.12 "$NVME/venv-vllm"
uv pip install -q --python "$NVME/venv-vllm/bin/python" "${VLLM_C[@]}" \
  "vllm==0.31.0" pyyaml httpx  # the version exp04 was checked against
uv pip install -q --python "$NVME/venv-vllm/bin/python" -e "$REPO" --no-deps

say "Training venv (reap, heal, quantize, eval, package, API)"
# gguf comes from the pinned llama.cpp checkout below, not from PyPI.
uv venv -q --python 3.12 "$NVME/venv-train"
uv pip install -q --python "$NVME/venv-train/bin/python" "${TRAIN_C[@]}" \
  "torch==2.14.0" "transformers==5.17.0" "peft>=0.17" "accelerate==1.15.0" safetensors \
  anthropic httpx fastapi uvicorn sentencepiece protobuf
uv pip install -q --python "$NVME/venv-train/bin/python" -e "$REPO" --no-deps

# Written before the llama.cpp build so the data stage works even if that step fails.
cat > "$REPO/.env.vm" <<ENV
export LOBBOT_MODELS=$NVME/models
export LOBBOT_LLAMA_CPP=$NVME/llama.cpp
export LOBBOT_VLLM_PY=$NVME/venv-vllm/bin/python
export LOBBOT_CODEBENCH=$NVME/codebench
export HF_HOME=$NVME/hf-cache
export UV_CACHE_DIR=$NVME/uv-cache
export LOBBOT_JOBS=$NVME/jobs
export PATH=$NVME/venv-train/bin:\$PATH
# Temp files (the sandbox's cases.json copies) and the vLLM/torch compile caches on the
# NVMe: the root disk is 14 GB.
export TMPDIR=$NVME/tmp
export XDG_CACHE_HOME=$NVME/cache
export VLLM_CACHE_ROOT=$NVME/cache/vllm
export TRITON_CACHE_DIR=$NVME/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$NVME/cache/torchinductor
mkdir -p \$TMPDIR \$VLLM_CACHE_ROOT \$TRITON_CACHE_DIR \$TORCHINDUCTOR_CACHE_DIR
# API keys (GEMINI_API_KEY, ANTHROPIC_API_KEY) live outside the repo, mode 600.
if [ -f \$HOME/.lobbot-env ]; then . \$HOME/.lobbot-env; fi
ENV
if [ -n "${UV_EXCLUDE_NEWER:-}" ]; then echo "export UV_EXCLUDE_NEWER=$UV_EXCLUDE_NEWER" >> "$REPO/.env.vm"; fi

say "vLLM smoke test"
"$NVME/venv-vllm/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda ok:', torch.cuda.is_available())" || echo 'WARNING: vLLM import failed; the data stage will not run'

say "llama.cpp with CUDA at $LLAMA_CPP_SHA"
if ! command -v nvcc >/dev/null && [ ! -x /usr/local/cuda/bin/nvcc ]; then
  # Just the compiler and the libraries llama.cpp links; the full toolkit does not fit the root disk.
  # evroc's image already has NVIDIA's apt repo; add it if not.
  if ! apt-cache show cuda-nvcc-12-9 >/dev/null 2>&1; then
    curl -fsSLo /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i /tmp/cuda-keyring.deb && sudo apt-get update -qq
  fi
  sudo apt-get install -y -qq --no-install-recommends cuda-nvcc-12-9 cuda-cudart-dev-12-9 libcublas-dev-12-9 cuda-nvrtc-dev-12-9
  sudo apt-get clean
fi
export PATH="/usr/local/cuda/bin:$PATH"
llama_cpp_checkout "$NVME/llama.cpp"
# The GPU's own architecture: 100 on a B200, 89 on an L40S.
CUDA_ARCH=${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.[:space:]')}
cmake -S "$NVME/llama.cpp" -B "$NVME/llama.cpp/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_BUILD_TYPE=Release >/dev/null
# perplexity: the KL-divergence checks; tokenize: llama-tokenize == HF; batched-bench: slot sizing.
cmake --build "$NVME/llama.cpp/build" -j"$(nproc)" --target llama-quantize llama-imatrix llama-server llama-cli \
  llama-perplexity llama-tokenize llama-batched-bench
# gguf-py from the same commit: qheal's and gguf_build's dequantization must be the
# checkout's. convert_hf_to_gguf.py's own requirements file pins a CPU torch and an
# older transformers, which would replace the venv's, so only gguf is installed and
# the converter's imports are checked instead.
uv pip install -q --python "$NVME/venv-train/bin/python" "${TRAIN_C[@]}" --reinstall-package gguf "$NVME/llama.cpp/gguf-py"
"$NVME/venv-train/bin/python" "$NVME/llama.cpp/convert_hf_to_gguf.py" --help >/dev/null

say "Code suites (scripts/fetch_codebench.py)"
SUITES="livecodebench codecontests apps"
[ -n "$SKIP_EXTRAS" ] || SUITES="multipl-e $SUITES general-calib"
MISSING=$(missing_suites "$NVME/codebench" $SUITES | tr '\n' ' ')
if [ -n "$SKIP_SUITES" ]; then
  echo "not fetched (SKIP_SUITES=1): the caller restores them from the bucket"
elif [ -n "${MISSING// }" ]; then
  uv run -q --python 3.12 --with datasets --with huggingface_hub \
    python "$REPO/scripts/fetch_codebench.py" --out "$NVME/codebench" --only $MISSING \
    || echo "WARNING: not every code suite downloaded ($MISSING): code eval fails without livecodebench, trace generation without codecontests"
else
  echo "already there: $SUITES"
fi

if [ -z "$SKIP_EXTRAS" ]; then
  say "Code eval in JavaScript/TypeScript: Node 22 + TypeScript (experiment 01)"
  if ! node -e 'process.exit(+process.versions.node.split(".")[0] < 22)' 2>/dev/null; then
    NODE_TGZ=$(curl -fsSL https://nodejs.org/dist/latest-v22.x/SHASUMS256.txt | grep -o 'node-v22[^ ]*-linux-x64.tar.xz' | head -1)
    curl -fsSL "https://nodejs.org/dist/latest-v22.x/$NODE_TGZ" | tar -xJ -C "$NVME"
    ln -sfn "$NVME/${NODE_TGZ%.tar.xz}" "$NVME/node"
  fi
  export PATH="$NVME/node/bin:$PATH"
  command -v tsc >/dev/null || npm install -g -s --prefix "$NVME/node" typescript
  grep -q "$NVME/node/bin" "$REPO/.env.vm" || echo "export PATH=$NVME/node/bin:\$PATH" >> "$REPO/.env.vm"

  say "llm-compressor in the training venv (quant_format w4a16: int4 experts, FP8 attention, for vLLM)"
  # 0.14 pins transformers 5.15-5.17, torch 2.10-2.14.0 and accelerate 1.15.0: this venv's pins
  # (transformers 5.17.0, torch 2.14.0, accelerate 1.15.0) are inside them, so it shares the venv.
  uv pip install -q --python "$NVME/venv-train/bin/python" "${TRAIN_C[@]}" "llmcompressor==0.14.0"
fi

say "Training venv check, after its last install"
"$NVME/venv-train/bin/python" -c "import torch; assert torch.cuda.is_available(), torch.__version__; print('train venv torch', torch.__version__, 'cuda ok')"

say "Venv freezes: $NVME/venvs (an evroc run pins the first VM's for the next ones)"
mkdir -p "$NVME/venvs"
for v in venv-vllm venv-train; do
  uv pip freeze --python "$NVME/$v/bin/python" | pinned_lines > "$NVME/venvs/$v.txt.part"
  mv "$NVME/venvs/$v.txt.part" "$NVME/venvs/$v.txt"
done

say "Versions: $NVME/versions.txt"
{
  echo "# scripts/setup_vm.sh $(date -u +%FT%TZ), repo $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/gpu: /'
  echo "nvcc: $(nvcc --version | tail -1)"
  echo "llama.cpp: $(git -C "$NVME/llama.cpp" rev-parse HEAD)"
  echo "## venv-vllm"
  uv pip freeze --python "$NVME/venv-vllm/bin/python"
  echo "## venv-train"
  uv pip freeze --python "$NVME/venv-train/bin/python"
} > "$NVME/versions.txt"
grep -E '^(gpu|nvcc|llama\.cpp): |^(vllm|torch|transformers|accelerate|gguf)==' "$NVME/versions.txt" || true

say "Done. Weights still downloading: tail -f $NVME/download.log"
echo "Then: source .env.vm && python pipeline.py --job $NVME/jobs/demo  (after copying a taskspec.json there)"

if [ -z "$SKIP_EXTRAS" ]; then
  say "Aider Polyglot: docker, the pinned aider and polyglot-benchmark, aider's benchmark image"
  # scripts/polyglot.py setup is idempotent; the image build takes ~15 minutes the first time.
  NVME="$NVME" "$NVME/venv-train/bin/python" "$REPO/scripts/polyglot.py" setup \
    || echo 'WARNING: Polyglot setup failed; re-run `python scripts/polyglot.py setup` before `polyglot.py run`'
fi
