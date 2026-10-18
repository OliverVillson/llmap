#!/usr/bin/env bash
# Qwen3.8-27B-FP8 behind vLLM on one GPU, for a Centipede crew (run sheet: deploy/README.md).
#
#   deploy/qwen.sh fresh      a bare Ubuntu VM to ready: apt packages, the data disk, the checkout, uv and
#                             bun, setup, download, bun install, preflight; safe to rerun (deploy/fresh.sh)
#   deploy/qwen.sh setup      vLLM $VLLM_VERSION in its own venv (uv), and a GPU/driver check
#   deploy/qwen.sh preflight  every start prerequisite (GPU, disk, venv, compilers, nvcc, weights, bun);
#                             each FAIL line says the command that fixes it; start runs it (PREFLIGHT=0 skips)
#   deploy/qwen.sh download   the weights (~31 GB) into the data folder; checks free space first, resumes
#   deploy/qwen.sh where      the data folder, its free space, and where the venv and weights are
#   deploy/qwen.sh start      the server, detached: it survives logging out of ssh
#   deploy/qwen.sh wait       block until it answers (or exit 1 with the log tail if it died)
#   deploy/qwen.sh status     running?, KV size, requests running/waiting, KV use, preemptions
#   deploy/qwen.sh check      one plain call, one thinking call, one JSON call, then the crew at once
#   deploy/qwen.sh bench      speed at 1, 13, 32 and 64 calls at once, saved per checkpoint
#   deploy/qwen.sh sweep      bench each server variant in turn (SWEEP="fp8 fp8+mtp3 nvfp4 ..."), then compare
#   deploy/qwen.sh compare    the saved bench results side by side
#   deploy/qwen.sh logs       follow the server log
#   deploy/qwen.sh stop
#   deploy/qwen.sh cmd        print the vllm command it would run, and change nothing
#
# The GPU profile (b200, h100, l40s) is read from nvidia-smi; GPU=b200 forces one. Any profile
# value can be overridden: MAX_MODEL_LEN, KV_DTYPE, MAX_NUM_SEQS, GPU_UTIL. MTP=3 turns on the
# model's own multi-token prediction (3 draft tokens). ATTENTION=FLASH_ATTN|TRITON_ATTN|FLASHINFER
# picks the attention kernels (default: vLLM's choice, FlashInfer on a B200, given nvcc). QUANT=nvfp4 serves NVIDIA's NVFP4
# checkpoint instead of Qwen's FP8 one (Blackwell only; give it to download and start, the rest
# read what is running). CREW_WORKERS=N (default 10) at start sizes the server for a crew of N
# workers, their leads and the orchestrator; the other commands and deploy/crew.env read it back. The server listens
# on 127.0.0.1 only; reach it from another machine through an ssh tunnel (README).
# DATA_DIR is where the weights, the vLLM venv and the caches go (about 50 GB). Unset, setup picks
# the local filesystem with the most free space this user can write to, and every later command
# uses the same folder (saved in ~/qwen/data.dir).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# before anything that needs python3: fresh installs it
[ "${1:-}" != fresh ] || { shift; exec bash "$HERE/fresh.sh" "$@"; }
# uv and bun land here; /usr/local/bin links are made by fresh, this covers older installs
export PATH="$PATH:$HOME/.local/bin:$HOME/.bun/bin"

QUANT="${QUANT:-fp8}"
case "$QUANT" in
  fp8) : "${MODEL:=Qwen/Qwen3.8-27B-FP8}" ;;
  # ModelOpt mixed precision: NVFP4 MLPs and lm_head, FP8 attention and DeltaNet projections
  nvfp4) : "${MODEL:=nvidia/Qwen3.8-27B-NVFP4}" ;;
  *) echo "qwen.sh: QUANT must be fp8 or nvfp4" >&2; exit 1 ;;
esac
SERVED="${SERVED:-qwen3.8-27b}"
VLLM_VERSION="${VLLM_VERSION:-0.31.0}"
STATE="${STATE:-$HOME/qwen}"
DATA_FILE="$STATE/data.dir"
if [ -z "${DATA_DIR:-}" ] && [ -f "$DATA_FILE" ]; then DATA_DIR="$(cat "$DATA_FILE")"; fi
DATA_DIR="${DATA_DIR:-$(python3 "$HERE/disk.py" pick)}"
# the weights, vLLM's compile cache and Triton's kernels all go under the data folder
export HF_HOME="${HF_HOME:-$DATA_DIR/hf}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DATA_DIR/cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$DATA_DIR/cache/triton}" UV_CACHE_DIR="${UV_CACHE_DIR:-$DATA_DIR/cache/uv}"
# FlashInfer's compiled kernels: <base>/.cache/flashinfer
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$DATA_DIR}"
# a venv made before DATA_DIR existed stays where it is
if [ -z "${VENV:-}" ] && [ -x "$HOME/venvs/qwen-vllm/bin/python" ]; then VENV="$HOME/venvs/qwen-vllm"; fi
VENV="${VENV:-$DATA_DIR/venv}"
VLLM="${VLLM:-$VENV/bin/vllm}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MTP="${MTP:-0}"
LOG="$STATE/serve.log"
PIDFILE="$STATE/serve.pid"
RUNNING="$STATE/serve.quant"
URL="http://127.0.0.1:$PORT"
# the crew this server is for: N workers, their leads and the orchestrator (deploy/crew.py)
# set it once at start; later commands (wait, check, bench, over ssh too) read it back
CREW_FILE="$STATE/serve.crew"
if [ -z "${CREW_WORKERS:-}" ] && [ -f "$CREW_FILE" ]; then CREW_WORKERS="$(cat "$CREW_FILE")"; fi
CREW_WORKERS="${CREW_WORKERS:-10}"
read -r _ CREW_LEADS _ CREW_TOTAL CREW_AGENTS < <(python3 "$HERE/crew.py" "$CREW_WORKERS" --sizes)

die() { echo "qwen.sh: $*" >&2; exit 1; }

# Use DATA_DIR, after checking it has NEED GB free; remember it for every later command.
use_data_dir() {
  local need="$1" free
  # an earlier download into the default cache (before DATA_DIR) is often what filled the home disk
  local old="$HOME/.cache/huggingface/hub/models--${MODEL//\//--}"
  if [ "$HF_HOME" != "$HOME/.cache/huggingface" ] && [ -d "$old" ]; then
    echo "note: an earlier download is in $old ($(du -sh "$old" 2>/dev/null | cut -f1)); it is not used, remove it: rm -rf $old"
  fi
  mkdir -p "$DATA_DIR" 2>/dev/null || die "cannot create DATA_DIR=$DATA_DIR; pick a folder you can write to: DATA_DIR=/path deploy/qwen.sh $CMD_NAME"
  # the server's log, pid and crew size live in STATE (in the home folder): it needs a little room too
  { mkdir -p "$STATE" && touch "$DATA_FILE.new"; } 2>/dev/null ||
    die "cannot write $STATE: its disk is full ($(python3 "$HERE/disk.py" free "$STATE") GB free). Free some space there first$([ -d "$old" ] && echo " (rm -rf $old)"), then run deploy/qwen.sh $CMD_NAME again"
  free="$(python3 "$HERE/disk.py" free "$DATA_DIR")"
  echo "data folder $DATA_DIR: ${free} GB free, ${need} GB needed"
  if [ "$free" -lt "$need" ]; then
    echo "qwen.sh: not enough space in $DATA_DIR (${free} GB free, ${need} GB needed)." >&2
    python3 "$HERE/disk.py" bigger "$DATA_DIR" | sed 's/^/  bigger: /' >&2
    rm -f "$DATA_FILE.new"
    die "free space there, or point DATA_DIR at a bigger disk you can write to (README \"Disk\"): DATA_DIR=/path deploy/qwen.sh $CMD_NAME"
  fi
  # remembered only once it has room, so a folder that failed the check is not picked again
  echo "$DATA_DIR" > "$DATA_FILE.new" && mv "$DATA_FILE.new" "$DATA_FILE"
}

gpu_name() { command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true; }

# Which profile: GPU=…, else the first GPU's name.
detect_gpu() {
  [ -n "${GPU:-}" ] && { echo "$GPU"; return; }
  case "$(gpu_name | tr '[:upper:]' '[:lower:]')" in
    *b200*|*b300*|*gb200*) echo b200 ;;
    *h100*|*h200*|*h800*) echo h100 ;;
    *l40s*|*l40*) echo l40s ;;
    *) echo unknown ;;
  esac
}

# Per-GPU defaults (the arithmetic is in README.md "Memory and slots"). The server admits at
# least every call the crew can have in flight (CREW_TOTAL).
profile() {
  local g; g="$(detect_gpu)"
  [ -z "${KV_DTYPE:-}" ] || KV_DTYPE_SET=1
  if [ -n "${MAX_NUM_SEQS:-}" ] && [ "$MAX_NUM_SEQS" -lt "$CREW_TOTAL" ]; then
    die "MAX_NUM_SEQS=$MAX_NUM_SEQS is below the crew's $CREW_TOTAL calls (CREW_WORKERS=$CREW_WORKERS)"
  fi
  case "$g" in
    b200) : "${MAX_MODEL_LEN:=65536}" "${KV_DTYPE:=auto}" "${MAX_NUM_SEQS:=64}" "${GPU_UTIL:=0.94}" ;;
    h100) : "${MAX_MODEL_LEN:=32768}" "${KV_DTYPE:=auto}" "${MAX_NUM_SEQS:=16}" "${GPU_UTIL:=0.90}" ;;
    l40s) : "${MAX_MODEL_LEN:=16384}" "${KV_DTYPE:=fp8}" "${MAX_NUM_SEQS:=13}" "${GPU_UTIL:=0.92}" ;;
    *) die "unknown GPU '$(gpu_name)': set GPU=b200, h100 or l40s" ;;
  esac
  PROFILE="$g"
  [ "$MAX_NUM_SEQS" -ge "$CREW_TOTAL" ] || MAX_NUM_SEQS="$CREW_TOTAL"
  if [ "$QUANT" = nvfp4 ]; then
    [ "$g" = b200 ] || die "NVFP4 needs a Blackwell GPU (B200); this profile is $g"
    # NVIDIA's serve command for this checkpoint uses an fp8 KV cache (KV_DTYPE=auto to compare like for like)
    [ "$KV_DTYPE" != auto ] || [ -n "${KV_DTYPE_SET:-}" ] || KV_DTYPE=fp8_e4m3
  fi
}

serve_cmd() {
  profile
  [ "$HOST" = 127.0.0.1 ] || [ "${ALLOW_PUBLIC:-}" = 1 ] || die "HOST=$HOST would expose the model; use an ssh tunnel (README), or ALLOW_PUBLIC=1 with VLLM_API_KEY set and a firewall"
  if [ "$HOST" != 127.0.0.1 ] && [ -z "${VLLM_API_KEY:-}" ]; then die "a public HOST needs VLLM_API_KEY"; fi
  CMD=("$VLLM" serve "$MODEL"
    --served-model-name "$SERVED" "$MODEL"
    --host "$HOST" --port "$PORT"
    --tensor-parallel-size "$TP"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$GPU_UTIL"
    --kv-cache-dtype "$KV_DTYPE"
    --enable-prefix-caching
    --language-model-only
    --reasoning-parser qwen3)
  if [ "$MTP" != 0 ]; then CMD+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}"); fi
  # attention kernels: unset lets vLLM choose (FlashInfer with TRT-LLM kernels on a B200, the
  # fastest). ATTENTION=FLASH_ATTN, TRITON_ATTN or FLASHINFER forces one. vLLM uses FlashInfer for
  # more than attention on this model, so nvcc is needed whichever is picked (cuda_env).
  cuda_env
  if [ -x "$VENV/bin/python" ] && [ -z "${CUDA_HOME:-}" ] && ! command -v nvcc >/dev/null && [ ! -x /usr/local/cuda/bin/nvcc ]; then
    echo "warning: no nvcc; FlashInfer will fail to compile its kernels at start (README \"nvcc\")" >&2
  fi
  if [ -n "${ATTENTION:-}" ]; then CMD+=(--attention-backend "$ATTENTION"); fi
  # anything else, e.g. EXTRA_ARGS="--max-num-batched-tokens 16384"
  if [ -n "${EXTRA_ARGS:-}" ]; then read -r -a extra <<< "$EXTRA_ARGS"; CMD+=("${extra[@]}"); fi
}

# the boot it was started in: after a reboot the saved pid may be some other process's
boot_id() { cat /proc/sys/kernel/random/boot_id 2>/dev/null || true; }
pid_alive() {
  [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null || return 1
  [ ! -f "$STATE/serve.boot" ] || [ "$(cat "$STATE/serve.boot")" = "$(boot_id)" ]
}

# Triton compiles small C helpers on the first start (and the model check fails without them), so
# the venv's Python headers and a C compiler must be installed. Installs them with apt when sudo
# needs no password, else says the apt line.
build_tools() {
  [ -x "$VENV/bin/python" ] || return 0
  local missing=() dev
  dev="$("$VENV/bin/python" -c 'import sys; print("python%d.%d-dev" % sys.version_info[:2])')"
  "$VENV/bin/python" -c 'import os, sys, sysconfig; sys.exit(not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")))' || missing+=("$dev")
  # gcc for Triton's helpers, g++ as nvcc's host compiler for FlashInfer's kernels
  { command -v gcc >/dev/null || command -v cc >/dev/null; } && command -v g++ >/dev/null || missing+=(build-essential)
  [ "${#missing[@]}" -gt 0 ] || return 0
  if command -v apt-get >/dev/null && sudo -n true 2>/dev/null; then
    echo "installing ${missing[*]} (Triton builds C helpers at start)"
    sudo apt-get install -y "${missing[@]}" >/dev/null || die "apt-get install ${missing[*]} failed"
  else
    die "vLLM's Triton kernels need ${missing[*]}: sudo apt-get install -y ${missing[*]}, then run this again"
  fi
}

# FlashInfer (vLLM's fastest attention on a B200) compiles parts of itself with nvcc on first use
# and only looks in CUDA_HOME, PATH and /usr/local/cuda. vLLM's own dependencies put nvcc 13 and
# the CUDA headers in the venv (site-packages/nvidia/cu13), so point CUDA_HOME there, with the
# lib64, libcudart.so and libcuda.so names FlashInfer's link line expects. No system toolkit needed.
cuda_env() {
  # FlashInfer builds with ninja, which vLLM installs in the venv; vllm serve runs without the venv
  # activated, so its bin goes on PATH
  [ ! -x "$VENV/bin/ninja" ] || export PATH="$VENV/bin:$PATH"
  [ -z "${CUDA_HOME:-}" ] || return 0
  command -v nvcc >/dev/null && return 0
  [ -x /usr/local/cuda/bin/nvcc ] && return 0
  local d libcuda
  d="$(ls -d "$VENV"/lib/python3*/site-packages/nvidia/cu13 2>/dev/null | head -1 || true)"
  [ -n "$d" ] && [ -x "$d/bin/nvcc" ] || return 0
  [ -e "$d/lib64" ] || ln -s lib "$d/lib64"
  [ -e "$d/lib/libcudart.so" ] || ln -s libcudart.so.13 "$d/lib/libcudart.so"
  if [ ! -e "$d/lib/stubs/libcuda.so" ]; then
    libcuda="$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1 /{print $NF; exit}')"
    [ -z "$libcuda" ] || { mkdir -p "$d/lib/stubs"; ln -sf "$libcuda" "$d/lib/stubs/libcuda.so"; }
  fi
  export CUDA_HOME="$d" PATH="$d/bin:$PATH"
  # nvcc must be the same CUDA 13.x as the runtime headers torch brought, or FlashInfer's build
  # stops at "CUDA compiler and CUDA toolkit headers are incompatible"; vLLM's deps pull the newest
  local versions uv
  versions="$("$VENV/bin/python" -c 'import importlib.metadata as m
v = lambda p: ".".join(m.version(p).split(".")[:2])
print(v("nvidia-cuda-runtime"), v("nvidia-cuda-nvcc"))' 2>/dev/null || true)"
  set -- $versions
  if [ $# -eq 2 ] && [ "$1" != "$2" ]; then
    uv="$(command -v uv || echo "$HOME/.local/bin/uv")"
    echo "nvcc $2 does not match the CUDA $1 runtime: installing nvcc $1 in the venv"
    VIRTUAL_ENV="$VENV" "$uv" pip install -q "nvidia-cuda-nvcc==$1.*" "nvidia-cuda-crt==$1.*" "nvidia-nvvm==$1.*" ||
      die "could not install nvcc $1: VIRTUAL_ENV=$VENV uv pip install 'nvidia-cuda-nvcc==$1.*' 'nvidia-cuda-crt==$1.*' 'nvidia-nvvm==$1.*'"
  fi
}

# Every prerequisite of a start, checked in a few seconds before anything loads, each failure with
# the line that fixes it: the failures seen on the B200 (disk full, Python.h, nvcc, ninja, g++,
# nvcc vs CUDA headers, bun off PATH) and the next likely ones (GPU held by an old server, port
# taken, half-downloaded weights, driver older than torch's CUDA). NO_GPU=1 skips the GPU lines.
preflight() {
  local fails=0 warns=0 py="$VENV/bin/python" tmp free v
  ok() { printf 'ok    %s\n' "$1"; }
  bad() { printf 'FAIL  %s\n      fix: %s\n' "$1" "$2"; fails=$((fails + 1)); }
  warn() { printf 'warn  %s\n' "$1"; warns=$((warns + 1)); }

  # the GPU: driver up, nothing else holding its memory, libcuda for Triton
  if [ -z "${NO_GPU:-}" ]; then
    if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
      bad "GPU: nvidia-smi does not work (no driver loaded)" "sudo reboot, or a VM made from evroc's GPU image (sudo ubuntu-drivers install --gpgpu && sudo reboot)"
    else
      local used total apps drv
      read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1 | tr -d ',')
      apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';')"
      if [ "${used:-0}" -gt 2048 ] && ! pid_alive; then
        bad "GPU: ${used} MiB of ${total} MiB already in use (${apps:-no process listed})" "kill the process holding it: nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill"
      else
        ok "GPU: $(gpu_name), ${used} MiB of ${total} MiB in use"
      fi
      drv="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$' || true)"
      v="$("$py" -c 'import importlib.metadata as m; print(".".join(m.version("nvidia-cuda-runtime").split(".")[:2]))' 2>/dev/null || true)"
      if [ -n "$drv" ] && [ -n "$v" ] && python3 -c 'import sys; a, b = (tuple(map(int, x.split("."))) for x in sys.argv[1:]); sys.exit(a <= b)' "$v" "$drv"; then
        bad "driver: CUDA $drv is older than the CUDA $v runtime in the venv" "rm -rf $VENV && TORCH_BACKEND=cu${drv/./} deploy/qwen.sh setup   (or a newer driver)"
      elif [ -n "$drv" ]; then ok "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1), CUDA $drv${v:+, venv runtime $v}"; fi
      ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1 ' && ok "libcuda.so.1 (Triton loads it)" ||
        bad "libcuda.so.1 is not in the linker cache (Triton cannot start)" "sudo ldconfig; if it is still missing the driver is not installed properly: sudo ubuntu-drivers install --gpgpu && sudo reboot"
    fi
  fi

  # disks: the data folder for weights, venv and caches; the home folder for the log and state
  if [ ! -d "$DATA_DIR" ] || [ ! -w "$DATA_DIR" ]; then
    bad "data folder $DATA_DIR missing or not writable$(mountpoint -q /data 2>/dev/null || echo ' (/data is not mounted)')" "deploy/qwen.sh fresh   (mounts the data disk and makes the folder)"
  else
    free="$(python3 "$HERE/disk.py" free "$DATA_DIR")"
    if [ "$free" -lt 10 ]; then bad "data folder $DATA_DIR: only ${free} GB free (caches and compiled kernels need ~10 GB)" "deploy/qwen.sh where; free space there, e.g. rm -rf $DATA_DIR/cache/uv"
    else ok "data folder $DATA_DIR: ${free} GB free"; fi
  fi
  free="$(python3 "$HERE/disk.py" free "$HOME")"
  [ "$free" -ge 1 ] && ok "home disk: ${free} GB free" ||
    bad "home disk: ${free} GB free (the server log and state go in $STATE)" "sudo apt-get clean; rm -rf ~/.cache/uv ~/.cache/huggingface ~/.cache/pip; df -h ~"

  # the venv and vLLM
  if ! "$py" -c '' 2>/dev/null; then
    bad "venv: $VENV has no working Python" "deploy/qwen.sh setup"
  else
    v="$("$py" -c 'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null || true)"
    [ "$v" = "$VLLM_VERSION" ] && ok "vLLM $v in $VENV" || bad "vLLM in $VENV is '${v:-not installed}', expected $VLLM_VERSION" "deploy/qwen.sh setup"
    # Triton compiles a C helper against Python.h and its own cuda.h at the first start
    local inc tinc
    inc="$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
    tinc="$("$py" -c 'import importlib.util, os; s = importlib.util.find_spec("triton"); print(os.path.join(os.path.dirname(s.origin), "backends", "nvidia", "include") if s else "")' 2>/dev/null || true)"
    if [ ! -f "$inc/Python.h" ]; then
      bad "Python.h missing ($inc): Triton cannot build its helpers" "sudo apt-get install -y python$("$py" -c 'import sys; print("%d.%d" % sys.version_info[:2])')-dev"
    elif ! command -v gcc >/dev/null; then
      bad "gcc missing: Triton cannot build its helpers" "sudo apt-get install -y build-essential"
    elif [ -n "$tinc" ] && ! printf '#include <Python.h>\n#include <cuda.h>\n' | gcc -fsyntax-only -x c -I"$inc" -I"$tinc" - 2>/dev/null; then
      bad "a test of Triton's helper build (Python.h + cuda.h with gcc) fails" "sudo apt-get install -y build-essential python3-dev"
    else ok "Triton helper build: gcc, Python.h${tinc:+, cuda.h}"; fi
  fi
  command -v g++ >/dev/null && ok "g++ (nvcc's host compiler)" || bad "g++ missing (nvcc needs it for FlashInfer's kernels)" "sudo apt-get install -y build-essential"

  # FlashInfer's kernels: ninja on PATH, nvcc matching the CUDA headers, and the link line it uses
  cuda_env
  command -v ninja >/dev/null && ok "ninja $(ninja --version)" || bad "ninja not on PATH (FlashInfer builds with it)" "sudo apt-get install -y ninja-build"
  local nvcc="${CUDA_HOME:+$CUDA_HOME/bin/nvcc}"; [ -n "$nvcc" ] || nvcc="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
  if [ ! -x "$nvcc" ]; then
    bad "no nvcc (FlashInfer compiles kernels at the first start)" "deploy/qwen.sh setup   (vLLM's install brings nvcc into the venv)"
  else
    tmp="$(mktemp -d)"
    printf '#include <cuda_runtime.h>\n#include <cuda/std/cstdint>\n__global__ void k(cuda::std::int32_t *x) { *x = 1; }\nint probe() { return cudaGetLastError(); }\n' > "$tmp/p.cu"
    local home="${CUDA_HOME:-$(dirname "$(dirname "$nvcc")")}"
    if ! "$nvcc" -std=c++17 -gencode arch=compute_100a,code=sm_100a -Xcompiler -fPIC -c "$tmp/p.cu" -o "$tmp/p.o" > "$tmp/out" 2>&1; then
      bad "nvcc $("$nvcc" --version | grep -o 'release [0-9.]*' | cut -d' ' -f2) cannot build a sm_100a kernel: $(grep -m1 -iE 'error|incompatible' "$tmp/out" | cut -c1-160)" "deploy/qwen.sh setup   (pins nvcc to the CUDA runtime's version)"
    elif ! "$nvcc" -shared "$tmp/p.o" -o "$tmp/p.so" -L"$home/lib64" -L"$home/lib64/stubs" -lcudart -lcuda > "$tmp/out" 2>&1; then
      bad "nvcc builds but cannot link like FlashInfer ($(grep -m1 -iE 'cannot find|error' "$tmp/out" | cut -c1-120))" "$(ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1 ' && echo 'deploy/qwen.sh setup   (makes the lib64, libcudart.so and stubs/libcuda.so links)' || echo 'the NVIDIA driver (libcuda.so.1) is missing: sudo ldconfig, else reinstall the driver')"
    else
      ok "nvcc $("$nvcc" --version | grep -o 'release [0-9.]*' | cut -d' ' -f2) builds and links a sm_100a kernel (CUDA_HOME=${CUDA_HOME:-$home})"
    fi
    rm -rf "$tmp"
  fi

  # the weights: every shard the index names is on the disk
  if [ -d "$MODEL" ]; then ok "weights: $MODEL (a folder)"
  else
    v="$(python3 - "$HF_HOME/hub/models--${MODEL//\//--}" <<'PY'
import json, os, sys
d = sys.argv[1]
snaps = os.path.join(d, "snapshots")
if not os.path.isdir(snaps):
    print("not downloaded"); sys.exit()
best = None
for s in os.listdir(snaps):
    p = os.path.join(snaps, s)
    idx = os.path.join(p, "model.safetensors.index.json")
    if not os.path.exists(os.path.join(p, "config.json")):
        continue
    want = set(json.load(open(idx))["weight_map"].values()) if os.path.exists(idx) else {"model.safetensors"}
    gone = [f for f in want if not os.path.exists(os.path.join(p, f))]
    best = "ok" if not gone else f"{len(gone)} of {len(want)} weight files missing"
    if best == "ok":
        break
part = [f for f in os.listdir(os.path.join(d, "blobs"))] if os.path.isdir(os.path.join(d, "blobs")) else []
if best == "ok" and any(f.endswith(".incomplete") for f in part):
    best = "ok (a stale .incomplete file is left over)"
print(best or "no complete snapshot")
PY
)"
    case "$v" in ok*) ok "weights: $MODEL in $HF_HOME${v#ok}" ;;
      *) bad "weights: $MODEL $v in $HF_HOME" "deploy/qwen.sh download   (resumes)" ;; esac
  fi

  # the port, unless it is this server
  if ! pid_alive && ! python3 -c 'import socket, sys; s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", int(sys.argv[1])))' "$PORT" 2>/dev/null; then
    bad "port $PORT is taken by another process" "ss -ltnp | grep :$PORT   (stop it), or PORT=8001 deploy/qwen.sh start"
  else ok "port $PORT free$(pid_alive && echo ' (this server is running on it)')"; fi

  # the demo side: bun and git on the plain PATH (nohup and `ssh evroc 'cmd'` do not read ~/.bashrc)
  if env -i PATH=/usr/local/bin:/usr/bin:/bin bun --version >/dev/null 2>&1; then ok "bun $(env -i PATH=/usr/local/bin:/usr/bin:/bin bun --version) on PATH"
  elif command -v bun >/dev/null; then bad "bun is only on this shell's PATH ($(command -v bun)); ssh commands and new logins will not find it" "sudo ln -sfn $(command -v bun) /usr/local/bin/bun"
  else bad "bun not installed" "deploy/qwen.sh fresh   (or: curl -fsSL https://bun.sh/install | bash && sudo ln -sfn ~/.bun/bin/bun /usr/local/bin/bun)"; fi
  command -v git >/dev/null && ok "git" || bad "git missing" "sudo apt-get install -y git"

  # worth knowing, not fatal
  # FlashInfer fetches NVIDIA's prebuilt TRT-LLM attention kernels from here at the first start;
  # unreachable, it quietly builds slower ones of its own with nvcc
  curl -s -o /dev/null -m 5 https://edge.urm.nvidia.com/ 2>/dev/null ||
    warn "edge.urm.nvidia.com unreachable: FlashInfer will use its own (slower) attention kernels instead of NVIDIA's prebuilt ones"
  if [ -z "${NO_GPU:-}" ]; then
    free="$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)"
    [ "$free" -ge 40 ] || warn "only ${free} GB of RAM available; loading 31 GB of weights wants more"
  fi
  free="$(df -B1G --output=avail /dev/shm 2>/dev/null | tail -1 | tr -d ' ' || echo 0)"
  [ "${free:-0}" -ge 1 ] || warn "/dev/shm has ${free} GB; vLLM's processes talk through it"

  if [ "$fails" -gt 0 ]; then echo "preflight: $fails problem(s); run each fix above, then deploy/qwen.sh preflight again"; return 1; fi
  echo "preflight: PASS$([ "$warns" = 0 ] || echo " ($warns warning(s))")"
}

# After a failed start: the known causes in the log, each with what to run next (README "If start fails").
start_hint() {
  local log="$1" h=""
  grep -q 'toolkit headers are incompatible' "$log" && h="nvcc does not match the CUDA headers: deploy/qwen.sh setup (pins it), then start again"
  [ -n "$h" ] || ! grep -q 'Python.h: No such file' "$log" || h="sudo apt-get install -y python3-dev build-essential, then start again"
  [ -n "$h" ] || ! grep -qE "ninja: (command )?not found|No such file or directory: 'ninja'|Ninja is required" "$log" || h="sudo apt-get install -y ninja-build, then start again"
  [ -n "$h" ] || ! grep -qE "nvcc: (command )?not found|No such file or directory: '[^']*nvcc'|Could not find nvcc|CUDA_HOME environment variable is not set" "$log" || h="no nvcc for FlashInfer: deploy/qwen.sh setup, then start again (or deploy/qwen.sh preflight)"
  [ -n "$h" ] || ! grep -qE "cannot find -lcuda|libcuda.so cannot found" "$log" || h="libcuda is not found: sudo ldconfig; deploy/qwen.sh preflight"
  [ -n "$h" ] || ! grep -qiE 'edge\.urm\.nvidia\.com|cubin' "$log" || h="FlashInfer could not fetch its prebuilt attention kernels: EXTRA_ARGS='--attention-config {\"use_trtllm_attention\":false}' deploy/qwen.sh start"
  [ -n "$h" ] || ! grep -qiE 'gdn_prefill|cute.?dsl' "$log" || h="the FlashInfer DeltaNet prefill kernel failed: EXTRA_ARGS='--gdn-prefill-backend triton' deploy/qwen.sh start"
  [ -n "$h" ] || ! grep -qiE 'deep_?gemm' "$log" || h="DeepGEMM failed: VLLM_USE_DEEP_GEMM=0 deploy/qwen.sh start"
  [ -n "$h" ] || ! grep -qiE 'sampling|top_k_top_p|flashinfer.*sampl' "$log" || h="FlashInfer's sampler failed: VLLM_USE_FLASHINFER_SAMPLER=0 deploy/qwen.sh start"
  [ -n "$h" ] || ! grep -qiE 'out of memory|CUDA error: out of memory|OutOfMemory' "$log" || h="out of GPU memory: nvidia-smi (anything else on it?), else GPU_UTIL=0.90 deploy/qwen.sh start"
  [ -n "$h" ] || ! grep -qiE 'free memory on device|less than desired GPU memory utilization' "$log" || h="something else holds GPU memory: nvidia-smi, kill it, start again"
  [ -n "$h" ] || ! grep -qiE 'cuda.?graph|capture' "$log" || h="CUDA graph capture failed: EXTRA_ARGS='-cc.cudagraph_mode=PIECEWISE' deploy/qwen.sh start (slower), or GPU_UTIL=0.90"
  [ -n "$h" ] || ! grep -qiE 'No space left on device' "$log" || h="a disk is full: deploy/qwen.sh where; df -h ~ /data"
  [ -n "$h" ] || ! grep -qiE 'Address already in use' "$log" || h="port $PORT is taken: ss -ltnp | grep :$PORT"
  [ -n "$h" ] || ! grep -qiE 'HTTPError|ConnectionError|huggingface\.co|LocalEntryNotFound' "$log" || h="could not reach the Hub: if the weights are downloaded, HF_HUB_OFFLINE=1 deploy/qwen.sh start"
  if [ -n "$h" ]; then echo "likely fix: $h"; else echo "no known cause matched; send the lines above (README \"If start fails\")"; fi
}

CMD_NAME="${1:-}"
case "${1:-}" in
  where)
    echo "data folder  $DATA_DIR ($(python3 "$HERE/disk.py" free "$DATA_DIR") GB free)$([ -f "$DATA_FILE" ] || echo ', not chosen yet: setup or download saves it')"
    echo "venv         $VENV$([ -x "$VENV/bin/python" ] || echo ' (not installed)')"
    m="$HF_HOME/hub/models--${MODEL//\//--}"
    echo "weights      $m$([ -d "$m" ] && echo " ($(du -sh "$m" 2>/dev/null | cut -f1))" || echo ' (not downloaded)')"
    echo "state, log   $STATE"
    ;;
  setup)
    if [ -z "${NO_GPU:-}" ]; then
      command -v nvidia-smi >/dev/null || die "no nvidia-smi: this needs an NVIDIA GPU with its driver installed"
      nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
      echo "profile: $(detect_gpu)"
    fi
    use_data_dir 50
    command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
    # a venv kept on the data disk whose Python went with the old root disk: made again (from the uv cache)
    if [ -e "$VENV" ] && ! "$VENV/bin/python" -c '' 2>/dev/null; then echo "the venv's Python is gone ($VENV): making it again"; rm -rf "$VENV"; fi
    [ -x "$VENV/bin/python" ] || uv venv -q --python-preference system -p 3.12 "$VENV"
    # --torch-backend=auto picks the PyTorch build that matches the installed driver's CUDA;
    # TORCH_BACKEND=cu130 (etc.) forces one, TORCH_BACKEND=pypi takes PyPI's default build
    tb=(--torch-backend="${TORCH_BACKEND:-auto}"); [ "${TORCH_BACKEND:-}" != pypi ] || tb=()
    VIRTUAL_ENV="$VENV" uv pip install "vllm==$VLLM_VERSION" "${tb[@]}"
    # the version from the package metadata: `vllm --version` needs a GPU to start
    "$VENV/bin/python" -c 'import importlib.metadata as m; print("vLLM", m.version("vllm"), "torch", m.version("torch"))'
    build_tools
    cuda_env
    if [ -n "${CUDA_HOME:-}" ] || command -v nvcc >/dev/null || [ -x /usr/local/cuda/bin/nvcc ]; then
      echo "nvcc for FlashInfer: $(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
    else
      echo "warning: no nvcc found; FlashInfer cannot compile its kernels: VIRTUAL_ENV=$VENV uv pip install nvidia-cuda-nvcc, then setup again" >&2
    fi
    echo "next: deploy/qwen.sh download"
    ;;
  download)
    [ -x "$VENV/bin/hf" ] || die "run deploy/qwen.sh setup first"
    read -r total_gb left_gb < <(python3 "$HERE/disk.py" model-gb "$MODEL" "$HF_HOME" "$VENV/bin/python")
    [ -z "${MODEL_GB:-}" ] || left_gb="$MODEL_GB"
    echo "$MODEL: ${total_gb} GB, ${left_gb} GB still to download"
    use_data_dir "$(python3 -c "import math,sys; print(math.ceil(float(sys.argv[1])) + 2)" "$left_gb")"
    # resumes: files already in the cache are kept, a half-written one continues where it stopped.
    # HF_TOKEN is not needed (the checkpoint is public) but avoids the anonymous rate limit
    "$VENV/bin/hf" download "$MODEL"
    echo "next: deploy/qwen.sh start && deploy/qwen.sh wait"
    ;;
  preflight)
    preflight
    ;;
  cmd)
    serve_cmd
    echo "# profile $PROFILE ($(gpu_name || true)), $MODEL"
    printf '%q ' "${CMD[@]}"; echo
    ;;
  start)
    if pid_alive; then echo "already running (pid $(cat "$PIDFILE")); deploy/qwen.sh status"; exit 0; fi
    serve_cmd
    [ -x "$VLLM" ] || die "$VLLM not found: run deploy/qwen.sh setup first (or set VLLM=)"
    build_tools
    if [ "${PREFLIGHT:-1}" != 0 ]; then
      preflight || die "not starting: fix the FAIL lines above (PREFLIGHT=0 deploy/qwen.sh start skips the checks)"
    fi
    [ -d "$MODEL" ] || [ -d "$HF_HOME/hub/models--${MODEL//\//--}/snapshots" ] ||
      echo "warning: $MODEL is not in $HF_HOME (deploy/qwen.sh download); vLLM will download it now" >&2
    mkdir -p "$STATE"
    { echo "# $(date -Is) profile $PROFILE quant $QUANT"; printf '%q ' "${CMD[@]}"; echo; } > "$LOG"
    # what is running, as bench labels it: fp8, fp8+mtp3, nvfp4, ...
    echo "$QUANT$([ "$MTP" = 0 ] || echo "+mtp$MTP")" > "$RUNNING"
    echo "$CREW_WORKERS" > "$CREW_FILE"
    # no usage report to vLLM's servers (it only adds a network call)
    export VLLM_NO_USAGE_STATS=1
    # its own session, no terminal: the server outlives the ssh login that started it
    setsid nohup "${CMD[@]}" >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"; boot_id > "$STATE/serve.boot"
    echo "started pid $! (profile $PROFILE, $MODEL), log $LOG; loading takes a few minutes: deploy/qwen.sh wait"
    ;;
  wait)
    timeout_s="${WAIT_S:-1800}"
    start=$(date +%s)
    until curl -fsS -m 5 "$URL/health" >/dev/null 2>&1; do
      if ! pid_alive; then
        # the cause is usually far above the last lines (vLLM wraps it in a validation error)
        echo "the server is not running. First errors in $LOG:" >&2
        grep -m 8 -E 'fatal error|Error:|ERROR .*(Error|error)[^a-z]|No module named|CUDA error|out of memory' "$LOG" | grep -v 'Traceback' | cut -c1-300 >&2 || true
        echo "last lines:" >&2; tail -n 25 "$LOG" >&2 || true
        start_hint "$LOG" >&2; exit 1
      fi
      [ $(( $(date +%s) - start )) -lt "$timeout_s" ] || { echo "not up after ${timeout_s}s; deploy/qwen.sh logs" >&2; exit 1; }
      sleep 5
    done
    echo "up after $(( $(date +%s) - start ))s"
    grep -m1 -o 'KV cache size: .*' "$LOG" || true
    # does the crew fit if every call is as long as it can get, all at once? (vLLM's own pool size)
    pool_gib="$(grep -m1 -o 'Available KV cache memory: [0-9.]* GiB' "$LOG" | grep -o '[0-9.]*' || true)"
    if [ -n "$pool_gib" ]; then
      case "$(sed -n 2p "$LOG" | grep -o -- '--kv-cache-dtype [^ ]*' | cut -d' ' -f2)" in fp8*) kvb=32768 ;; *) kvb=65536 ;; esac
      python3 - "$pool_gib" "$(python3 "$HERE/crew.py" "$CREW_WORKERS" --need "$kvb")" "$CREW_WORKERS" "$CREW_LEADS" "$kvb" <<'PY'
import sys
pool, need, workers, leads, kvb = float(sys.argv[1]) * 2**30, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
per_worker = 26624 * kvb + 153_944_064
print(f"crew: {workers} workers + {leads} leads + 1 orchestrator need at most {need / 1e9:.0f} GB of KV cache and state; the pool holds {pool / 1e9:.0f} GB ({100 * need / pool:.0f}%)")
print(f"      by memory, about {int((pool - need) / per_worker) + workers} workers fit with every call at its longest; calls are rarely all that long at once")
if need > pool:
    print("warning: if every call is long at once, vLLM will pause some (preemptions in deploy/qwen.sh status);"
          " if that happens a lot, restart with KV_DTYPE=fp8 (half the KV memory) or fewer CREW_WORKERS", file=sys.stderr)
PY
    fi
    ;;
  status)
    if pid_alive; then echo "running  pid $(cat "$PIDFILE")"; else echo "stopped"; fi
    [ -f "$LOG" ] && { head -1 "$LOG"; grep -m1 -o 'KV cache size: .*' "$LOG" || true
      sed -n 2p "$LOG" | grep -q -- '--enable-prefix-caching' && echo "prefix caching on" || echo "prefix caching OFF"
      sed -n 2p "$LOG" | grep -o -- '--speculative-config [^ ]*' || echo "MTP off"; }
    exec python3 "$HERE/check.py" --url "$URL" --status
    ;;
  check)
    shift
    exec python3 "$HERE/check.py" --url "$URL" --model "$SERVED" --concurrency "$CREW_TOTAL" "$@"
    ;;
  bench)
    shift
    q="$(cat "$RUNNING" 2>/dev/null || echo "$QUANT")"
    exec python3 "$HERE/check.py" --url "$URL" --model "$SERVED" --bench --label "$q" --save "$STATE/bench-$q.json" "$@"
    ;;
  sweep)
    # one bench per server variant, back to back, then the comparison; the server is left stopped.
    # Takes about 15 minutes a variant. Run it detached:  setsid nohup deploy/qwen.sh sweep > ~/qwen/sweep.log 2>&1 &
    shift
    for v in ${SWEEP:-fp8 fp8+mtp3}; do
      q="${v%%+*}"; m=0; [ "$v" = "$q" ] || m="${v#*+mtp}"
      echo "=== $v ($(date -Is))"
      "$0" stop >/dev/null
      if ! QUANT="$q" MTP="$m" "$0" start || ! "$0" wait; then echo "$v did not start; skipped (deploy/qwen.sh logs)"; continue; fi
      "$0" bench "$@" || echo "$v: bench failed"
    done
    "$0" stop >/dev/null
    exec "$0" compare
    ;;
  compare)
    exec python3 "$HERE/check.py" --compare "$STATE"/bench-*.json
    ;;
  logs)
    exec tail -n "${LINES:-100}" -F "$LOG"
    ;;
  stop)
    pid_alive || { echo "not running"; rm -f "$PIDFILE" "$STATE/serve.boot"; exit 0; }
    pid="$(cat "$PIDFILE")"
    # the whole process group: vLLM's engine core runs in child processes
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && { echo "still up after 60 s: killing"; kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid"; }
    rm -f "$PIDFILE" "$RUNNING" "$STATE/serve.boot"
    echo stopped
    ;;
  *)
    sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
