#!/usr/bin/env bash
# Qwen3.8-27B-FP8 behind vLLM on one GPU, for a Centipede crew (run sheet: deploy/README.md).
#
#   deploy/qwen.sh setup      vLLM $VLLM_VERSION in its own venv (uv), and a GPU/driver check
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
# model's own multi-token prediction (3 draft tokens). QUANT=nvfp4 serves NVIDIA's NVFP4
# checkpoint instead of Qwen's FP8 one (Blackwell only; give it to download and start, the rest
# read what is running). CREW_WORKERS=N (default 10) at start sizes the server for a crew of N
# workers, their leads and the orchestrator; the other commands and deploy/crew.env read it back. The server listens
# on 127.0.0.1 only; reach it from another machine through an ssh tunnel (README).
# DATA_DIR is where the weights, the vLLM venv and the caches go (about 50 GB). Unset, setup picks
# the local filesystem with the most free space this user can write to, and every later command
# uses the same folder (saved in ~/qwen/data.dir).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

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
  command -v gcc >/dev/null || command -v cc >/dev/null || missing+=(build-essential)
  [ "${#missing[@]}" -gt 0 ] || return 0
  if command -v apt-get >/dev/null && sudo -n true 2>/dev/null; then
    echo "installing ${missing[*]} (Triton builds C helpers at start)"
    sudo apt-get install -y "${missing[@]}" >/dev/null || die "apt-get install ${missing[*]} failed"
  else
    die "vLLM's Triton kernels need ${missing[*]}: sudo apt-get install -y ${missing[*]}, then run this again"
  fi
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
    command -v nvidia-smi >/dev/null || die "no nvidia-smi: this needs an NVIDIA GPU with its driver installed"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "profile: $(detect_gpu)"
    use_data_dir 50
    command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
    [ -x "$VENV/bin/python" ] || uv venv -p 3.12 "$VENV"
    # --torch-backend=auto picks the PyTorch build that matches the installed driver's CUDA
    VIRTUAL_ENV="$VENV" uv pip install "vllm==$VLLM_VERSION" --torch-backend=auto
    "$VLLM" --version
    build_tools
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
    [ -d "$MODEL" ] || [ -d "$HF_HOME/hub/models--${MODEL//\//--}/snapshots" ] ||
      echo "warning: $MODEL is not in $HF_HOME (deploy/qwen.sh download); vLLM will download it now" >&2
    mkdir -p "$STATE"
    { echo "# $(date -Is) profile $PROFILE quant $QUANT"; printf '%q ' "${CMD[@]}"; echo; } > "$LOG"
    # what is running, as bench labels it: fp8, fp8+mtp3, nvfp4, ...
    echo "$QUANT$([ "$MTP" = 0 ] || echo "+mtp$MTP")" > "$RUNNING"
    echo "$CREW_WORKERS" > "$CREW_FILE"
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
        echo "last lines:" >&2; tail -n 25 "$LOG" >&2 || true; exit 1
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
