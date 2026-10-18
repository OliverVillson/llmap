#!/usr/bin/env bash
# Qwen3.8-27B-FP8 behind vLLM on one GPU, for a Centipede crew (run sheet: deploy/README.md).
#
#   deploy/qwen.sh setup      vLLM $VLLM_VERSION in its own venv (uv), and a GPU/driver check
#   deploy/qwen.sh download   the weights (~31 GB) into the Hugging Face cache
#   deploy/qwen.sh start      the server, detached: it survives logging out of ssh
#   deploy/qwen.sh wait       block until it answers (or exit 1 with the log tail if it died)
#   deploy/qwen.sh status     running?, KV size, requests running/waiting, KV use, preemptions
#   deploy/qwen.sh check      one plain call, one thinking call, one JSON call, then 13 at once
#   deploy/qwen.sh bench      decode speed with 13 calls at once, saved per checkpoint
#   deploy/qwen.sh compare    the saved bench results side by side (FP8 against NVFP4)
#   deploy/qwen.sh logs       follow the server log
#   deploy/qwen.sh stop
#   deploy/qwen.sh cmd        print the vllm command it would run, and change nothing
#
# The GPU profile (b200, h100, l40s) is read from nvidia-smi; GPU=b200 forces one. Any profile
# value can be overridden: MAX_MODEL_LEN, KV_DTYPE, MAX_NUM_SEQS, GPU_UTIL. MTP=3 turns on the
# model's own multi-token prediction (3 draft tokens). QUANT=nvfp4 serves NVIDIA's NVFP4
# checkpoint instead of Qwen's FP8 one (Blackwell only; give it to download and start, the rest
# read what is running). The server listens on 127.0.0.1 only;
# reach it from another machine through an ssh tunnel (README), never an open port.
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
VENV="${VENV:-$HOME/venvs/qwen-vllm}"
VLLM="${VLLM:-$VENV/bin/vllm}"
STATE="${STATE:-$HOME/qwen}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MTP="${MTP:-0}"
LOG="$STATE/serve.log"
PIDFILE="$STATE/serve.pid"
RUNNING="$STATE/serve.quant"
URL="http://127.0.0.1:$PORT"

die() { echo "qwen.sh: $*" >&2; exit 1; }

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

# Per-GPU defaults (the arithmetic is in README.md "Memory and slots"). A crew of 10 workers,
# 2 leads and 1 orchestrator keeps 13 calls in flight; every profile admits at least that many.
profile() {
  local g; g="$(detect_gpu)"
  [ -z "${KV_DTYPE:-}" ] || KV_DTYPE_SET=1
  case "$g" in
    b200) : "${MAX_MODEL_LEN:=65536}" "${KV_DTYPE:=auto}" "${MAX_NUM_SEQS:=32}" "${GPU_UTIL:=0.90}" ;;
    h100) : "${MAX_MODEL_LEN:=32768}" "${KV_DTYPE:=auto}" "${MAX_NUM_SEQS:=16}" "${GPU_UTIL:=0.90}" ;;
    l40s) : "${MAX_MODEL_LEN:=16384}" "${KV_DTYPE:=fp8}" "${MAX_NUM_SEQS:=13}" "${GPU_UTIL:=0.92}" ;;
    *) die "unknown GPU '$(gpu_name)': set GPU=b200, h100 or l40s" ;;
  esac
  PROFILE="$g"
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

pid_alive() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-}" in
  setup)
    command -v nvidia-smi >/dev/null || die "no nvidia-smi: this needs an NVIDIA GPU with its driver installed"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "profile: $(detect_gpu)"
    free_gb="$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)"
    [ "$free_gb" -ge 50 ] || echo "warning: ${free_gb} GB free in $HOME; the venv and weights need about 45 GB" >&2
    command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
    [ -x "$VENV/bin/python" ] || uv venv -p 3.12 "$VENV"
    # --torch-backend=auto picks the PyTorch build that matches the installed driver's CUDA
    VIRTUAL_ENV="$VENV" uv pip install "vllm==$VLLM_VERSION" --torch-backend=auto
    "$VLLM" --version
    echo "next: deploy/qwen.sh download"
    ;;
  download)
    [ -x "$VENV/bin/hf" ] || die "run deploy/qwen.sh setup first"
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
    mkdir -p "$STATE"
    { echo "# $(date -Is) profile $PROFILE quant $QUANT"; printf '%q ' "${CMD[@]}"; echo; } > "$LOG"
    echo "$QUANT" > "$RUNNING"
    # its own session, no terminal: the server outlives the ssh login that started it
    setsid nohup "${CMD[@]}" >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "started pid $! (profile $PROFILE, $MODEL), log $LOG; loading takes a few minutes: deploy/qwen.sh wait"
    ;;
  wait)
    timeout_s="${WAIT_S:-1800}"
    start=$(date +%s)
    until curl -fsS -m 5 "$URL/health" >/dev/null 2>&1; do
      if ! pid_alive; then echo "the server is not running; last lines of $LOG:" >&2; tail -n 40 "$LOG" >&2 || true; exit 1; fi
      [ $(( $(date +%s) - start )) -lt "$timeout_s" ] || { echo "not up after ${timeout_s}s; deploy/qwen.sh logs" >&2; exit 1; }
      sleep 5
    done
    echo "up after $(( $(date +%s) - start ))s"
    grep -m1 -o 'KV cache size: .*' "$LOG" || true
    ;;
  status)
    if pid_alive; then echo "running  pid $(cat "$PIDFILE")"; else echo "stopped"; fi
    [ -f "$LOG" ] && { head -1 "$LOG"; grep -m1 -o 'KV cache size: .*' "$LOG" || true; }
    exec python3 "$HERE/check.py" --url "$URL" --status
    ;;
  check)
    shift
    exec python3 "$HERE/check.py" --url "$URL" --model "$SERVED" "$@"
    ;;
  bench)
    shift
    q="$(cat "$RUNNING" 2>/dev/null || echo "$QUANT")"
    exec python3 "$HERE/check.py" --url "$URL" --model "$SERVED" --bench --label "$q" --save "$STATE/bench-$q.json" "$@"
    ;;
  compare)
    exec python3 "$HERE/check.py" --compare "$STATE"/bench-*.json
    ;;
  logs)
    exec tail -n "${LINES:-100}" -F "$LOG"
    ;;
  stop)
    pid_alive || { echo "not running"; rm -f "$PIDFILE"; exit 0; }
    pid="$(cat "$PIDFILE")"
    # the whole process group: vLLM's engine core runs in child processes
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && { echo "still up after 60 s: killing"; kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid"; }
    rm -f "$PIDFILE" "$RUNNING"
    echo stopped
    ;;
  *)
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
