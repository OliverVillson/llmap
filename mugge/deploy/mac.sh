#!/usr/bin/env bash
# Qwen3.5 behind llama-server (Metal) on an Apple-silicon Mac, for a Centipede crew (run sheet:
# deploy/MAC.md). The B200 counterpart is deploy/qwen.sh; this one needs no GPU box.
#
#   deploy/mac.sh setup      llama.cpp from Homebrew (upgraded if it is too old for the flags or for Qwen3.5 on Metal), bun, python3
#   deploy/mac.sh wired      let the GPU use 12 GB of a 16 GB Mac until the next restart (asks for your password)
#   deploy/mac.sh download   the model GGUF (Qwen3.5-9B Q4_K_M, 5.7 GB) into the data folder; resumes
#   deploy/mac.sh crew       the crew this Mac fits, with the memory arithmetic (deploy/mac_crew.py)
#   deploy/mac.sh cmd        print the llama-server command it would run, and change nothing
#   deploy/mac.sh preflight  every start prerequisite; each FAIL line says the command that fixes it
#   deploy/mac.sh start      the server, detached (it outlives the terminal; the Mac stays awake while it runs)
#   deploy/mac.sh wait       block until it answers (or exit 1 with the log's errors if it died)
#   deploy/mac.sh status     running?, the crew it serves, calls running and waiting, memory and swap
#   deploy/mac.sh check      one plain call, one thinking call, two JSON calls, then the whole crew at once
#   deploy/mac.sh bench      speed at 1, 4, 8 and all-slots calls at once
#   deploy/mac.sh logs       follow the server log
#   deploy/mac.sh stop
#
# Knobs (give them to start; the other commands and deploy/mac.env read back what is running):
#   MODEL=9b|4b|/path/to.gguf   9b (default) is Qwen3.5-9B; 4b is Qwen3.5-4B (faster, codes worse);
#                               a path serves a GGUF you have, with PRESET=qwen3.5-9b|qwen3.5-4b|qwen3.6-35b-a3b
#   QUANT=Q4_K_M|Q5_K_M|Q6_K|Q8_0   the GGUF's quantization (9b: Q4_K_M, 4b: Q6_K)
#   KV=q8_0|q4_0|f16 (or KV_K, KV_V apart)   KV cache type (default q8_0 for both)
#   CTX=16384                   tokens of context per slot (every call fits in it, at its longest)
#   CREW_WORKERS=N              this many worker slots, whatever memory says (default: as many as fit, at most MAX_WORKERS=8)
#   PORT=8080 HOST=127.0.0.1 EXTRA_ARGS="..." (appended to llama-server) LLAMA_SERVER=/path/to/llama-server
#   MODELS_DIR (default ~/qwen-mac/models) STATE (default ~/qwen-mac: log, pid, what is running)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin:$HOME/.bun/bin"

STATE="${STATE:-$HOME/qwen-mac}"
MODELS_DIR="${MODELS_DIR:-$STATE/models}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
URL="http://127.0.0.1:$PORT"
LOG="$STATE/serve.log"
PIDFILE="$STATE/serve.pid"
# what start ran with: later commands (and deploy/mac.env) read these back
PLAN_FILE="$STATE/serve.plan.json"
BACKENDS_FILE="$STATE/backends.json"
KNOBS_FILE="$STATE/serve.knobs"
# llama-server ends a thinking that reaches its budget (the engine's MUGGE_THINKING_BUDGET) with
# this line and the end-of-thinking tag, so the answer still comes
BUDGET_MESSAGE="${BUDGET_MESSAGE:-Considering the limited time, I have to write the answer now.}"

die() { echo "mac.sh: $*" >&2; exit 1; }
is_mac() { [ "$(uname -s)" = Darwin ]; }

# The knobs start saved, unless this command was given its own: a later `mac.sh check` or
# `mac.sh status` describes the server that is running, not the defaults.
load_knobs() {
  [ -f "$KNOBS_FILE" ] || return 0
  local k v line
  while IFS= read -r line; do
    k="${line%%=*}"; v="${line#*=}"
    case "$k" in MODEL|QUANT|PRESET|KV|KV_K|KV_V|CTX|CREW_WORKERS|MAX_WORKERS) ;; *) continue ;; esac
    # the environment wins over what was saved
    if [ -z "$(eval "echo \${$k:-}")" ]; then eval "$k=\$v"; fi
  done < "$KNOBS_FILE"
}
save_knobs() {
  mkdir -p "$STATE"
  : > "$KNOBS_FILE.new"
  for k in MODEL QUANT PRESET KV KV_K KV_V CTX CREW_WORKERS MAX_WORKERS; do
    v="$(eval "echo \${$k:-}")"; [ -z "$v" ] || echo "$k=$v" >> "$KNOBS_FILE.new"
  done
  mv "$KNOBS_FILE.new" "$KNOBS_FILE"
}

# the crew plan for this Mac and these knobs, as JSON (deploy/mac_crew.py)
plan_json() { MODEL="${MODEL:-}" QUANT="${QUANT:-}" PRESET="${PRESET:-}" KV="${KV:-}" KV_K="${KV_K:-}" KV_V="${KV_V:-}" CTX="${CTX:-}" CREW_WORKERS="${CREW_WORKERS:-}" MAX_WORKERS="${MAX_WORKERS:-}" python3 "$HERE/mac_crew.py" --url "$URL/v1" "$@"; }
# one field of a plan held in $PLAN
field() { python3 -c 'import json, sys; v = json.loads(sys.argv[1])[sys.argv[2]]; print("" if v is None else v)' "$PLAN" "$1"; }

read_plan() {
  PLAN="$(plan_json --json)" || die "deploy/mac_crew.py failed (MODEL=${MODEL:-9b} QUANT=${QUANT:-} PRESET=${PRESET:-})"
  FILE="$(field file)"; SERVED="$(field served)"; SLOTS="$(field slots)"; CTX_SLOT="$(field ctx)"
  K="$(field k_type)"; V="$(field v_type)"; REPO="$(field repo)"; BYTES="$(field weights_bytes)"
  case "${MODEL:-9b}" in
    */*|*.gguf) GGUF="${MODEL/#\~/$HOME}" ;;
    *) GGUF="$MODELS_DIR/$FILE" ;;
  esac
}

llama_bin() {
  if [ -n "${LLAMA_SERVER:-}" ]; then echo "$LLAMA_SERVER"; return; fi
  command -v llama-server 2>/dev/null || true
}

# flags this script passes that older llama.cpp builds do not know
NEEDED_FLAGS="--reasoning-budget-message --ctx-checkpoints --cache-ram --reasoning-format"
# the oldest llama.cpp build measured running Qwen3.5 well on Metal: on an M5 Air, build 11146
# (brew 0.5.0) made 8 calls at once barely faster than one; build 11669 (brew 0.7.0) did 1.7x
MIN_BUILD=11669
missing_flags() {
  local help f out="" build
  help="$("$1" --help 2>&1 || true)"
  # (no grep -q in a pipe: under pipefail its early exit fails the pipe)
  for f in $NEEDED_FLAGS; do case "$help" in *"$f"*) ;; *) out="$out, no $f" ;; esac; done
  build="$("$1" --version 2>&1 | sed -n 's/.*(build \([0-9][0-9]*\).*/\1/p' | head -1)"
  [ -z "$build" ] || [ "$build" -ge "$MIN_BUILD" ] || out="$out, build $build is older than $MIN_BUILD"
  echo "${out#, }"
}

serve_cmd() {
  read_plan
  local bin; bin="$(llama_bin)"
  [ -n "$bin" ] || bin=llama-server
  # one slot per call, each with its own CTX of KV cache: no --kv-unified, because a shared pool
  # that runs out fails every running call at once (deploy/MAC.md "Why no shared KV pool")
  CMD=("$bin" -m "$GGUF" --alias "$SERVED" --host "$HOST" --port "$PORT"
    -ngl 99 -fa on
    -np "$SLOTS" -c "$((SLOTS * CTX_SLOT))"
    -ctk "$K" -ctv "$V"
    --jinja --reasoning-format deepseek --reasoning-budget-message "$BUDGET_MESSAGE"
    # Qwen3.5 is hybrid: reusing a prompt prefix needs a saved recurrent state (~50 MB each, in
    # RAM); 2 a slot keep a ticket's fix call from reading its pack again. No host prompt cache
    # (its default is 8 GB, half this Mac's memory).
    --ctx-checkpoints 2 --cache-ram 0
    --metrics)
  if [ -n "${EXTRA_ARGS:-}" ]; then read -r -a extra <<< "$EXTRA_ARGS"; CMD+=("${extra[@]}"); fi
}

pid_alive() {
  [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

port_taken() {
  if command -v lsof >/dev/null; then lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
  else python3 -c 'import socket, sys; s = socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1])))' "$1" 2>/dev/null && return 1 || return 0; fi
}

gb() { python3 -c 'import sys; print(f"{int(sys.argv[1]) / 1e9:.1f}")' "$1"; }
free_gb() { df -Pk "$1" 2>/dev/null | awk 'NR == 2 {print int($4 / 1048576)}'; }

preflight() {
  local fails=0 warns=0 bin miss
  ok() { echo "ok    $*"; }
  bad() { echo "FAIL  $1"; echo "      fix: $2"; fails=$((fails + 1)); }
  warn() { echo "warn  $*"; warns=$((warns + 1)); }
  read_plan
  if is_mac; then
    [ "$(uname -m)" = arm64 ] && ok "Apple silicon ($(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo arm64))" || bad "not Apple silicon: Metal needs an M-series Mac" "run the B200 path instead (deploy/qwen.sh)"
  else
    [ "${MAC_SH_ANY_OS:-}" = 1 ] && warn "not macOS ($(uname -s)): running anyway (MAC_SH_ANY_OS=1; CPU only)" || bad "this is $(uname -s), not macOS" "use deploy/qwen.sh on a Linux GPU box (MAC_SH_ANY_OS=1 forces this script)"
  fi
  bin="$(llama_bin)"
  if [ -z "$bin" ]; then bad "llama-server not installed" "deploy/mac.sh setup   (brew install llama.cpp)"
  else
    miss="$(missing_flags "$bin")"
    if [ -n "$miss" ]; then bad "llama-server at $bin is too old ($miss)" "brew upgrade llama.cpp   (deploy/mac.sh setup)"
    else ok "llama-server $bin ($("$bin" --version 2>&1 | grep -m1 -oE 'version: [^ ]+( \([^)]*\))?' || echo 'version unknown'))"; fi
  fi
  if [ -s "$GGUF" ]; then
    if [ -n "$REPO" ] && [ "$(wc -c < "$GGUF" | tr -d ' ')" != "$BYTES" ]; then bad "model $GGUF is $(wc -c < "$GGUF" | tr -d ' ') bytes, not $BYTES" "deploy/mac.sh download   (resumes)"
    else ok "model $GGUF ($(gb "$(wc -c < "$GGUF" | tr -d ' ')") GB)"; fi
  else bad "model $GGUF not downloaded" "deploy/mac.sh download"; fi
  if plan_json > "$STATE/.plan.txt" 2>&1; then
    ok "crew fits in GPU memory: $(grep '^crew:' "$STATE/.plan.txt" | cut -d: -f2- | sed 's/^ //')"
  else
    bad "the crew does not fit in GPU memory: $(grep '^GPU total' "$STATE/.plan.txt" | cut -d: -f2-)" "KV=q4_0, CTX=12288, fewer CREW_WORKERS, MODEL=4b, or deploy/mac.sh wired (deploy/mac.sh crew shows the arithmetic)"
  fi
  rm -f "$STATE/.plan.txt"
  if ! pid_alive && port_taken "$PORT"; then bad "port $PORT is taken by another process" "lsof -nP -iTCP:$PORT -sTCP:LISTEN   (stop it), or PORT=8081 deploy/mac.sh start"
  else ok "port $PORT free$(pid_alive && echo ' (this server is running on it)' || true)"; fi
  command -v bun >/dev/null && ok "bun $(bun --version)" || bad "bun not installed" "curl -fsSL https://bun.sh/install | bash"
  command -v git >/dev/null && ok "git" || bad "git missing" "xcode-select --install"
  python3 -c 'import sys; sys.exit(sys.version_info < (3, 8))' && ok "python3 $(python3 -c 'import platform; print(platform.python_version())')" || bad "python3 older than 3.8" "brew install python"
  if is_mac; then
    pmset -g 2>/dev/null | grep -q 'lowpowermode *1' && warn "Low Power Mode is on: the GPU runs slower (System Settings > Battery)"
    pmset -g batt 2>/dev/null | grep -q "'Battery Power'" && warn "on battery: plug in for a long run (a fanless Air also throttles when hot)"
    local swap; swap="$(sysctl -n vm.swapusage 2>/dev/null | awk '{print $6}' | tr -d 'M' | tr ',' '.' | cut -d. -f1)"
    [ -z "$swap" ] || [ "$swap" -lt 2048 ] || warn "${swap} MB of swap already in use: close big apps (browsers, Docker, LM Studio, Ollama) first"
    pgrep -qf 'LM Studio|ollama serve|Ollama' 2>/dev/null && warn "LM Studio or Ollama is running and may hold a model in memory: quit it first"
  fi
  if [ "$fails" -gt 0 ]; then echo "preflight: $fails problem(s); run each fix above, then deploy/mac.sh preflight again"; return 1; fi
  echo "preflight: PASS$([ "$warns" = 0 ] || echo " ($warns warning(s))")"
}

start_hint() {
  local log="$1" h=""
  grep -qiE 'unknown argument|invalid argument|error: unrecognized' "$log" && h="this llama.cpp is too old for a flag: brew upgrade llama.cpp, then start again"
  [ -n "$h" ] || ! grep -qiE 'failed to allocate|out of memory|insufficient memory|kIOGPUCommandBufferCallbackErrorOutOfMemory|failed to create context' "$log" || h="not enough GPU memory: quit other apps, then KV=q4_0 or CREW_WORKERS=6 deploy/mac.sh start (or deploy/mac.sh wired)"
  [ -n "$h" ] || ! grep -qiE 'failed to load model|error loading model|invalid magic|unknown model architecture' "$log" || h="the model file is broken or unsupported: deploy/mac.sh download (it checks the size), and brew upgrade llama.cpp"
  [ -n "$h" ] || ! grep -qiE 'couldn.t bind|address already in use' "$log" || h="port $PORT is taken: lsof -nP -iTCP:$PORT -sTCP:LISTEN"
  if [ -n "$h" ]; then echo "likely fix: $h"; else echo "no known cause matched; paste the lines above in the project chat"; fi
}

CMD_NAME="${1:-}"
load_knobs

case "$CMD_NAME" in
  setup)
    is_mac || [ "${MAC_SH_ANY_OS:-}" = 1 ] || die "this is for macOS (deploy/qwen.sh is the Linux GPU path)"
    if ! command -v brew >/dev/null; then die "Homebrew is needed: https://brew.sh (then run deploy/mac.sh setup again)"; fi
    bin="$(llama_bin)"
    if [ -z "$bin" ]; then brew install llama.cpp
    elif [ -n "$(missing_flags "$bin")" ]; then echo "llama-server at $bin is too old ($(missing_flags "$bin")): upgrading"; brew upgrade llama.cpp || brew install llama.cpp; fi
    bin="$(llama_bin)"; [ -n "$bin" ] || die "llama-server still not found after brew install llama.cpp"
    [ -z "$(missing_flags "$bin")" ] || die "llama-server at $bin is still too old ($(missing_flags "$bin")); brew update && brew upgrade llama.cpp"
    "$bin" --version 2>&1 | head -2
    command -v bun >/dev/null || { curl -fsSL https://bun.sh/install | bash; export PATH="$HOME/.bun/bin:$PATH"; }
    command -v python3 >/dev/null || brew install python
    mkdir -p "$STATE" "$MODELS_DIR"
    echo "next: deploy/mac.sh download"
    ;;
  wired)
    is_mac || die "only on macOS"
    mem_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
    want="${WIRED_MB:-$([ "$mem_gb" -le 16 ] && echo 12288 || echo $(( mem_gb * 1024 * 3 / 4 )))}"
    echo "letting the GPU use $want MB of this ${mem_gb} GB Mac until the next restart (macOS asks for your password)"
    sudo sysctl iogpu.wired_limit_mb="$want" || sudo sysctl debug.iogpu.wired_limit=$((want * 1048576))
    echo "back to the default: sudo sysctl iogpu.wired_limit_mb=0 (or restart). The crew for the new limit: deploy/mac.sh crew"
    ;;
  download)
    read_plan
    [ -n "$REPO" ] || { [ -s "$GGUF" ] && { echo "$GGUF is already on disk ($(gb "$(wc -c < "$GGUF" | tr -d ' ')") GB)"; exit 0; } || die "$GGUF not found (a local MODEL path is not downloaded)"; }
    mkdir -p "$MODELS_DIR"
    if [ -s "$GGUF" ] && [ "$(wc -c < "$GGUF" | tr -d ' ')" = "$BYTES" ]; then echo "$GGUF already downloaded ($(gb "$BYTES") GB)"; exit 0; fi
    have=0; [ -f "$GGUF.part" ] && have="$(wc -c < "$GGUF.part" | tr -d ' ')"
    need_gb=$(( (BYTES - have) / 1000000000 + 2 ))
    free="$(free_gb "$MODELS_DIR")"
    [ "${free:-0}" -ge "$need_gb" ] || die "not enough space in $MODELS_DIR (${free} GB free, ${need_gb} GB needed); MODELS_DIR=/path deploy/mac.sh download"
    echo "$REPO $FILE: $(gb "$BYTES") GB$([ "$have" = 0 ] || echo ", $(gb "$have") GB already here") -> $MODELS_DIR"
    # resumes a cut download (-C -); HF_TOKEN is not needed (the repo is public) but lifts the anonymous rate limit
    auth=(); [ -z "${HF_TOKEN:-}" ] || auth=(-H "Authorization: Bearer $HF_TOKEN")
    curl -fL --retry 5 --retry-delay 3 -C - ${auth[@]+"${auth[@]}"} -o "$GGUF.part" "https://huggingface.co/$REPO/resolve/main/$FILE"
    got="$(wc -c < "$GGUF.part" | tr -d ' ')"
    [ "$got" = "$BYTES" ] || die "downloaded $got bytes, expected $BYTES: run deploy/mac.sh download again (it resumes)"
    mv "$GGUF.part" "$GGUF"
    echo "next: deploy/mac.sh start && deploy/mac.sh wait"
    ;;
  crew)
    shift
    plan_json "$@"
    ;;
  cmd)
    serve_cmd
    printf '%q ' "${CMD[@]}"; echo
    ;;
  preflight)
    mkdir -p "$STATE"
    preflight
    ;;
  start)
    if pid_alive; then echo "already running (pid $(cat "$PIDFILE")); deploy/mac.sh status"; exit 0; fi
    mkdir -p "$STATE"
    serve_cmd
    if [ "${PREFLIGHT:-1}" != 0 ]; then preflight || die "not starting: fix the FAIL lines above (PREFLIGHT=0 deploy/mac.sh start skips the checks)"; fi
    save_knobs
    printf '%s\n' "$PLAN" > "$PLAN_FILE"
    plan_json --backends > "$BACKENDS_FILE"
    { echo "# $(date '+%Y-%m-%dT%H:%M:%S') $FILE, $SLOTS slots of $CTX_SLOT, KV $K/$V"; printf '%q ' "${CMD[@]}"; echo; } > "$LOG"
    # nohup and no terminal: the server outlives this shell; caffeinate keeps the Mac awake (not the
    # display) for as long as the server runs, so an idle Mac does not sleep through a run
    nohup "${CMD[@]}" >> "$LOG" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "$PIDFILE"
    if command -v caffeinate >/dev/null; then nohup caffeinate -i -s -w "$pid" >/dev/null 2>&1 < /dev/null & fi
    echo "started pid $pid ($FILE, $SLOTS slots of $CTX_SLOT tokens), log $LOG; loading takes under a minute: deploy/mac.sh wait"
    ;;
  wait)
    timeout_s="${WAIT_S:-600}"
    t0=$(date +%s)
    until curl -fsS -m 5 "$URL/health" >/dev/null 2>&1; do
      if ! pid_alive; then
        echo "the server is not running. Errors in $LOG:" >&2
        grep -m 10 -iE 'error|failed|unknown argument|out of memory' "$LOG" | cut -c1-300 >&2 || true
        echo "last lines:" >&2; tail -n 15 "$LOG" >&2 || true
        start_hint "$LOG" >&2; exit 1
      fi
      [ $(( $(date +%s) - t0 )) -lt "$timeout_s" ] || { echo "not up after ${timeout_s}s but still running: deploy/mac.sh logs" >&2; exit 1; }
      sleep 2
    done
    echo "up after $(( $(date +%s) - t0 ))s"
    if [ -f "$PLAN_FILE" ]; then
      python3 - "$PLAN_FILE" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print(f"crew: {p['workers']} workers + {p['leads']} leads + 1 orchestrator on {p['slots']} slots of {p['ctx']} tokens;"
      f" GPU memory planned {p['need_bytes'] / 1e9:.1f} of {p['wired_bytes'] / 1e9:.1f} GB")
PY
    fi
    grep -m1 -oE 'recommendedMaxWorkingSetSize *= *[0-9.]+ MB' "$LOG" || true
    echo "next: deploy/mac.sh check"
    ;;
  status)
    if pid_alive; then echo "running  pid $(cat "$PIDFILE")"; else echo "stopped"; fi
    [ -f "$LOG" ] && head -1 "$LOG"
    if is_mac; then
      echo "swap     $(sysctl -n vm.swapusage 2>/dev/null | awk '{print $6}') used (growing during a run means the crew is too big for this Mac: fewer CREW_WORKERS or KV=q4_0)"
      memory_pressure 2>/dev/null | grep -m1 'free percentage' | sed 's/^.*: /free     /;s/$/ of memory/' || true
    fi
    exec python3 "$HERE/check.py" --url "$URL" --status
    ;;
  check)
    shift
    read_plan
    # the thinking calls get a budget, as the crew's do (deploy/mac.env), so they answer in time
    exec python3 "$HERE/check.py" --url "$URL" --model "$SERVED" --concurrency "$SLOTS" --thinking-budget "${CHECK_THINKING_BUDGET:-1024}" "$@"
    ;;
  bench)
    shift
    read_plan
    exec python3 "$HERE/check.py" --url "$URL" --model "$SERVED" --bench --levels "1,4,8,$SLOTS" --prompt-tokens 4000 --answers thinking:1024,short:512 \
      --label "${SERVED}-$K" --save "$STATE/bench-${SERVED}-$K.json" "$@"
    ;;
  logs)
    exec tail -n "${LINES:-100}" -F "$LOG"
    ;;
  stop)
    pid_alive || { echo "not running"; rm -f "$PIDFILE"; exit 0; }
    pid="$(cat "$PIDFILE")"
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && { echo "still up after 30 s: killing"; kill -KILL "$pid" 2>/dev/null || true; }
    rm -f "$PIDFILE"
    echo stopped
    ;;
  *)
    sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
