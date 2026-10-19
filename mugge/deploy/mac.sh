#!/usr/bin/env bash
# Qwen3.5 behind llama-server (Metal) on an Apple-silicon Mac, for a Centipede crew (run sheet:
# deploy/MAC.md). The B200 counterpart is deploy/qwen.sh; this one needs no GPU box.
# The same script runs llama.cpp's CUDA build on a Linux box with an NVIDIA GPU (centipede-server,
# deploy/SERVER.md; deploy/box.sh is this file): setup installs llama.cpp's release build there,
# and `helper add URL` lets the crew use another box's llama-server too.
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
#   deploy/mac.sh helper add|list|clear [URL]   (Linux head) another box's llama-server whose slots the crew uses
#   deploy/mac.sh ssh-config  (Linux head) the lines for your Mac's ~/.ssh/config: ssh centipede-server
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
#   ROLE=helper (Linux) every slot a worker slot, for another box's crew; with HOST=0.0.0.0
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin:$HOME/.bun/bin"

STATE="${STATE:-$HOME/qwen-mac}"
MODELS_DIR="${MODELS_DIR:-$STATE/models}"
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
has_nvidia() { ! is_mac && command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; }
HELPERS_FILE="$STATE/helpers"
# llama.cpp's release build for Linux + CUDA (setup), when no LLAMA_SERVER is given
LLAMA_DIR="$STATE/llama.cpp"

# The knobs start saved, unless this command was given its own: a later `mac.sh check` or
# `mac.sh status` describes the server that is running, not the defaults.
load_knobs() {
  [ -f "$KNOBS_FILE" ] || return 0
  local k v line
  while IFS= read -r line; do
    k="${line%%=*}"; v="${line#*=}"
    case "$k" in MODEL|QUANT|PRESET|KV|KV_K|KV_V|CTX|CREW_WORKERS|MAX_WORKERS|ROLE|HOST) ;; *) continue ;; esac
    # the environment wins over what was saved
    if [ -z "$(eval "echo \${$k:-}")" ]; then eval "$k=\$v"; fi
  done < "$KNOBS_FILE"
}
save_knobs() {
  mkdir -p "$STATE"
  : > "$KNOBS_FILE.new"
  for k in MODEL QUANT PRESET KV KV_K KV_V CTX CREW_WORKERS MAX_WORKERS ROLE HOST; do
    v="$(eval "echo \${$k:-}")"; [ -z "$v" ] || echo "$k=$v" >> "$KNOBS_FILE.new"
  done
  mv "$KNOBS_FILE.new" "$KNOBS_FILE"
}

# the crew plan for this Mac and these knobs, as JSON (deploy/mac_crew.py)
plan_json() { MODEL="${MODEL:-}" QUANT="${QUANT:-}" PRESET="${PRESET:-}" KV="${KV:-}" KV_K="${KV_K:-}" KV_V="${KV_V:-}" CTX="${CTX:-}" CREW_WORKERS="${CREW_WORKERS:-}" MAX_WORKERS="${MAX_WORKERS:-}" ROLE="${ROLE:-}" python3 "$HERE/mac_crew.py" --url "$URL/v1" "$@"; }
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
  if ! is_mac && [ -x "$LLAMA_DIR/current/llama-server" ]; then echo "$LLAMA_DIR/current/llama-server"; return; fi
  command -v llama-server 2>/dev/null || true
}

# The b<number> build whose binaries to fetch. llama.cpp's "latest" release is a version (v0.7.0)
# that carries no binaries, only nightly-tag.txt naming its b<number> build; the b<number> builds
# are marked prerelease. LLAMA_TAG may be either form.
llama_release_tag() {
  local tag gh=https://github.com/ggml-org/llama.cpp/releases/download
  tag="${LLAMA_TAG:-$(curl -fsSL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | python3 -c 'import json, sys; print(json.load(sys.stdin)["tag_name"])')}" || tag=""
  case "$tag" in b[0-9]*) echo "$tag"; return 0 ;; esac
  [ -z "$tag" ] || tag="$(curl -fsSL "$gh/$tag/nightly-tag.txt" | tr -d '[:space:]')" || tag=""
  case "$tag" in b[0-9]*) echo "$tag"; return 0 ;; esac
  # no nightly-tag.txt: the newest b<number> release
  tag="$(curl -fsSL 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30' \
    | python3 -c 'import json, re, sys; print(next(r["tag_name"] for r in json.load(sys.stdin) if re.fullmatch(r"b[0-9]+", r["tag_name"])))')" || return 1
  echo "$tag"
}

# Linux + NVIDIA: llama.cpp's latest release build for CUDA (the binaries and the CUDA runtime
# libraries beside them) into $LLAMA_DIR/llama-<tag>, linked as current. CUDA 13.4 needs driver
# 580 or newer, CUDA 12.8 driver 570.
install_llama_linux() {
  local drv cuda tag base t
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)"
  [ -n "$drv" ] || die "nvidia-smi does not answer: install the NVIDIA driver (sudo ubuntu-drivers install && sudo reboot), then deploy/mac.sh setup"
  if [ "$drv" -ge 580 ]; then cuda=13.4; elif [ "$drv" -ge 570 ]; then cuda=12.8
  else die "NVIDIA driver $drv is too old for llama.cpp's CUDA build (570 or newer): sudo ubuntu-drivers install && sudo reboot"; fi
  tag="$(llama_release_tag)" || die "could not find llama.cpp's latest release build on GitHub; LLAMA_TAG=b<number> deploy/mac.sh setup"
  [ "${tag#b}" -ge "$MIN_BUILD" ] 2>/dev/null || die "llama.cpp release $tag is older than build $MIN_BUILD"
  if [ -x "$LLAMA_DIR/llama-$tag/llama-server" ]; then ln -sfn "llama-$tag" "$LLAMA_DIR/current"; echo "llama.cpp $tag (CUDA) already here"; return; fi
  base="https://github.com/ggml-org/llama.cpp/releases/download/$tag"
  mkdir -p "$LLAMA_DIR"
  t="$(mktemp -d)"
  echo "llama.cpp $tag for CUDA $cuda (driver $drv) -> $LLAMA_DIR/llama-$tag"
  curl -fL --retry 3 -o "$t/bin.tar.gz" "$base/llama-$tag-bin-ubuntu-cuda-$cuda-x64.tar.gz"
  curl -fL --retry 3 -o "$t/cudart.tar.gz" "$base/cudart-llama-$tag-bin-ubuntu-cuda-$cuda-x64.tar.gz"
  tar -xzf "$t/bin.tar.gz" -C "$LLAMA_DIR"
  # the runtime goes next to the binaries, which find it through their $ORIGIN rpath
  tar -xzf "$t/cudart.tar.gz" -C "$t"
  cp -L "$t"/cudart-llama-*/* "$LLAMA_DIR/llama-$tag/"
  rm -rf "$t"
  [ -x "$LLAMA_DIR/llama-$tag/llama-server" ] || die "no llama-server in the $tag release archive"
  ln -sfn "llama-$tag" "$LLAMA_DIR/current"
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
  elif has_nvidia; then
    ok "NVIDIA GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader 2>/dev/null | head -1 | awk -F', ' '{print $1 ", " $2 " (" $3 " in use), driver " $4}')"
  else
    [ "${MAC_SH_ANY_OS:-}" = 1 ] && warn "no Mac and no NVIDIA GPU ($(uname -s)): running anyway (MAC_SH_ANY_OS=1; CPU only)" || bad "this is $(uname -s) with no NVIDIA GPU (nvidia-smi)" "install the NVIDIA driver: sudo ubuntu-drivers install && sudo reboot"
  fi
  bin="$(llama_bin)"
  if [ -z "$bin" ]; then bad "llama-server not installed" "deploy/mac.sh setup   ($(is_mac && echo 'brew install llama.cpp' || echo "llama.cpp's CUDA release build"))"
  else
    miss="$(missing_flags "$bin")"
    if [ -n "$miss" ]; then bad "llama-server at $bin is too old ($miss)" "$(is_mac && echo 'brew upgrade llama.cpp   (deploy/mac.sh setup)' || echo 'deploy/mac.sh setup   (the latest llama.cpp release)')"
    else ok "llama-server $bin ($("$bin" --version 2>&1 | grep -m1 -oE 'version: [^ ]+( \([^)]*\))?' || echo 'version unknown'))"; fi
  fi
  if [ -s "$GGUF" ]; then
    if [ -n "$REPO" ] && [ "$(wc -c < "$GGUF" | tr -d ' ')" != "$BYTES" ]; then bad "model $GGUF is $(wc -c < "$GGUF" | tr -d ' ') bytes, not $BYTES" "deploy/mac.sh download   (resumes)"
    else ok "model $GGUF ($(gb "$(wc -c < "$GGUF" | tr -d ' ')") GB)"; fi
  else bad "model $GGUF not downloaded" "deploy/mac.sh download"; fi
  if plan_json > "$STATE/.plan.txt" 2>&1; then
    ok "crew fits in GPU memory: $(grep '^crew:' "$STATE/.plan.txt" | cut -d: -f2- | sed 's/^ //')"
  else
    bad "the crew does not fit in GPU memory: $(grep '^GPU total' "$STATE/.plan.txt" | cut -d: -f2-)" "KV=q4_0, a smaller CTX, fewer CREW_WORKERS, MODEL=4b$(is_mac && echo ', or deploy/mac.sh wired') (deploy/mac.sh crew shows the arithmetic)"
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
  [ -n "$h" ] || ! grep -qiE 'failed to allocate|out of memory|insufficient memory|kIOGPUCommandBufferCallbackErrorOutOfMemory|failed to create context|cudaMalloc failed|CUDA error: out of memory' "$log" || h="not enough GPU memory: quit other GPU programs, then CREW_WORKERS=<fewer> deploy/mac.sh start (or KV=q4_0, MODEL=4b$(is_mac && echo '; or deploy/mac.sh wired'))"
  [ -n "$h" ] || ! grep -qiE 'failed to load model|error loading model|invalid magic|unknown model architecture' "$log" || h="the model file is broken or unsupported: deploy/mac.sh download (it checks the size), and brew upgrade llama.cpp"
  [ -n "$h" ] || ! grep -qiE 'couldn.t bind|address already in use' "$log" || h="port $PORT is taken: lsof -nP -iTCP:$PORT -sTCP:LISTEN"
  if [ -n "$h" ]; then echo "likely fix: $h"; else echo "no known cause matched; paste the lines above in the project chat"; fi
}

CMD_NAME="${1:-}"
load_knobs
HOST="${HOST:-127.0.0.1}"

case "$CMD_NAME" in
  setup)
    if ! is_mac && has_nvidia; then
      miss=""; for c in curl git python3 tar unzip lsof; do command -v "$c" >/dev/null || miss="$miss $c"; done
      # llama.cpp's release build links OpenMP
      [ -e /usr/lib/x86_64-linux-gnu/libgomp.so.1 ] || miss="$miss libgomp1"
      [ -z "$miss" ] || { echo "installing:$miss"; sudo apt-get update -q && sudo apt-get install -y $miss; }
      [ -n "${LLAMA_SERVER:-}" ] || install_llama_linux
      bin="$(llama_bin)"; [ -n "$bin" ] || die "llama-server not found after setup"
      libs="$(ldd "$bin" 2>/dev/null | awk '/not found/ {print $1}' | tr '\n' ' ')"
      [ -z "$libs" ] || die "llama-server needs libraries this box lacks: $libs(apt search them, or paste this line in the project chat)"
      [ -z "$(missing_flags "$bin")" ] || die "llama-server at $bin is too old ($(missing_flags "$bin")); LLAMA_TAG=b<newer> deploy/mac.sh setup"
      "$bin" --version 2>&1 | grep -m2 -E 'version|built' || true
      command -v bun >/dev/null || { curl -fsSL https://bun.sh/install | bash; export PATH="$HOME/.bun/bin:$PATH"; }
      grep -q '.bun/bin' "$HOME/.bashrc" 2>/dev/null || echo 'export PATH="$HOME/.bun/bin:$PATH"' >> "$HOME/.bashrc"
      mkdir -p "$STATE" "$MODELS_DIR"
      echo "next: deploy/mac.sh download"
      exit 0
    fi
    is_mac || [ "${MAC_SH_ANY_OS:-}" = 1 ] || die "this is for macOS, or Linux with an NVIDIA GPU (no nvidia-smi here: sudo ubuntu-drivers install && sudo reboot)"
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
crew = (f"{p['workers']} worker slots for another box's crew" if p.get("role") == "helper"
        else f"{p['workers']} workers + {p['leads']} leads + 1 orchestrator")
print(f"crew: {crew} on {p['slots']} slots of {p['ctx']} tokens;"
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
    elif has_nvidia; then
      echo "gpu      $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | head -1 | awk -F', ' '{print $1 ": " $2 " of " $3 " in use, " $4 " busy"}')"
    fi
    if [ -s "$HELPERS_FILE" ]; then
      while read -r h; do
        [ -n "$h" ] || continue
        curl -fsS -m 5 "$h/health" >/dev/null 2>&1 && echo "helper   $h up" || echo "helper   $h NOT ANSWERING (the crew runs without it)"
      done < "$HELPERS_FILE"
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
  helper)
    # another box's llama-server (deploy/helper.ps1 on Windows, or ROLE=helper HOST=0.0.0.0 deploy/mac.sh start
    # on Linux): its slots join the crew's workers when deploy/mac.env loads (deploy/SERVER.md)
    sub="${2:-list}"; mkdir -p "$STATE"; touch "$HELPERS_FILE"
    case "$sub" in
      add)
        h="${3:-}"; [ -n "$h" ] || die "usage: deploy/mac.sh helper add http://ADDRESS:8080"
        case "$h" in http://*|https://*) ;; *) h="http://$h" ;; esac
        h="${h%/}"; h="${h%/v1}"
        case "$h" in *:[0-9]*) ;; *) h="$h:8080" ;; esac
        props="$(curl -fsS -m 10 "$h/props" 2>/dev/null)" || die "$h does not answer: is its llama-server running, and is the address right? (on the helper: the window deploy/helper.ps1 opened; Windows asks once to allow it on the network)"
        python3 - "$props" "$h" <<'PY'
import json, sys
p = json.loads(sys.argv[1])
print(f"{sys.argv[2]}: {p.get('model_alias')}, {p.get('total_slots')} slots of {(p.get('default_generation_settings') or {}).get('n_ctx')} tokens")
PY
        grep -qxF "$h" "$HELPERS_FILE" || echo "$h" >> "$HELPERS_FILE"
        echo "added; the next demo run uses it (set -a; . deploy/mac.env; set +a loads it)"
        ;;
      list) if [ -s "$HELPERS_FILE" ]; then cat "$HELPERS_FILE"; else echo "no helpers (deploy/mac.sh helper add http://ADDRESS:8080)"; fi ;;
      clear) : > "$HELPERS_FILE"; echo "no helpers" ;;
      *) die "usage: deploy/mac.sh helper add URL | list | clear" ;;
    esac
    ;;
  release-tag)
    # the llama.cpp build setup fetches on Linux
    llama_release_tag ;;
  ssh-config)
    # what your Mac needs for `ssh centipede-server`, with this box's addresses and user; the live view
    # comes along as http://127.0.0.1:7341/ on the Mac while that ssh session is open
    ips="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+\.' | grep -v '^127\.' || true)"
    first="$(echo "$ips" | head -1)"
    echo "# on your Mac: add this to ~/.ssh/config (one address of this box: ${ips//$'\n'/, })"
    printf 'Host %s\n  HostName %s\n  User %s\n  LocalForward 7341 127.0.0.1:7340\n  ServerAliveInterval 30\n' "${SSH_NAME:-centipede-server}" "${first:-ADDRESS}" "$(id -un)"
    command -v sshd >/dev/null || [ -x /usr/sbin/sshd ] || echo "# this box has no ssh server yet: sudo apt-get install -y openssh-server"
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
