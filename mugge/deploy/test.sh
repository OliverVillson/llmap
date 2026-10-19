#!/usr/bin/env bash
# Dry run of the deploy scripts with no GPU: deploy/mock_vllm.py stands in for vLLM, and the crew
# file is checked against it the way `mugge run` checks machines at start. CI runs this.
set -euo pipefail
cd "$(dirname "$0")/.."
export STATE="$(mktemp -d)" VLLM="$PWD/deploy/mock_vllm.py" GPU=b200 PORT="${PORT:-18000}" MOCK_LOAD_S=2
# the mock is not a real install: start skips its preflight here (tested on its own at the end)
export PREFLIGHT=0
export DATA_DIR="$STATE/data"
trap 'deploy/qwen.sh stop >/dev/null 2>&1 || true; rm -rf "$STATE"' EXIT
deploy/qwen.sh cmd
deploy/qwen.sh start
deploy/qwen.sh wait
deploy/qwen.sh check
deploy/qwen.sh status
# the crew on the same server: every entry answers and each kind of call has its slots
sed "s/:8000/:$PORT/" deploy/crew-b200.json > "$STATE/crew.json"
MUGGE_BACKENDS="@$STATE/crew.json" bun src/cli.ts backends | tee "$STATE/backends.txt"
grep -q 'b200-workers .* worker  10 slots' "$STATE/backends.txt"
grep -q 'b200-leads .* lead  2 slots' "$STATE/backends.txt"
grep -q 'b200-orchestrator .* manager  1 slots' "$STATE/backends.txt"
[ "$(grep -c '^  up ' "$STATE/backends.txt")" = 3 ]
! grep -q 'llama-server' "$STATE/backends.txt"
# speed: bench saves per checkpoint, compare reads them back
deploy/qwen.sh status > "$STATE/status.txt"; grep -q 'prefix caching on' "$STATE/status.txt"
deploy/qwen.sh bench --levels 1,13,32 --prompt-tokens 500 --answers thinking:64,short:16 | tee "$STATE/bench.txt"
grep -q '13 -> 32 at once' "$STATE/bench.txt"
# the sweep restarts the server per variant (MTP off, then on) and compares
SWEEP="fp8 fp8+mtp3" deploy/qwen.sh sweep --levels 1,13 --prompt-tokens 200 --answers thinking:32,short:8 | tee "$STATE/sweep.txt"
grep -q 'fp8+mtp3 .* 13 ' "$STATE/sweep.txt"
grep -q '^short answers' "$STATE/sweep.txt"
grep -q 'vs fp8$' "$STATE/sweep.txt"
ls "$STATE"/bench-fp8.json "$STATE"/bench-fp8+mtp3.json >/dev/null
deploy/qwen.sh start && deploy/qwen.sh wait >/dev/null
# NVFP4: NVIDIA's checkpoint with an fp8 KV cache, on Blackwell only
QUANT=nvfp4 deploy/qwen.sh cmd | grep -q 'nvidia/Qwen3.8-27B-NVFP4 .*--kv-cache-dtype fp8_e4m3'
QUANT=nvfp4 KV_DTYPE=auto deploy/qwen.sh cmd | grep -q -- '--kv-cache-dtype auto'
if QUANT=nvfp4 GPU=h100 deploy/qwen.sh cmd >/dev/null 2>&1; then echo "NVFP4 allowed on an H100"; exit 1; fi
# one knob for the crew size: the machine list, the agents and the server agree
diff <(python3 deploy/crew.py 10) deploy/crew-b200.json
( CREW_WORKERS=24; set -a; . deploy/crew.env; set +a
  [ "$MUGGE_CONCURRENCY" = 30 ]
  echo "$MUGGE_BACKENDS" | grep -q '"b200-workers".*"slots": 24'
  echo "$MUGGE_BACKENDS" | grep -q '"b200-leads".*"slots": 3' )
( CREW_WORKERS=64; set -a; . deploy/crew.env; set +a
  [ "$MUGGE_CONCURRENCY" = 80 ]
  [ "$MUGGE_REASONING_EFFORT" = worker=medium,lead=low,manager=xhigh ]
  echo "$MUGGE_BACKENDS" | grep -q '"b200-workers".*"slots": 64'
  echo "$MUGGE_BACKENDS" | grep -q '"b200-leads".*"slots": 8' )
CREW_WORKERS=64 deploy/qwen.sh cmd | grep -q -- '--max-num-seqs 73 .*--gpu-memory-utilization 0.94'
CREW_WORKERS=80 deploy/qwen.sh cmd | grep -q -- '--max-num-seqs 91'
# the 64-agent start: wait reports the crew against the pool
deploy/qwen.sh stop >/dev/null
MOCK_KV_GIB=100 CREW_WORKERS=64 deploy/qwen.sh start >/dev/null
# the size sticks: wait, check and crew.env read it back without being told
deploy/qwen.sh wait 2>&1 | tee "$STATE/wait64.txt"
grep -q 'crew: 64 workers + 8 leads + 1 orchestrator need at most 137 GB' "$STATE/wait64.txt"
grep -q 'warning: if every call is long' "$STATE/wait64.txt"
deploy/qwen.sh check > "$STATE/check64.txt"; grep -q 'ok    73 calls at once' "$STATE/check64.txt"
( unset CREW_WORKERS; set -a; . deploy/crew.env; set +a; [ "$MUGGE_CONCURRENCY" = 80 ] )
# the same from sh (dash) and from another folder; CREW_WORKERS stays out of the environment
env -u CREW_WORKERS sh -c 'set -a; . deploy/crew.env; set +a; [ "$MUGGE_CONCURRENCY" = 80 ] && [ -z "${CREW_WORKERS:-}" ]'
( cd /; unset CREW_WORKERS; set -a; . "$OLDPWD/deploy/crew.env"; set +a; [ "$MUGGE_CONCURRENCY" = 80 ] )
if MAX_NUM_SEQS=8 deploy/qwen.sh cmd >/dev/null 2>&1; then echo "MAX_NUM_SEQS below the crew was allowed"; exit 1; fi
# a public host is refused without a key
if HOST=0.0.0.0 ALLOW_PUBLIC=1 deploy/qwen.sh cmd >/dev/null 2>&1; then echo "public host without a key was allowed"; exit 1; fi
deploy/qwen.sh stop
# the data folder: download checks free space before it starts, saves the folder, and fetches into it
mkdir -p "$STATE/venv/bin"; ln -s "$(command -v python3)" "$STATE/venv/bin/python"
printf '#!/bin/sh\necho "hf $* into $HF_HOME"\n' > "$STATE/venv/bin/hf"; chmod +x "$STATE/venv/bin/hf"
if MODEL_GB=999999 VENV="$STATE/venv" deploy/qwen.sh download > "$STATE/dl.txt" 2>&1; then echo "download should refuse a full disk"; exit 1; fi
grep -q 'not enough space' "$STATE/dl.txt"
if grep -q '^hf download' "$STATE/dl.txt"; then echo "download ran on a full disk"; exit 1; fi
# a folder that failed the check is not remembered (the next plain download picks again)
if [ -f "$STATE/data.dir" ]; then echo "a data folder without room was saved"; exit 1; fi
MODEL_GB=0 VENV="$STATE/venv" deploy/qwen.sh download > "$STATE/dl.txt" 2>&1 || { cat "$STATE/dl.txt"; exit 1; }
grep -q "hf download Qwen/Qwen3.8-27B-FP8 into $STATE/data/hf" "$STATE/dl.txt"
[ "$(cat "$STATE/data.dir")" = "$STATE/data" ]
( unset DATA_DIR; deploy/qwen.sh where ) > "$STATE/where.txt"; grep -q "data folder  $STATE/data " "$STATE/where.txt"
# nvcc for FlashInfer: the venv's pip CUDA (nvidia/cu13) becomes CUDA_HOME; without it, a warning
cu="$STATE/cv/lib/python3.12/site-packages/nvidia/cu13"; mkdir -p "$cu/bin" "$cu/lib" "$STATE/cv/bin"
ln -s "$(command -v python3)" "$STATE/cv/bin/python"; touch "$cu/lib/libcudart.so.13"
VENV="$STATE/cv" deploy/qwen.sh cmd > "$STATE/cmd.txt" 2>&1; grep -q 'warning: no nvcc' "$STATE/cmd.txt"
printf '#!/bin/sh\necho nvcc\n' > "$cu/bin/nvcc"; chmod +x "$cu/bin/nvcc"
( unset CUDA_HOME; VENV="$STATE/cv" deploy/qwen.sh cmd ) > "$STATE/cmd.txt" 2>&1
if grep -q -- '--attention-backend' "$STATE/cmd.txt"; then cat "$STATE/cmd.txt"; echo "nvcc in the venv should keep FlashInfer"; exit 1; fi
[ -L "$cu/lib64" ] && [ -L "$cu/lib/libcudart.so" ]
# ninja from the venv is on the server's PATH
touch "$STATE/cv/bin/ninja"; chmod +x "$STATE/cv/bin/ninja"
( unset CUDA_HOME; VENV="$STATE/cv" bash -c 'source <(sed -n "/^cuda_env()/,/^}/p" deploy/qwen.sh); cuda_env; command -v ninja; echo "$CUDA_HOME"' ) > "$STATE/env.txt"
grep -qx "$STATE/cv/bin/ninja" "$STATE/env.txt"; grep -qx "$cu" "$STATE/env.txt"
ATTENTION=TRITON_ATTN deploy/qwen.sh cmd | grep -q -- '--attention-backend TRITON_ATTN'
# preflight: on a box with no GPU, venv or weights it fails, and every FAIL comes with a fix line
if NO_GPU=1 VENV="$STATE/novenv" deploy/qwen.sh preflight > "$STATE/pf.txt" 2>&1; then cat "$STATE/pf.txt"; echo "preflight passed with no venv"; exit 1; fi
grep -q '^FAIL  venv: ' "$STATE/pf.txt"; grep -q '^FAIL  weights: ' "$STATE/pf.txt"
[ "$(grep -c '^FAIL' "$STATE/pf.txt")" = "$(grep -c '^      fix: ' "$STATE/pf.txt")" ]
# start refuses to launch when preflight fails, and says so
if PREFLIGHT=1 NO_GPU=1 deploy/qwen.sh start > "$STATE/st.txt" 2>&1; then deploy/qwen.sh stop; echo "start ran past a failed preflight"; exit 1; fi
grep -q 'not starting: fix the FAIL lines' "$STATE/st.txt"
# a complete snapshot counts as downloaded; a missing shard does not
snap="$STATE/data/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/abc"; mkdir -p "$snap"
echo '{}' > "$snap/config.json"; echo '{"weight_map": {"a": "m-1.safetensors", "b": "m-2.safetensors"}}' > "$snap/model.safetensors.index.json"
touch "$snap/m-1.safetensors"
NO_GPU=1 deploy/qwen.sh preflight > "$STATE/pf.txt" 2>&1 || true; grep -q 'FAIL  weights: .*1 of 2 weight files missing' "$STATE/pf.txt"
touch "$snap/m-2.safetensors"
NO_GPU=1 deploy/qwen.sh preflight > "$STATE/pf.txt" 2>&1 || true; grep -q '^ok    weights: ' "$STATE/pf.txt"
# a failed start names the likely fix (the B200's nvcc/header mismatch)
printf '# x\nvllm serve\n#error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"\n' > "$STATE/serve.log"
echo 999999 > "$STATE/serve.pid"
deploy/qwen.sh wait > "$STATE/w.txt" 2>&1 && exit 1; grep -q 'likely fix: nvcc does not match' "$STATE/w.txt"
rm -f "$STATE/serve.pid"
# the bootstrap parses, and refuses to format anything that is not a blank disk
bash -n deploy/fresh.sh
if bash -c 'source <(sed -n "/^SUDO=/p;/^mounted_at()/p" deploy/fresh.sh; sed -n "/^empty_disk()/,/^}/p" deploy/fresh.sh); empty_disk "$(findmnt -n -o SOURCE /)"'; then echo "the root disk counted as empty"; exit 1; fi
# crew.env says when it falls back to the default crew
env -u CREW_WORKERS STATE="$STATE/nostate" bash -c 'set -a; . deploy/crew.env; set +a' 2>&1 | grep -q 'no crew size saved'
# the Mac path (deploy/mac.sh): deploy/mock_llama.py stands in for llama-server, a stub file for the GGUF
( export STATE="$STATE/mac" PORT=$((PORT + 1)) MAC_SH_ANY_OS=1 LLAMA_SERVER="$PWD/deploy/mock_llama.py" MOCK_LOAD_S=1
  mkdir -p "$STATE"; head -c 1000 /dev/zero > "$STATE/stub.gguf"
  trap 'deploy/mac.sh stop >/dev/null 2>&1 || true' EXIT
  # the 16 GB plan: Qwen3.5-9B Q4_K_M, 8 workers + 2 leads + 1 orchestrator, one slot of 16384 each, no shared pool
  python3 deploy/mac_crew.py --mem-gb 16 --wired-mb 0 | tee "$STATE/crew.txt"
  grep -q 'Qwen3.5-9B-Q4_K_M.gguf (5.68 GB)' "$STATE/crew.txt"; grep -q 'crew: 8 workers + 2 leads + 1 orchestrator = 11 slots' "$STATE/crew.txt"
  MODEL=9b deploy/mac.sh cmd | grep -q -- "-m $STATE/models/Qwen3.5-9B-Q4_K_M.gguf --alias qwen3.5-9b .*-np 11 -c 180224 -ctk q8_0 -ctv q8_0 .*--reasoning-format deepseek"
  if MODEL=9b deploy/mac.sh cmd | grep -q -- '--kv-unified'; then echo "mac.sh must not share one KV pool"; exit 1; fi
  MODEL=4b KV_V=q4_0 deploy/mac.sh cmd | grep -q -- 'Qwen3.5-4B-Q6_K.gguf .*-ctk q8_0 -ctv q4_0'
  # a crew that does not fit is refused, with what to change
  if python3 deploy/mac_crew.py --mem-gb 8 --wired-mb 0 --workers 8 > "$STATE/no.txt" 2>&1; then echo "an 8 GB Mac took 8 workers of 9B"; exit 1; fi
  grep -q 'does not fit' "$STATE/no.txt"
  # no model yet: preflight fails with the fix, start refuses
  if MODEL=9b deploy/mac.sh preflight > "$STATE/pf.txt" 2>&1; then echo "preflight passed with no model"; exit 1; fi
  grep -q 'FAIL  model .* not downloaded' "$STATE/pf.txt"; [ "$(grep -c '^FAIL' "$STATE/pf.txt")" = "$(grep -c '^      fix: ' "$STATE/pf.txt")" ]
  # an old llama-server (no --reasoning-budget-message) is caught before start
  printf '#!/bin/sh\necho "usage: llama-server -np N"\n' > "$STATE/old-llama"; chmod +x "$STATE/old-llama"
  if LLAMA_SERVER="$STATE/old-llama" MODEL="$STATE/stub.gguf" PRESET=qwen3.5-9b deploy/mac.sh preflight > "$STATE/old.txt" 2>&1; then echo "preflight passed an old llama-server"; exit 1; fi
  grep -q 'FAIL  llama-server at .* is too old' "$STATE/old.txt"; grep -qE 'fix: (brew upgrade llama.cpp|deploy/mac.sh setup)' "$STATE/old.txt"
  # the run: start, wait, check (thinking with a budget, JSON with thinking, the whole crew at once), status
  MODEL="$STATE/stub.gguf" PRESET=qwen3.5-9b deploy/mac.sh start
  deploy/mac.sh wait | tee "$STATE/wait.txt"; grep -q 'crew: 8 workers + 2 leads + 1 orchestrator on 11 slots of 16384 tokens' "$STATE/wait.txt"
  deploy/mac.sh check | tee "$STATE/check.txt"; grep -q 'ok    11 calls at once' "$STATE/check.txt"; grep -q '^PASS' "$STATE/check.txt"
  deploy/mac.sh status > "$STATE/status.txt"; grep -q 'health  ok' "$STATE/status.txt"
  # the crew file matches what started, from bash and from sh (zsh users source it the same way)
  ( set -a; . deploy/mac.env; set +a
    [ "$MUGGE_CONCURRENCY" = 10 ] && [ "$MUGGE_MODEL" = stub ] && [ "$MUGGE_THINKING_BUDGET" = low=1024,medium=4096,xhigh=8192 ]
    echo "$MUGGE_BACKENDS" | grep -q '"mac-workers".*"slots": 8'
    bun src/cli.ts backends > "$STATE/backends.txt"; [ "$(grep -c '^  up ' "$STATE/backends.txt")" = 3 ] )
  sh -c 'set -a; . deploy/mac.env; set +a; [ "$MUGGE_CONCURRENCY" = 10 ]'
  deploy/mac.sh bench --levels 1,4 --prompt-tokens 200 --answers thinking:32,short:8 > "$STATE/bench.txt"; grep -q '1 -> 4 at once' "$STATE/bench.txt"
  deploy/mac.sh stop )
# centipede-server (deploy/box.sh: the same script on Linux with an NVIDIA GPU): a mock nvidia-smi
# (an 8 GB laptop RTX 3070), the head and a helper llama-server on this machine
( export STATE="$STATE/box" PORT=$((PORT + 2)) LLAMA_SERVER="$PWD/deploy/mock_llama.py" MOCK_LOAD_S=1
  HPORT=$((PORT + 1)) HSTATE="$STATE/helper"
  mkdir -p "$STATE/bin"; ln -sf "$PWD/deploy/mock_nvidia_smi.py" "$STATE/bin/nvidia-smi"; export PATH="$STATE/bin:$PATH"
  head -c 1000 /dev/zero > "$STATE/stub.gguf"
  trap 'deploy/box.sh stop >/dev/null 2>&1 || true; STATE="$HSTATE" PORT="$HPORT" deploy/box.sh stop >/dev/null 2>&1 || true' EXIT
  # an 8 GB card: q4_0 KV and 12,288 tokens a slot; the head 4 workers + 2 leads + 1 orchestrator, a helper with 600 MiB in use 4 slots
  deploy/box.sh crew | tee "$STATE/crew.txt"
  grep -q 'GPU: NVIDIA GeForce RTX 3070 Laptop GPU' "$STATE/crew.txt"; grep -q 'crew: 4 workers + 2 leads + 1 orchestrator = 7 slots' "$STATE/crew.txt"
  deploy/box.sh cmd > "$STATE/cmd.txt"; grep -q -- '-np 7 -c 86016 -ctk q4_0 -ctv q4_0' "$STATE/cmd.txt"
  MOCK_GPU_USED_MIB=600 ROLE=helper deploy/box.sh crew > "$STATE/out1.txt"; grep -q 'crew: 4 worker slots' "$STATE/out1.txt"
  MODEL="$STATE/stub.gguf" PRESET=qwen3.5-9b deploy/box.sh preflight > "$STATE/out2.txt"; grep -q 'ok    NVIDIA GPU: NVIDIA GeForce RTX 3070 Laptop GPU, 8192 MiB' "$STATE/out2.txt"
  # an old driver is refused before anything downloads
  if MOCK_DRIVER=550.1 LLAMA_SERVER= deploy/box.sh setup > "$STATE/drv.txt" 2>&1; then echo "setup took driver 550"; exit 1; fi
  grep -q 'driver 550 is too old' "$STATE/drv.txt"
  # llama.cpp's latest release is a version without binaries: setup takes the build it names
  mkdir -p "$STATE/gh"; cat > "$STATE/gh/curl" <<'SH'
#!/bin/sh
for a; do u="$a"; done
case "$u" in
  */releases/latest) echo '{"tag_name": "v0.7.0"}' ;;
  */v0.7.0/nightly-tag.txt) [ -z "$NO_NIGHTLY" ] || exit 22; echo b11669 ;;
  *'/releases?per_page=30') echo '[{"tag_name": "v0.7.0"}, {"tag_name": "b11732"}, {"tag_name": "b11731"}]' ;;
  *) exit 22 ;;
esac
SH
  chmod +x "$STATE/gh/curl"
  [ "$(PATH="$STATE/gh:$PATH" deploy/box.sh release-tag)" = b11669 ]
  [ "$(NO_NIGHTLY=1 PATH="$STATE/gh:$PATH" deploy/box.sh release-tag)" = b11732 ]
  [ "$(LLAMA_TAG=b11700 PATH="$STATE/gh:$PATH" deploy/box.sh release-tag)" = b11700 ]
  # the helper, then the head, then the head's crew with the helper's slots
  STATE="$HSTATE" PORT="$HPORT" CREW_WORKERS=4 ROLE=helper HOST=0.0.0.0 MODEL="$STATE/stub.gguf" PRESET=qwen3.5-9b deploy/box.sh start
  STATE="$HSTATE" PORT="$HPORT" deploy/box.sh wait > "$STATE/out3.txt"; grep -q "crew: 4 worker slots for another box's crew on 4 slots of 12288 tokens" "$STATE/out3.txt"
  CREW_WORKERS=4 MODEL="$STATE/stub.gguf" PRESET=qwen3.5-9b deploy/box.sh start
  deploy/box.sh wait > "$STATE/out4.txt"; grep -q 'crew: 4 workers + 2 leads + 1 orchestrator on 7 slots of 12288 tokens' "$STATE/out4.txt"
  if deploy/box.sh helper add "127.0.0.1:$((PORT + 9))" > "$STATE/dead.txt" 2>&1; then echo "added a helper that is not there"; exit 1; fi
  if deploy/box.sh helper add http://WINDOWS-ADDRESS:8080 > "$STATE/ph.txt" 2>&1; then echo "added a placeholder"; exit 1; fi
  grep -q 'is a placeholder' "$STATE/ph.txt"
  deploy/box.sh helper add "127.0.0.1:$HPORT" > "$STATE/out5.txt"; grep -q '4 slots of 12288 tokens' "$STATE/out5.txt"
  deploy/box.sh status > "$STATE/out6.txt"; grep -q "helper   http://127.0.0.1:$HPORT up" "$STATE/out6.txt"
  ( set -a; . deploy/mac.env; set +a
    [ "$MUGGE_CONCURRENCY" = 10 ] && [ "$MUGGE_MAX_CALL_TOKENS" = 6144 ] && [ "$MUGGE_PLAN_MAX_CALL_TOKENS" = 8192 ]
    [[ "$(echo "$MUGGE_BACKENDS" | tr -d '\n ')" == *'"name":"helper-1"'*'"slots":4'* ]]
    bun src/cli.ts backends > "$STATE/backends.txt"; [ "$(grep -c '^  up ' "$STATE/backends.txt")" = 4 ] )
  deploy/box.sh check | tee "$STATE/check.txt"; grep -q '^PASS' "$STATE/check.txt"
  deploy/box.sh ssh-config > "$STATE/out7.txt"; grep -q '^Host centipede-server' "$STATE/out7.txt"
  deploy/box.sh stop; STATE="$HSTATE" PORT="$HPORT" deploy/box.sh stop )
echo "deploy dry run: PASS"
