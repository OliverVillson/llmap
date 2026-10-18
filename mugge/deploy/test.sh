#!/usr/bin/env bash
# Dry run of the deploy scripts with no GPU: deploy/mock_vllm.py stands in for vLLM, and the crew
# file is checked against it the way `mugge run` checks machines at start. CI runs this.
set -euo pipefail
cd "$(dirname "$0")/.."
export STATE="$(mktemp -d)" VLLM="$PWD/deploy/mock_vllm.py" GPU=b200 PORT="${PORT:-18000}" MOCK_LOAD_S=2
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
( unset DATA_DIR; MODEL_GB=0 VENV="$STATE/venv" deploy/qwen.sh download > "$STATE/dl.txt" 2>&1 ) || { cat "$STATE/dl.txt"; exit 1; }
grep -q "hf download Qwen/Qwen3.8-27B-FP8 into $STATE/data/hf" "$STATE/dl.txt"
[ "$(cat "$STATE/data.dir")" = "$STATE/data" ]
( unset DATA_DIR; deploy/qwen.sh where ) > "$STATE/where.txt"; grep -q "data folder  $STATE/data " "$STATE/where.txt"
echo "deploy dry run: PASS"
