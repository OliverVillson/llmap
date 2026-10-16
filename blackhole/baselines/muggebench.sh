#!/usr/bin/env bash
# MuggeBench for one GGUF on a laptop: the RTX 4060 laptop under WSL2 (or any Linux box with
# an NVIDIA GPU), and Apple Silicon Macs. It is the same scoring the Centipede run uses on the
# B200: llmap's eval stage serves the GGUF with llama-server, puts its forced-answer proxy in
# front, and runs every bench ticket through the Mugge engine's write, check and fix loop,
# graded on hidden tests.
#
#   GGUF=~/models/BH-0.1-1.7B-Q4_K_M.gguf bash muggebench.sh
#   GGUF=... SEEDS=1 TICKETS=20 bash muggebench.sh                     # a quick look first
#   GGUF=... BENCH_TAR=~/muggebench-dev.tar.gz bash muggebench.sh      # the private dev set
#
# Without BENCH_TAR (or ~/muggebench-dev.tar.gz), only the public spike project is scored
# (15 tickets). That is enough to see the pipeline work, but it is not the real Mugge score.
#
# Prints the Mugge score with its 95% interval and the model's footprint, then checks them
# against the BlackHole target: a score of at least 50% (TARGET_SCORE) with a footprint of at
# most 2 GB (TARGET_GB, in 1e9 bytes, the unit the Centipede run uses). The footprint is what
# one agent needs: the model plus its working memory at a 16k context (FOOTPRINT_CTX), measured
# by serving the GGUF once with one slot of that context and the bench's KV cache type, and
# summing every buffer llama-server allocates on every device (weights, KV cache, compute and
# output buffers). With nvidia-smi, the GPU memory that load took is printed beside it.
#
# Sampling is BlackHole's one set (SAMPLING=blackhole, the default): thinking on at
# temperature 0.6, top_p 0.95, top_k 20; thinking off at 0.7, 0.8, 20; min_p 0, no penalties.
# SAMPLING=centipede is the Centipede run's MuggeBench set (scripts/centipede.py EVAL:
# temperature 1.0, top_p 0.95, no top_k), to compare with VM numbers; its results are kept
# apart, under NAME-centipede. TEMP, TOP_P, TOP_K and NOTHINK_TEMP, NOTHINK_TOP_P, NOTHINK_TOP_K
# override one value. Other settings: THINKING=1, TOKENS=8192 (per call), SEEDS=3, KV=q8_0
# (8-bit KV cache, which fits more slots in 8 GB), CUT_RETRY=nothink (a thinking call that runs
# out of tokens is asked again at once with thinking off, and the ticket's later calls go
# without thinking, as Centipede's harness does). Change them only for every model you compare.
# CTX16=1 scores in a 16k context per call instead (TOKENS=4096, LOBBOT_EVAL_CTX=16384), the
# context the footprint is measured at; its results are kept apart, under NAME-ctx16k.
#
# Re-running is safe: setup steps that are done are skipped. A finished model is reported
# again from its results. Delete $WORK/jobs/<NAME> to score it again.
set -euo pipefail

GGUF=${GGUF:?set GGUF to the model file}
SAMPLING=${SAMPLING:-blackhole}
NAME=${NAME:-$(basename "$GGUF" .gguf)$([ "${CTX16:-0}" = 1 ] && echo -ctx16k)$([ "$SAMPLING" = blackhole ] || echo "-$SAMPLING")}
WORK=${WORK:-$HOME/muggebench}
LLMAP=${LLMAP:-$WORK/llmap}
LLMAP_REF=${LLMAP_REF:-776570b890a4cc67e5823d3c98d8b73fa5f84dfc}  # llmap branch claude/project-thread-3t4k6m: MuggeBench plus Centipede's thinking-cap retry
LLAMA_CPP=${LLAMA_CPP:-$WORK/llama.cpp}
LLAMA_CPP_SHA=${LLAMA_CPP_SHA:-af50f2f248b475d9760f8e67503d01c0f194f5f4}  # the Centipede run's
BUN_VERSION=${BUN_VERSION:-1.3.14}
BENCH_TAR=${BENCH_TAR:-$HOME/muggebench-dev.tar.gz}
SEEDS=${SEEDS:-3}
TICKETS=${TICKETS:-}
THINKING=${THINKING:-1}
CTX16=${CTX16:-0}
if [ "$CTX16" = 1 ]; then
  # 16K mode: every call fits in the 16k context the footprint is measured at (prompt up to
  # ~8k, 4k of thinking, 4k forced answer), instead of the Centipede run's ~28.7k.
  TOKENS=${TOKENS:-4096}
  export LOBBOT_EVAL_CTX=16384
fi
TOKENS=${TOKENS:-8192}
case $SAMPLING in
  blackhole) TEMP=${TEMP:-0.6} TOP_P=${TOP_P:-0.95} TOP_K=${TOP_K:-20}
             NOTHINK_TEMP=${NOTHINK_TEMP:-0.7} NOTHINK_TOP_P=${NOTHINK_TOP_P:-0.8} NOTHINK_TOP_K=${NOTHINK_TOP_K:-20} ;;
  centipede) TEMP=${TEMP:-1.0} TOP_P=${TOP_P:-0.95} TOP_K=${TOP_K:-0}  # 0: no top_k (the server gets 100, see stages/eval.py)
             NOTHINK_TEMP=${NOTHINK_TEMP:-1.0} NOTHINK_TOP_P=${NOTHINK_TOP_P:-0.95} NOTHINK_TOP_K=${NOTHINK_TOP_K:-0} ;;
  *) echo "SAMPLING must be blackhole or centipede"; exit 1 ;;
esac
KV=${KV:-q8_0}
PARSER=${PARSER:-qwen3}
CUT_RETRY=${CUT_RETRY:-nothink}  # nothink: a thinking call that runs out is asked again without thinking, as Centipede does; force: the old forced answer
TARGET_SCORE=${TARGET_SCORE:-50}
TARGET_GB=${TARGET_GB:-2.0}
FOOTPRINT_CTX=${FOOTPRINT_CTX:-16384}
FOOTPRINT_PROMPT=${FOOTPRINT_PROMPT:-12000}
FOOTPRINT_UBATCH=${FOOTPRINT_UBATCH-128}  # also measured at this -ub (empty: the default -ub only)

say() { printf '\n==> %s\n' "$*"; }
OS=$(uname -s)
[ -s "$GGUF" ] || { echo "No model at $GGUF"; exit 1; }
GGUF=$(cd "$(dirname "$GGUF")" && pwd)/$(basename "$GGUF")
mkdir -p "$WORK/jobs"

say "Tools"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
export PATH="$HOME/.bun/bin:$PATH"
if ! command -v bun >/dev/null; then
  command -v unzip >/dev/null || { [ "$OS" = Linux ] && sudo apt-get install -y -qq unzip; }
  curl -fsSL https://bun.sh/install | bash -s "bun-v$BUN_VERSION"
fi
SERVER=$LLAMA_CPP/build/bin/llama-server
if [ ! -x "$SERVER" ]; then
  if [ "$OS" = Darwin ]; then
    command -v llama-server >/dev/null || brew install llama.cpp
    SERVER=$(command -v llama-server)
  else
    say "Building llama.cpp with CUDA (once, about 10 minutes)"
    command -v nvcc >/dev/null || [ -x /usr/local/cuda/bin/nvcc ] || {
      echo "No CUDA toolkit. In WSL2: sudo apt-get install -y nvidia-cuda-toolkit (the Windows NVIDIA driver provides the GPU)."; exit 1; }
    export PATH="/usr/local/cuda/bin:$PATH"
    command -v cmake >/dev/null || sudo apt-get install -y -qq cmake build-essential git
    [ -d "$LLAMA_CPP/.git" ] || git clone -q https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
    git -C "$LLAMA_CPP" fetch -q origin "$LLAMA_CPP_SHA" && git -C "$LLAMA_CPP" checkout -q "$LLAMA_CPP_SHA"
    cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$LLAMA_CPP/build" --target llama-server -j "$(nproc)" >/dev/null
  fi
fi
echo "llama-server: $SERVER"

if [ ! -e "$LLMAP/.git" ]; then
  say "llmap (the eval stage and the Mugge engine)"
  git clone -q https://github.com/OliverVillson/llmap "$LLMAP"
fi
git -C "$LLMAP" cat-file -e "$LLMAP_REF^{commit}" 2>/dev/null || git -C "$LLMAP" fetch -q origin "$LLMAP_REF" || git -C "$LLMAP" fetch -q origin
[ "$(git -C "$LLMAP" rev-parse HEAD)" = "$(git -C "$LLMAP" rev-parse "$LLMAP_REF")" ] || git -C "$LLMAP" checkout -q "$LLMAP_REF"
[ -d "$LLMAP/mugge/node_modules" ] || (cd "$LLMAP/mugge" && bun install --silent)
[ -x "$WORK/venv/bin/python" ] || uv venv -q --python 3.12 "$WORK/venv"
"$WORK/venv/bin/python" -c "import httpx" 2>/dev/null || uv pip install -q --python "$WORK/venv/bin/python" httpx
# The local sandbox and the bench's Python tickets run `python3`: the venv's comes first.
export PATH="$WORK/venv/bin:$PATH"

say "Bench projects"
BENCH=$WORK/bench
if [ ! -f "$BENCH/.ready" ] || { [ -s "$BENCH_TAR" ] && [ "$BENCH_TAR" -nt "$BENCH/.ready" ]; }; then
  rm -rf "$BENCH" && mkdir -p "$BENCH"
  cp -R "$LLMAP/mugge/spike" "$BENCH/spike"
  if [ -s "$BENCH_TAR" ]; then
    tar xzf "$BENCH_TAR" -C "$BENCH"
  else
    echo "No dev set at $BENCH_TAR: scoring the public spike only (not the real Mugge score)."
  fi
  touch "$BENCH/.ready"
fi
PROJECTS=$(find "$BENCH" -name plan.json -not -path '*/node_modules/*' | wc -l | tr -d ' ')
IMAGES=$(python3 - "$BENCH" <<'PY'
import json, sys
from pathlib import Path
print(" ".join(sorted({t.get("image", "node") for p in Path(sys.argv[1]).rglob("plan.json") for t in json.loads(p.read_text())["tickets"]})))
PY
)
echo "$PROJECTS projects in $BENCH, toolchain images: $IMAGES"

say "Sandbox"
SANDBOX_ASKED=${SANDBOX:-}
if [ -z "${SANDBOX:-}" ]; then
  if command -v docker >/dev/null && docker info >/dev/null 2>&1; then SANDBOX=docker
  elif command -v podman >/dev/null && podman info >/dev/null 2>&1; then SANDBOX=podman
  else SANDBOX=local; fi
fi
if [ "$SANDBOX" = local ] && [ "$PROJECTS" -gt 1 ] && [ "$SANDBOX_ASKED" != local ]; then
  echo "The dev set needs Docker or Podman, and neither is running. In the local sandbox its ml and dotnet tickets"
  echo "cannot run and fix prompts show this machine's paths, not /work as in training, so the score is not the one"
  echo "that counts. Start Docker (Docker Desktop with WSL integration on Windows), or set SANDBOX=local to score anyway."
  exit 1
fi
if [ "$SANDBOX" = local ]; then
  echo "No Docker or Podman: tickets run on this machine, with its own toolchains."
  echo "That is fine for the spike. The dev set's ml and dotnet tickets need Docker (Docker Desktop with WSL integration)."
else
  for img in $IMAGES; do
    "$SANDBOX" image inspect "mugge-$img" >/dev/null 2>&1 || (cd "$LLMAP/mugge" && ENGINE=$SANDBOX bash sandbox/build.sh "$img")
  done
  echo "$SANDBOX images ready: $(for i in $IMAGES; do printf 'mugge-%s ' "$i"; done)"
fi
# Tickets left out of every score, so every BlackHole number is on the same tickets: the dev
# set's four Playwright tickets (the web image installs Chromium where the sandbox's HOME=/tmp
# does not look) and booking-net/dn-repo (its hidden test fails even with the reference
# answer, most often under load). They are left out whether or not they fail on a given run.
UNSCORABLE=${UNSCORABLE-rota-ui/fe-grid-keys,rota-ui/fe-layout,rota-ui/fe-form,rota-ui/fe-grid,booking-net/dn-repo}
EXCLUDED_FILE=$BENCH/.excluded-$SANDBOX
if [ ! -f "$BENCH/.preflight-$SANDBOX" ]; then
  say "Preflight: every scorable ticket once with its reference answer (no model), to check the sandbox"
  rm -rf "$WORK/preflight" "$WORK"/preflight-again-*
  (cd "$LLMAP/mugge" && bun bench/run.ts --bench "$BENCH" --out "$WORK/preflight" --fake --seeds 1 \
    --concurrency 4 --sandbox "$SANDBOX" ${UNSCORABLE:+--exclude "$UNSCORABLE"} > "$WORK/preflight.log" 2>&1) || true
  [ -s "$WORK/preflight/results.jsonl" ] || { echo "The preflight did not run. Full log: $WORK/preflight.log"; exit 1; }
  failed=$(python3 -c 'import json,sys; print(" ".join(r["key"] for r in map(json.loads, open(sys.argv[1])) if not r["solved"]))' "$WORK/preflight/results.jsonl")
  still=""
  for key in $failed; do  # once more, alone: four sandboxes at once can starve a small VM
    echo "  $key failed; trying it again on its own"
    out=$WORK/preflight-again-${key//\//_}
    (cd "$LLMAP/mugge" && bun bench/run.ts --bench "$BENCH" --out "$out" --fake --seeds 1 --concurrency 1 \
      --sandbox "$SANDBOX" --filter "$key" > "$out.log" 2>&1) || true
    python3 -c 'import json,sys; sys.exit(0 if any(r["key"] == sys.argv[2] and r["solved"] for r in map(json.loads, open(sys.argv[1]))) else 1)' \
      "$out/results.jsonl" "$key" 2>/dev/null || still="$still $key"
  done
  if [ -n "$still" ]; then
    echo "Reference answers fail for these tickets, twice, so the sandbox cannot run them:"
    for key in $still; do
      out=$WORK/preflight-again-${key//\//_}
      echo "  $key: $(python3 -c 'import json,sys; r=[r for r in map(json.loads, open(sys.argv[1])) if r["key"] == sys.argv[2]]; print(((r[0].get("error") or "").strip().splitlines() or [""])[-1][:200] if r else "no result")' "$out/results.jsonl" "$key" 2>/dev/null || echo "see $out.log")"
    done
    echo "Full logs: $WORK/preflight.log and $WORK/preflight-again-*.log. On a Mac, give Podman or Docker 6 GB and 8 CPUs (see the README)."
    exit 1
  fi
  # the known unscorable tickets this bench has (the spike has none)
  python3 - "$BENCH" "$UNSCORABLE" "$EXCLUDED_FILE" <<'PY'
import json, sys
from pathlib import Path
bench, known, out = Path(sys.argv[1]), [k for k in sys.argv[2].split(",") if k], Path(sys.argv[3])
keys = {f"{p.parent.name}/{t['id']}" for p in bench.rglob("plan.json") if "node_modules" not in p.parts for t in json.loads(p.read_text())["tickets"]}
out.write_text(",".join(k for k in known if k in keys))
PY
  touch "$BENCH/.preflight-$SANDBOX"
  echo "reference answers: every scorable ticket solved$( [ -n "$failed" ] && echo " (after a second try on its own for:$(printf ' %s' $failed))")"
fi
EXCLUDE=$(cat "$EXCLUDED_FILE" 2>/dev/null || true)
[ -n "$EXCLUDE" ] && echo "left out of every score: ${EXCLUDE//,/, }"

JOB=$WORK/jobs/$NAME
mkdir -p "$JOB"
cp "$LLMAP/examples/python-utils.code.taskspec.json" "$JOB/taskspec.json"
THINK=$([ "$THINKING" = 1 ] && echo true || echo false)
cat > "$JOB/config.json" <<EOF
{
  "llama_cpp": "$(dirname "$(dirname "$(dirname "$SERVER")")")",
  "eval_candidates": {"$NAME": "$GGUF"},
  "code_eval_suites": [],
  "code_eval_thinking": $THINK,
  "code_eval_thinking_temperature": $TEMP,
  "code_eval_temperature": $NOTHINK_TEMP,
  "thinking_top_k": $TOP_K,
  "eval_top_p": $TOP_P,
  "eval_nothink_top_p": $NOTHINK_TOP_P,
  "eval_nothink_top_k": $NOTHINK_TOP_K,
  "eval_min_p": 0.0,
  "code_eval_max_tokens": $TOKENS,
  "eval_reasoning_parser": "$PARSER",
  "eval_kv_type": "$KV",
  "mugge_bench_dir": "$BENCH",
  "mugge_bench_seeds": $SEEDS,
  "mugge_bench_split": "",
  "mugge_bench_sandbox": "$SANDBOX",
  "mugge_bench_cut_retry": "$CUT_RETRY",
  "mugge_bench_exclude": $(python3 -c 'import json,sys; print(json.dumps([k for k in sys.argv[1].split(",") if k]))' "$EXCLUDE")
}
EOF
if [ "$OS" = Darwin ]; then
  export LOBBOT_EVAL_SLOTS=${SLOTS:-2}  # no nvidia-smi to size them from
elif [ -n "${SLOTS:-}" ]; then
  export LOBBOT_EVAL_SLOTS=$SLOTS
fi
# The bench runs as many tickets at once as llama-server has slots.
CONC=${LOBBOT_EVAL_SLOTS:-$("$WORK/venv/bin/python" - "$LLMAP" "$JOB" "$GGUF" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from stages._util import Job
from stages import eval as ev
print(ev.llama_slots(Job(sys.argv[2]).config, sys.argv[3]))
PY
)}
python3 - "$JOB/config.json" "$CONC" "$TICKETS" <<'PY'
import json, sys
p = sys.argv[1]; c = json.load(open(p)); c["mugge_bench_concurrency"] = int(sys.argv[2])
if sys.argv[3]:  # here, not in the heredoc: bash 3.2 (macOS /bin/bash) keeps \" inside ${VAR:+...} there
    c["mugge_bench_tickets"] = int(sys.argv[3])
json.dump(c, open(p, "w"), indent=2)
PY

if [ ! -s "$JOB/footprint.json" ]; then
  say "Footprint: one agent at a ${FOOTPRINT_CTX}-token context, after a ${FOOTPRINT_PROMPT}-token prompt"
  FPORT=${FOOTPRINT_PORT:-8092}
  # Code to fill the prompt with: llmap's own sources, cut to FOOTPRINT_PROMPT tokens.
  cat "$LLMAP"/stages/*.py "$LLMAP"/mugge/src/*.ts 2>/dev/null | head -c 400000 > "$JOB/footprint-prompt.txt" || true
  for ub in "" $FOOTPRINT_UBATCH; do
    tag=${ub:-default}
    peakf=$JOB/footprint-gpu-$tag.log; : > "$peakf"
    before=""
    if command -v nvidia-smi >/dev/null; then
      before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$peakf" 2>/dev/null &
      smi=$!
    fi
    # buffer sizes are logged only at verbosity 4 in newer builds (brew's 0.5.0)
    LLAMA_ARG_LOG_VERBOSITY=4 "$SERVER" -m "$GGUF" -ngl 999 --host 127.0.0.1 --port "$FPORT" -c "$FOOTPRINT_CTX" -np 1 --jinja \
      ${ub:+-ub "$ub"} ${KV:+-ctk "$KV" -ctv "$KV" -fa on} > "$JOB/footprint-server-$tag.log" 2>&1 &
    fp=$!
    for _ in $(seq 300); do curl -sf "http://127.0.0.1:$FPORT/health" >/dev/null && break; kill -0 $fp 2>/dev/null || break; sleep 1; done
    python3 - "http://127.0.0.1:$FPORT" "$JOB/footprint-prompt.txt" "$FOOTPRINT_PROMPT" > "$JOB/footprint-prompt-$tag.txt" <<'PY' || true
import json, sys, urllib.request
url, path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
def post(route, body):
    req = urllib.request.Request(url + route, json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))
tokens = post("/tokenize", {"content": open(path, errors="replace").read()})["tokens"][:n]
r = post("/completion", {"prompt": tokens, "n_predict": 64, "temperature": 0, "cache_prompt": False})
print(r.get("timings", {}).get("prompt_n", r.get("tokens_evaluated", len(tokens))))
PY
    kill $fp 2>/dev/null; wait $fp 2>/dev/null || true
    [ -n "${smi:-}" ] && { kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null || true; smi=""; }
    echo "${before:-}" > "$JOB/footprint-gpu-$tag.before"
  done
  BUILD=$("$SERVER" --version 2>&1 | grep -m1 -iE "version|build" || echo unknown)
  python3 - "$JOB" "$FOOTPRINT_CTX" "$KV" "$FOOTPRINT_PROMPT" "$GGUF" "$BUILD" $FOOTPRINT_UBATCH <<'PY'
import json, re, sys
from pathlib import Path
job, ctx, kv, want, gguf, build, ubs = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), Path(sys.argv[5]), sys.argv[6], sys.argv[7:]
# The weights count as the GGUF file: llama.cpp maps it, and its model buffers on several
# devices (CPU_Mapped and MTL0_Mapped on Metal) can be views of the same pages.
shards = sorted(gguf.parent.glob(gguf.name.replace("-00001-of-", "-*-of-"))) if "-00001-of-" in gguf.name else [gguf]
file_gb = sum(p.stat().st_size for p in shards) / 1e9
runs = {}
for tag in ["default", *ubs]:
    log = job / f"footprint-server-{tag}.log"
    buffers = {}
    for dev, kind, mib in re.findall(r"(\S+)\s+(model|KV|compute|output|RS) buffer size\s*=\s*([\d.]+) MiB", log.read_text(errors="replace")):
        buffers[f"{dev} {kind}"] = buffers.get(f"{dev} {kind}", 0) + float(mib)
    if not buffers:
        sys.exit(f"llama-server logged no buffer sizes; see {log}")
    prompt = (job / f"footprint-prompt-{tag}.txt").read_text().strip()
    if not prompt.isdigit():
        sys.exit(f"the {want}-token prompt failed (-ub {tag}); see {log}")
    by = {k: sum(v for b, v in buffers.items() if b.endswith(" " + k)) * 2**20 / 1e9 for k in ("model", "KV", "RS", "compute", "output")}
    before = (job / f"footprint-gpu-{tag}.before").read_text().strip()
    samples = [int(x) for x in (job / f"footprint-gpu-{tag}.log").read_text().split() if x.isdigit()]
    peak = (max(samples) - int(before)) * 2**20 / 1e9 if before and samples else None
    work = sum(v for k, v in by.items() if k != "model")
    runs[tag] = {"ubatch": tag, "prompt_tokens": int(prompt), "total_gb": round(file_gb + work, 3),
                 "file_gb": round(file_gb, 3), "working_gb": round(work, 3),
                 "buffer_sum_gb": round(sum(buffers.values()) * 2**20 / 1e9, 3),
                 "by_kind_gb": {k: round(v, 3) for k, v in by.items() if v}, "buffers_mib": buffers,
                 "gpu_peak_gb": round(peak, 3) if peak is not None else None}
    print(f"  -ub {tag}: {runs[tag]['total_gb']:.2f} GB (file {file_gb:.2f} + working memory {work:.2f}) after {prompt} prompt tokens; buffers: "
          + ", ".join(f"{k} {v:.2f}" for k, v in by.items() if v)
          + (f"; peak GPU memory {peak:.2f} GB, CUDA context included" if peak is not None else ""))
best = min(runs.values(), key=lambda r: r["total_gb"])
json.dump({"ctx": ctx, "kv_type": kv, "prompt_tokens": want, "llama_server": build,
           **{k: best[k] for k in ("ubatch", "total_gb", "file_gb", "working_gb", "buffer_sum_gb", "by_kind_gb", "gpu_peak_gb")},
           "runs": runs}, open(job / "footprint.json", "w"), indent=2)
print(f"  footprint: {best['total_gb']:.2f} GB (-ub {best['ubatch']}, the smaller; -ub changes speed, not answers)")
PY
fi

if [ ! -f "$JOB/.done/eval" ]; then
  say "Scoring $NAME: $CONC tickets at once, $SEEDS seed(s) (log: $JOB/run.log)"
  baseline=""
  if command -v nvidia-smi >/dev/null; then
    baseline=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    (while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; sleep 2; done) > "$JOB/gpu-mem.log" 2>/dev/null &
    sampler=$!
  fi
  keepawake=()
  [ "$OS" = Darwin ] && keepawake=(caffeinate -i)
  set +e
  (cd "$LLMAP" && ${keepawake[@]+"${keepawake[@]}"} "$WORK/venv/bin/python" pipeline.py --job "$JOB" --only eval) > "$JOB/run.log" 2>&1 &
  run=$!
  while kill -0 $run 2>/dev/null; do
    line=$(grep -E "^ *[0-9]+/[0-9]+ " "$JOB"/work/mugge-bench-*.log 2>/dev/null | tail -1)
    [ -n "$line" ] && printf '\r  %s' "$(echo "$line" | cut -c1-90)                    "
    sleep 5
  done
  wait $run; code=$?
  set -e
  [ -n "${sampler:-}" ] && kill "$sampler" 2>/dev/null
  echo
  [ -n "$baseline" ] && echo "$baseline" > "$JOB/gpu-baseline.txt"
  [ -f "$JOB/.done/eval" ] || { echo "The eval failed (exit $code). Last lines of $JOB/run.log:"; tail -25 "$JOB/run.log"; exit 1; }
fi

say "Result"
RUN=$JOB/work/mugge_bench/$NAME
[ -s "$RUN/report.json" ] || { echo "MuggeBench gave no score. Last lines of $JOB/work/mugge-bench-$NAME.log:"; tail -20 "$JOB/work/mugge-bench-$NAME.log"; rm -f "$JOB/.done/eval"; exit 1; }
mkdir -p "$JOB/rollouts"  # one row per thinking call, named after the model, for scripts/thinking_report.py
[ -s "$RUN/transcripts.jsonl" ] && [ ! -s "$JOB/rollouts/$NAME.jsonl" ] && \
  "$WORK/venv/bin/python" "$LLMAP/scripts/mugge_rollouts.py" "$RUN" --out "$JOB/rollouts/$NAME.jsonl" >/dev/null || true
"$WORK/venv/bin/python" - "$JOB" "$NAME" "$GGUF" "$TARGET_SCORE" "$TARGET_GB" "$PROJECTS" "$LLMAP" <<'PY'
import json, sys
from pathlib import Path
job, name, gguf, target, budget, projects = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
r = json.loads((job / "work" / "mugge_bench" / name / "report.json").read_text())
fp = json.loads((job / "footprint.json").read_text())
shards = sorted(gguf.parent.glob(gguf.name.replace("-00001-of-", "-*-of-"))) if "-00001-of-" in gguf.name else [gguf]
size = sum(p.stat().st_size for p in shards) / 1e9
lo, hi = r["scoreCI"]
excluded = json.loads((job / "config.json").read_text()).get("mugge_bench_exclude", [])
print(f"{name}: Mugge score {r['score']}% (95% {lo}-{hi}) on {r['tickets']} scorable tickets x {r['seeds']} seeds, {projects} projects"
      + (f" ({len(excluded)} known unscorable left out: {', '.join(excluded)})" if excluded else ""))
parts = ", ".join(f"{k} {v:.2f}" for k, v in fp["by_kind_gb"].items())
print(f"  footprint {fp['total_gb']:.2f} GB for one agent at {fp['ctx']} context after a {fp.get('prompt_tokens', '?')}-token prompt: "
      f"GGUF file {size:.2f} GB + working memory {fp.get('working_gb', fp['total_gb'] - size):.2f} GB "
      f"({fp['kv_type'] or 'f16'} KV, -ub {fp.get('ubatch', 'default')})")
if "buffer_sum_gb" in fp:
    print(f"  every buffer llama-server allocated, summed: {fp['buffer_sum_gb']:.2f} GB ({parts}); mapped views of the file can count twice")
print(f"  llama-server: {fp.get('llama_server', 'unknown')}")
for tag, run in fp.get("runs", {}).items():
    print(f"    -ub {tag}: {run['total_gb']:.2f} GB, compute buffer {run['by_kind_gb'].get('compute', 0):.2f} GB"
          + (f", peak GPU {run['gpu_peak_gb']:.2f} GB" if run.get("gpu_peak_gb") is not None else ""))
mem, base = job / "gpu-mem.log", job / "gpu-baseline.txt"
if mem.exists() and base.exists() and mem.read_text().split():
    peak = max(int(x) for x in mem.read_text().split()) - int(base.read_text())
    print(f"  peak GPU memory during the bench, every slot: {peak * 2**20 / 1e9:.2f} GB")
by = r.get("byTheme", {})
if by:
    print("  by theme: " + ", ".join(f"{k} {v['score']}%" for k, v in by.items()))
print(f"  first-call solves {r['firstTryRate']}%, answers with no owned file {r['emptyAnswerRate']}%, "
      f"passed acceptance but failed hidden {r['hiddenFailRate']}%")
import re
log = job / "work" / f"mugge-bench-{name}.log"
m = re.search(r"model calls (\d+), answers forced after the thinking (\d+), thinking calls asked again without thinking (\d+)", log.read_text(errors="replace")) if log.exists() else None
if m:
    calls, forced, nothink = map(int, m.groups())
    print(f"  model calls {calls}: {nothink} ran out of thinking and were asked again without thinking"
          + (f", {forced} got a forced answer" if forced else "") + (f" ({100 * (nothink + forced) / calls:.0f}%)" if calls else ""))
roll = job / "rollouts" / f"{name}.jsonl"
if roll.exists():
    print(f"  thinking per call (looping or just long): python3 {sys.argv[7]}/scripts/thinking_report.py {roll}")
ok_size = fp["total_gb"] <= budget
verdict = "MEETS" if lo >= target and ok_size else "MISSES" if hi < target or not ok_size else "TOO CLOSE TO CALL for"
print(f"\n{verdict} the target (Mugge score >= {target:g}% with one agent in <= {budget:g} GB at {fp['ctx']} context): "
      f"score {'clears' if lo >= target else 'below' if hi < target else 'straddles'} {target:g}% at 95%, "
      f"footprint {'within' if ok_size else 'over'} budget")
if json.loads((job / "config.json").read_text()).get("mugge_bench_sandbox") == "local" and projects > 1:
    print("Scored in the local sandbox: not a deciding score (ml and dotnet tickets cannot run, fix prompts show host paths).")
if projects <= 1:
    print("Only the public spike was scored: this is a smoke test, not the Mugge score. Set BENCH_TAR to the dev set.")
PY
