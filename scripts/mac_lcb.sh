#!/usr/bin/env bash
# LiveCodeBench with thinking off and on, for one compressed model, on an
# Apple Silicon Mac (16 GB or more).
#   bash scripts/mac_lcb.sh                  # r50w95s, the first 40 problems from 2025-01-01
#   N=0 bash scripts/mac_lcb.sh              # every problem in the window (overnight)
#   MODEL=r50mix bash scripts/mac_lcb.sh     # another GGUF from the mugge-library bucket
# Needs Homebrew. The first run installs llama.cpp, uv and the evroc CLI, logs in to
# evroc (opens your browser) and downloads the model (~9 GB) and LiveCodeBench.
# Re-running skips finished runs; delete $WORK/jobs/<run> to redo one.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$HOME/llmap-mac}
MODEL=${MODEL:-r50w95s}
N=${N:-40}
SLOTS=${SLOTS:-2}
THINK_TOKENS=${THINK_TOKENS:-24576}

say() { printf '\n==> %s\n' "$*"; }

[ "$(uname -s)" = Darwin ] || { echo "This script is for macOS."; exit 1; }
MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
[ "$MEM_GB" -ge 16 ] || { echo "Needs 16 GB of memory; this Mac has ${MEM_GB} GB."; exit 1; }
command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }

say "Tools"
command -v llama-server >/dev/null || brew install llama.cpp
command -v uv >/dev/null || brew install uv
mkdir -p "$WORK/models" "$WORK/codebench" "$WORK/jobs"
[ -x "$WORK/venv/bin/python" ] || uv venv -q --python 3.12 "$WORK/venv"
uv pip install -q --python "$WORK/venv/bin/python" httpx huggingface_hub
# The sandbox runs answers with `python3`, so put the 3.12 venv first.
export PATH="$WORK/venv/bin:$PATH"

GGUF="$WORK/models/qwen36-$MODEL.gguf"
if [ ! -s "$GGUF" ]; then
  say "Model qwen36-$MODEL.gguf from the mugge-library bucket"
  command -v evroc >/dev/null || curl -fsSL https://docs.evroc.com/install.sh | bash
  # Credentials last about an hour; a fresh login is only needed when this fails.
  evroc storage bucket get-s3-credentials >/dev/null 2>&1 || {
    evroc login; evroc config set-project "${EVROC_PROJECT:-mugge-11a5}"; evroc storage bucket get-s3-credentials >/dev/null; }
  evroc storage bucket copy --from "bucket://mugge-library/gguf/qwen36-$MODEL.gguf" --to "$GGUF.part" >/dev/null
  mv "$GGUF.part" "$GGUF"
fi
ls -lh "$GGUF"

if [ ! -s "$WORK/codebench/livecodebench.jsonl" ]; then
  say "LiveCodeBench problems"
  "$WORK/venv/bin/python" "$REPO/scripts/fetch_codebench.py" --out "$WORK/codebench" --only livecodebench
fi

if [ "$MEM_GB" -le 16 ]; then
  # macOS lets the GPU use about two thirds of memory (~10.7 GB of 16), and the model
  # plus its context needs ~11. This raises it to 12 GB until the next restart.
  say "Letting the GPU use 12 GB of memory until the next restart (asks for your password)"
  sudo sysctl iogpu.wired_limit_mb=12288 || sudo sysctl debug.iogpu.wired_limit=$((12288 << 20)) || true
fi

LIMIT=$([ "$N" = 0 ] && echo null || echo "$N")
run() {  # run name, thinking (true/false), answer token cap
  local d="$WORK/jobs/$1"
  mkdir -p "$d"
  cp "$REPO/examples/python-utils.code.taskspec.json" "$d/taskspec.json"
  cat > "$d/config.json" <<EOF
{
  "teacher": "Qwen/Qwen3.6-35B-A3B",
  "code_eval_dir": "$WORK/codebench",
  "code_eval_suites": ["livecodebench"],
  "lcb_since": "2025-01-01",
  "code_eval_limit": $LIMIT,
  "code_eval_thinking": $2,
  "code_eval_max_tokens": $3,
  "eval_candidates": {"$MODEL": "$GGUF"}
}
EOF
  if [ -f "$d/.done/eval" ]; then echo "$1 already done"; return; fi
  say "$1 (started $(date +%H:%M); log in $d/run.log)"
  # caffeinate keeps the Mac awake while it runs.
  (cd "$REPO" && LOBBOT_EVAL_SLOTS=$SLOTS caffeinate -i "$WORK/venv/bin/python" pipeline.py --job "$d" --only eval) \
    2>&1 | tee "$d/run.log" | grep --line-buffered -E 'answering|pass@1|error|Error' || true
  [ -f "$d/.done/eval" ] || { echo "$1 failed; last lines of $d/run.log:"; tail -20 "$d/run.log"; exit 1; }
}

run "lcb-$MODEL-think-off" false 4096
run "lcb-$MODEL-think-on" true "$THINK_TOKENS"

say "Result"
"$WORK/venv/bin/python" - "$WORK/jobs" "$MODEL" <<'PY'
import json, sys
from pathlib import Path
jobs, model = Path(sys.argv[1]), sys.argv[2]
for mode in ("off", "on"):
    d = jobs / f"lcb-{model}-think-{mode}"
    c = json.loads((d / "out" / "eval.json").read_text())["candidates"][0]
    s = c["code"]["suites"]["livecodebench"]
    rows = [json.loads(l) for l in (d / "work" / "code_eval" / f"{model}.jsonl").read_text().splitlines() if l.strip()]
    empty = sum(not r["answer"].strip() for r in rows)
    note = f", {empty} ran out of thinking budget" if mode == "on" and empty else ""
    print(f"{model} thinking {mode:3}: LiveCodeBench {s['pass@1']:.0%} on {s['n']} problems, "
          f"{c['tok_s_vm'] or '?'} tok/s{note}")
PY
