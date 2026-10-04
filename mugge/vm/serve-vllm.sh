#!/usr/bin/env bash
# Serves the base model and every specialist adapter from one vLLM server on the project VM.
# Continuous batching serves many agents at once; requests for different adapters batch together.
#   MODEL=/models/mugge-small ADAPTERS_DIR=/models/adapters vm/serve-vllm.sh
# Each folder in ADAPTERS_DIR becomes a served model name (coder-ts, coder-py, fixer-c, ...), which is
# what tickets' `model` fields map to (see MUGGE_MODELS in src/config.ts).
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
SERVED="${SERVED:-mugge-small}"
PORT="${PORT:-8000}"
ADAPTERS_DIR="${ADAPTERS_DIR:-/models/adapters}"
TP="${TP:-1}"
lora=()
if [ -d "$ADAPTERS_DIR" ] && [ -n "$(ls -A "$ADAPTERS_DIR" 2>/dev/null)" ]; then
  mods=()
  for d in "$ADAPTERS_DIR"/*/; do mods+=("$(basename "$d")=$d"); done
  lora=(--enable-lora --max-loras "${#mods[@]}" --max-lora-rank "${MAX_LORA_RANK:-64}" --lora-modules "${mods[@]}")
fi
exec "${VLLM:-$HOME/venvs/vllm/bin/vllm}" serve "$MODEL" \
  --served-model-name "$SERVED" \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-num-seqs "${MAX_SEQS:-64}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.92}" \
  --enable-prefix-caching \
  "${lora[@]}"
