#!/usr/bin/env bash
# BlackHole footprint measurement (same definition as training-plan.md and eval):
# one-slot llama-server, -c 16384, q8_0 KV, -fa on, after a ~12K-token prompt.
# Footprint = weights + KV cache + recurrent state + compute buffers, read from llama-server's own
# buffer report; peak process/GPU memory is logged beside it as a cross-check. Goal: <= 2.0e9 bytes.
#   ./footprint.sh BH0.1.gguf            # measures at the default -ub and at -ub 128
#   UB_LIST="512 256 128" ./footprint.sh model.gguf
set -euo pipefail
MODEL=${1:?gguf path}; PORT=${PORT:-8099}; UB_LIST=${UB_LIST:-"default 128"}
BIN=${LLAMA_SERVER:-llama-server}; OUT=${OUT:-footprint-$(basename "$MODEL" .gguf).jsonl}
for UB in $UB_LIST; do
  LOG=$(mktemp); UBARG=(); [ "$UB" != default ] && UBARG=(-ub "$UB")
  "$BIN" -m "$MODEL" -ngl 99 -np 1 -c 16384 -fa on -ctk q8_0 -ctv q8_0 "${UBARG[@]}" \
    --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 & PID=$!
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do kill -0 $PID 2>/dev/null || { cat "$LOG"; exit 1; }; sleep 1; done
  PEAK=0
  ( while kill -0 $PID 2>/dev/null; do
      if command -v nvidia-smi >/dev/null; then
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$PID -F', ' '$1==p{print $2*1048576}'
      else ps -o rss= -p $PID | awk '{print $1*1024}'; fi
      sleep 0.5; done ) >"$LOG.mem" & MON=$!
  PROMPT_N=$(python3 - "$PORT" <<'PY'
import json, sys, urllib.request
line = "def handler_{i}(req):\n    return {{'id': {i}, 'ok': req.get('k{i}') is not None}}\n"
text = "".join(line.format(i=i) for i in range(560))   # ~12K tokens on Qwen-family tokenizers
body = json.dumps({"prompt": text, "n_predict": 64, "cache_prompt": False}).encode()
r = json.load(urllib.request.urlopen(urllib.request.Request(
    f"http://127.0.0.1:{sys.argv[1]}/completion", body, {"Content-Type": "application/json"}), timeout=3600))
print(r.get("timings", {}).get("prompt_n", r.get("tokens_evaluated", 0)))
PY
)
  sleep 1; kill $PID; wait $PID 2>/dev/null || true; kill $MON 2>/dev/null || true
  python3 - "$LOG" "$LOG.mem" "$UB" "$PROMPT_N" "$MODEL" <<'PY' | tee -a "$OUT"
import json, re, sys
log, mem, ub, prompt_n, model = sys.argv[1:]
txt = open(log).read()
def mib(kind):   # every device and host buffer of one kind, e.g. "CUDA0 compute buffer size =   312.00 MiB"
    return sum(float(m.group(1)) for m in re.finditer(rf"\b{kind} buffer size\s*=\s*([\d.]+) MiB", txt))
parts = {k: mib(k) for k in ("model", "KV", "RS", "compute")}
total = sum(parts.values()) * 2**20
peak = max([float(x) for x in open(mem).read().split()] or [0])
print(json.dumps({"model": model, "ub": ub, "prompt_tokens": int(prompt_n or 0),
                  **{f"{k}_MiB": round(v, 1) for k, v in parts.items()},
                  "footprint_bytes": int(total), "footprint_GB": round(total / 1e9, 3),
                  "under_2GB": total <= 2e9, "peak_process_bytes": int(peak)}))
PY
  rm -f "$LOG" "$LOG.mem"
done
