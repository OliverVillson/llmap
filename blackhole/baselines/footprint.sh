#!/usr/bin/env bash
# BlackHole footprint measurement (same definition as training-plan.md and eval):
# one-slot llama-server, -c 16384, q8_0 KV, -fa on, after a 12,000-token prompt.
# Footprint = GGUF file size + KV + recurrent state (RS) + compute + output buffers, at the better -ub.
# The file is counted once: on Metal the CPU_Mapped and MTL0_Mapped model buffers are views of the
# same mmapped file. The raw sum of every reported buffer and the peak process/GPU memory are logged
# beside it. Goal: footprint <= 2.0e9 bytes.
#   ./footprint.sh BH0.1.gguf            # measures at the default -ub and at -ub 128
#   UB_LIST="512 256 128" ./footprint.sh model.gguf
# Works with macOS bash 3.2. Needs a llama-server that logs buffer sizes (-lv 4 is passed).
set -euo pipefail
MODEL=${1:?gguf path}; PORT=${PORT:-8099}; UB_LIST=${UB_LIST:-"default 128"}; PROMPT_TOKENS=${PROMPT_TOKENS:-12000}
BIN=${LLAMA_SERVER:-llama-server}; OUT=${OUT:-footprint-$(basename "$MODEL" .gguf).jsonl}
FILE_BYTES=$(python3 -c 'import os,sys; print(os.path.getsize(sys.argv[1]))' "$MODEL")
RUNS=$(mktemp)
for UB in $UB_LIST; do
  LOG=$(mktemp); UBARG=(); [ "$UB" != default ] && UBARG=(-ub "$UB")
  LLAMA_ARG_LOG_VERBOSITY=4 "$BIN" -m "$MODEL" -ngl 99 -np 1 -c 16384 -fa on -ctk q8_0 -ctv q8_0 -lv 4 \
    ${UBARG[@]+"${UBARG[@]}"} --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 & PID=$!
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do kill -0 $PID 2>/dev/null || { tail -50 "$LOG"; exit 1; }; sleep 1; done
  ( while kill -0 $PID 2>/dev/null; do
      if command -v nvidia-smi >/dev/null; then
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$PID -F', ' '$1==p{print $2*1048576}'
      fi
      ps -o rss= -p $PID | awk '{print "rss", $1*1024}'
      sleep 0.5; done ) >"$LOG.mem" 2>/dev/null & MON=$!
  PROMPT_N=$(python3 - "$PORT" "$PROMPT_TOKENS" <<'PY'
import json, sys, urllib.request
port, want = sys.argv[1], int(sys.argv[2])
def post(path, obj):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(obj).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))
line = "def handler_{i}(req):\n    return {{'id': {i}, 'ok': req.get('k{i}') is not None}}\n"
text = "".join(line.format(i=i) for i in range(1200))
toks = post("/tokenize", {"content": text})["tokens"]
assert len(toks) >= want, f"filler too short: {len(toks)} tokens"
r = post("/completion", {"prompt": toks[:want], "n_predict": 64, "cache_prompt": False})
print(r.get("timings", {}).get("prompt_n", r.get("tokens_evaluated", 0)))
PY
)
  sleep 1; kill $PID; wait $PID 2>/dev/null || true; kill $MON 2>/dev/null || true
  python3 - "$LOG" "$LOG.mem" "$UB" "$PROMPT_N" "$MODEL" "$FILE_BYTES" <<'PY' | tee -a "$OUT" >>"$RUNS"
import json, re, sys
log, mem, ub, prompt_n, model, file_bytes = sys.argv[1:]; file_bytes = int(file_bytes)
txt = open(log).read()
def mib(kind):   # every device and host buffer of one kind, e.g. "CUDA0 compute buffer size =   312.00 MiB"
    return sum(float(m.group(1)) for m in re.finditer(rf"\b{kind} buffer size\s*=\s*([\d.]+) MiB", txt))
parts = {k: mib(k) for k in ("model", "KV", "RS", "compute", "output")}
if not any(parts.values()):
    sys.exit("no buffer sizes in the log: is this llama-server too old to log them at -lv 4?")
working = (parts["KV"] + parts["RS"] + parts["compute"] + parts["output"]) * 2**20
footprint = file_bytes + working
peak_gpu = max([float(x) for x in re.findall(r"^(\d+(?:\.\d+)?)$", open(mem).read(), re.M)] or [0])
peak_rss = max([float(x) for x in re.findall(r"^rss (\d+)$", open(mem).read(), re.M)] or [0])
print(json.dumps({"model": model, "ub": ub, "prompt_tokens": int(prompt_n or 0), "file_bytes": file_bytes,
                  **{f"{k}_MiB": round(v, 1) for k, v in parts.items()},
                  "footprint_bytes": int(footprint), "footprint_GB": round(footprint / 1e9, 3),
                  "under_2GB": footprint <= 2e9,
                  "raw_buffer_sum_bytes": int(sum(parts.values()) * 2**20),
                  "peak_gpu_bytes": int(peak_gpu), "peak_rss_bytes": int(peak_rss)}))
PY
  rm -f "$LOG" "$LOG.mem"
done
python3 - "$RUNS" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
best = min(rows, key=lambda r: r["footprint_bytes"])
print(f"FOOTPRINT {best['footprint_GB']} GB at -ub {best['ub']} ({'OK' if best['under_2GB'] else 'OVER 2 GB'}): {best['model']}")
PY
rm -f "$RUNS"
