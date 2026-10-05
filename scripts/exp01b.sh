#!/usr/bin/env bash
# Experiment 01 follow-ups on the B200, in one go (each step is skipped once done):
#   1. 2..8-bit mixed quant of r50mix's pruned+healed model at 7.5 and 9.5 GB, and at
#      7.5 and 9.5 GB with every non-expert tensor at 8-bit (r50w75, r50w95, r50w95s, r50w75s)
#   2. heal on Python, C, JavaScript and TypeScript together (multi)
#   3. an x86-64 assembly-only model (asm), and the full model on its held-out tasks (asm-ref)
#
#   source .env.vm && bash scripts/exp01b.sh 2>&1 | tee ~/exp01b.log
#   bash scripts/exp01b.sh summary
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
J=${LOBBOT_JOBS:-/mnt/nvme/jobs}
cd "$REPO"

# edit KEY=VALUE pairs into a job's config.json; taskspec target via target.max_size_gb=N
setcfg() {
  python - "$@" <<'PY'
import json, sys
d, pairs = sys.argv[1], sys.argv[2:]
c = json.load(open(f"{d}/config.json")); s = json.load(open(f"{d}/taskspec.json"))
for p in pairs:
    k, v = p.split("=", 1)
    v = json.loads(v) if v[:1] in "[{0123456789" or v in ("true", "false", "null") else v
    if k == "-":
        c.pop(v, None)
    elif k.startswith("target."):
        s["target"][k[7:]] = v
    else:
        c[k] = v
json.dump(c, open(f"{d}/config.json", "w"), indent=2); json.dump(s, open(f"{d}/taskspec.json", "w"), indent=2)
PY
}

# a job that reuses r50mix's data, pruned and healed model, and redoes quantize onwards
requant() {
  local name=$1; shift
  local src=$J/code10x-r50mix dst=$J/code10x-$name
  [ -f "$dst/.done/eval" ] && { echo "== $name: done"; return; }
  echo "== $name"
  mkdir -p "$dst/work" && cp -r "$src/data" "$src/.done" "$src/config.json" "$src/taskspec.json" "$dst/"
  cp "$src"/work/*.json "$dst/work/" && ln -sfn "$src/work/healed" "$dst/work/healed"
  rm -f "$dst"/.done/{quantize,eval,package}
  setcfg "$dst" "$@"
  python pipeline.py --job "$dst" --from quantize 2>&1 | tee "$J/$name.log" | grep -E 'pass@1|GB|rror' || true
}

summary() {
  python - "$J" <<'PY'
import json, sys, collections
from pathlib import Path
j = Path(sys.argv[1])
for name in ["ref", "r50mix", "r50q4", "r50q6", "r50w75", "r50w95", "r50w95s", "r50w75s", "multi", "asm", "asm-ref"]:
    p = j / f"code10x-{name}" / "out" / "eval.json"
    if not p.exists():
        print(f"{name:9} not run"); continue
    e = json.loads(p.read_text())
    c = next((x for x in e["candidates"] if x["name"] in ("lobbot-moe", name)), e["candidates"][0])
    code = c.get("code") or {}
    suites = " ".join(f"{s} {r['pass@1']:.0%}" for s, r in code.get("suites", {}).items() if "pass@1" in r)
    share = code.get("share_of_ref") or {}
    vs = f"  vs ref {share['mean']:.0%} (lowest {share['min']:.0%})" if share else ""
    bits = ""
    a = j / f"code10x-{name}" / "work" / "allocation.json"
    if a.exists() and "bit_widths" in json.loads(a.read_text()):
        n = collections.Counter(w["type"] for w in json.loads(a.read_text())["bit_widths"])
        bits = "  bits " + ", ".join(f"{k} {v}" for k, v in sorted(n.items()))
    print(f"{name:9} {c.get('size_gb', 0):5.1f} GB  {suites}{vs}{bits}")
PY
}

[ "${1:-}" = summary ] && { summary; exit 0; }
[ -f "$J/code10x-r50mix/.done/heal" ] || { echo "run code10x-r50mix first"; exit 1; }

# 1. quantization
requant r50w75  bit_floor=q2_k bit_ceiling=q8_0 target.max_size_gb=7.5
requant r50w95  bit_floor=q2_k bit_ceiling=q8_0 target.max_size_gb=9.5
requant r50w95s bit_floor=q2_k bit_ceiling=q8_0 static_type=q8_0 target.max_size_gb=9.5
requant r50w75s bit_floor=q2_k bit_ceiling=q8_0 static_type=q8_0 target.max_size_gb=7.5

# 2. multi-language heal (Python data comes from r50mix)
python scripts/code10x.py lang-jobs --taskspec examples/c-strings.code.taskspec.json \
  examples/javascript-utils.code.taskspec.json examples/typescript-utils.code.taskspec.json --n 800
for l in c javascript typescript; do
  [ -f "$J/code10x-data-$l/.done/data" ] && continue
  echo "== data $l"
  python pipeline.py --job "$J/code10x-data-$l" --only data 2>&1 | tee "$J/data-$l.log" | grep -E '"done"|passed|rror' || true
done
if [ ! -f "$J/code10x-multi/.done/eval" ]; then
  echo "== multi"
  [ -f "$J/code10x-multi/.done/data" ] || python scripts/code10x.py merge-data --into code10x-multi \
    --sources code10x-r50mix code10x-data-c code10x-data-javascript code10x-data-typescript
  python pipeline.py --job "$J/code10x-multi" 2>&1 | tee "$J/multi.log" | grep -E 'pass@1|GB|rror' || true
fi

# 3. assembly only, scored on its own held-out tasks
if [ ! -f "$J/code10x-asm/.done/eval" ]; then
  echo "== asm"
  if [ ! -d "$J/code10x-asm" ]; then
    python scripts/code10x.py lang-jobs --taskspec examples/asm-x86-64.code.taskspec.json --n 2000 --heldout 100
    mv "$J/code10x-data-asm" "$J/code10x-asm"
    setcfg "$J/code10x-asm" 'code_eval_suites=["heldout"]' -=code_eval_ref
  fi
  python pipeline.py --job "$J/code10x-asm" 2>&1 | tee "$J/asm.log" | grep -E 'passed|pass@1|GB|rror' || true
fi
if [ -f "$J/code10x-asm/.done/data" ] && [ ! -f "$J/code10x-asm-ref/.done/eval" ]; then
  echo "== asm-ref"
  mkdir -p "$J/code10x-asm-ref/.done"
  cp -r "$J/code10x-asm/data" "$J/code10x-asm/taskspec.json" "$J/code10x-asm-ref/"
  cp "$J/code10x-asm/.done/data" "$J/code10x-asm-ref/.done/"
  cp "$J/code10x-ref/config.json" "$J/code10x-asm-ref/config.json"
  setcfg "$J/code10x-asm-ref" 'code_eval_suites=["heldout"]'
  python pipeline.py --job "$J/code10x-asm-ref" --only eval 2>&1 | tee "$J/asm-ref.log" | grep -E 'pass@1|rror' || true
fi

echo; summary
