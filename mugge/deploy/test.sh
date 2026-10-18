#!/usr/bin/env bash
# Dry run of the deploy scripts with no GPU: deploy/mock_vllm.py stands in for vLLM, and the crew
# file is checked against it the way `mugge run` checks machines at start. CI runs this.
set -euo pipefail
cd "$(dirname "$0")/.."
export STATE="$(mktemp -d)" VLLM="$PWD/deploy/mock_vllm.py" GPU=b200 PORT="${PORT:-18000}" MOCK_LOAD_S=2
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
# a public host is refused without a key
if HOST=0.0.0.0 ALLOW_PUBLIC=1 deploy/qwen.sh cmd >/dev/null 2>&1; then echo "public host without a key was allowed"; exit 1; fi
deploy/qwen.sh stop
echo "deploy dry run: PASS"
