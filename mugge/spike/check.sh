#!/usr/bin/env bash
# Verifies the spike test project against plan.json and the canned solutions:
#   (a) every ticket's acceptance FAILS on the bare scaffold
#   (a') ...and still fails with only its dependencies' solutions applied (its test needs its own file)
#   (b) with all solutions applied, every acceptance command PASSES
#   (c) with only its transitive dependencies' solutions plus its own, each ticket's acceptance PASSES
set -uo pipefail
SPIKE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$SPIKE/project"
PLAN="$SPIKE/plan.json"
SOLUTIONS="$SPIKE/solutions"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Emits one line per ticket: id<TAB>acceptance commands (\x1f separated)<TAB>transitive deps (space separated)
python3 - "$PLAN" "$SOLUTIONS" > "$WORK/tickets.tsv" <<'PY' || exit 1
import json, os, sys
plan = json.load(open(sys.argv[1]))
by = {t['id']: t for t in plan['tickets']}
def anc(i, seen=None):
    seen = set() if seen is None else seen
    for d in by[i].get('depends_on', []):
        if d not in seen:
            seen.add(d); anc(d, seen)
    return seen
for t in plan['tickets']:
    sol = json.load(open(os.path.join(sys.argv[2], t['id'] + '.json')))
    extra = set(sol['files']) - set(t['owns'])
    missing = set(t['owns']) - set(sol['files'])
    if extra or missing:
        sys.exit(f"solution {t['id']}: writes {sorted(extra)} outside owns / misses {sorted(missing)}")
    print(t['id'], '\x1f'.join(t['acceptance']), ' '.join(sorted(anc(t['id']))), sep='\t')
PY

fresh() { # fresh <dir> <ticket ids...>: copy the scaffold, apply those tickets' solutions
  local dir="$1"; shift
  rm -rf "$dir"; mkdir -p "$dir"; cp -r "$PROJECT/." "$dir/"
  python3 - "$dir" "$SOLUTIONS" "$@" <<'PY'
import json, os, sys
root, sols = sys.argv[1], sys.argv[2]
for tid in sys.argv[3:]:
    for path, content in json.load(open(os.path.join(sols, tid + '.json')))['files'].items():
        full = os.path.join(root, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, 'w').write(content)
PY
}

run_all() { # run_all <dir> <cmds>: 0 iff every acceptance command exits 0
  local dir="$1" cmds="$2" cmd
  IFS=$'\x1f' read -r -a list <<< "$cmds"
  for cmd in "${list[@]}"; do
    (cd "$dir" && timeout 60 bash -c "$cmd") > "$WORK/last.log" 2>&1 || return 1
  done
  return 0
}

fails=0
bad() { echo "  FAIL: $*"; tail -n 15 "$WORK/last.log" | sed 's/^/    | /'; fails=$((fails + 1)); }

echo "(a) acceptance fails on the bare scaffold, and with only dependencies applied"
while IFS=$'\t' read -r id cmds deps; do
  fresh "$WORK/t" ; if run_all "$WORK/t" "$cmds"; then bad "$id passes on bare scaffold"; else echo "  ok   $id fails on scaffold"; fi
  if [ -n "$deps" ]; then
    fresh "$WORK/t" $deps
    if run_all "$WORK/t" "$cmds"; then bad "$id passes with only deps ($deps)"; else echo "  ok   $id fails with deps only"; fi
  fi
done < "$WORK/tickets.tsv"

echo "(b) all solutions applied: every acceptance command passes"
fresh "$WORK/all" $(cut -f1 "$WORK/tickets.tsv")
while IFS=$'\t' read -r id cmds deps; do
  if run_all "$WORK/all" "$cmds"; then echo "  ok   $id"; else bad "$id fails with all solutions"; fi
done < "$WORK/tickets.tsv"

echo "(c) depends_on is sufficient: deps + own solution passes"
while IFS=$'\t' read -r id cmds deps; do
  fresh "$WORK/t" $deps "$id"
  if run_all "$WORK/t" "$cmds"; then echo "  ok   $id (deps: ${deps:-none})"; else bad "$id fails with deps [${deps}] + own"; fi
done < "$WORK/tickets.tsv"

echo
if [ "$fails" -eq 0 ]; then echo "check.sh: all checks passed"; else echo "check.sh: $fails check(s) failed"; exit 1; fi
