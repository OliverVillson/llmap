#!/usr/bin/env python3
"""
The crew's machine list (MUGGE_BACKENDS) for N workers on one Qwen server, printed as JSON.
crew.env runs it with CREW_WORKERS; qwen.sh uses the same sizes for the server.

  crew.py 10                      10 workers, 2 leads, 1 orchestrator
  crew.py 24 --url http://127.0.0.1:8000/v1 --ctx 32768
  crew.py 24 --sizes              "workers leads orchestrator total", for scripts

Leads: one per 5 workers, at least 2 (a lead review is one short call per passing ticket).
"""
import argparse
import json
import math


def sizes(workers):
    leads = max(2, math.ceil(workers / 5))
    return workers, leads, 1


def machines(workers, url, ctx):
    w, l, o = sizes(workers)
    entry = lambda name, role, slots: {"name": name, "url": url, "server": "vllm", "roles": [role], "slots": slots,
                                       "model": "qwen3.8-27b", "preset": "qwen3.8-27b", "cache_type": "bf16", "ctx": ctx}
    return [entry("b200-workers", "worker", w), entry("b200-leads", "lead", l), entry("b200-orchestrator", "manager", o)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workers", type=int, nargs="?", default=10)
    p.add_argument("--url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--ctx", type=int, default=32768, help="tokens one call may reach: pack plus MUGGE_MAX_CALL_TOKENS")
    p.add_argument("--sizes", action="store_true")
    a = p.parse_args()
    if a.workers < 1:
        p.error("workers must be at least 1")
    if a.sizes:
        w, l, o = sizes(a.workers)
        print(w, l, o, w + l + o)
    else:
        ms = machines(a.workers, a.url, a.ctx)
        print("[\n" + ",\n".join("  " + json.dumps(m) for m in ms) + "\n]")


if __name__ == "__main__":
    main()
