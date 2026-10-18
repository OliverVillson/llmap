#!/usr/bin/env python3
"""
The crew's machine list (MUGGE_BACKENDS) for N workers on one Qwen server, printed as JSON.
crew.env runs it with CREW_WORKERS; qwen.sh uses the same sizes for the server.

  crew.py 64                      64 workers, 4 leads, 1 orchestrator
  crew.py 64 --url http://127.0.0.1:8000/v1
  crew.py 64 --sizes              "workers leads orchestrator total", for scripts
  crew.py 64 --need 65536         bytes of KV cache and state the crew needs at most, at that many
                                  KV bytes per token (65536 bf16, 32768 fp8)

Leads: one per 16 workers, at least 2 (a lead review is one short call per passing ticket).
The longest each kind of call can get, which is what a slot costs at most:
  worker  26,624 tokens: a pack of up to ~10K plus MUGGE_MAX_CALL_TOKENS (16,384)
  lead     8,192 tokens: the review prompt and diff plus a 2,048-token answer
  manager 24,576 tokens: a plan call with thinking, capped at MUGGE_MAX_CALL_TOKENS
"""
import argparse
import json
import math

CTX = {"worker": 26624, "lead": 8192, "manager": 24576}
# Qwen3.8-27B's Gated DeltaNet state per sequence (48 layers): the engine's qwen3.8-27b preset
STATE_BYTES = 153_944_064


def sizes(workers):
    leads = max(2, math.ceil(workers / 16))
    return workers, leads, 1


def machines(workers, url):
    w, l, o = sizes(workers)
    entry = lambda name, role, slots: {"name": name, "url": url, "server": "vllm", "roles": [role], "slots": slots,
                                       "model": "qwen3.8-27b", "preset": "qwen3.8-27b", "cache_type": "bf16", "ctx": CTX[role]}
    return [entry("b200-workers", "worker", w), entry("b200-leads", "lead", l), entry("b200-orchestrator", "manager", o)]


def need(workers, kv_bytes_per_token):
    w, l, o = sizes(workers)
    per = lambda role: CTX[role] * kv_bytes_per_token + STATE_BYTES
    return w * per("worker") + l * per("lead") + o * per("manager")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workers", type=int, nargs="?", default=10)
    p.add_argument("--url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--sizes", action="store_true")
    p.add_argument("--need", type=int, metavar="KV_BYTES_PER_TOKEN")
    a = p.parse_args()
    if a.workers < 1:
        p.error("workers must be at least 1")
    if a.sizes:
        w, l, o = sizes(a.workers)
        print(w, l, o, w + l + o)
    elif a.need:
        print(need(a.workers, a.need))
    else:
        ms = machines(a.workers, a.url)
        print("[\n" + ",\n".join("  " + json.dumps(m) for m in ms) + "\n]")


if __name__ == "__main__":
    main()
