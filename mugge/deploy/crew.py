#!/usr/bin/env python3
"""
The crew's machine list (MUGGE_BACKENDS) for N workers on one Qwen server, printed as JSON.
crew.env runs it with CREW_WORKERS; qwen.sh uses the same sizes for the server.

  crew.py 64                      64 worker slots, 8 lead slots, 1 orchestrator slot
  crew.py 64 --url http://127.0.0.1:8000/v1
  crew.py 64 --sizes              "workers leads orchestrator slots agents", for scripts
  crew.py 64 --need 65536         bytes of KV cache and state the crew needs at most, at that many
                                  KV bytes per token (65536 bf16, 32768 fp8)

Leads: one slot per 8 worker slots, at least 2. Agents (MUGGE_CONCURRENCY, tickets at once):
1.25 per worker slot, so a slot freed by an agent running its checks or waiting for review goes
to another agent's call at once; the calls in flight never exceed the slots, so the extra agents
cost no GPU memory. (Both from the engine's crew-scale measurements.)
The longest each kind of call can get, which is what a slot costs at most:
  worker  26,624 tokens: a pack of up to ~10K plus MUGGE_MAX_CALL_TOKENS (16,384)
  lead    24,576 tokens: the review prompt and diff (up to ~8K) plus a thinking review capped at 16,384
  manager 24,576 tokens: a plan call with thinking, capped at MUGGE_MAX_CALL_TOKENS
"""
import argparse
import json
import math

CTX = {"worker": 26624, "lead": 24576, "manager": 24576}
# Qwen3.8-27B's Gated DeltaNet state per sequence (48 layers): the engine's qwen3.8-27b preset
STATE_BYTES = 153_944_064


def sizes(workers):
    leads = max(2, math.ceil(workers / 8))
    return workers, leads, 1


def agents(workers):
    return math.ceil(workers * 1.25)


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
        print(w, l, o, w + l + o, agents(a.workers))
    elif a.need:
        print(need(a.workers, a.need))
    else:
        ms = machines(a.workers, a.url)
        print("[\n" + ",\n".join("  " + json.dumps(m) for m in ms) + "\n]")


if __name__ == "__main__":
    main()
