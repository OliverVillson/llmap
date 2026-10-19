#!/usr/bin/env python3
"""A stand-in nvidia-smi for dry runs of the Linux GPU path (deploy/test.sh): an 8 GB laptop RTX 3070
with MOCK_GPU_USED_MIB in use (default 100), driver MOCK_DRIVER (default 580.65)."""
import os
import sys

FIELDS = {
    "name": "NVIDIA GeForce RTX 3070 Laptop GPU",
    "memory.total": "8192",
    "memory.used": os.environ.get("MOCK_GPU_USED_MIB", "100"),
    "driver_version": os.environ.get("MOCK_DRIVER", "580.65"),
    "utilization.gpu": "0 %",
}
args = sys.argv[1:]
if "-L" in args:
    print(f"GPU 0: {FIELDS['name']} (UUID: GPU-mock)")
    sys.exit(0)
q = next((a.split("=", 1)[1] for a in args if a.startswith("--query-gpu=")), None)
if q is None:
    print("NVIDIA-SMI mock")
    sys.exit(0)
units = not any("nounits" in a for a in args)
out = []
for f in q.split(","):
    v = FIELDS[f]
    if units and f.startswith("memory."):
        v += " MiB"
    out.append(v)
print(", ".join(out))
