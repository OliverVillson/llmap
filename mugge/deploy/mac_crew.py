#!/usr/bin/env python3
"""
The Centipede crew on one Apple-silicon Mac: which model, how many llama-server slots of how much
context, and the machine list (MUGGE_BACKENDS) for them. deploy/mac.sh starts the server from it
and deploy/mac.env loads the crew from it; deploy/crew.py is the B200 counterpart.

  mac_crew.py                        the plan for this Mac, with the memory arithmetic
  mac_crew.py --mem-gb 16 --wired-mb 0 --model 9b      the same for another Mac (wired 0 = the macOS default)
  mac_crew.py --json                 the plan as JSON (mac.sh reads it)
  mac_crew.py --backends             MUGGE_BACKENDS for the plan
  mac_crew.py --table                16, 24, 32 and 64 GB side by side

Memory. The GPU may use at most the wired limit (iogpu.wired_limit_mb; 0 means the macOS default,
about two thirds of memory below 64 GB and three quarters from 64 GB). In it go the weights, the
Metal compute buffers, and per slot its KV cache (context x KV bytes a token) plus the model's
fixed recurrent state (Qwen3.5 is hybrid: 8 of 32 layers keep a KV cache, 24 keep a fixed Gated
DeltaNet state). A margin stays free. Slots are never overcommitted: llama-server fails every
running call at once when a shared (unified) KV pool runs out, so each slot gets its own context
(-np N -c N*ctx, no --kv-unified) and the crew must fit with every call at its longest.

Crew: leads = max(2, ceil(workers / 8)), 1 orchestrator, agents (tickets at once) = 1.25 per
worker slot, as on the B200. Workers are capped (--max-workers, default 8): past that the Mac's GPU
is busy anyway and more slots only make every call slower.

On Linux with an NVIDIA GPU (deploy/mac.sh on centipede-server, deploy/SERVER.md) the limit is the
GPU's free memory (nvidia-smi), the compute buffers add the CUDA context, and an 8 GB card gets
q4_0 KV and 12,288 tokens a slot by default so it holds a crew. --role helper gives every slot to
workers: a second GPU box whose slots the head's crew uses (--add-helpers, deploy/mac.env).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import urllib.request

GB = 1e9
MIB = 2**20
# bytes per KV element in llama.cpp's cache types (q8_0: 34 bytes a block of 32)
KV_TYPE_BYTES = {"f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32, "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}

# KV layout from each model's config.json. Gated DeltaNet state per sequence (fp32): per linear
# layer, value heads x key dim x value dim, plus the conv state (kernel - 1) x (value dim x value
# heads + 2 x key heads x key dim).
def deltanet_state(layers, v_heads=32, k_heads=16, dim=128, kernel=4):
    return layers * 4 * (v_heads * dim * dim + (kernel - 1) * (v_heads * dim + 2 * k_heads * dim))

PRESETS = {
    "qwen3.5-9b": {"kv_layers": 8, "kv_heads": 4, "head_dim": 256, "state": deltanet_state(24), "compute_gb": 1.0},
    "qwen3.5-4b": {"kv_layers": 8, "kv_heads": 4, "head_dim": 256, "state": deltanet_state(24), "compute_gb": 0.7},
    # Qwen3.6-35B-A3B and its expert-pruned GGUFs (e.g. the REAP-50% qwen36-r50w95s): 10 of 40 layers
    "qwen3.6-35b-a3b": {"kv_layers": 10, "kv_heads": 2, "head_dim": 256, "state": deltanet_state(30), "compute_gb": 0.8},
}

# The models mac.sh can download (unsloth's GGUFs; sizes from the Hub, checked after download).
MODELS = {
    "9b": {"repo": "unsloth/Qwen3.5-9B-GGUF", "quant": "Q4_K_M", "preset": "qwen3.5-9b", "served": "qwen3.5-9b",
           "files": {"Q4_K_M": 5680522464, "Q5_K_M": 6577841376, "Q6_K": 7458301152, "Q8_0": 9527502048},
           "pattern": "Qwen3.5-9B-{quant}.gguf"},
    "4b": {"repo": "unsloth/Qwen3.5-4B-GGUF", "quant": "Q6_K", "preset": "qwen3.5-4b", "served": "qwen3.5-4b",
           "files": {"Q4_K_M": 2740937888, "Q5_K_M": 3143656608, "Q6_K": 3525956768, "Q8_0": 4482403488},
           "pattern": "Qwen3.5-4B-{quant}.gguf"},
}

CTX = 16384
MARGIN_GB = 0.5
# NVIDIA: the CUDA context comes on top of the compute buffers; the card is the GPU's alone
CUDA_CONTEXT_GB = 0.3
NVIDIA_MARGIN_GB = 0.25
# an NVIDIA card's defaults when KV and CTX are not given: q4_0 and 12,288 tokens a slot, so an
# 8 GB card holds 5-7 slots of the 9B (q8_0 and 16,384 would give 3)
NVIDIA_KV = "q4_0"
NVIDIA_CTX = 12288


def default_wired_bytes(mem_bytes):
    return mem_bytes * (3 / 4 if mem_bytes >= 64 * 2**30 else 2 / 3)


def kv_bytes_per_token(p, k_type, v_type):
    per = p["kv_layers"] * p["kv_heads"] * p["head_dim"]
    return per * (KV_TYPE_BYTES[k_type] + KV_TYPE_BYTES[v_type])


def sizes(workers, role="head"):
    if role == "helper":
        return workers, 0, 0
    leads = max(2, math.ceil(workers / 8))
    return workers, leads, 1


def model_info(model, quant=None, path=None, preset=None):
    """(file name, weight bytes, preset name, served name, repo or None) for a model key or a local .gguf."""
    if model in MODELS:
        m = MODELS[model]
        q = quant or m["quant"]
        if q not in m["files"]:
            raise SystemExit(f"mac_crew: QUANT {q} is not one of {', '.join(m['files'])} for {model}")
        return m["pattern"].format(quant=q), m["files"][q], m["preset"], m["served"], m["repo"]
    # a GGUF already on disk (e.g. MODEL=~/llmap-mac/models/qwen36-r50w95s.gguf PRESET=qwen3.6-35b-a3b)
    p = os.path.expanduser(path or model)
    if not p.endswith(".gguf"):
        raise SystemExit(f"mac_crew: MODEL must be one of {', '.join(MODELS)} or a path to a .gguf file")
    if not preset:
        raise SystemExit(f"mac_crew: a local GGUF needs PRESET (one of {', '.join(PRESETS)}) for its KV layout")
    size = os.path.getsize(p) if os.path.exists(p) else 0
    return os.path.basename(p), size, preset, os.path.basename(p)[:-5].lower(), None


def plan(mem_bytes, wired_mb, model="9b", quant=None, preset=None, k_type="q8_0", v_type="q8_0", ctx=CTX, max_workers=8, workers=None,
         gpu="metal", free_bytes=None, gpu_name="", role="head"):
    """gpu "metal": the GPU may use the wired limit of mem_bytes; gpu "nvidia": free_bytes of the card's mem_bytes."""
    if preset and preset not in PRESETS:
        raise SystemExit(f"mac_crew: PRESET must be one of {', '.join(PRESETS)}")
    file, weights, preset_name, served, repo = model_info(model, quant, preset=preset)
    preset_name = preset or preset_name
    p = PRESETS[preset_name]
    for t in (k_type, v_type):
        if t not in KV_TYPE_BYTES:
            raise SystemExit(f"mac_crew: KV type {t} is not one of {', '.join(KV_TYPE_BYTES)}")
    nvidia = gpu == "nvidia"
    wired = free_bytes if nvidia else (wired_mb * MIB if wired_mb else default_wired_bytes(mem_bytes))
    compute = (p["compute_gb"] + (CUDA_CONTEXT_GB if nvidia else 0)) * GB
    margin = (NVIDIA_MARGIN_GB if nvidia else MARGIN_GB) * GB
    kv_tok = kv_bytes_per_token(p, k_type, v_type)
    slot = ctx * kv_tok + p["state"]
    fixed = weights + compute + margin
    room = wired - fixed
    fits = lambda w: sum(sizes(w, role)) * slot <= room
    if workers is None:
        w = max_workers
        while w > 1 and not fits(w):
            w -= 1
    else:
        w = workers
    w, l, o = sizes(w, role)
    slots = w + l + o
    need = fixed + slots * slot
    return {
        "model": model, "file": file, "repo": repo, "preset": preset_name, "served": served,
        "gpu": gpu, "gpu_name": gpu_name, "role": role,
        "weights_bytes": weights, "mem_bytes": mem_bytes, "wired_bytes": round(wired), "wired_set": bool(wired_mb),
        "compute_bytes": round(compute), "margin_bytes": round(margin),
        "k_type": k_type, "v_type": v_type, "ctx": ctx, "kv_bytes_per_token": kv_tok, "state_bytes": p["state"],
        "slot_bytes": round(slot), "workers": w, "leads": l, "orchestrator": o, "slots": slots,
        "agents": math.ceil(w * 1.25), "need_bytes": round(need), "fits": need <= wired,
        "max_slots_by_memory": max(0, int(room // slot)),
        "left_for_macos_bytes": round(mem_bytes - need),
    }


def explain(pl):
    g = lambda b: f"{b / GB:.2f} GB"
    nvidia = pl.get("gpu") == "nvidia"
    wired = f"{g(pl['wired_bytes'])} ({'iogpu.wired_limit_mb' if pl['wired_set'] else 'the macOS default, about 2/3 of memory'})"
    where = (f"GPU: {pl['gpu_name'] or 'NVIDIA'}, {pl['mem_bytes'] / 2**30:.1f} GiB, {g(pl['wired_bytes'])} free" if nvidia
             else f"Mac: {pl['mem_bytes'] / 2**30:.0f} GiB of memory; the GPU may use {wired}")
    crew = (f"crew: {pl['workers']} worker slots for the head's crew (a helper: no leads, no orchestrator)" if pl.get("role") == "helper"
            else f"crew: {pl['workers']} workers + {pl['leads']} leads + {pl['orchestrator']} orchestrator = {pl['slots']} slots, "
                 f"{pl['agents']} agents (tickets at once)")
    lines = [
        where,
        f"model: {pl['file']} ({g(pl['weights_bytes'])}), preset {pl['preset']}, served as {pl['served']}",
        f"KV cache: K {pl['k_type']}, V {pl['v_type']}: {pl['kv_bytes_per_token'] / 1024:.1f} KiB a token; "
        f"recurrent state {pl['state_bytes'] / 2**20:.1f} MiB a sequence",
        f"one slot: {pl['ctx']} tokens x {pl['kv_bytes_per_token'] / 1024:.1f} KiB + state = {pl['slot_bytes'] / 2**20:.0f} MiB",
        f"fixed: weights {g(pl['weights_bytes'])} + {'CUDA context and compute' if nvidia else 'Metal compute'} {g(pl['compute_bytes'])} + margin {g(pl['margin_bytes'])}",
        crew,
        f"GPU total: {g(pl['need_bytes'] - pl['margin_bytes'])} + margin = {g(pl['need_bytes'])} of {g(pl['wired_bytes'])}"
        f" -> {'fits' if pl['fits'] else 'DOES NOT FIT'}; memory alone would allow {pl['max_slots_by_memory']} slots",
    ]
    if not nvidia:
        lines.append(f"left for macOS, the engine, the checks and the browser: about {g(pl['left_for_macos_bytes'])}")
    return "\n".join(lines)


def backends(pl, url):
    entry = lambda name, role, slots: {"name": name, "url": url, "server": "llama-server", "roles": [role], "slots": slots,
                                       "model": pl["served"], "preset": pl["preset"], "cache_type": pl["k_type"], "ctx": pl["ctx"]}
    at = "server" if pl.get("gpu") == "nvidia" else "mac"
    ms = [entry(f"{at}-workers", "worker", pl["workers"]), entry(f"{at}-leads", "lead", pl["leads"]), entry(f"{at}-orchestrator", "manager", pl["orchestrator"])]
    return dump(ms)


def dump(ms):
    return "[\n" + ",\n".join("  " + json.dumps(m) for m in ms) + "\n]"


def add_helpers(machines, urls, timeout=5):
    """The head's machine list plus one worker entry per helper llama-server (its slots, alias and per-slot
    context read from its /props), and the total worker slots. A helper that does not answer is left out
    with a warning, so the crew runs on what is up."""
    head = next((m for m in machines if "worker" in m["roles"]), machines[0])
    out = list(machines)
    for i, u in enumerate(urls, 1):
        base = u.rstrip("/").removesuffix("/v1")
        try:
            with urllib.request.urlopen(f"{base}/props", timeout=timeout) as r:
                props = json.load(r)
        except (OSError, ValueError) as e:
            print(f"mac_crew: helper {base} does not answer ({e}); the crew runs without it", file=sys.stderr)
            continue
        slots = int(props.get("total_slots") or 0)
        ctx = int((props.get("default_generation_settings") or {}).get("n_ctx") or head["ctx"])
        if slots < 1:
            print(f"mac_crew: helper {base} reports no slots; left out", file=sys.stderr)
            continue
        out.append({"name": f"helper-{i}", "url": f"{base}/v1", "server": "llama-server", "roles": ["worker"], "slots": slots,
                    "model": props.get("model_alias") or head["model"], "preset": head.get("preset"), "cache_type": head.get("cache_type"), "ctx": ctx})
    return out, sum(m["slots"] for m in out if "worker" in m["roles"])


def nvidia_gpu():
    """(name, total bytes, free bytes) of the first NVIDIA GPU from nvidia-smi, or None (a Mac, or no driver)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        name, total, used = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return name, int(float(total)) * MIB, (int(float(total)) - int(float(used))) * MIB
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def this_mac():
    """(memory bytes, wired limit MiB or 0) from sysctl; (16 GiB, 0) where there is no sysctl (tests on Linux)."""
    def sysctl(name):
        try:
            return int(subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5).stdout.strip() or 0)
        except (OSError, ValueError, subprocess.SubprocessError):
            return 0
    return sysctl("hw.memsize") or 16 * 2**30, sysctl("iogpu.wired_limit_mb")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    env = os.environ
    p.add_argument("--mem-gb", type=float, help="memory in GiB (default: this Mac's)")
    p.add_argument("--wired-mb", type=int, help="the GPU's wired limit in MiB, 0 = macOS default (default: this Mac's)")
    p.add_argument("--model", default=env.get("MODEL") or "9b", help="9b, 4b, or a path to a .gguf (with --preset)")
    p.add_argument("--quant", default=env.get("QUANT") or None)
    p.add_argument("--preset", default=env.get("PRESET") or None)
    p.add_argument("--k-type", default=env.get("KV_K") or env.get("KV") or None, help="q8_0 (an NVIDIA card: q4_0)")
    p.add_argument("--v-type", default=env.get("KV_V") or env.get("KV") or None, help="q8_0 (an NVIDIA card: q4_0)")
    p.add_argument("--ctx", type=int, default=int(env["CTX"]) if env.get("CTX") else None, help=f"tokens of context per slot ({CTX}; an NVIDIA card: {NVIDIA_CTX})")
    p.add_argument("--max-workers", type=int, default=int(env["MAX_WORKERS"]) if env.get("MAX_WORKERS") else None, help="8 (an NVIDIA card: 16)")
    p.add_argument("--gpu", choices=["auto", "metal", "nvidia"], default=env.get("GPU_KIND") or "auto",
                   help="auto: NVIDIA when nvidia-smi answers on a machine that is not a Mac")
    p.add_argument("--vram-mb", type=int, help="NVIDIA: the card's memory in MiB (default: nvidia-smi)")
    p.add_argument("--vram-free-mb", type=int, help="NVIDIA: its free memory in MiB (default: nvidia-smi)")
    p.add_argument("--role", choices=["head", "helper"], default=env.get("ROLE") or "head",
                   help="helper: every slot a worker slot, for another box's crew")
    p.add_argument("--add-helpers", metavar="URLS", help="read a machine list on stdin and add these helper servers (space-separated base URLs)")
    p.add_argument("--workers", type=int, default=int(env["CREW_WORKERS"]) if env.get("CREW_WORKERS") else None,
                   help="this many workers whatever the memory says (CREW_WORKERS)")
    p.add_argument("--url", default="http://127.0.0.1:8080/v1")
    p.add_argument("--json", action="store_true")
    p.add_argument("--backends", action="store_true")
    p.add_argument("--sizes", action="store_true", help='"workers leads orchestrator slots agents", like crew.py --sizes')
    p.add_argument("--table", action="store_true")
    a = p.parse_args()
    if a.add_helpers is not None:
        machines, workers = add_helpers(json.load(sys.stdin), a.add_helpers.split())
        print(json.dumps({"machines": machines, "workers": workers, "agents": math.ceil(workers * 1.25)}))
        return
    gpu, name, free = a.gpu, "", None
    found = None
    if gpu in ("auto", "nvidia") and (a.vram_mb is None or a.vram_free_mb is None) and sys.platform != "darwin":
        found = nvidia_gpu()
    if gpu == "auto":
        gpu = "nvidia" if (found or a.vram_mb) else "metal"
    if gpu == "nvidia":
        if not found and (a.vram_mb is None or a.vram_free_mb is None):
            p.error("no NVIDIA GPU answers (nvidia-smi); give --vram-mb and --vram-free-mb to plan for one")
        name = found[0] if found else ""
        mem = a.vram_mb * MIB if a.vram_mb is not None else found[1]
        free = a.vram_free_mb * MIB if a.vram_free_mb is not None else found[2]
        wired = 0
    else:
        mem, wired = this_mac()
        if a.mem_gb:
            mem = int(a.mem_gb * 2**30)
        if a.wired_mb is not None:
            wired = a.wired_mb
    nvidia = gpu == "nvidia"
    a.k_type = a.k_type or (NVIDIA_KV if nvidia else "q8_0")
    a.v_type = a.v_type or (NVIDIA_KV if nvidia else "q8_0")
    a.ctx = a.ctx or (NVIDIA_CTX if nvidia else CTX)
    a.max_workers = a.max_workers or (16 if nvidia else 8)
    if a.ctx < 2048 or a.ctx % 256:
        p.error("--ctx must be a multiple of 256, at least 2048")
    if a.workers is not None and a.workers < 1:
        p.error("CREW_WORKERS must be at least 1")
    if a.table:
        print(f"{'Mac':>6}  {'GPU limit':>9}  {'model':<24} {'KV':<11} {'ctx':>6}  crew (w+l+o)  agents  GPU total  left for macOS")
        for gb, model, quant, wmb in ((16, "9b", None, 0), (16, "9b", None, 12288), (16, "4b", None, 0), (24, "9b", "Q6_K", 0), (32, "9b", "Q8_0", 0), (64, "9b", "Q8_0", 0)):
            pl = plan(gb * 2**30, wmb, model, quant, None, a.k_type, a.v_type, a.ctx, 16 if gb >= 32 else a.max_workers)
            lim = f"{pl['wired_bytes'] / GB:.1f} GB" + ("*" if wmb else "")
            print(f"{gb:>4} GB  {lim:>9}  {pl['file']:<24} {pl['k_type'] + '/' + pl['v_type']:<11} {pl['ctx']:>6}  "
                  f"{pl['workers']:>3}+{pl['leads']}+{pl['orchestrator']}        {pl['agents']:>4}  {pl['need_bytes'] / GB:>6.1f} GB  {pl['left_for_macos_bytes'] / GB:>5.1f} GB")
        print("* with sudo sysctl iogpu.wired_limit_mb=12288 (deploy/mac.sh wired)")
        return
    pl = plan(mem, wired, a.model, a.quant, a.preset, a.k_type, a.v_type, a.ctx, a.max_workers, a.workers, gpu, free, name, a.role)
    if a.json:
        print(json.dumps(pl))
    elif a.backends:
        print(backends(pl, a.url))
    elif a.sizes:
        print(pl["workers"], pl["leads"], pl["orchestrator"], pl["slots"], pl["agents"])
    else:
        print(explain(pl))
        if not pl["fits"]:
            print("does not fit: use KV=q4_0 (or KV_V=q4_0), a smaller CTX, fewer CREW_WORKERS, MODEL=4b" + ("" if nvidia else ", or raise the limit (deploy/mac.sh wired)"), file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
