#!/usr/bin/env python3
"""The footprint, as Oliver defined it: one llama-server slot at 16K
(`-c 16384 -np 1 -fa on -ctk q8_0 -ctv q8_0`) after a ~12K-token prompt; weights + KV
cache + recurrent state + compute buffers, at most 2.0e9 bytes. Measured with the
default -ub and with -ub 128; the smaller counts.

    python3 scripts/bh/footprint.py --gguf model.gguf --llama-cpp <checkout> --text calib.txt --out fp.json
        [--prompt-tokens 12000] [--footprint-sh blackhole/serving/footprint.sh]

The number is the sum of the buffers llama.cpp reports at load (model, KV, RS, compute,
output; every device, host included), so the same GGUF measures the same on the 4060
and the Mac; nvidia-smi's peak during the prompt is kept beside it as a cross-check
(it adds the CUDA context, which the definition leaves out). With --footprint-sh,
BlackHole's own script is run too and its number wins. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

BUF = re.compile(r"(\S+)\s+(model|KV|RS|compute|output) buffer size\s*=\s*([\d.]+)\s*MiB")
PORT = 8011


def buffers(log: str) -> dict:
    """{"<device> <kind>": bytes} from a llama-server log, the last load's lines."""
    out = {}
    for dev, kind, mib in BUF.findall(log):
        out[f"{dev} {kind}"] = out.get(f"{dev} {kind}", 0) + int(float(mib) * 1024 * 1024)
    return out


def name(version: str, nbytes: int) -> str:
    """BH<version>-<GB>: the footprint in 1e9 bytes to one decimal, rounded up, so the
    name never claims less than was measured (1.21e9 -> 1.3)."""
    return f"BH{version}-{math.ceil(nbytes / 1e8 - 1e-9) / 10:.1f}"


def _get(url, body=None, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _gpu_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return int(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def measure(gguf: str, llama_cpp: str, text: str, prompt_tokens: int, ub: int | None, log_path: Path) -> dict:
    srv = Path(llama_cpp) / "build" / "bin" / "llama-server"
    cmd = [str(srv), "-m", gguf, "-ngl", "999", "--host", "127.0.0.1", "--port", str(PORT), "-c", "16384",
           "-np", "1", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"] + (["-ub", str(ub)] if ub else [])
    idle = _gpu_mib()
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{PORT}"
    peak, stop = [0], threading.Event()
    try:
        t = time.monotonic()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited {proc.returncode}: see {log_path}")
            try:
                if _get(f"{url}/health", timeout=5).get("status") == "ok":
                    break
            except OSError:
                pass
            if time.monotonic() - t > 600:
                raise RuntimeError(f"llama-server not up after 600 s: see {log_path}")
            time.sleep(1)
        ids = _get(f"{url}/tokenize", {"content": text})["tokens"]
        while len(ids) < prompt_tokens:
            ids = ids + ids
        ids = ids[:prompt_tokens]

        def watch():
            while not stop.wait(0.5):
                peak[0] = max(peak[0], _gpu_mib() or 0)

        threading.Thread(target=watch, daemon=True).start()
        r = _get(f"{url}/completion", {"prompt": ids, "n_predict": 16, "temperature": 0, "top_k": 20, "min_p": 0.0,
                                       "cache_prompt": False})
        stop.set()
        n_prompt = (r.get("timings") or {}).get("prompt_n") or len(ids)
    finally:
        stop.set()
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
                proc.wait(timeout=30)
                break
            except (ProcessLookupError, subprocess.TimeoutExpired):
                continue
    bufs = buffers(log_path.read_text(errors="replace"))
    if not bufs:
        raise RuntimeError(f"no buffer sizes in {log_path}: is this llama.cpp's log format?")
    return {"ub": ub or "default", "bytes": sum(bufs.values()), "buffers": bufs, "prompt_tokens": n_prompt,
            "nvidia_smi_peak_mib": peak[0] or None, "nvidia_smi_idle_mib": idle}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--llama-cpp", required=True)
    ap.add_argument("--text", required=True, help="text to make the prompt from (the calibration text)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=12000)
    ap.add_argument("--footprint-sh", default="")
    ap.add_argument("--version", default="0.1")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = Path(a.text).read_text()
    runs = [measure(a.gguf, a.llama_cpp, text, a.prompt_tokens, ub, out.with_name(f"{out.stem}.ub{ub or 'd'}.log"))
            for ub in (None, 128)]
    best = min(runs, key=lambda r: r["bytes"])
    rep = {"gguf": a.gguf, "file_bytes": os.path.getsize(a.gguf), "footprint": best["bytes"], "runs": runs,
           "source": "llama.cpp buffers"}
    if a.footprint_sh:
        r = subprocess.run(["bash", a.footprint_sh, a.gguf], capture_output=True, text=True)
        m = re.search(r"footprint[^\d]*(\d{6,})", r.stdout)
        rep["footprint_sh"] = {"exit": r.returncode, "stdout": r.stdout[-2000:]}
        if r.returncode == 0 and m:
            rep.update(footprint=int(m.group(1)), source=a.footprint_sh)
    rep["name"] = name(a.version, rep["footprint"])
    out.with_suffix(".part").write_text(json.dumps(rep, indent=2) + "\n")
    out.with_suffix(".part").replace(out)
    print(json.dumps({k: rep[k] for k in ("footprint", "file_bytes", "name", "source")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
