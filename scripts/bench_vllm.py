"""Output throughput of a vLLM-served model at several request counts at once.

    $LOBBOT_VLLM_PY scripts/bench_vllm.py --model <dir> --out throughput.json --concurrency 1 32 128

Serves the model, runs `vllm bench serve` on random prompts (2k tokens in, 1k out,
no early stop) once per concurrency, and writes {concurrency: {output_tok_s,
tpot_ms, ttft_ms}}. One request at once is one Mugge agent's speed; 32 and 128 are
a project's worth of agents sharing the GPU. Run it with the vLLM venv's python.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

PORT = 8471


def wait_healthy(proc: subprocess.Popen, timeout: float = 1200) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with {proc.returncode} while loading")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(5)
    raise RuntimeError("vLLM did not come up")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 32, 128])
    ap.add_argument("--input-len", type=int, default=2048)
    ap.add_argument("--output-len", type=int, default=1024)
    args = ap.parse_args()

    vllm = str(Path(sys.executable).parent / "vllm")
    out = Path(args.out)
    log = open(out.with_suffix(".server.log"), "w")
    server = subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", args.model,
                               "--host", "127.0.0.1", "--port", str(PORT), "--max-model-len", "8192",
                               "--max-num-seqs", str(max(256, max(args.concurrency))), "--gpu-memory-utilization", "0.9"],
                              stdout=log, stderr=subprocess.STDOUT)
    results = {}
    try:
        wait_healthy(server)
        with tempfile.TemporaryDirectory() as tmp:
            for c in args.concurrency:
                name = f"c{c}.json"
                subprocess.run([vllm, "bench", "serve", "--backend", "vllm", "--base-url", f"http://127.0.0.1:{PORT}",
                                "--model", args.model, "--dataset-name", "random",
                                "--random-input-len", str(args.input_len), "--random-output-len", str(args.output_len),
                                "--num-prompts", str(max(8, 4 * c)), "--max-concurrency", str(c), "--ignore-eos",
                                "--save-result", "--result-dir", tmp, "--result-filename", name], check=True)
                r = json.loads((Path(tmp) / name).read_text())
                results[str(c)] = {"output_tok_s": r.get("output_throughput"), "tpot_ms": r.get("mean_tpot_ms"),
                                   "ttft_ms": r.get("mean_ttft_ms")}
                print(f"{c} at once: {results[str(c)]}", flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
    out.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
