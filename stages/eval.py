"""Stage 5: evaluate teacher vs. candidates on the held-out set.

Each candidate is served on the VM GPU and answers the held-out inputs with the
same system prompt the training data used. A GGUF is served with llama-server
(as many fixed slots of the eval's context as the free GPU memory holds); an HF
checkpoint dir (one with a config.json: BF16, or llm-compressor's
compressed-tensors NVFP4/W4A16) with vLLM's OpenAI server, run as a subprocess
from its own venv, with --max-num-seqs answers at once so the GPU stays busy
through long thinking answers. Every answer is scored two ways:
  agreement  match with the teacher's reference answer (field-level for JSON
             outputs, token F1 otherwise). Needs no API key.
  judge      an LLM judge (Gemini by default, cfg.judge_model) scores each
             answer 0-10 against the TaskSpec criteria (teacher references
             included). Used as `score` when available.
  pass@k     when Config.code_eval_suites is set, every candidate also answers
             the code benchmarks (stages/codebench.py) and each answer is
             compiled and run against its tests (stages/sandbox.py). The mean
             pass@1 over suites then becomes `score` (score_method "pass@1"),
             with judge and agreement still reported next to it.
Also records the decode speed measured on the VM; for code, the completion
tokens per answer, the shares of answers that hit the cap, needed a forced
answer, were cut after their thinking or ran out of time on a test case, and
the output tokens/s of the whole eval; and the GPU's mean busy % and peak memory
while each candidate answered (nvidia-smi). Writes out/eval.json, the contract
the scoreboard screen reads, and work/code_eval/<candidate>.jsonl with every
code answer and its test status.

A reading is written only when every answer came back. Requests stream, so no
limit applies to a whole answer: one that goes IDLE_TIMEOUT s without a byte is
asked again, and counts as a failure only when no other answer got a token
meanwhile (otherwise it was waiting in the server's queue). A server that died
(its process exited, or it refuses or drops connections and /health stops
answering) raises ServerDead instead of turning the remaining answers into empty
failures, and the candidate's server is started again once. Answers that still
errored are asked again, and the stage fails rather than score them. Each code
answer is appended to work/code_eval/<candidate>.partial.jsonl as it arrives, so
a re-run after a stop asks only for the (problem, sample) pairs missing there,
and starts no server when none are. Every answer is in, and the server stopped,
before the sandbox runs, so the tests' wall-clock limits do not share the CPU
with the engine. The server runs as a process group of its own, which stop() and
a SIGTERM to the stage end as a whole; a server a killed eval left on the port is
refused rather than spoken to.

Config.eval_candidates replaces work/allocation.json with named GGUFs or
checkpoint dirs, so an eval-only job (pipeline.py --only eval) can score an
uncompressed reference; without data/heldout.jsonl such a job is scored on code
alone. Config.size_gb_override sets the size eval.json reports, e.g. the shipped
GGUF's for a candidate scored as its bf16 dequantization. work/provenance.json
(also under "provenance" in eval.json) records the GPU, CPU count, package
versions, git HEAD, llama.cpp commit and code suite files each reading ran on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import statistics
import struct
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.progress import emit
from stages import codebench, data, harness, sandbox
from stages._util import DRY_RUN, Job, read_jsonl, write_jsonl
from stages.taskdata import think_tags

STAGE = "eval"
REPO = Path(__file__).resolve().parent.parent
PORT = int(os.environ.get("LOBBOT_EVAL_PORT", "8091"))
# llama-server's parallel requests when they cannot be sized from free GPU memory (no
# nvidia-smi, e.g. a Mac: 2 on a 16 GB one); setting it always wins.
SLOTS = int(os.environ.get("LOBBOT_EVAL_SLOTS", 8))
# vLLM batches far more: its --max-num-seqs, and the requests kept in flight
# (Config.vllm_seqs wins).
VLLM_SEQS = int(os.environ.get("LOBBOT_VLLM_SEQS", 256))
VLLM_GPU_UTIL = float(os.environ.get("LOBBOT_VLLM_GPU_UTIL", 0.9))
# Must cover the longest teacher answer the data stage keeps
# (LOBBOT_DATA_ANSWER_MAX_TOKENS), or long correct answers are judged as truncated.
MAX_TOKENS = int(os.environ.get("LOBBOT_EVAL_MAX_TOKENS", max(2048, data.ANSWER_MAX_TOKENS)))
# Room per slot for the system prompt, an input of up to data.MAX_INPUT_CHARS
# (~2k tokens) and the answer.
CTX_PER_SLOT = int(os.environ.get("LOBBOT_EVAL_CTX", MAX_TOKENS + 4096))

# With thinking, the answer cap for the step that makes the model answer when its
# thinking ended without one (see generate).
ANSWER_TOKENS = 4096
# Qwen's wording for cutting thinking short (the Qwen3 model card's thinking budget).
EARLY_STOP = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now."

# Requests stream, so no limit applies to a whole answer: one that goes this many seconds
# without a byte is given up, however long it has run. If another answer got a token
# meanwhile, the server was busy and this one waited in its queue (vLLM sends nothing to
# a request it has queued or preempted): it is asked again without counting as a
# failure, up to QUEUED_RETRIES times. If nothing came for anyone, the server stalled.
IDLE_TIMEOUT = 600.0
QUEUED_RETRIES = 12
RETRIES = 2  # after the first try, on stalls, cut streams and 5xx, while the server lives
BACKOFF = 5.0  # seconds before the first retry, doubling
HEALTH_WAIT = 10.0  # seconds between the /health tries that decide a server is dead
HEALTH_TIMEOUT = {"vllm": 1200, "llama-server": 600}  # seconds to come up; a vLLM load takes minutes
STOP_WAIT = 30.0  # seconds a stopped server's processes get to exit before SIGKILL
# On SIGTERM: pipeline.py SIGKILLs a stage 30 s after its SIGTERM, and a server in a
# session of its own would outlive that, so it gets less time to exit.
SIGTERM_WAIT = 15.0
REASK = 2  # rounds of asking errored answers again before the stage fails
RESTARTS = 1  # times a candidate's server that died mid-eval is started again
# llama-server: GPU memory left out of the KV slots, for compute buffers and the CUDA context.
SLOT_RESERVE_GB = float(os.environ.get("LOBBOT_EVAL_SLOT_RESERVE_GB", 2.0))
MAX_SLOTS = 128
# Bytes per K or V value in llama.cpp's cache types (a block's bytes over its 32 values).
KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
            "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}

# The vLLM server serve() started last ({"name", "dir"}); empty while llama-server
# serves. generate() and force_answer() speak to whichever runs.
VLLM: dict[str, str] = {}
THINK_TOP_K = 20  # top_k for thinking answers; serve() takes the job's thinking_top_k
MIN_P = 0.0  # serve() takes the job's eval_min_p
# The server serve() started ({"proc", "path", "name"}): alive() watches its process, and
# the SIGTERM handler stops its process group.
_SERVER: dict = {}
# Answers in flight on the running server, by backend: llama-server's slots and vLLM's
# --max-num-seqs, which serve() sets. generate() calls running at once (code_eval's sample
# passes) share one gate, so the rest wait here and not as preempted requests in the server.
IN_FLIGHT: dict[str, int] = {}
_gates: dict[tuple[str, int], threading.BoundedSemaphore] = {}
_gate_lock = threading.Lock()
_http: dict = {}
_http_lock = threading.Lock()
_tokenizers: dict[str, object] = {}
tokenizer_lock = threading.Lock()
# When any streamed reply last carried an event (time.monotonic()), for request() to tell
# a request that waited in the server's queue from a server that stalled.
_last_event = [0.0]
# Set by the SIGTERM handler: requests stop at once instead of retrying or waiting on /health.
_stopping = threading.Event()


class ServerDead(RuntimeError):
    """The server's process exited, or it refuses or drops connections and /health stopped
    answering: no retry can help, and no answer may be scored as empty because of it."""


class RequestFailed(RuntimeError):
    """A request the live server refused (4xx), or that failed RETRIES + 1 times."""


class StreamCut(Exception):
    """A streamed reply that ended before its finish, or carried an error."""


def workers() -> int:
    """Answers to keep in flight on the running server."""
    if VLLM:
        return IN_FLIGHT.get("vllm") or VLLM_SEQS
    return IN_FLIGHT.get("llama") or SLOTS


def gate() -> threading.BoundedSemaphore:
    """The semaphore every request to the running server holds while it runs."""
    key = ("vllm" if VLLM else "llama", workers())
    with _gate_lock:
        return _gates.setdefault(key, threading.BoundedSemaphore(key[1]))


def limits(cfg) -> tuple[int, int]:
    """(max answer tokens, context per slot) for this job: follows the job's
    data answer cap (Config data_answer_max_tokens) unless LOBBOT_EVAL_* is set.
    The context also fits the code eval's answer cap (code_eval_max_tokens), plus
    the forced answer after it when thinking."""
    max_tokens = int(os.environ.get("LOBBOT_EVAL_MAX_TOKENS", max(2048, data.answer_max_tokens(cfg))))
    code = cfg.code_eval_max_tokens + (ANSWER_TOKENS if cfg.code_eval_thinking else 0)
    longest = max(max_tokens, code if cfg.code_eval_suites else 0)
    return max_tokens, int(os.environ.get("LOBBOT_EVAL_CTX", longest + 4096))


def gguf_arch(path: str) -> str:
    """general.architecture from a GGUF header (no gguf-py, which a Mac lacks)."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as f:
        magic, _version, _tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
        if magic != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")

        def string() -> bytes:
            return f.read(struct.unpack("<Q", f.read(8))[0])

        def skip(vtype: int) -> None:
            if vtype == 8:
                string()
            elif vtype == 9:
                item, n = struct.unpack("<IQ", f.read(12))
                for _ in range(n):
                    skip(item)
            else:
                f.seek(sizes[vtype], 1)

        for _ in range(n_kv):
            key, vtype = string(), struct.unpack("<I", f.read(4))[0]
            if key == b"general.architecture" and vtype == 8:
                return string().decode()
            skip(vtype)
    raise ValueError(f"{path} has no general.architecture")


_GGUF_NUMBER = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def gguf_meta(path: str) -> dict:
    """A GGUF header's strings, numbers and number arrays (e.g. a per-layer KV head
    count); string arrays such as the vocabulary are skipped."""
    with open(path, "rb") as f:
        magic, _version, _tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
        if magic != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")

        def string() -> bytes:
            return f.read(struct.unpack("<Q", f.read(8))[0])

        def value(vtype: int):
            if vtype == 8:
                return string().decode(errors="replace")
            if vtype == 9:
                item, n = struct.unpack("<IQ", f.read(12))
                if item in _GGUF_NUMBER:
                    return list(struct.unpack(f"<{n}{_GGUF_NUMBER[item]}",
                                              f.read(n * struct.calcsize(_GGUF_NUMBER[item]))))
                for _ in range(n):
                    value(item)
                return None
            fmt = "<" + _GGUF_NUMBER[vtype]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        out = {}
        for _ in range(n_kv):
            key = string().decode()
            v = value(struct.unpack("<I", f.read(4))[0])
            if v is not None:
                out[key] = v
    return out


def kv_bytes_per_token(meta: dict, kv_type: str = "") -> float:
    """KV-cache bytes one token takes in llama.cpp: K and V of every KV head in every
    layer (head_count_kv may be per layer, 0 on a recurrent one) at kv_type's size."""
    arch = meta["general.architecture"]
    layers = int(meta[f"{arch}.block_count"])
    per_layer = lambda v: [int(x) for x in v] if isinstance(v, list) else [int(v)] * layers
    heads = per_layer(meta[f"{arch}.attention.head_count"])
    kv = per_layer(meta.get(f"{arch}.attention.head_count_kv", meta[f"{arch}.attention.head_count"]))
    embd = int(meta[f"{arch}.embedding_length"])
    values = 0
    for h, n in zip(heads, kv):
        k_len = int(meta.get(f"{arch}.attention.key_length") or (embd // h if h else 0))
        values += n * (k_len + int(meta.get(f"{arch}.attention.value_length") or k_len))
    return values * KV_BYTES[kv_type or "f16"]


def free_vram_bytes() -> int | None:
    """Free memory over every GPU nvidia-smi lists, in bytes; None without one."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        mib = [float(x) for x in out.split()]
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return int(sum(mib) * 2 ** 20) if mib else None


def model_files(p: Path) -> list[Path]:
    """The files of a model: a checkpoint dir's safetensors, every shard of a split GGUF,
    else the file itself."""
    if p.is_dir():
        return sorted(p.glob("*.safetensors"))
    if "-00001-of-" in p.name:
        return sorted(p.parent.glob(p.name.replace("-00001-of-", "-*-of-")))
    return [p]


def llama_slots(cfg, gguf: str) -> int:
    """llama-server's slot count: as many fixed slots of the eval's context as the free GPU
    memory holds next to the weights and SLOT_RESERVE_GB. llama-server cannot preempt, so
    each slot must hold an answer at full length. LOBBOT_EVAL_SLOTS wins; SLOTS without
    nvidia-smi or a header to size the cache from."""
    if os.environ.get("LOBBOT_EVAL_SLOTS"):
        return SLOTS
    free = free_vram_bytes()
    if free is None:
        return SLOTS
    try:
        per_token = kv_bytes_per_token(gguf_meta(gguf), cfg.eval_kv_type)
    except (OSError, KeyError, ValueError, struct.error) as e:
        print(f"[eval] cannot size llama-server's slots from {gguf} ({e!r}); using {SLOTS}", flush=True)
        return SLOTS
    files = model_files(Path(gguf)) + ([Path(cfg.eval_lora)] if cfg.eval_lora else [])
    room = free - sum(f.stat().st_size for f in files if f.exists()) - SLOT_RESERVE_GB * 1e9
    return max(1, min(MAX_SLOTS, int(room // (limits(cfg)[1] * per_token))))


def server_args(cfg, gguf: str, slots: int | None = None) -> list[str]:
    slots = slots or llama_slots(cfg, gguf)
    args = ["-m", gguf, "-ngl", "999", "--host", "127.0.0.1", "--port", str(PORT),
            "-c", str(slots * limits(cfg)[1]), "-np", str(slots), "--jinja"]
    if cfg.eval_experts_used:
        args += ["--override-kv", f"{gguf_arch(gguf)}.expert_used_count=int:{cfg.eval_experts_used}"]
    if cfg.eval_lora:
        args += ["--lora", cfg.eval_lora]
    if cfg.eval_kv_type:  # a quantized V cache needs flash attention
        args += ["-ctk", cfg.eval_kv_type, "-ctv", cfg.eval_kv_type, "-fa", "on"]
    return args


def is_checkpoint(path: str) -> bool:
    """An HF checkpoint dir (BF16 safetensors, or compressed-tensors), served with
    vLLM; anything else is a GGUF for llama-server."""
    return (Path(path) / "config.json").is_file()


def vllm_command(model_dir: str) -> tuple[list[str], dict]:
    """(command that serves model_dir, its environment). The eval runs in the training
    venv, which has no vllm, so this is the vLLM venv's (LOBBOT_VLLM_PY, see
    setup_vm.sh): its `vllm` script, else its interpreter on the API server module,
    with its bin first on PATH and no PYTHONPATH, so the training venv's packages
    don't leak in. Without LOBBOT_VLLM_PY, vllm from PATH."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    py = os.environ.get("LOBBOT_VLLM_PY")
    if not py:
        return [shutil.which("vllm") or "vllm", "serve", model_dir], env
    bin_dir = Path(py).parent
    env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")])
    if (bin_dir / "vllm").exists():
        return [str(bin_dir / "vllm"), "serve", model_dir], env
    return [py, "-m", "vllm.entrypoints.openai.api_server", "--model", model_dir], env


def vllm_seqs(cfg) -> int:
    """vLLM's --max-num-seqs, and the answers kept in flight: the last --max-num-seqs in
    eval_vllm_args (vLLM takes the last), else Config.vllm_seqs, else VLLM_SEQS."""
    args = list(cfg.eval_vllm_args)
    for k in reversed(range(len(args))):
        if args[k] == "--max-num-seqs" and k + 1 < len(args):
            return int(args[k + 1])
        if args[k].startswith("--max-num-seqs="):
            return int(args[k].split("=", 1)[1])
    return cfg.vllm_seqs or VLLM_SEQS


def vllm_args(cfg, model_dir: str, name: str) -> list[str]:
    """`vllm serve` flags after the model; eval_vllm_args come last, so they win."""
    args = ["--host", "127.0.0.1", "--port", str(PORT), "--served-model-name", name,
            "--max-model-len", str(limits(cfg)[1]), "--max-num-seqs", str(cfg.vllm_seqs or VLLM_SEQS),
            "--gpu-memory-utilization", str(VLLM_GPU_UTIL), "--enable-prefix-caching"]
    if cfg.eval_reasoning_parser:
        args += ["--reasoning-parser", cfg.eval_reasoning_parser]
    if cfg.eval_experts_used:  # nested where the checkpoint keeps it: Qwen3.5/3.6 ship a text_config
        hf = json.loads((Path(model_dir) / "config.json").read_text())
        text = hf.get("text_config") or {}
        key = "top_k_experts" if "top_k_experts" in {*hf, *text} else "num_experts_per_tok"  # Gemma 4's name
        k = {key: cfg.eval_experts_used}
        nested = key not in hf and key in text
        args += ["--hf-overrides", json.dumps({"text_config": k} if nested else k)]
    return args + list(cfg.eval_vllm_args)


def make_client(transport=None):
    """An HTTP client for the server: no limit on a whole request, IDLE_TIMEOUT s on each
    read (between streamed tokens), short connect and write limits, and no limit on the
    wait for a connection (the gate already bounds the requests in flight)."""
    import httpx

    return httpx.Client(timeout=httpx.Timeout(connect=30, write=60, pool=None, read=IDLE_TIMEOUT),
                        limits=httpx.Limits(max_connections=None, max_keepalive_connections=64), transport=transport)


def _client():
    with _http_lock:
        if "client" not in _http:
            _http["client"] = make_client()
        return _http["client"]


def alive() -> bool:
    """False when the server is gone: the process serve() started has exited, or /health
    failed three times in a row, HEALTH_WAIT s apart."""
    import httpx

    proc = _SERVER.get("proc")
    if proc is not None and proc.poll() is not None:
        return False
    for i in range(3):
        try:
            if _client().get(f"http://127.0.0.1:{PORT}/health", timeout=30).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        if i < 2:
            time.sleep(HEALTH_WAIT)
    return False


def port_taken() -> bool:
    """Whether something already listens on PORT."""
    try:
        socket.create_connection(("127.0.0.1", PORT), timeout=2).close()
    except OSError:
        return False
    return True


def serve(job: Job, path: str, name: str) -> subprocess.Popen:
    """Starts the server for one candidate, vLLM for a checkpoint dir and llama-server
    for a GGUF, and waits until it is up; generate() then speaks to it. It runs as a
    session of its own, so stop() reaches every process it starts (vLLM's engine core).
    A server already on PORT (one an earlier eval left running when it was killed)
    is refused: its /health would pass for this one's, and it would answer for it."""
    import httpx

    global VLLM, THINK_TOP_K, MIN_P
    if port_taken():
        raise RuntimeError(f"port {PORT} is already in use, e.g. by a server an earlier eval left running; "
                           "stop it (or set LOBBOT_EVAL_PORT) and run again")
    cfg = job.config
    THINK_TOP_K, MIN_P = cfg.thinking_top_k, cfg.eval_min_p
    if is_checkpoint(path):
        cmd, env = vllm_command(path)
        cmd += vllm_args(cfg, path, name)
        server = "vllm"
        IN_FLIGHT["vllm"] = vllm_seqs(cfg)
        VLLM = {"name": name, "dir": path}
    else:
        binary = Path(cfg.llama_cpp) / "build/bin/llama-server"
        if not binary.exists() and shutil.which("llama-server"):
            binary = Path(shutil.which("llama-server"))  # e.g. Homebrew's llama.cpp on a Mac
        IN_FLIGHT["llama"] = llama_slots(cfg, path)
        cmd, env = [str(binary), *server_args(cfg, path, IN_FLIGHT["llama"])], None
        server = "llama-server"
        VLLM = {}
    log_path = job.path("work", f"{server}-{name}.log")
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    _SERVER.clear()
    _SERVER.update(proc=proc, path=path, name=name)
    deadline = time.monotonic() + HEALTH_TIMEOUT[server]
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stop(proc)  # what it started may outlive it
            raise RuntimeError(f"{server} exited for {name}:\n" + "\n".join(log_path.read_text().splitlines()[-20:]))
        try:
            if _client().get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(1)
    stop(proc)
    raise RuntimeError(f"{server} did not become healthy for {name} in {HEALTH_TIMEOUT[server]} s; see {log_path}")


def _signal(proc: subprocess.Popen, sig: int) -> None:
    """sig to proc's process group (serve() makes proc its leader), else to proc alone."""
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        if proc.poll() is None:
            proc.send_signal(sig)


def _group_alive(pgid: int) -> bool:
    """Whether a process of the group still runs. A zombie does not count: its parent (the
    stopped leader, then whatever adopted it) may never reap it, and it holds no GPU memory."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    if not Path("/proc").is_dir():  # e.g. a Mac: no way to tell a zombie
        return True
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            state, _ppid, group = stat.read_text().rsplit(")", 1)[1].split()[:3]
        except (OSError, ValueError):
            continue
        if state != "Z" and int(group) == pgid:
            return True
    return False


def stop(proc: subprocess.Popen, wait: float | None = None) -> None:
    """Stops a server and every process it started: SIGTERM to its process group, up to
    `wait` s (STOP_WAIT) for the group to exit, then SIGKILL to whatever is left. Killing
    only the API server could leave vLLM's engine core holding the GPU for the next job.
    Stopping it again does nothing: by then its group id may belong to another process."""
    if getattr(proc, "group_stopped", False):
        return
    wait = STOP_WAIT if wait is None else wait
    _signal(proc, signal.SIGTERM)
    deadline = time.monotonic() + wait
    try:
        proc.wait(timeout=wait)
    except subprocess.TimeoutExpired:
        pass
    while _group_alive(proc.pid) and time.monotonic() < deadline:  # the leader is reaped; its children?
        time.sleep(0.2)
    _signal(proc, signal.SIGKILL)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    deadline = time.monotonic() + 10
    while _group_alive(proc.pid) and time.monotonic() < deadline:  # gone, so its GPU memory is free
        time.sleep(0.1)
    proc.group_stopped = True
    if _SERVER.get("proc") is proc:
        _SERVER.clear()


def raise_on_sigterm() -> None:
    """From now on SIGTERM (pipeline.py stopping the stage, a killed tmux) stops the
    running server's process group (SIGKILL after SIGTERM_WAIT s), makes every request in
    flight give up, and raises SystemExit, so the finally blocks run and the answers in
    the partial file stay for the next run."""
    def handler(signum, _frame):
        _stopping.set()
        proc = _SERVER.get("proc")
        if proc is not None:
            stop(proc, SIGTERM_WAIT)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handler)


def split_reasoning(message: dict, cut: bool = False) -> tuple[str, str]:
    """(reasoning, answer) from a chat completion message. llama-server returns
    the reasoning separately, vLLM too (as reasoning_content or reasoning, by
    version); when they do not, the tags are still in the content. cut: thinking
    was on and the cap was hit, so untagged content with no reasoning beside it is
    a thinking the prompt opened (vLLM's qwen3 parser leaves that in content)."""
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if "</think>" in content:
        head, content = content.rsplit("</think>", 1)
        reasoning = reasoning or head.replace("<think>", "", 1)
    elif "<think>" in content:  # thinking that never closed
        reasoning, content = reasoning or content.split("<think>", 1)[1], ""
    elif "</think>" in reasoning and not content.strip():  # answer left inside the reasoning
        reasoning, content = reasoning.rsplit("</think>", 1)
    elif cut and not reasoning:
        reasoning, content = content, ""
    return reasoning.strip(), content.strip()


def sampling_params(seed: int | None, thinking: bool) -> dict:
    """Sampling fields for a request to the running server. min_p (Config.eval_min_p) and
    a repetition penalty of 1 always go explicitly, in the engine's own field names, so
    neither engine's default applies (llama-server's min_p is 0.05)."""
    penalty = {"repetition_penalty": 1.0} if VLLM else {"repeat_penalty": 1.0}
    return {**({"seed": seed, "top_p": 0.95} if seed is not None else {}), **({"top_k": THINK_TOP_K} if thinking else {}),
            "min_p": MIN_P, **penalty}


def tokenizer(model_dir: str):
    """model_dir's own HF tokenizer, loaded once per dir. Use it under tokenizer_lock:
    a fast tokenizer is not safe to share across threads."""
    with tokenizer_lock:
        if model_dir not in _tokenizers:
            from transformers import AutoTokenizer

            _tokenizers[model_dir] = AutoTokenizer.from_pretrained(model_dir)
        return _tokenizers[model_dir]


def chat_template(model_dir: str, messages: list[dict]) -> str:
    """messages in model_dir's chat template with thinking on, by its own tokenizer:
    vLLM has no /apply-template."""
    tok = tokenizer(model_dir)
    with tokenizer_lock:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)


def _events(r):
    """The JSON events of a server-sent-events reply, up to its [DONE]."""
    for line in r.iter_lines():
        if not line.startswith("data:"):  # llama-server's keep-alive pings are ":" lines
            continue
        _last_event[0] = time.monotonic()
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as e:
            raise StreamCut(f"unreadable event {payload[:200]!r}") from e
        if "error" in event or event.get("object") == "error":
            raise StreamCut(f"server error: {str(event.get('error') or event.get('message'))[:300]}")
        yield event


def request(path: str, body: dict, read=None):
    """POSTs body to the running server: read(events) of a streamed reply, or the JSON of
    a plain one (read=None). After any failure the server is checked (its process, then
    /health): once it is gone this raises ServerDead, which is what a refused or dropped
    connection nearly always means. While it lives, a reply that went IDLE_TIMEOUT s
    without a byte while other replies got tokens was queued, and is sent again (see
    IDLE_TIMEOUT); stalls, dropped or cut streams and 5xx are retried RETRIES times with
    backoff, then raise RequestFailed. A 4xx raises RequestFailed at once: the request
    itself is wrong."""
    import httpx

    url = f"http://127.0.0.1:{PORT}{path}"
    failures = queued = 0
    while True:
        if _stopping.is_set():
            raise ServerDead(f"{path}: the eval is stopping")
        idle = False
        try:
            with _client().stream("POST", url, json=body) as r:
                if r.status_code < 400:
                    return read(_events(r)) if read else json.loads(r.read())
                r.read()
                err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code < 500:
                    raise RequestFailed(f"{path}: {err}")
        except httpx.ReadTimeout:
            err, idle = f"nothing for {IDLE_TIMEOUT:.0f} s", True
        except (httpx.TransportError, StreamCut) as e:  # refused, dropped, cut short
            err = f"{type(e).__name__}: {e}"
        if _stopping.is_set() or not alive():
            raise ServerDead(f"{path}: the server is down ({err})")
        if idle and time.monotonic() - _last_event[0] < IDLE_TIMEOUT and queued < QUEUED_RETRIES:
            queued += 1
            print(f"[eval] {path}: {err} while the server answered others; asking again", flush=True)
            continue
        if failures == RETRIES:
            raise RequestFailed(f"{path} failed {RETRIES + 1} times; last: {err}")
        failures += 1
        print(f"[eval] {path} failed ({err}); retry {failures}/{RETRIES}", flush=True)
        time.sleep(BACKOFF * 2 ** (failures - 1))


def _read_chat(events) -> dict:
    """A streamed chat completion put together: {message: {content, reasoning_content},
    finish_reason, usage, timings (llama-server's), decode_s (first token to last)}."""
    content, reasoning, finish, usage, timings, first = [], [], None, None, None, None
    for e in events:
        for c in e.get("choices") or []:
            delta = c.get("delta") or {}
            text, thought = delta.get("content") or "", delta.get("reasoning_content") or delta.get("reasoning") or ""
            if (text or thought) and first is None:
                first = time.monotonic()
            content.append(text)
            reasoning.append(thought)
            finish = c.get("finish_reason") or finish
        usage, timings = e.get("usage") or usage, e.get("timings") or timings
    if finish is None:
        raise StreamCut("the stream ended without a finish reason")
    return {"message": {"content": "".join(content), "reasoning_content": "".join(reasoning)}, "finish_reason": finish,
            "usage": usage or {}, "timings": timings or {}, "decode_s": time.monotonic() - first if first else 0.0}


def _read_text(events) -> tuple[str, int]:
    """(text, tokens) of a streamed text completion: vLLM's /v1/completions or
    llama-server's /completion (whose last event has stop: true)."""
    text, finish, tokens = [], None, 0
    for e in events:
        for c in e.get("choices") or []:
            text.append(c.get("text") or "")
            finish = c.get("finish_reason") or finish
        text.append(e.get("content") or "")
        if e.get("stop") is True:
            finish, tokens = e.get("stop_type") or "stop", e.get("tokens_predicted") or 0
        tokens = (e.get("usage") or {}).get("completion_tokens") or tokens
    if finish is None:
        raise StreamCut("the stream ended without a finish reason")
    return "".join(text).strip(), tokens


def chat(messages: list[dict], max_tokens: int, temperature: float, sampling: dict, thinking: bool) -> dict:
    """One streamed chat completion from the running server (see _read_chat)."""
    return request("/v1/chat/completions", {
        **({"model": VLLM["name"]} if VLLM else {}),
        "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking}, **sampling,
        "stream": True, "stream_options": {"include_usage": True}}, _read_chat)


def thinking_prompt(messages: list[dict]) -> str:
    """The chat-template prompt up to and including the opened thinking (<think>,
    or Gemma 4's thought channel)."""
    if VLLM:
        prompt = chat_template(VLLM["dir"], messages)
    else:
        prompt = request("/apply-template", {"messages": messages,
                                             "chat_template_kwargs": {"enable_thinking": True}})["prompt"]
    opening = think_tags(prompt)[0]
    if not prompt.rstrip().endswith(opening.strip()):  # templates that let the model open it
        prompt += opening
    return prompt


def force_answer(messages: list[dict], reasoning: str, why: str, temperature: float,
                 sampling: dict) -> tuple[str, int]:
    """Closes the thinking (with Qwen's early-stop line when it ran out of budget)
    and has the model answer on top of it: (answer, tokens). Streamed like every
    request, so a long wait for a free slot fails nothing."""
    prompt = thinking_prompt(messages)
    prompt += reasoning + (EARLY_STOP if why == "length" else "") + think_tags(prompt)[1]
    if VLLM:  # the rendered template already holds its special tokens
        return request("/v1/completions", {
            "model": VLLM["name"], "prompt": prompt, "max_tokens": ANSWER_TOKENS, "temperature": temperature,
            "add_special_tokens": False, **sampling, "stream": True, "stream_options": {"include_usage": True}},
            _read_text)
    return request("/completion", {"prompt": prompt, "n_predict": ANSWER_TOKENS, "temperature": temperature,
                                   "cache_prompt": True, **sampling, "stream": True}, _read_text)


def generate(system: str, inputs: list[str], max_tokens: int = MAX_TOKENS,
             temperature: float = 0.0, seed: int | None = None,
             thinking: bool = False, on_answer=None, infos: list | None = None,
             keep_reasoning: bool = False, on_result=None) -> tuple[list[str], float | None]:
    """Answers plus the median decode speed (tok/s) the server reported (vLLM
    reports none: there, completion tokens over each answer's first to last token).

    With thinking, a thinking that ends without an answer (out of max_tokens, or
    the model stopped inside it) is closed and the model is asked for the answer
    on top of it, with up to ANSWER_TOKENS more; an answer the cap cut after the
    thinking closed is marked answer_cut and left as it is (LiveCodeBench does the
    same). on_result(k, answer, info) and then on_answer() are called from a worker
    thread as answer k finishes. infos, when given, receives one dict per input:
    finish reason, tokens, reasoning size and how the answer came, and with
    keep_reasoning the whole thinking. A request that keeps failing while the
    server lives gives an answer "" with finish "error", which callers must not
    score (answer_all and code_eval ask it again); a dead server raises ServerDead."""
    speeds: list[float] = []
    sampling = sampling_params(seed, thinking)
    held = gate()

    def one(text: str) -> tuple[str, dict]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": text}]
        try:
            got = chat(messages, max_tokens, temperature, sampling, thinking)
        except RequestFailed as e:
            print(f"[eval] generation failed: {e}", flush=True)
            return "", {"finish": "error", "error": str(e)[:300], "forced": None}
        tokens = got["usage"].get("completion_tokens")
        tps = got["timings"].get("predicted_per_second")
        if VLLM:
            tps = tokens / got["decode_s"] if tokens and got["decode_s"] > 0 else None
        if tps:
            speeds.append(float(tps))
        finish = got["finish_reason"]
        reasoning, answer = split_reasoning(got["message"], cut=bool(VLLM) and thinking and finish == "length")
        info = {"finish": finish, "tokens": tokens, "reasoning_chars": len(reasoning), "forced": None,
                **({"reasoning": reasoning} if keep_reasoning else {})}
        if thinking and answer and finish == "length":
            info["answer_cut"] = True
        if thinking and not answer:
            # "length": out of budget; "stop": the model ended inside its thinking;
            # "no_reasoning": nothing came back at all (then this is a thinking-off answer).
            info["forced"] = "no_reasoning" if not reasoning else "length" if finish == "length" else "stop"
            info["reasoning_tail"] = reasoning[-1500:]
            try:
                answer, info["forced_tokens"] = force_answer(messages, reasoning, info["forced"], temperature, sampling)
            except RequestFailed as e:  # an errored answer: asked again from the start
                print(f"[eval] answer after thinking failed: {e}", flush=True)
                info["error"] = str(e)[:300]
        return answer, info

    def counted(k: int, text: str) -> tuple[str, dict]:
        with held:
            got = one(text)
        if on_result:
            on_result(k, *got)
        if on_answer:
            on_answer()
        return got

    with ThreadPoolExecutor(workers()) as ex:
        futures = [ex.submit(counted, k, text) for k, text in enumerate(inputs)]
        try:
            got = [f.result() for f in futures]
        except BaseException:  # a dead server or a stop: start no more requests
            for f in futures:
                f.cancel()
            raise
    if infos is not None:
        infos.extend(i for _, i in got)
    return [a for a, _ in got], (round(statistics.median(speeds), 1) if speeds else None)


def answer_all(system: str, inputs: list[str], max_tokens: int = MAX_TOKENS, temperature: float = 0.0,
               seed: int | None = None, thinking: bool = False, on_answer=None) -> tuple[list[str], float | None]:
    """generate() for answers that are scored straight away (held-out inputs, fix turns):
    each errored answer is asked again, up to REASK more rounds, and if any is still
    errored this raises rather than hand it back to be scored as empty."""
    infos: list[dict] = []
    answers, tps = generate(system, inputs, max_tokens, temperature, seed, thinking, on_answer, infos=infos)
    bad = [k for k, info in enumerate(infos) if errored(info)]
    for _ in range(REASK):
        if not bad:
            break
        print(f"[eval] {len(bad)} answers errored; asking them again", flush=True)
        again: list[dict] = []
        got, _ = generate(system, [inputs[k] for k in bad], max_tokens, temperature, seed, thinking, infos=again)
        for k, a, info in zip(bad, got, again):
            answers[k], infos[k] = a, info
        bad = [k for k in bad if errored(infos[k])]
    if bad:
        raise RuntimeError(f"{len(bad)} answers still errored after {REASK} more tries "
                           f"(first: {infos[bad[0]].get('error')})")
    return answers, tps


def _tokens(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def _f1(a: str, b: str) -> float:
    ta, tb = Counter(_tokens(a)), Counter(_tokens(b))
    common = sum((ta & tb).values())
    if not ta or not tb:
        return float(ta == tb)
    if common == 0:
        return 0.0
    p, r = common / sum(ta.values()), common / sum(tb.values())
    return 2 * p * r / (p + r)


def _as_json(s: str):
    s = s.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.S)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None


def agreement(answer: str, reference: str) -> float:
    """1.0 means the answer matches the teacher. For JSON objects: short fields
    must match exactly, long text fields score by token F1; invalid JSON is 0."""
    ref = _as_json(reference)
    if isinstance(ref, dict) and ref:
        ans = _as_json(answer)
        if not isinstance(ans, dict):
            return 0.0
        scores = []
        for k, v in ref.items():
            a = ans.get(k)
            if isinstance(v, str) and len(v.split()) > 3:
                scores.append(_f1(str(a or ""), v))
            else:
                scores.append(float(str(a).strip().lower() == str(v).strip().lower()))
        return sum(scores) / len(scores)
    return _f1(answer, reference)


def judge_key(model: str) -> str | None:
    """The API key the judge model needs, or None if it is not set."""
    if model.startswith("gemini"):
        from stages.gemini import api_key

        return api_key()
    return os.environ.get("ANTHROPIC_API_KEY")


def judge(spec, model: str, inputs: list[str], answers: list[str]) -> float | None:
    """Mean judge score in [0, 1]; None if no answer could be judged. gemini-*
    models go to the Gemini API, claude-* to Anthropic; both through
    condense.chat when CONDENSE_API_KEY is set (stages/gemini.py, stages/condense.py)."""
    if model.startswith("gemini"):
        from stages import gemini

        ask = lambda prompt: gemini.chat(model, [{"role": "user", "content": prompt}])
    else:
        from stages.condense import anthropic_client

        client = anthropic_client(max_retries=6)
        ask = lambda prompt: client.messages.create(
            model=model, max_tokens=20, messages=[{"role": "user", "content": prompt}]).content[0].text

    broken = threading.Event()  # a 4xx (bad key or model id) will not fix itself: stop calling

    def one(pair) -> float | None:
        text, answer = pair
        if broken.is_set():
            return None
        try:
            reply = ask(
                f"Task: {spec.description}\nOutput format: {spec.output_format}\nCriteria: {spec.eval_criteria}\n\n"
                f"<input>\n{text}\n</input>\n<answer>\n{answer}\n</answer>\n\n"
                "Score how well the answer meets the criteria from 0 to 10. Reply with the number only.")
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None) or getattr(e, "status_code", None)
            if isinstance(status, int) and 400 <= status < 500 and status != 429 and not broken.is_set():
                broken.set()
                print(f"[eval] judge disabled after HTTP {status}: {e}", flush=True)
            elif not broken.is_set():
                print(f"[eval] judge call failed: {e}", flush=True)
            return None
        m = re.search(r"\d+(\.\d+)?", reply)
        return min(10.0, float(m.group(0))) / 10 if m else None

    with ThreadPoolExecutor(16) as ex:
        scores = [s for s in ex.map(one, zip(inputs, answers)) if s is not None]
    return round(sum(scores) / len(scores), 3) if scores else None


def watch_gpu(every: float = 30.0):
    """Samples nvidia-smi every `every` s in a background thread, to see whether the
    server kept the GPU fed. Returns end(), which stops it and gives {busy_pct: mean
    over samples and GPUs, mem_peak_mib: peak of the GPUs' total}; {} without
    nvidia-smi or before the first sample."""
    smi, busy, mem, done = shutil.which("nvidia-smi"), [], [], threading.Event()

    def sample() -> None:
        while not done.wait(every):
            try:
                out = subprocess.run([smi, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=20).stdout
                gpus = [(float(u), float(m)) for u, m in (line.split(",") for line in out.splitlines() if line.strip())]
            except (OSError, ValueError, subprocess.SubprocessError):
                continue
            if gpus:
                busy.append(statistics.mean(u for u, _ in gpus))
                mem.append(sum(m for _, m in gpus))

    thread = threading.Thread(target=sample, daemon=True)
    if smi:
        thread.start()

    def end() -> dict:
        done.set()
        if smi:
            thread.join()
        return {"busy_pct": round(statistics.mean(busy), 1), "mem_peak_mib": round(max(mem))} if busy else {}

    return end


def cap_cases(p: dict, n: int) -> dict:
    """p with only its first n test cases (the examples come first); n = 0 keeps all."""
    tests = p.get("tests")
    if not n or not isinstance(tests, (list, dict)):
        return p
    tests = {**tests, "cases": tests["cases"][:n]} if isinstance(tests, dict) else tests[:n]
    return {**p, "tests": tests, **({"public": min(p["public"], n)} if p.get("public") is not None else {})}


def lcb_problems(cfg) -> list[dict]:
    """LiveCodeBench's problems from lcb_since to lcb_until (a row without a date is kept,
    as codebench.load_suite does), only code_eval_ids when set, each with its first
    lcb_max_cases cases. Read a row at a time: scripts/fetch_codebench.py stores every
    case, so the file runs to gigabytes and one problem's cases to hundreds of MB, and
    only the kept rows, cut, stay in memory."""
    path = codebench.suite_path("livecodebench", cfg.code_eval_dir)
    if not path.exists():
        raise FileNotFoundError(f"code suite livecodebench not found at {path}; run scripts/fetch_codebench.py")
    want = set(cfg.code_eval_ids)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            day = (r.get("date") or "")[:10]
            if day and (cfg.lcb_since and day < cfg.lcb_since or cfg.lcb_until and day > cfg.lcb_until):
                continue
            if want and r["id"] not in want:
                continue
            r.update(language=codebench.normalize_language(r["language"]), suite="livecodebench")
            rows.append(cap_cases(r, cfg.lcb_max_cases))
    return rows


def code_problems(job: Job, held: list[dict]) -> list[dict]:
    """Every problem of the configured code suites (LiveCodeBench's as lcb_problems
    selects them), then only code_eval_ids when set, then code_eval_limit per suite."""
    cfg = job.config
    want = set(cfg.code_eval_ids)
    problems: list[dict] = []
    for suite in cfg.code_eval_suites:
        if suite == "heldout":
            rows = codebench.heldout_problems(held)
        elif suite == "livecodebench":
            rows = lcb_problems(cfg)
        else:
            rows = codebench.load_suite(suite, cfg.code_eval_dir)
        if want:
            rows = [r for r in rows if r["id"] in want]
        rows = rows[: cfg.code_eval_limit or None]
        if not rows:
            emit(STAGE, msg=f"code suite {suite} has no problems; skipping it")
        problems += rows
    missing = want - {p["id"] for p in problems}
    if missing:
        raise ValueError(f"{len(missing)} code_eval_ids are in none of the suites {cfg.code_eval_suites}: "
                         f"{', '.join(sorted(missing)[:5])}")
    return problems


def ids_sha256(ids) -> str:
    """The sha256 of a problem set: its sorted ids, one per line."""
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def prompting(cfg, task_system: str):
    """(system_of(p), prompt(p), code_of(p, answer)) for the configured code eval format."""
    mugge = cfg.code_eval_format == "harness"
    prompt = (lambda p: harness.pack_text(codebench.ticket(p))) if mugge else codebench.build_prompt
    code_of = codebench.harness_code if mugge else lambda p, a: codebench.extract_code(a)

    def system_of(p: dict) -> str:
        # Held-out rows of a merged multi-language job carry their own task prompt.
        if mugge:
            return harness.SYSTEM
        return (p.get("system") or task_system) if p["suite"] == "heldout" else codebench.SYSTEM

    return system_of, prompt, code_of


# LiveCodeBench runs every test case in one process, so it gets a longer budget. Problems
# fetched with their cases as data get CASE_TIMEOUT per case instead (codebench.time_limit).
SUITE_TIMEOUT = {"livecodebench": max(sandbox.TIMEOUT, 60.0)}


def sample_counts(cfg, problems: list[dict]) -> tuple[list[dict], list[int]]:
    """(problems, answers to each): code_eval_samples_by_id for the ids it lists, else
    max(code_eval_samples, max(code_eval_k)). Problems given 0 are left out."""
    n = max(cfg.code_eval_samples, max(cfg.code_eval_k))
    by_id = cfg.code_eval_samples_by_id
    unknown = set(by_id) - {p["id"] for p in problems}
    if unknown:
        emit(STAGE, msg=f"code_eval_samples_by_id names {len(unknown)} ids that are not among the problems, "
                        f"e.g. {sorted(unknown)[0]}")
    counts = [int(by_id.get(p["id"], n)) for p in problems]
    if any(k < 0 for k in counts):
        raise ValueError("code_eval_samples_by_id has a negative count")
    kept = [(p, k) for p, k in zip(problems, counts) if k > 0]
    return [p for p, _ in kept], [k for _, k in kept]


def errored(info: dict | None) -> bool:
    """An answer the server never gave in full: it must be asked again, never scored."""
    return info is None or info.get("finish") == "error" or "error" in info


def run_key(cfg, path: str, temperature: float, keep_reasoning: bool) -> str:
    """What the answers depend on besides the problem and the seed: the model at path
    (size of its files, and the caller's eval_model_fingerprint or else their modification
    time), the cap, sampling and the server settings that change what it writes. Answers
    resumed from a partial file must have been made under the same key. Known before any
    server starts, so a run with every answer in the partial file starts none. The mtime
    only holds on the VM that wrote the answers: a new VM downloads or rebuilds the same
    model with new mtimes, so a caller that knows the model by content passes that."""
    files = [f for f in model_files(Path(path)) if f.exists()] if path else []
    made = cfg.eval_model_fingerprint or max((f.stat().st_mtime for f in files), default=0)
    key = {"model": [path, sum(f.stat().st_size for f in files), made],
           "max_tokens": cfg.code_eval_max_tokens, "thinking": cfg.code_eval_thinking, "temperature": temperature,
           "top_k": cfg.thinking_top_k, "min_p": cfg.eval_min_p, "format": cfg.code_eval_format, "kv": cfg.eval_kv_type,
           "lora": cfg.eval_lora, "vllm_args": list(cfg.eval_vllm_args), "parser": cfg.eval_reasoning_parser,
           "experts": cfg.eval_experts_used, "keep_reasoning": keep_reasoning}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


# Fields of a partial-file row that are not the answer's info.
_ROW_KEYS = {"suite", "id", "sample", "seed", "run", "prompt_sha", "answer"}


def resume(path: Path, run: str) -> tuple[dict[tuple, dict], int]:
    """({(suite, id, sample): row}, rows made under another run key) from a partial file;
    errored answers are left out, so they are asked again."""
    rows, stale = {}, 0
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # empty, or cut by the stop
                continue
            if r.get("run") != run:
                stale += 1
            elif not errored({k: v for k, v in r.items() if k not in _ROW_KEYS}):
                rows[(r["suite"], r["id"], r["sample"])] = r
    return rows, stale


def code_eval(job: Job, name: str, problems: list[dict], task_system: str, *, start=None, release=None,
              path: str = "") -> dict:
    """Answer every problem on the server (code_eval_samples times, or its count in
    code_eval_samples_by_id; answer i with seed code_eval_seed_offset + i), then run the
    tests in the sandbox and return per-suite pass@k and answer_stats().

    Each answer is appended to work/code_eval/<name>.partial.jsonl as it arrives, and
    answers already there (same run_key for the model at path) are not asked again.
    Errored answers are asked again up to REASK times; if any remain this raises, and
    no result is returned. start(), when given, is called before anything is asked
    (run_candidate starts the server there, so a run with every answer in the partial
    file starts none); release() once every answer is in and before the sandbox runs
    (run_candidate stops the server there). The greedy fix turn needs the server, so
    with code_eval_fix release() comes after it. Per-answer results go to
    work/code_eval/<name>.jsonl. Held-out task rows are asked with the task's own
    system prompt (the one the model was healed on), benchmarks with codebench.SYSTEM."""
    cfg = job.config
    problems, counts = sample_counts(cfg, problems)
    if not problems:
        raise ValueError("no code problem is left to answer (every count in code_eval_samples_by_id is 0)")
    n = max(counts)
    ks = [k for k in cfg.code_eval_k if k <= min(counts)]
    if len(ks) < len(cfg.code_eval_k):
        emit(STAGE, msg=f"pass@{', '.join(str(k) for k in cfg.code_eval_k if k not in ks)} left out: "
                        f"some problems get only {min(counts)} answers")
    mugge = cfg.code_eval_format == "harness"
    system_of, prompt, code_of = prompting(cfg, task_system)
    groups: dict[tuple[str, float], list[int]] = {}
    for j, p in enumerate(problems):
        groups.setdefault((system_of(p), SUITE_TIMEOUT.get(p["suite"], sandbox.TIMEOUT)), []).append(j)
    greedy = n == 1
    # Greedy decoding loops when thinking, so thinking always samples at the model
    # card's temperature, seeded for repeatability, however many samples there are.
    temperature = (cfg.code_eval_thinking_temperature if cfg.code_eval_thinking
                   else 0.0 if greedy else cfg.code_eval_temperature)
    seeds = [cfg.code_eval_seed_offset + i if cfg.code_eval_thinking or not greedy else None for i in range(n)]
    keep = bool(cfg.code_eval_budgets)
    out_dir = job.path("work", "code_eval")
    out_dir.mkdir(exist_ok=True)
    partial = out_dir / f"{name}.partial.jsonl"
    run = run_key(cfg, path or _SERVER.get("path") or "", temperature, keep)
    prompts = [prompt(p) for p in problems]
    shas = [hashlib.sha256((system_of(p) + "\0" + text).encode()).hexdigest()[:16] for p, text in zip(problems, prompts)]
    answers: list[list] = [[None] * k for k in counts]
    infos: list[list] = [[None] * k for k in counts]
    have, stale = resume(partial, run)
    for j, p in enumerate(problems):
        for i in range(counts[j]):
            r = have.get((p["suite"], p["id"], i))
            if r and r.get("seed") == seeds[i] and r.get("prompt_sha") == shas[j]:
                answers[j][i], infos[j][i] = r["answer"], {k: v for k, v in r.items() if k not in _ROW_KEYS}
    todo = [(j, i) for i in range(n) for j in range(len(problems)) if i < counts[j] and infos[j][i] is None]
    total = sum(counts)
    done, lock = total - len(todo), threading.Lock()
    if done or stale:
        emit(STAGE, msg=f"{name}: {done} answers kept from {partial.name}"
                        + (f"; {stale} made under other settings are asked again" if stale else ""))
    speeds: list[float] = []
    finished: list[float] = []  # when each pass got its last answer
    fresh: set[tuple[int, int]] = set()  # (j, i) answered in this run, for its tokens/s

    def answered():  # progress events with answered/total, at most ~100 per eval
        nonlocal done
        with lock:
            done += 1
            if done == total or done % max(1, total // 100) == 0:
                emit(STAGE, msg=f"{name}: {done}/{total} code answers", answered=done, total=total)

    def record(j: int, i: int, answer: str, info: dict) -> None:
        p = problems[j]
        with lock:
            answers[j][i], infos[j][i] = answer, info
            fresh.add((j, i))
            with partial.open("a") as f:
                f.write(json.dumps({"suite": p["suite"], "id": p["id"], "sample": i, "seed": seeds[i], "run": run,
                                    "prompt_sha": shas[j], "answer": answer, **info}, ensure_ascii=False) + "\n")

    def ask(pairs: list[tuple[int, int]]) -> None:
        """Answers each (problem j, sample i): one generate() per sample pass and prompt
        group, every pass at once, so one pass's long tail does not idle the GPU."""
        by_pass: dict[int, set[int]] = {}
        for j, i in pairs:
            by_pass.setdefault(i, set()).add(j)

        def one_pass(i: int) -> None:
            for (system, _), group in groups.items():
                idx = [j for j in group if j in by_pass[i]]
                if not idx:
                    continue
                got_info: list[dict] = []
                got, tps = generate(system, [prompts[j] for j in idx], cfg.code_eval_max_tokens, temperature, seeds[i],
                                    cfg.code_eval_thinking, answered, infos=got_info, keep_reasoning=keep,
                                    on_result=lambda k, a, info, idx=idx: record(idx[k], i, a, info))
                speeds.extend([tps] if tps else [])
                for k, (j, a) in enumerate(zip(idx, got)):  # any answer on_result did not see
                    if infos[j][i] is None:
                        record(j, i, a, got_info[k] if k < len(got_info) else {})
            finished.append(time.monotonic())

        with ThreadPoolExecutor(len(by_pass)) as ex:
            list(ex.map(one_pass, sorted(by_pass)))

    emit(STAGE, msg=f"{name}: {done}/{total} code answers", answered=done, total=total)
    began, reasked = time.monotonic(), 0
    for attempt in range(REASK + 1):
        if todo:
            if start:
                start()
            ask(todo)
        todo = [(j, i) for j in range(len(problems)) for i in range(counts[j]) if errored(infos[j][i])]
        if not todo or attempt == REASK:
            break
        emit(STAGE, msg=f"{name}: {len(todo)} answers errored; asking them again")
        reasked += len(todo)
        with lock:
            total += len(todo)
            for j, i in todo:
                answers[j][i] = infos[j][i] = None
    if todo:
        j, i = todo[0]
        raise RuntimeError(f"{name}: {len(todo)} answers still errored after {REASK} more tries (first: "
                           f"{(infos[j][i] or {}).get('error')}); answers so far are kept in {partial}, "
                           "and a re-run asks only for the rest")
    seconds = max(finished, default=began) - began
    fix_turn = cfg.code_eval_fix and greedy
    if release and not fix_turn:
        release()

    pairs = [(j, i) for i in range(n) for j in range(len(problems)) if i < counts[j]]
    ran: dict[tuple[int, int], sandbox.Result] = {}
    for (_, timeout), group in groups.items():
        members = set(group)
        sel = [(j, i) for j, i in pairs if j in members]
        items = [codebench.job(problems[j], answers[j][i], code_of(problems[j], answers[j][i])) for j, i in sel]
        ran.update(zip(sel, sandbox.run_many(items, timeout=timeout)))
    # Sampled: whether each sample passes the examples in its problem's text, for pick@examples
    examples: list[list[bool]] = [[] for _ in problems]
    if not greedy:
        check = [(j, i) for j, i in pairs if problems[j].get("public") and not ran[j, i].passed]
        shown = {j: codebench.examples_only(problems[j]) for j, _ in check}
        items = [codebench.job(shown[j], answers[j][i], code_of(shown[j], answers[j][i])) for j, i in check]
        ex = dict(zip(check, (r.passed for r in sandbox.run_many(items)))) if items else {}
        for j, p in enumerate(problems):
            if p.get("public"):
                examples[j] = [ran[j, i].passed or ex[j, i] for i in range(counts[j])]  # a full pass passes them too
    fixed: list = [None] * len(problems)
    fix_answers = [""] * len(problems)
    if fix_turn:
        todo_fix = {j for j in range(len(problems)) if not ran[j, 0].passed and ran[j, 0].reason not in codebench.CANNOT_RUN}
        with lock:
            total += len(todo_fix)
        emit(STAGE, msg=f"{name}: one fix turn for {len(todo_fix)} failed answers", answered=done, total=total)
        fix_text = ((lambda j: codebench.harness_fix_text(problems[j], code_of(problems[j], answers[j][0]), ran[j, 0]))
                    if mugge else (lambda j: codebench.fix_prompt(problems[j], answers[j][0], ran[j, 0])))
        fix_groups = [(system, timeout, sel) for (system, timeout), group in groups.items()
                      if (sel := [j for j in group if j in todo_fix])]
        if fix_groups and start:
            start()
        for system, _, sel in fix_groups:
            got, tps = answer_all(system, [fix_text(j) for j in sel], cfg.code_eval_max_tokens, temperature,
                                  seeds[0], cfg.code_eval_thinking, answered)
            speeds.extend([tps] if tps else [])
            for j, a in zip(sel, got):
                fix_answers[j] = a
        if release:
            release()
        for _, timeout, sel in fix_groups:  # tested once the server is stopped, as the first answers were
            items = [codebench.job(problems[j], fix_answers[j], code_of(problems[j], fix_answers[j])) for j in sel]
            for j, r in zip(sel, sandbox.run_many(items, timeout=timeout)):
                fixed[j] = r

    rows = []
    for j, i in pairs:
        p, r = problems[j], ran[j, i]
        fix = ({"fix_passed": fixed[j].passed, "fix_reason": fixed[j].reason, "fix_output": fixed[j].output[-500:],
                "fix_answer": fix_answers[j]} if fixed[j] else {})
        rows.append({"suite": p["suite"], "id": p["id"], "sample": i, "seed": seeds[i], "passed": r.passed,
                     "reason": r.reason, "output": r.output[-500:], "answer": answers[j][i], **infos[j][i], **fix,
                     **({"examples_passed": examples[j][i]} if len(examples[j]) > i else {})})
    write_jsonl(out_dir / f"{name}.jsonl", rows)
    results = [[ran[j, i] for i in range(counts[j])] for j in range(len(problems))]
    suites = codebench.summarize(problems, results, ks)
    for suite, v in suites.items():  # the problem set, to check a paired reading against
        v["ids_sha256"] = ids_sha256(p["id"] for p in problems if p["suite"] == suite)
        if suite == "livecodebench":
            v.update(since=cfg.lcb_since, until=cfg.lcb_until, max_cases=cfg.lcb_max_cases)
    if not greedy:
        codebench.add_pick_scores(suites, problems, results, examples)
    fix = {}
    if fix_turn:
        codebench.add_fix_scores(suites, problems, [rs[0] for rs in results], fixed)
        fix = {"mean_fix@1": codebench.mean_fix1(suites)}
    for suite, v in suites.items():
        broken = sum(v["status"].get(r, 0) for r in ("missing_toolchain", "unsupported_language"))
        if broken:
            emit(STAGE, msg=f"{suite}: {broken} answers could not run ({', '.join(k for k in v['status'] if k in ('missing_toolchain', 'unsupported_language'))}); check the VM toolchains")
    return {"suites": suites, "mean_pass@1": codebench.mean_pass1(suites), **fix, "samples_per_problem": n,
            "answers": len(rows), "seed_offset": cfg.code_eval_seed_offset, "reasked": reasked,
            "tok_s_vm": round(statistics.median(speeds), 1) if speeds else None,
            **answer_stats(rows, seconds, [rows[k] for k, pair in enumerate(pairs) if pair in fresh])}


def answer_stats(rows: list[dict], seconds: float, timed: list[dict] | None = None) -> dict:
    """How the code answers ran: completion tokens per answer (thinking in, a forced
    answer not); the shares that errored, hit the cap, needed a forced answer, were
    cut by the cap after their thinking closed (answer_cut), ran out of the sandbox's
    time (timeout) or of one test case's (case_timeout); and output tokens/s of the
    whole eval: every token written (forced answers in, fix turns not) over the wall
    time of answering. timed: the rows answered in those seconds (all of them unless
    some were resumed from an earlier run)."""
    tokens = [r["tokens"] for r in rows if r.get("tokens") is not None]
    timed = rows if timed is None else timed
    written = sum(r.get("tokens") or 0 for r in timed) + sum(r.get("forced_tokens") or 0 for r in timed)
    share = lambda hit: round(sum(map(hit, rows)) / len(rows), 4) if rows else None
    return {"tokens_mean": round(statistics.mean(tokens)) if tokens else None,
            "tokens_median": round(statistics.median(tokens)) if tokens else None,
            "error_share": share(lambda r: r.get("finish") == "error" or "error" in r),
            "cap_hit_share": share(lambda r: r.get("finish") == "length"),
            "forced_share": share(lambda r: bool(r.get("forced"))),
            "answer_cut_share": share(lambda r: bool(r.get("answer_cut"))),
            "timeout_share": share(lambda r: r.get("reason") == "timeout"),
            "case_timeout_share": share(lambda r: r.get("reason") == "case_timeout"),
            "tok_s_total": round(written / seconds, 1) if seconds > 0 else None, "answer_s": round(seconds, 1)}


def load_code_ref(job: Job) -> dict:
    """The reference job's per-suite code results ({} when unset or not run yet)."""
    if not job.config.code_eval_ref:
        return {}
    p = Path(job.config.code_eval_ref)
    p = p if p.is_absolute() else job.root.parent / p
    if p.is_dir():
        p = p / "out" / "eval.json"
    if not p.exists():
        emit(STAGE, msg=f"reference results {p} not found; run the ref job first for scores as a share of ref")
        return {}
    return codebench.ref_suites(json.loads(p.read_text()))


def candidates_to_eval(job: Job) -> dict[str, dict]:
    """{name: {path, size_gb, tok_s_est}} from Config.eval_candidates or allocation.json;
    size_gb_override wins for the names it lists."""
    cfg = job.config
    if cfg.eval_candidates:
        out = {}
        for name, path in cfg.eval_candidates.items():
            p = Path(path)
            if not p.exists() and not DRY_RUN:
                raise FileNotFoundError(f"eval candidate {name}: {p} not found")
            size = sum(f.stat().st_size for f in model_files(p) if f.exists())
            out[name] = {"path": str(p), "size_gb": round(size / 1e9, 3), "tok_s_est": None}
    else:
        out = json.loads(job.path("work", "allocation.json").read_text())["candidates"]
    for name, gb in cfg.size_gb_override.items():
        if name in out:
            out[name] = {**out[name], "size_gb": gb}
    return out


def teacher_size_gb(job: Job) -> float:
    d = Path(job.model_path(job.config.teacher))
    if d.is_dir():
        n = sum(f.stat().st_size for f in d.glob("*.safetensors"))
        if n:
            return round(n / 1e9, 1)
    return 61.0  # Qwen3-30B-A3B in bf16


def suite_meta(cfg) -> dict:
    """What scripts/fetch_codebench.py recorded for each code suite's file (repo, commit,
    case cap, the file's sha256, its ids' sha256 and windows), without the id lists."""
    out = {}
    for suite in cfg.code_eval_suites:
        try:
            meta = json.loads((Path(cfg.code_eval_dir) / f"{suite}.meta.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(meta.get("windows"), dict):
            meta["windows"] = {k: {x: v for x, v in w.items() if x != "ids"} for k, w in meta["windows"].items()}
        out[suite] = meta
    return out


def provenance(cfg) -> dict:
    """What a reading ran on: GPU, CPU count (logical and physical), sandbox workers,
    package versions (the eval's venv and vLLM's), the repo's git HEAD (and whether
    tracked files differ from it), the llama.cpp commit, and the code suites' fetch
    records. A value that cannot be read is None."""
    import importlib.metadata as md

    def out(*cmd: str) -> str | None:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
        return p.stdout.strip() or None if p.returncode == 0 else None

    def version(pkg: str) -> str | None:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    pkgs = ("vllm", "torch", "transformers")
    py = os.environ.get("LOBBOT_VLLM_PY")
    serving = None
    if py:  # the versions vLLM serves with, from its own venv
        script = ("import importlib.metadata as m, json\n"
                  "def v(p):\n"
                  "    try:\n        return m.version(p)\n"
                  "    except m.PackageNotFoundError:\n        return None\n"
                  f"print(json.dumps({{p: v(p) for p in {pkgs!r}}}))")
        try:
            serving = json.loads(out(py, "-c", script) or "null")
        except json.JSONDecodeError:
            serving = None
    smi = shutil.which("nvidia-smi")
    gpus = out(smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader") if smi else None
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": platform.node(),
            "gpu": gpus.splitlines() if gpus else None, "nproc": os.cpu_count(),
            "physical_cores": sandbox.physical_cores(), "sandbox_workers": sandbox.WORKERS,
            "python": platform.python_version(), "eval_venv": {p: version(p) for p in pkgs},
            "vllm_venv": serving, "git_head": out("git", "-C", str(REPO), "rev-parse", "HEAD"),
            "git_dirty": bool(out("git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no")),
            "llama_cpp": out("git", "-C", cfg.llama_cpp, "rev-parse", "HEAD"), "code_suites": suite_meta(cfg),
            "env": {k: v for k, v in sorted(os.environ.items())
                    if k.startswith(("LOBBOT_EVAL_", "LOBBOT_VLLM_", "LOBBOT_SANDBOX_", "LOBBOT_LCB_"))}}


def run_candidate(job: Job, name: str, path: str, inputs: list[str], problems: list[dict],
                  system: str) -> tuple[list[str], float | None, dict | None, dict]:
    """(answers to the held-out inputs, their tok/s, code results, GPU record) for one
    candidate. Its server starts when the first answer is asked (none when the partial
    file holds every code answer and there are no held-out inputs), stops once every
    answer is in (before the sandbox), and is started again RESTARTS times if it dies;
    code answers made before resume from the partial file."""
    cfg = job.config
    answers: list[str] = []
    tps = None
    gpu: dict = {}
    for attempt in range(RESTARTS + 1):
        server: dict = {}

        def start() -> None:
            if not server:
                server["proc"] = serve(job, path, name)
                server["end_watch"] = watch_gpu()  # from here on: the load is not the eval

        def release() -> None:
            if "end_watch" in server:
                gpu.update(server.pop("end_watch")())
            if "proc" in server:
                stop(server.pop("proc"))

        try:
            if inputs and not answers:
                start()
                answers, tps = answer_all(system, inputs, limits(cfg)[0])
            code = None
            if problems:
                emit(STAGE, msg=f"{name}: answering {len(problems)} code problems")
                code = code_eval(job, name, problems, system, start=start, release=release, path=path)
            release()
            return answers, tps, code, gpu
        except ServerDead as e:
            if attempt == RESTARTS:
                raise
            emit(STAGE, msg=f"{name}: the server died ({e}); starting it again, answers so far are kept")
        finally:
            release()
    raise AssertionError("unreachable")


def run_stage(job: Job) -> None:
    from stages.data import system_prompt

    cfg, spec = job.config, job.spec
    alloc_path = job.path("work", "allocation.json")
    alloc = json.loads(alloc_path.read_text()) if alloc_path.exists() and not cfg.eval_candidates else {}
    cands = candidates_to_eval(job)
    held_path = job.path("data", "heldout.jsonl")
    held = read_jsonl(held_path) if held_path.exists() else []
    inputs = [h["input"] for h in held]
    refs = [h["reference"] for h in held]
    system = system_prompt(spec)
    teacher_name = cfg.teacher.split("/")[-1]
    problems = code_problems(job, held) if cfg.code_eval_suites else []
    if not inputs and not problems:
        raise RuntimeError("nothing to evaluate: no data/heldout.jsonl and no code_eval_suites")
    use_judge = bool(inputs) and bool(judge_key(cfg.judge_model)) and not DRY_RUN
    if inputs and not use_judge and not DRY_RUN:
        need = "GEMINI_API_KEY" if cfg.judge_model.startswith("gemini") else "ANTHROPIC_API_KEY"
        emit(STAGE, msg=f"{need} not set on the VM; scoring by agreement with the teacher")
    if problems:
        sets = ", ".join(f"{s} {sum(p['suite'] == s for p in problems)} (ids {ids_sha256(p['id'] for p in problems if p['suite'] == s)[:12]})"
                         for s in dict.fromkeys(p["suite"] for p in problems))
        emit(STAGE, msg=f"code eval: {len(problems)} problems: {sets}")

    results: dict[str, dict] = {}
    teacher_judge = None
    prov = None
    if DRY_RUN:
        for i, name in enumerate(cands):
            code = None
            if problems:
                langs = {p["suite"]: p["language"] for p in problems}
                suites = {s: {"language": lang, "n": sum(p["suite"] == s for p in problems), "status": {},
                              **{f"pass@{k}": round(0.8 - 0.07 * i, 4) for k in cfg.code_eval_k},
                              **({"fix@1": round(0.9 - 0.07 * i, 4)} if cfg.code_eval_fix else {})}
                          for s, lang in langs.items()}
                code = {"suites": suites, "mean_pass@1": codebench.mean_pass1(suites), "samples_per_problem": 1,
                        **({"mean_fix@1": codebench.mean_fix1(suites)} if cfg.code_eval_fix else {})}
            results[name] = {"agreement": 0.85 - 0.07 * i if inputs else None, "judge": None, "tok_s_vm": None,
                             "code": code, "samples": [], "gpu": {}}
    else:
        job.save_config()
        prov = provenance(cfg)
        job.path("work", "provenance.json").write_text(json.dumps(prov, indent=2))
        if use_judge:
            emit(STAGE, pct=5, msg=f"judging teacher references on {len(inputs)} held-out inputs")
            teacher_judge = judge(spec, cfg.judge_model, inputs, refs)
        n = len(cands)
        for i, (name, c) in enumerate(cands.items()):
            emit(STAGE, pct=10 + 85 * i / n, msg=f"running {name}")
            answers, tps, code, gpu = run_candidate(job, name, c["path"], inputs, problems, system)
            tps = tps or (code or {}).get("tok_s_vm")
            agree = score = None
            if inputs:
                agree = round(sum(agreement(a, r) for a, r in zip(answers, refs)) / max(len(refs), 1), 3)
                emit(STAGE, pct=10 + 85 * (i + 0.6) / n, msg=f"{name}: {agree:.0%} agreement with teacher, {tps} tok/s on the VM")
                score = judge(spec, cfg.judge_model, inputs, answers) if use_judge else None
            if code:
                per = ", ".join(f"{k} {v['pass@1']:.0%}" for k, v in code["suites"].items() if "pass@1" in v)
                emit(STAGE, pct=10 + 85 * (i + 0.9) / n, msg=f"{name}: pass@1 {per}")
            results[name] = {
                "agreement": agree, "judge": score, "tok_s_vm": tps, "code": code, "gpu": gpu,
                "samples": [{"input": a, "output": b, "reference": r} for a, b, r in list(zip(inputs, answers, refs))[:3]],
            }

    # pass@1 when code was run for every model; else judge when every model has
    # one; else agreement (teacher = 1.0 by definition). The teacher itself is
    # not run on code here: an eval-only `ref` job scores it.
    coded = bool(problems) and all(r["code"] and r["code"]["mean_pass@1"] is not None for r in results.values())
    judged = teacher_judge is not None and all(r["judge"] is not None for r in results.values())
    method = "pass@1" if coded else "judge" if judged else "agreement"
    teacher_score = None if coded else teacher_judge if judged else 1.0

    candidates = []
    for name, c in cands.items():
        r = results[name]
        score = r["code"]["mean_pass@1"] if coded else r["judge"] if judged else r["agreement"]
        tok_s_est = c.get("tok_s_est")
        candidates.append({
            "name": name, "size_gb": c["size_gb"], "score": score,
            "agreement": r["agreement"], "judge_score": r["judge"], "code": r["code"],
            "tok_s_est": tok_s_est, "tok_s_vm": r["tok_s_vm"], "gpu": r["gpu"],
            "meets_target": c["size_gb"] <= spec.target.max_size_gb and (tok_s_est or 0) >= spec.target.min_tok_s,
        })
    ref = load_code_ref(job) if coded else {}
    for c in candidates:
        if ref and c["code"]:
            c["code"]["share_of_ref"] = codebench.share_of_ref(c["code"]["suites"], ref)
    eligible = [c for c in candidates if c["meets_target"]] or candidates
    best = max(eligible, key=lambda c: c["score"] if c["score"] is not None else -1)
    # Prefer the compressed MoE when it is within 2 points of the best.
    moe = next((c for c in eligible if c["name"] == "lobbot-moe"), None)
    winner = moe if moe and moe["score"] is not None and moe["score"] >= best["score"] - 0.02 else best

    report = {
        "teacher": {"name": teacher_name, "size_gb": teacher_size_gb(job), "score": teacher_score},
        "candidates": candidates,
        "winner": winner["name"],
        "score_method": method,
        "bit_widths": alloc.get("bit_widths", []),
        "samples": {k: v["samples"] for k, v in results.items()},
        **({"provenance": prov} if prov else {}),
    }
    job.path("out", "eval.json").write_text(json.dumps(report, indent=2))
    job.mark_done(STAGE, {"winner": winner["name"]})
    for name in cands:  # the reading is complete: a later run must not resume from it
        job.path("work", "code_eval", f"{name}.partial.jsonl").unlink(missing_ok=True)
    emit(STAGE, "done", 100, f"winner {winner['name']}: score {winner['score']} vs teacher {teacher_score} ({method})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    raise_on_sigterm()
    run_stage(Job(ap.parse_args().job))
