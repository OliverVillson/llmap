"""The servers the train harness starts: vLLM for the teacher (and the bf16 student on the
B200), llama-server for GGUFs. One at a time per GPU: start() stops the one before and
waits until the GPU's memory is back before launching (run 1 lost steps to a server that
had not let go). A server outlives the step that started it (a pidfile in <work>/servers),
so `teacher-answers` reuses the teacher `teacher-serve` booted, and a stage run alone
starts what it needs.

Sampling is never left to a server's defaults: every request goes through the recording
proxy (proxy.py), which sets top_k, top_p, temperature, min_p and thinking from the
config. The server flags below only size the server.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from bh.exec import Cmd, Exec, dump, load

PORT = 8000


@dataclass
class Spec:
    """A server to start: its kind (vllm or llama), the model path or id, the name it
    serves under, and its command line."""
    kind: str
    model: str
    name: str
    sh: str
    ctx: int = 0
    slots: int = 0

    def same(self, other: dict | None) -> bool:
        return bool(other) and other.get("sh") == self.sh


def vllm_spec(cfg, model: str, name: str, *, extra: tuple[str, ...] = (), seqs: int | None = None) -> Spec:
    """`vllm serve` from the vLLM venv (.env.vm's LOBBOT_VLLM_PY): 16K slots, prefix
    caching (a ticket's write and fix calls share the pack), the top-k log-probs the
    proxy asks for, and the request's own sampling (--generation-config vllm)."""
    args = [model, "--served-model-name", name, "--host", "127.0.0.1", "--port", str(PORT),
            "--max-model-len", str(cfg.slot_ctx), "--max-num-seqs", str(seqs or cfg.vllm_seqs),
            "--gpu-memory-utilization", str(cfg.vllm_gpu_util), "--enable-prefix-caching",
            "--max-logprobs", str(max(cfg.teacher_logprobs, 1)), "--generation-config", "vllm", *extra]
    sh = '"$(dirname "${LOBBOT_VLLM_PY:-python}")/vllm" serve ' + " ".join(shlex.quote(a) for a in args)
    return Spec("vllm", model, name, sh, cfg.slot_ctx, seqs or cfg.vllm_seqs)


def llama_bin(cfg, name: str) -> str:
    return str(Path(cfg.nvme) / "llama.cpp" / "build" / "bin" / name)


def llama_spec(cfg, gguf: str, name: str, *, slots: int, ctx: int | None = None, ub: int | None = None) -> Spec:
    """One llama-server serving `slots` contexts of slot_ctx tokens each from one copy of
    the weights (-np, --kv-unified), q8 KV with flash attention, the GGUF's own chat
    template (--jinja)."""
    ctx = ctx or cfg.slot_ctx
    args = ["-m", gguf, "-a", name, "-ngl", "999", "--host", "127.0.0.1", "--port", str(PORT),
            "-c", str(ctx * slots), "-np", str(slots), "--kv-unified", "--jinja", "-fa", "on",
            "-ctk", cfg.kv_type, "-ctv", cfg.kv_type, "--metrics"]
    if ub:
        args += ["-ub", str(ub)]
    sh = shlex.quote(llama_bin(cfg, "llama-server")) + " " + " ".join(shlex.quote(a) for a in args)
    return Spec("llama", gguf, name, sh, ctx, slots)


def gpu_free_sh(tries: int = 120, every: int = 5, limit_mib: int = 2000) -> str:
    """Shell that waits for the GPU's memory to come back and fails, naming the holders,
    when it does not (centipede.py's)."""
    used = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1"
    return (f"command -v nvidia-smi >/dev/null || exit 0; "
            f"for i in $(seq {tries}); do m=$({used}); [ -n \"$m\" ] && [ \"$m\" -lt {limit_mib} ] && break; "
            f"sleep {every}; done; "
            f"m=$({used}); if [ -z \"$m\" ] || [ \"$m\" -ge {limit_mib} ]; then "
            f"echo \"GPU not free after {tries * every}s: ${{m:-no reading of}} MiB in use. Holders:\"; "
            "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; exit 1; fi")


class Servers:
    def __init__(self, cfg, ex: Exec):
        self.cfg, self.ex = cfg, ex
        self.dir = Path(cfg.work) / "servers"
        self.pidfile = self.dir / "server.json"

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{PORT}"

    def current(self) -> dict | None:
        """The running server we started, or None (a stale pidfile is cleared)."""
        cur = load(self.pidfile)
        if not cur:
            return None
        if self.ex.dry or _alive(cur.get("pid", 0)):
            return cur
        self.pidfile.unlink(missing_ok=True)
        return None

    def ensure(self, step, spec: Spec, health_s: int = 1800) -> dict:
        """spec's server running and healthy: reused when it is the one already up,
        else the old one stopped, the GPU waited for, and spec started."""
        cur = self.current()
        if spec.same(cur) and (self.ex.dry or _healthy(self.url)):
            return cur
        self.stop(step)
        self.ex.run(step, Cmd(gpu_free_sh(), "gpu-free"))
        self.dir.mkdir(parents=True, exist_ok=True)
        log = self.dir / f"{spec.kind}-{spec.name}.log"
        info = {"kind": spec.kind, "model": spec.model, "name": spec.name, "sh": spec.sh, "log": str(log),
                "ctx": spec.ctx, "slots": spec.slots, "started": time.strftime("%F %T")}
        if self.ex.dry:
            self.ex.run(step, Cmd(spec.sh, f"serve-{spec.kind}", {"model": spec.model, "name": spec.name}))
            dump(self.pidfile, {**info, "pid": 0})
            return load(self.pidfile)
        self.ex.commands.append(Cmd(spec.sh, f"serve-{spec.kind}", {"model": spec.model, "name": spec.name}))
        repo = Path(__file__).resolve().parents[2]
        full = f"cd {repo} && {{ [ -f .env.vm ] && source .env.vm; true; }} && exec {spec.sh}"
        with open(log, "a") as f:
            f.write(f"\n===== {time.strftime('%F %T')} $ {spec.sh}\n")
            f.flush()
            proc = subprocess.Popen(["bash", "-c", full], stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        dump(self.pidfile, {**info, "pid": proc.pid})
        deadline = time.monotonic() + health_s
        while not _healthy(self.url):
            if proc.poll() is not None:
                self.pidfile.unlink(missing_ok=True)
                raise RuntimeError(f"{spec.kind} server for {spec.model} exited ({proc.returncode}): "
                                   + _tail(log))
            if time.monotonic() > deadline:
                self.stop(step)
                raise RuntimeError(f"{spec.kind} server for {spec.model} not healthy after {health_s} s: " + _tail(log))
            time.sleep(3)
        return load(self.pidfile)

    def stop(self, step=None) -> None:
        """Stops our server, if any, and waits until the GPU is free."""
        cur = self.current()
        if not cur:
            return
        if not self.ex.dry:
            pid = cur.get("pid", 0)
            for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 15)):
                try:
                    os.killpg(pid, sig)
                except ProcessLookupError:
                    break
                t = time.monotonic()
                while _alive(pid) and time.monotonic() - t < wait:
                    time.sleep(1)
                if not _alive(pid):
                    break
        self.ex.commands.append(Cmd(f"# stop {cur.get('kind')} {cur.get('model')}", "serve-stop"))
        self.pidfile.unlink(missing_ok=True)
        self.ex.run(step, Cmd(gpu_free_sh(), "gpu-free"))


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def _tail(log: Path, n: int = 12) -> str:
    try:
        return "\n".join(log.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return f"(no log at {log})"


def get_json(url: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def post_json(url: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
