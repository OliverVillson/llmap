#!/usr/bin/env python3
"""Aider Polyglot: score a model dir with aider's own benchmark, and heal data in aider's format.

    python scripts/polyglot.py setup         # docker, pinned aider + polyglot-benchmark, the image
    python scripts/polyglot.py selftest --out DIR      # no model: reference solutions through the tests
    python scripts/polyglot.py run --model-dir DIR --name NAME     # serve, benchmark, <run>/out/polyglot.json
    python scripts/polyglot.py run --model-dir DIR --name NAME --sample-per-language 1   # 6 exercises
    python scripts/polyglot.py exercism      # training exercises: Exercism minus every Polyglot slug
    python scripts/polyglot.py run --model-dir TEACHER --name teacher-ex --out teacher-ex.json \
        --exercises $NVME/polyglot/exercism --log teacher-ex.requests.jsonl
    python scripts/polyglot.py rows --results teacher-ex.json --log teacher-ex.requests.jsonl --out rows.jsonl

Polyglot is aider's benchmark: 225 Exercism exercises in C++, Go, Java, JavaScript, Python
and Rust. The model edits the exercise's stub files in aider's edit format, the tests run,
and after a failure it gets a second try with the test output; pass_rate_2 is the headline.
`run` serves the model dir with stages.eval.serve (vLLM) and runs benchmark/benchmark.py from
a pinned aider inside aider's Docker image, which has every toolchain the tests need, on the
host network. Thinking, sampling (Qwen's thinking settings by default; Gemma 4 takes
--temperature 1.0 --top-k 64 --reasoning-parser gemma4) and limits go in through aider's
model settings; the results JSON is aider's --stats as fractions, per language too, with the
GPU's busy share, and per exercise ("cases": {"<lang>/<exercise>": [passed by try 1, by try 2]})
for comparing two runs exercise by exercise.
A rerun resumes: aider keeps each finished exercise's .aider.results.json.
--sample-per-language K runs the first K exercises (by name) of each language, a pre-flight
check of serving and parsing in minutes, in its own run dir (runs/NAME-sampleK).

`selftest` needs no model: the first N exercises of each language with their reference solution
(Exercism's .meta example files) in place of the stub go through aider's benchmark with
--no-aider, which runs each exercise's tests once as a real run does (same image, threads,
caches and test timeout). A broken toolchain, no network for Rust crates or Gradle, or a test
timeout too short for the machine fails it in minutes, and it warms the Gradle and cargo caches.

Heal data: `exercism` builds a Polyglot-layout dir from the Exercism tracks at the commits
Polyglot was copied from, without any slug Polyglot has in any language. A teacher `run` on
it with --log records every chat completion through a logging proxy, and `rows` keeps the
passing transcripts as chat rows for Config.data_extra_rows: a first-try pass as its last
exchange, a second-try pass as the whole history (failed answer, test output, fix).

Under $NVME/polyglot: aider/, polyglot-benchmark/, exercism-src/<lang>/, exercism/, gradle/
and cargo/ (the Java and Rust tests' caches), runs/<name>/ (a job dir: work/ logs, bench/
aider's output, out/).
"""

from __future__ import annotations

import argparse
import grp
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from common.progress import emit  # noqa: E402
from stages import eval as ev  # noqa: E402
from stages._util import Job, read_jsonl, run, write_jsonl  # noqa: E402
from stages.data import _chat_row  # noqa: E402

STAGE = "polyglot"
NVME = Path(os.environ.get("NVME", "/mnt/nvme"))
HOME = NVME / "polyglot"
AIDER, POLYGLOT, RUNS = HOME / "aider", HOME / "polyglot-benchmark", HOME / "runs"
EXERCISM_SRC, EXERCISM = HOME / "exercism-src", HOME / "exercism"
AIDER_COMMIT = "5dc9490bb35f9729ef2c95d00a19ccd30c26339c"  # main on 2026-05-22 (0.86.3.dev, litellm 1.82.3)
POLYGLOT_COMMIT = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"  # 2024-12-22
# The Exercism commits Polyglot's exercises were copied from: byte for byte the same files but
# .gitignore. Their build files are the ones aider's image runs (Jest and Babel from its
# /npm-install, the Gradle wrapper, Catch2 v2, go 1.18); Exercism has moved on since (ESLint
# flat config, jest.config.js, Catch2 v3, a newer Gradle).
TRACKS = {"python": "7bec634f5c51cce82d233ad88f7ae81a3e98242a", "go": "26635e19868f0b4fbed09a986b824cd67efdbc2e",
          "java": "1123e5b3dddeb9e57b802835285a886b4428bfcc", "javascript": "dbfbeae34dd4fce7fad7afd740d075d33cc51b21",
          "rust": "6c321382019fa969401b8e923a5e12ff69065561", "cpp": "acc32555eab077cf786f892b6ab27a22184d81c7"}
IMAGE = f"aider-benchmark:{AIDER_COMMIT[:12]}"
# benchmark.py run_unit_tests: the test files' extension picks the command, which needs these.
TEST_EXTS = {".py", ".rs", ".go", ".js", ".cpp", ".java"}
BUILD = {"go": "go.mod", "java": "gradlew", "javascript": "package.json", "rust": "Cargo.toml", "cpp": "CMakeLists.txt"}

MAX_TOKENS = 32768
# aider caps the chat history at a 16th of the model's context (8k at most) and summarizes
# what is over; leaderboard models get the 8k. The model is served with this much context
# (or its own maximum) and aider is told so (model-metadata.json).
CTX = 131072
# Per request. aider's default is 600 s, and 32k tokens of thinking with 63 other requests in
# flight can take far longer; a timed-out request is aborted and asked again from the start.
TIMEOUT = 4 * 3600
MEMORY = "64g"  # the container's cap: 64 exercises building at once (Gradle, rustc, Catch2)
# benchmark.py run from aider's README, after registering the model's context with aider as
# `aider --model-metadata-file` would (the benchmark has no such option). Without it aider takes
# the context as unknown and summarizes the chat once it passes 1k tokens.
BOOT = ("import sys; sys.path.insert(0, '/aider/benchmark'); from aider import models; "
        "models.register_litellm_models(['/benchmarks/model-metadata.json']); "
        "import benchmark; sys.argv[0] = 'benchmark.py'; benchmark.app()")
TEST_TIMEOUT = 180  # benchmark.py run_unit_tests: seconds per test run, fixed in aider
# The selftest's benchmark.py, with each exercise's test run timed into .selftest.seconds.
SELFTEST_BOOT = ("import sys, time; sys.path.insert(0, '/aider/benchmark'); import benchmark\n"
                 "def timed(original, testdir, history, files, tests=benchmark.run_unit_tests):\n"
                 "    start = time.time()\n"
                 "    try:\n"
                 "        return tests(original, testdir, history, files)\n"
                 "    finally:\n"
                 "        (testdir / '.selftest.seconds').write_text(str(round(time.time() - start, 1)))\n"
                 "benchmark.run_unit_tests = timed; sys.argv[0] = 'benchmark.py'; benchmark.app()")
# The harness's own words (aider benchmark/prompts.py): the first user message is the
# exercise's docs, then ADDENDUM and the file list; the second try's ends in RETRY.
ADDENDUM = "\n####\n\nUse the above instructions to modify the supplied files: "
RETRY = "\n####\n\nSee the testing errors above."


# ---------- setup ----------


def docker() -> list[str]:
    """docker, through sudo until a new login picks up the docker group."""
    return ["docker"] if os.access("/var/run/docker.sock", os.R_OK | os.W_OK) else ["sudo", "docker"]


def clone(url: str, dest: Path, commit: str) -> None:
    """dest checked out at commit; a checkout at another commit is moved to it."""
    if not (dest / ".git").exists():
        run(["git", "clone", "-q", url, str(dest)], STAGE)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != commit:
        if subprocess.run(["git", "-C", str(dest), "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True).returncode:
            run(["git", "-C", str(dest), "fetch", "-q", "origin"], STAGE)
        run(["git", "-C", str(dest), "checkout", "-q", commit], STAGE)


def install_docker() -> None:
    """Ubuntu's docker.io, its images on the NVMe (the root disk is 14 GB and aider's image
    several), and this user in the docker group for later logins."""
    if not shutil.which("docker"):
        run(["sudo", "apt-get", "update", "-qq"], STAGE)
        run(["sudo", "apt-get", "install", "-y", "-qq", "docker.io"], STAGE)
    daemon = Path("/etc/docker/daemon.json")
    conf = json.loads(daemon.read_text()) if daemon.exists() else {}
    if conf.get("data-root") != str(NVME / "docker"):
        conf["data-root"] = str(NVME / "docker")
        subprocess.run(["sudo", "mkdir", "-p", str(daemon.parent)], check=True)
        subprocess.run(["sudo", "tee", str(daemon)], input=json.dumps(conf, indent=2), text=True, check=True,
                       stdout=subprocess.DEVNULL)
        run(["sudo", "systemctl", "restart", "docker"], STAGE)
    run(["sudo", "systemctl", "enable", "--now", "docker"], STAGE)
    user = os.environ.get("USER", "root")
    if user != "root" and user not in grp.getgrnam("docker").gr_mem:
        run(["sudo", "usermod", "-aG", "docker", user], STAGE)


def image_exists() -> bool:
    return subprocess.run([*docker(), "image", "inspect", IMAGE], capture_output=True).returncode == 0


def setup(a) -> None:
    install_docker()
    clone("https://github.com/Aider-AI/aider", AIDER, AIDER_COMMIT)
    clone("https://github.com/Aider-AI/polyglot-benchmark", POLYGLOT, POLYGLOT_COMMIT)
    if not image_exists():  # benchmark/docker_build.sh, tagged with the pin
        run([*docker(), "build", "--file", "benchmark/Dockerfile", "-t", IMAGE, "."], STAGE, cwd=AIDER)
    emit(STAGE, "done", 100, f"aider {AIDER_COMMIT[:7]}, polyglot-benchmark {POLYGLOT_COMMIT[:7]}, image {IMAGE}")


# ---------- run ----------


def context(model_dir: str) -> int:
    """CTX, or the model's own maximum when that is smaller."""
    hf = json.loads((Path(model_dir) / "config.json").read_text())
    most = hf.get("max_position_embeddings") or (hf.get("text_config") or {}).get("max_position_embeddings")
    return min(CTX, most) if most else CTX


def model_settings(name: str, edit_format: str, temperature: float = 0.6, top_p: float = 0.95,
                   top_k: int = 20) -> list[dict]:
    """aider's model settings (--read-model-settings), sent with every request. litellm drops
    what OpenAI's API lacks (aider sets drop_params), so top_k and the chat template's thinking
    switch (Qwen's and Gemma 4's) go in extra_body; `timeout` replaces aider's 600 s request
    timeout (models.send_completion). The reasoning parser keeps the thinking out of the
    content; reasoning_tag strips a <think> that gets through anyway."""
    return [{"name": f"openai/{name}", "edit_format": edit_format, "use_temperature": float(temperature),
             "reasoning_tag": "think",
             "extra_params": {"max_tokens": MAX_TOKENS, "top_p": top_p, "timeout": TIMEOUT,
                              "extra_body": {"top_k": top_k, "chat_template_kwargs": {"enable_thinking": True}}}}]


def model_metadata(name: str, ctx: int) -> dict:
    return {f"openai/{name}": {"max_input_tokens": ctx, "max_tokens": MAX_TOKENS, "max_output_tokens": MAX_TOKENS,
                               "input_cost_per_token": 0, "output_cost_per_token": 0,
                               "litellm_provider": "openai", "mode": "chat"}}


def container(name: str) -> str:
    return "polyglot-" + re.sub(r"[^A-Za-z0-9_.-]", "-", name)


def docker_run(name: str, run_dir: Path, exercises: Path, inner: str, env: dict[str, str], gradle: Path,
               cargo: Path, prefix: tuple[str, ...]) -> list[str]:
    """`inner` in aider's image, as benchmark/docker.sh starts it but on the host network:
    run_dir at /benchmarks, the exercises read-only at /exercises, and the Gradle and cargo
    caches kept on the host across runs."""
    return [*prefix, "run", "--rm", "--name", container(name), "--network", "host",
            "--memory", MEMORY, "--memory-swap", MEMORY,
            "-v", f"{run_dir}:/benchmarks", "-v", f"{exercises}:/exercises:ro", "-v", f"{gradle}:/root/.gradle",
            "-v", f"{cargo}:/root/.cargo/registry",
            "-e", "AIDER_DOCKER=1", "-e", "AIDER_BENCHMARK_DIR=/benchmarks",
            *(x for k, v in env.items() for x in ("-e", f"{k}={v}")),
            IMAGE, "bash", "-c", inner]


def docker_command(name: str, run_dir: Path, exercises: Path, api_base: str, edit_format: str = "diff",
                   threads: int = 64, num_tests: int = 0, owner: str = "0:0", gradle: Path = HOME / "gradle",
                   prefix: tuple[str, ...] = ("docker",), cargo: Path = HOME / "cargo") -> list[str]:
    """aider's benchmark in its image (docker_run), so OPENAI_API_BASE reaches vLLM (or the
    proxy) on 127.0.0.1. aider writes bench/ under run_dir, and the files it wrote as root are
    handed to `owner` at the end."""
    bench = ["/benchmarks/bench", "--model", f"openai/{name}", "--edit-format", edit_format,
             "--threads", str(threads), "--exercises-dir", "/exercises",
             "--read-model-settings", "/benchmarks/model-settings.yml",
             *(["--num-tests", str(num_tests)] if num_tests > 0 else [])]
    inner = f"python3 -c \"{BOOT}\" {' '.join(bench)}; rc=$?; chown -R {owner} /benchmarks; exit $rc"
    return docker_run(name, run_dir, exercises, inner, {"OPENAI_API_BASE": api_base, "OPENAI_API_KEY": "local"},
                      gradle, cargo, prefix)


def selftest_command(out: Path, exercises: Path, threads: int = 64, owner: str = "0:0",
                     gradle: Path = HOME / "gradle", cargo: Path = HOME / "cargo",
                     prefix: tuple[str, ...] = ("docker",)) -> list[str]:
    """aider's benchmark with --no-aider (docker_run): each exercise's files as they are, then
    its tests, one try, into a fresh out/bench."""
    bench = ["/benchmarks/bench", "--no-aider", "--tries", "1", "--threads", str(threads),
             "--exercises-dir", "/exercises"]
    inner = (f"rm -rf /benchmarks/bench; python3 -c \"{SELFTEST_BOOT}\" {' '.join(bench)}; rc=$?; "
             f"chown -R {owner} /benchmarks; exit $rc")
    return docker_run("selftest", out, exercises, inner, {"OPENAI_API_KEY": "local"}, gradle, cargo, prefix)


def make_caches() -> None:
    """The Gradle and cargo cache dirs, made here so docker does not make them as root."""
    for d in (HOME / "gradle", HOME / "cargo"):
        d.mkdir(parents=True, exist_ok=True)


def passed(r: dict, tries: int) -> bool:
    """aider's pass within `tries` tries: the last test run passed, and it was one of the first `tries`."""
    o = r.get("tests_outcomes") or []
    return bool(o and o[-1] and len(o) <= tries)


def summarize(bench: Path) -> dict:
    """aider's --stats (benchmark.py summarize_results) for one run, rates as fractions, plus
    pass rates per language and each exercise's pass by try 1 and 2 (cases). Rates are over
    the exercises that finished (n), as aider's are; total counts every exercise in the run."""
    done = []
    for f in sorted(bench.glob("*/exercises/practice/*/.aider.results.json")):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:  # being written
            continue
        if r:
            lang, _, _, slug, _ = f.relative_to(bench).parts
            done.append((lang, slug, r))

    def rate(rs: list[dict], tries: int) -> float | None:
        ok = [passed(r, tries) for r in rs]
        return round(sum(ok) / len(ok), 4) if ok else None

    rs = [r for _, _, r in done]
    total = lambda k: sum(r.get(k) or 0 for r in rs)  # noqa: E731
    langs = {lang: [r for l2, _, r in done if l2 == lang] for lang in sorted({lang for lang, _, _ in done})}
    return {
        "pass_rate_1": rate(rs, 1), "pass_rate_2": rate(rs, 2),
        "percent_cases_well_formed": round(1 - sum(bool(r.get("num_malformed_responses")) for r in rs) / len(rs), 4)
        if rs else None,
        "per_language": {lang: {"n": len(v), "pass_rate_1": rate(v, 1), "pass_rate_2": rate(v, 2)}
                         for lang, v in langs.items()},
        "n": len(rs), "total": sum(1 for d in bench.glob("*/exercises/practice/*") if d.is_dir()),
        "errors": sum("exception" in r for r in rs),  # the harness crashed on the exercise
        "test_timeouts": total("test_timeouts"), "exhausted_context_windows": total("num_exhausted_context_windows"),
        "error_outputs": total("num_error_outputs"), "malformed_responses": total("num_malformed_responses"),
        "prompt_tokens": total("prompt_tokens"), "completion_tokens": total("completion_tokens"),
        "seconds_per_case": round(total("duration") / len(rs), 1) if rs else None,
        "cases": {f"{lang}/{slug}": [passed(r, 1), passed(r, 2)] for lang, slug, r in done},
    }


# Hop-by-hop headers, and the ones httpx sets itself; no Accept-Encoding, so answers come plain.
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
       "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding"}
SAMPLING = ("temperature", "top_p", "top_k", "max_tokens", "chat_template_kwargs")


def log_entry(request: dict, response: dict, seconds: float) -> dict:
    """One logged chat completion: what was asked and what came back."""
    choice = (response.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return {"time": round(time.time(), 3), "seconds": round(seconds, 1), "model": request.get("model"),
            "messages": request.get("messages"), "sampling": {k: request[k] for k in SAMPLING if k in request},
            "reasoning_content": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "content": msg.get("content") or "", "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage")}


class _ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # a line per request is noise at 64 threads
        pass

    def do_GET(self) -> None:
        import httpx

        p = self.server
        if not self.path.startswith("/v1/"):
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        start, got, sent = time.monotonic(), [], False
        try:
            with p.client.stream(self.command, p.upstream + self.path, content=body, headers=headers) as r:
                self.send_response(r.status_code)
                for k, v in r.headers.items():
                    if k.lower() not in HOP - {"content-length"}:
                        self.send_header(k, v)
                self.end_headers()
                sent = True
                for chunk in r.iter_raw():  # as it arrives, so a streamed answer streams
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    got.append(chunk)
        except httpx.HTTPError as e:
            if not sent:
                self.send_error(502, str(e)[:200])
            return
        except (BrokenPipeError, ConnectionResetError):
            return  # the client gave up (its timeout) and will ask again; not logged
        if r.status_code == 200 and self.path.endswith("/chat/completions"):
            p.record(body, b"".join(got), time.monotonic() - start)

    do_POST = do_GET


class LoggingProxy(ThreadingHTTPServer):
    """Forwards /v1/* to upstream (vLLM) and appends one log_entry() line per chat completion
    to log. A streamed answer is relayed but not logged; the benchmark asks unstreamed
    (Coder.create(stream=False)). Serves on 127.0.0.1:port (0: a free one, see .url) in a
    background thread until close()."""

    daemon_threads = True
    request_queue_size = 256

    def __init__(self, upstream: str, log: Path, port: int = 0):
        import httpx

        super().__init__(("127.0.0.1", port), _ProxyHandler)
        self.upstream, self.log, self.lock = upstream.rstrip("/"), Path(log), threading.Lock()
        self.client = httpx.Client(timeout=httpx.Timeout(TIMEOUT + 60, connect=30),
                                   limits=httpx.Limits(max_connections=None, max_keepalive_connections=256))
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def record(self, request: bytes, response: bytes, seconds: float) -> None:
        try:
            line = json.dumps(log_entry(json.loads(request), json.loads(response), seconds), ensure_ascii=False)
        except (json.JSONDecodeError, UnicodeDecodeError):  # a streamed answer (server-sent events)
            return
        with self.lock, self.log.open("a") as f:
            f.write(line + "\n")

    def close(self) -> None:
        self.shutdown()
        self.server_close()
        self.client.close()


def default_threads() -> int:
    """Exercises at once: 4 per CPU, as each runs its test build on the CPUs, at most 64."""
    return min(64, 4 * (os.cpu_count() or 16))


def show_threads(threads: int) -> None:
    print(f"[{STAGE}] {os.cpu_count()} CPUs, {threads} threads", flush=True)


def follow(proc, bench: Path, total: int, label: str, check=lambda: None) -> None:
    """Wait for proc, a progress line whenever another exercise finishes; check() may raise."""
    shown = -1
    while proc.poll() is None:
        check()
        done = sum(1 for _ in bench.glob("*/exercises/practice/*/.aider.results.json"))
        if done != shown:
            emit(STAGE, msg=f"{label}: {done}/{total} exercises", answered=done, total=total)
            shown = done
        time.sleep(10)


def pick(root: Path, k: int, keep=lambda ex: True) -> list[Path]:
    """The first k exercises (by name) of each language under root that keep() takes."""
    return [ex for practice in sorted(root.glob("*/exercises/practice"))
            for ex in [d for d in sorted(practice.iterdir()) if d.is_dir() and keep(d)][:k]]


def case(ex: Path) -> str:
    """<lang>/<exercise>, the key of summarize()'s cases."""
    return f"{ex.parts[-4]}/{ex.name}"


def copy_exercises(picked: list[Path], dest: Path, solve: bool = False) -> Path:
    """A fresh dest in Polyglot's layout holding the picked exercises; with solve, each with its
    reference solution (reference()) in place of the stub."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for ex in picked:
        to = dest / ex.parts[-4] / "exercises" / "practice" / ex.name
        shutil.copytree(ex, to)  # copy2: gradlew stays executable
        for f, src in (reference(ex) if solve else {}).items():
            (to / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, to / f)
    return dest


def run_bench(a) -> int:
    sample = f"-sample{a.sample_per_language}" if a.sample_per_language > 0 else ""
    run_dir = Path(a.dir or RUNS / (a.name + sample))
    out = Path(a.out or run_dir / "out" / "polyglot.json")
    if out.exists() and not a.force:
        print(f"[{STAGE}] {out} exists; skipping {a.name} (--force runs it again)", flush=True)
        return 0
    source = exercises = Path(a.exercises).resolve()
    if not any(d.is_dir() for d in source.glob("*/exercises/practice/*")):
        raise SystemExit(f"no exercises under {source}")
    if not ev.is_checkpoint(a.model_dir):
        raise SystemExit(f"{a.model_dir} is not a checkpoint dir (no config.json)")
    if not image_exists():
        raise SystemExit(f"no docker image {IMAGE}; run `python scripts/polyglot.py setup` first")
    if sample:  # a copy, as aider reads /exercises in the container, where links out of it break
        exercises = copy_exercises(pick(source, a.sample_per_language), (run_dir / "sample").resolve())
    total = sum(1 for d in exercises.glob("*/exercises/practice/*") if d.is_dir())
    total = min(total, a.num_tests) if a.num_tests > 0 else total

    job = Job(run_dir)  # serve() reads its Config (a config.json there can add eval_vllm_args) and logs to work/
    job.config.eval_reasoning_parser, job.config.eval_experts_used = a.reasoning_parser, a.experts_used
    os.environ.setdefault("LOBBOT_EVAL_CTX", str(context(a.model_dir)))
    settings = model_settings(a.name, a.edit_format, a.temperature, a.top_p, a.top_k)
    (run_dir / "model-settings.yml").write_text(json.dumps(settings, indent=2))
    (run_dir / "model-metadata.json").write_text(json.dumps(model_metadata(a.name, ev.limits(job.config)[1]), indent=2))
    make_caches()
    subprocess.run([*docker(), "rm", "-f", container(a.name)], capture_output=True)  # left over from a stopped run
    show_threads(a.threads)

    emit(STAGE, msg=f"{a.name}: serving {a.model_dir}")
    server = ev.serve(job, a.model_dir, a.name)
    end_gpu, proxy, start = ev.watch_gpu(), None, time.monotonic()
    bench, log_path = run_dir / "bench", job.path("work", "aider.log")

    def vllm_up() -> None:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited during the benchmark; see {job.path('work', f'vllm-{a.name}.log')}")

    try:
        if a.log:
            Path(a.log).parent.mkdir(parents=True, exist_ok=True)
            proxy = LoggingProxy(f"http://127.0.0.1:{ev.PORT}", Path(a.log))
        api = (proxy.url if proxy else f"http://127.0.0.1:{ev.PORT}") + "/v1"
        cmd = docker_command(a.name, run_dir.resolve(), exercises, api, a.edit_format, a.threads, a.num_tests,
                             f"{os.getuid()}:{os.getgid()}", prefix=tuple(docker()))
        with open(log_path, "a") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            follow(proc, bench, total, a.name, vllm_up)
        if proc.returncode:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-15:])
            raise RuntimeError(f"aider's benchmark exited with {proc.returncode}; see {log_path}:\n{tail}")
    finally:
        seconds = time.monotonic() - start
        subprocess.run([*docker(), "rm", "-f", container(a.name)], capture_output=True)
        gpu = end_gpu()
        if proxy:
            proxy.close()
        ev.stop(server)

    res = {"name": a.name, "model_dir": str(a.model_dir), "run_dir": str(run_dir.resolve()),
           "exercises": str(source), "sample_per_language": a.sample_per_language, "edit_format": a.edit_format,
           "sampling": {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k},
           "experts_used": a.experts_used, "reasoning_parser": a.reasoning_parser,
           **summarize(bench), "seconds": round(seconds, 1), "gpu": gpu}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    emit(STAGE, "done", 100, f"{a.name}: pass_rate_2 {res['pass_rate_2']}, pass_rate_1 {res['pass_rate_1']} "
                             f"on {res['n']}/{res['total']} exercises; {out}")
    return 0


# ---------- selftest ----------


def reference(ex: Path) -> dict[str, Path]:
    """{file in ex: the reference file that goes there}, from .meta/config.json: each example
    file over the solution file of its name, else over the one solution file of its extension,
    else beside that file under its own name (Java's helper classes); Rust's
    .meta/Cargo-example.toml (the crates the reference uses) over Cargo.toml. {} when there is
    no example, or one is missing or has no single place to go."""
    try:
        files = json.loads((ex / ".meta/config.json").read_text()).get("files") or {}
    except (OSError, json.JSONDecodeError):
        return {}
    solution, example = files.get("solution") or [], files.get("example") or []
    names = {Path(e).name for e in example}
    placed = {}
    for e in example:
        same = [Path(s) for s in solution if Path(s).suffix == Path(e).suffix]
        named = [s for s in same if s.name == Path(e).name]
        if named:
            to = named[0]
        elif len(same) == 1:
            to = same[0] if same[0].name not in names else same[0].parent / Path(e).name
        else:
            return {}
        if not (ex / e).is_file() or str(to) in placed:
            return {}
        placed[str(to)] = ex / e
    if placed and (ex / ".meta/Cargo-example.toml").is_file():
        placed["Cargo.toml"] = ex / ".meta/Cargo-example.toml"
    return placed


def failure_output(d: Path, r: dict, lines: int = 30) -> str:
    """The end of why an exercise did not pass: its test output, which benchmark.py appends to
    the chat history, or the harness's exception."""
    history = d / ".aider.chat.history.md"
    if not r:
        text = "no results; see selftest.log"
    elif r.get("exception"):
        text = r["exception"]
    elif r.get("test_timeouts"):
        text = f"the tests ran over {TEST_TIMEOUT} s"
    else:
        text = history.read_text(errors="replace") if history.is_file() else ""
    return "\n".join([x for x in text.splitlines() if x.strip() != "```"][-lines:])


def selftest_results(bench: Path, names: list[str]) -> tuple[dict, dict]:
    """Per language {passed, failed, timed_out, seconds: its slowest test run}, and per exercise
    that did not pass whether it timed out, its test run's seconds and the end of its output."""
    per, failing = {}, {}
    for name in names:
        lang, slug = name.split("/")
        d = bench / lang / "exercises" / "practice" / slug
        try:
            r = json.loads((d / ".aider.results.json").read_text())
        except (OSError, json.JSONDecodeError):
            r = {}
        try:
            seconds = float((d / ".selftest.seconds").read_text())
        except (OSError, ValueError):
            seconds = None
        timed_out = bool(r.get("test_timeouts"))
        kind = "passed" if passed(r, 1) else "timed_out" if timed_out else "failed"
        s = per.setdefault(lang, {"passed": 0, "failed": 0, "timed_out": 0, "seconds": 0.0})
        s[kind] += 1
        s["seconds"] = max(s["seconds"], seconds or 0.0)
        if kind != "passed":
            failing[name] = {"timed_out": timed_out, "seconds": seconds, "output": failure_output(d, r)}
    return per, failing


def selftest(a) -> int:
    out, source = Path(a.out).resolve(), Path(a.exercises).resolve()
    picked = pick(source, a.per_language, lambda ex: bool(reference(ex)))
    if not picked:
        raise SystemExit(f"no exercises with a reference solution under {source}")
    if not image_exists():
        raise SystemExit(f"no docker image {IMAGE}; run `python scripts/polyglot.py setup` first")
    show_threads(a.threads)
    exercises = copy_exercises(picked, out / "exercises", solve=True)
    shutil.rmtree(out / "bench", ignore_errors=True)  # what a stopped run left as root, the container clears
    make_caches()
    names = [case(ex) for ex in picked]
    cmd = selftest_command(out, exercises, a.threads, f"{os.getuid()}:{os.getgid()}", prefix=tuple(docker()))
    subprocess.run([*docker(), "rm", "-f", container("selftest")], capture_output=True)
    emit(STAGE, msg=f"selftest: {len(names)} reference solutions through their tests")
    start = time.monotonic()
    try:
        with open(out / "selftest.log", "w") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            follow(proc, out / "bench", len(names), "selftest")
    finally:
        subprocess.run([*docker(), "rm", "-f", container("selftest")], capture_output=True)

    per, failing = selftest_results(out / "bench", names)
    ok = not failing and not proc.returncode
    res = {"exercises": str(source), "exercises_per_language": a.per_language, "n": len(names),
           "cpus": os.cpu_count(), "threads": a.threads, "test_timeout": TEST_TIMEOUT, "exit_code": proc.returncode,
           "seconds": round(time.monotonic() - start, 1), "ok": ok, "per_language": per, "failing": failing}
    (out / "selftest.json").write_text(json.dumps(res, indent=2) + "\n")
    for lang, s in per.items():
        print(f"[{STAGE}] {lang:10} {s['passed']:3} passed {s['failed']:3} failed {s['timed_out']:3} timed out; "
              f"slowest test run {s['seconds']:.0f} s", flush=True)
    for name, f in failing.items():
        print(f"[{STAGE}] {'timed out' if f['timed_out'] else 'failed'}: {name}", flush=True)
        for line in f["output"].splitlines()[-5:]:
            print(f"[{STAGE}]     {line}", flush=True)
    good = sum(s["passed"] for s in per.values())
    emit(STAGE, "done" if ok else "error", 100,
         f"selftest: {good}/{len(names)} reference solutions pass"
         f"{'' if proc.returncode == 0 else f', benchmark.py exited with {proc.returncode}'}; {out / 'selftest.json'}")
    return 0 if ok else 1


# ---------- training exercises ----------


def solution_files(ex: Path) -> list[str]:
    """The files the harness gives the model to edit (benchmark.py run_test_real)."""
    files = json.loads((ex / ".meta/config.json").read_text()).get("files") or {}
    ignore = {"CMakeLists.txt", "Cargo.toml", *(files.get("test") or []), *(files.get("example") or [])}
    return [f for f in files.get("solution") or []
            if f not in ignore and not f.startswith((".meta/", ".docs/")) and (ex / f).is_file()]


def why_skip(lang: str, ex: Path, status: str | None, taken: set[str]) -> str:
    """"" when the exercise may train and aider's harness can run it, else why not."""
    if ex.name in taken:
        return "in_polyglot"
    if status in (None, "deprecated"):
        return "deprecated" if status else "not_in_track"
    try:
        files = json.loads((ex / ".meta/config.json").read_text()).get("files") or {}
    except (OSError, json.JSONDecodeError):
        return "no_config"
    tests = files.get("test") or []
    if not (ex / ".docs/instructions.md").is_file():  # the harness reads it unconditionally
        return "no_instructions"
    if not solution_files(ex):
        return "no_solution"
    if not tests or not all((ex / t).is_file() for t in tests):
        return "no_tests"
    if not {Path(t).suffix for t in tests} & TEST_EXTS:
        return "no_test_command"
    if lang in BUILD and not (ex / BUILD[lang]).is_file() or lang == "java" and not os.access(ex / "gradlew", os.X_OK):
        return "no_build_file"
    return ""


def build_exercism(tracks: dict[str, Path], polyglot: Path, out: Path) -> dict:
    """out/<lang>/exercises/practice/<slug> from each track's practice exercises, as Polyglot
    lays them out, without every slug Polyglot has in any language (so no Polyglot problem
    trains in another language), deprecated ones, and ones the harness cannot run. Returns
    {lang: {"kept", "dropped": {why: n}}}, also written to out/counts.json; out/.done marks a
    finished build."""
    taken = {d.name for d in polyglot.glob("*/exercises/practice/*") if d.is_dir()}
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    counts = {}
    for lang, track in tracks.items():
        status = {e["slug"]: e.get("status", "active")
                  for e in json.loads((track / "config.json").read_text())["exercises"]["practice"]}
        kept, dropped = 0, Counter()
        for ex in sorted(d for d in (track / "exercises" / "practice").iterdir() if d.is_dir()):
            why = why_skip(lang, ex, status.get(ex.name), taken)
            if why:
                dropped[why] += 1
                continue
            shutil.copytree(ex, out / lang / "exercises" / "practice" / ex.name,
                            ignore=shutil.ignore_patterns(".gitignore"))  # copy2: gradlew stays executable
            kept += 1
        counts[lang] = {"kept": kept, "dropped": dict(dropped)}
    (out / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    (out / ".done").touch()
    return counts


def exercism(a) -> None:
    clone("https://github.com/Aider-AI/polyglot-benchmark", POLYGLOT, POLYGLOT_COMMIT)
    for lang, commit in TRACKS.items():
        clone(f"https://github.com/exercism/{lang}", EXERCISM_SRC / lang, commit)
    out = Path(a.out)
    counts = build_exercism({lang: EXERCISM_SRC / lang for lang in TRACKS}, POLYGLOT, out)
    for lang, c in counts.items():
        print(f"[{STAGE}] {lang:10} {c['kept']:4} exercises  (dropped {c['dropped']})", flush=True)
    emit(STAGE, "done", 100, f"{sum(c['kept'] for c in counts.values())} training exercises in {out}")


# ---------- rows ----------


def instructions(ex: Path) -> str:
    """The exercise text the harness sends ahead of ADDENDUM."""
    docs = ex / ".docs"
    return "".join((docs / f).read_text() for f in ("introduction.md", "instructions.md", "instructions.append.md")
                   if (docs / f).is_file())


def exercise_index(root: Path) -> dict[str, list[tuple[tuple[str, str], frozenset]]]:
    """{exercise text: [((lang, slug), its solution file names)]} for the exercises under root."""
    index = defaultdict(list)
    for ex in sorted(root.glob("*/exercises/practice/*")):
        if (ex / ".docs" / "instructions.md").is_file() and (ex / ".meta" / "config.json").is_file():
            index[instructions(ex)].append(((ex.parts[-4], ex.name), frozenset(Path(f).name for f in solution_files(ex))))
    return index


def match(messages: list[dict], index: dict) -> tuple[tuple[str, str] | None, int]:
    """(lang, slug) of the exercise a logged request is about, or None, and its try: 2 once the
    test output is in. The exercise is the first user message that is an exercise's text plus
    the harness's addendum with its files (the text alone can repeat across languages)."""
    key = None
    for m in messages:
        text = m.get("content") if m.get("role") == "user" else None
        head, sep, tail = text.partition(ADDENDUM) if isinstance(text, str) else ("", "", "")
        if sep and head in index:
            names = frozenset(tail.split("\n", 1)[0].split())
            hits = [k for k, n in index[head] if n == names]
            if len(hits) == 1:
                key = hits[0]
                break
    retry = any(m.get("role") == "user" and RETRY in str(m.get("content")) for m in messages)
    return key, 2 if retry else 1


def rows_from_run(bench: Path, log: Path, max_chars: int = 0) -> tuple[list[dict], dict]:
    """Chat rows from a logged run: per exercise that passed, the last logged request of the
    passing try (its whole history: aider's prompt, any format or lint retries, and on a second
    try the failed answer and the test output) with the answer as the assistant message. Dropped:
    exercises that never passed, tries not in the log, answers cut at max_tokens (finish
    "length"), thinking that never closed, empty answers, and rows over max_chars (0: no cap).
    Each row's "meta" names its source, language and exercise; training ignores it."""
    index = exercise_index(bench)
    last: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    entries = read_jsonl(log)
    unmatched = 0
    for e in entries:  # in the order they finished; a try's later requests carry its earlier ones
        key, attempt = match(e.get("messages") or [], index)
        if key:
            last[key][attempt] = e
        else:
            unmatched += 1
    kept, per = [], {}
    for f in sorted(bench.glob("*/exercises/practice/*/.aider.results.json")):
        lang, _, _, slug, _ = f.relative_to(bench).parts
        s = per.setdefault(lang, {"exercises": 0, "kept": 0, "first_try": 0, "fix": 0, "dropped": Counter()})
        s["exercises"] += 1
        res = json.loads(f.read_text())
        outcomes = res.get("tests_outcomes") or []
        attempt = {(True,): 1, (False, True): 2}.get(tuple(outcomes), 0)
        e = last[(lang, slug)].get(attempt) if attempt else None
        why = ("error" if "exception" in res else "failed") if not attempt else "" if e else "not_logged"
        if not why:
            reasoning, content = ev.split_reasoning(e)
            msgs = [{"role": m["role"], "content": m["content"]} for m in e["messages"]]
            size = sum(len(m["content"]) for m in msgs) + len(reasoning) + len(content)
            why = ("truncated" if e.get("finish_reason") == "length" else "no_thinking_end" if not reasoning
                   else "empty" if not content else "too_long" if max_chars and size > max_chars else "")
        if why:
            s["dropped"][why] += 1
            continue
        kept.append({**_chat_row(msgs, content, reasoning),
                     "meta": {"source": "aider", "language": lang, "exercise": slug}})
        s["kept"] += 1
        s["first_try" if attempt == 1 else "fix"] += 1
    for s in per.values():
        s["dropped"] = dict(s["dropped"])
    thinking = [len(r["messages"][-1]["reasoning_content"]) for r in kept]
    return kept, {"rows": len(kept), "fix_rows": sum(s["fix"] for s in per.values()), "requests": len(entries),
                  "unmatched_requests": unmatched,
                  "thinking_chars_median": statistics.median(thinking) if thinking else 0, "per_language": per}


def make_rows(a) -> None:
    run_dir = Path(json.loads(Path(a.results).read_text())["run_dir"])
    got, stats = rows_from_run(run_dir / "bench", Path(a.log), a.max_chars)
    out = Path(a.out)
    write_jsonl(out, got)
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    for lang, s in stats["per_language"].items():
        print(f"[{STAGE}] {lang:10} kept {s['kept']:4} of {s['exercises']:4} ({s['first_try']} first try, "
              f"{s['fix']} fixes); dropped {s['dropped']}", flush=True)
    emit(STAGE, "done", 100, f"{stats['rows']} rows ({stats['fix_rows']} fixes) in {out}; "
                             f"{stats['unmatched_requests']} of {stats['requests']} logged requests matched no exercise")


def name_arg(s: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", s):
        raise argparse.ArgumentTypeError("letters, digits, '.', '_' and '-' only")
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="install docker, clone the pinned repos, build aider's benchmark image")
    threads = dict(type=int, default=default_threads(), help="exercises at once (default: 4 per CPU, at most 64)")
    s = sub.add_parser("selftest", help="no model: reference solutions through each language's tests in aider's image")
    s.add_argument("--out", required=True, help="selftest.json, selftest.log and the exercises as run go here")
    s.add_argument("--per-language", type=int, default=3, help="the first N exercises (by name) of each language")
    s.add_argument("--exercises", default=str(POLYGLOT), help="exercises in Polyglot's layout (default: Polyglot)")
    s.add_argument("--threads", **threads)
    r = sub.add_parser("run", help="serve a model dir and run aider's benchmark against it")
    r.add_argument("--model-dir", required=True,
                   help="an HF checkpoint dir (BF16 or compressed-tensors), served with vLLM")
    r.add_argument("--name", required=True, type=name_arg, help="the run's name: served model name and run dir")
    r.add_argument("--exercises", default=str(POLYGLOT), help="exercises in Polyglot's layout (default: Polyglot)")
    r.add_argument("--edit-format", default="diff")
    r.add_argument("--threads", **threads)
    r.add_argument("--num-tests", type=int, default=0, help="run only this many exercises, picked at random")
    r.add_argument("--sample-per-language", type=int, default=0, metavar="K",
                   help="run only the first K exercises (by name) of each language, in runs/NAME-sampleK")
    r.add_argument("--log", help="log every chat completion to this jsonl (a proxy between aider and vLLM)")
    r.add_argument("--experts-used", type=int, default=0, help="routed experts per token (0: the model's own)")
    r.add_argument("--reasoning-parser", default="qwen3", help="vLLM's --reasoning-parser: qwen3, gemma4, '' for none")
    r.add_argument("--temperature", type=float, default=0.6, help="sampling for every request (Gemma 4: 1.0)")
    r.add_argument("--top-p", type=float, default=0.95)
    r.add_argument("--top-k", type=int, default=20, help="(Gemma 4: 64)")
    r.add_argument("--dir", help="run dir (default $NVME/polyglot/runs/NAME)")
    r.add_argument("--out", help="results JSON (default <run dir>/out/polyglot.json)")
    r.add_argument("--force", action="store_true", help="run even when --out exists (finished exercises are kept)")
    e = sub.add_parser("exercism", help="build the training exercises from Exercism minus Polyglot's slugs")
    e.add_argument("--out", default=str(EXERCISM))
    w = sub.add_parser("rows", help="chat rows for data_extra_rows from a logged run")
    w.add_argument("--results", required=True, help="the run's results JSON (its --out), which names its run dir")
    w.add_argument("--log", required=True, help="the run's --log file")
    w.add_argument("--out", required=True, help="rows jsonl; stats go next to it as .stats.json")
    w.add_argument("--max-chars", type=int, default=0, help="drop rows longer than this (0: keep all)")
    a = ap.parse_args(argv)
    cmds = {"setup": setup, "selftest": selftest, "run": run_bench, "exercism": exercism, "rows": make_rows}
    return cmds[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
