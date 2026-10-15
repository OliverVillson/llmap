#!/usr/bin/env python3
"""The train harness's data path: tickets in, training rows out, every row a real engine call.

    python3 scripts/bh/data.py export     --work W          # bhdata kits -> train-bench, heldout-bench, extra rows
    python3 scripts/bh/data.py check      --work W          # reference answers + decontam of both benches
    python3 scripts/bh/data.py teacher    --work W --runs R1 [R2 ...]   # harvest the teacher's bench runs
    python3 scripts/bh/data.py extra      --work W          # bhdata's free rows, checked and decontaminated
    python3 scripts/bh/data.py repair-requests --work W --runs S1 ...   # round 2: the student's failures as fix calls
    python3 scripts/bh/data.py answer     --requests Q --out A --url U --model M [--samples 2]
    python3 scripts/bh/data.py repair     --work W --answers A          # round 2: own-repair rows
    python3 scripts/bh/data.py rft        --work W --runs S1 ...        # round 2: the student's passes
    python3 scripts/bh/data.py build      --work W [--round 2] [--tokenizer DIR]   # train.jsonl
    python3 scripts/bh/data.py smoke      --work W          # the dry run on the 15 public spike tickets

End to end (scripts/bh_run.py drives it):
  1. export: bhdata writes MuggeBench-format projects, split by kit (bhdata_io.py).
  2. check: every reference answer runs in the sandbox; tickets whose reference fails are
     broken (reported, never rows). Tickets that share text with MuggeBench, or carry its
     canary, are excluded (decontam.py).
  3. the teacher works train-bench through the engine's own loop, 2-4 seeds, thinking off
     (`bun bench/run.ts --bench W/data/export/train-bench --transcripts --thinking off ...`,
     bh_run.py's step); teacher: harvest.py turns its passing calls into write and fix rows,
     each re-checked in the sandbox.
  4. extra: bhdata's free rows, held to the same wire shape (rendered back through the
     engine), checked in the sandbox when they name their ticket, decontaminated.
  5. build: all row files, decontaminated again, deduplicated, shuffled -> W/data/train.jsonl
     with the tokenizer round trip on one real row.
  Round 2: the student runs heldout-bench (laptops, --transcripts); repair-requests renders the
  fix call the engine would make next on each failure; the teacher answers them (answer);
  repair keeps the shortest passing answer per failure (own-repair rows); rft keeps the
  student's own passes; build --round 2 adds both.

Every prompt and answer is rendered by the engine (render.py -> mugge/bench/render.ts); this
file never builds one. Row format: {"messages": [system, user, assistant], "meta": {...}},
what stages/sft.py and Config.data_extra_rows read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in (None, ""):  # run as a script: python3 scripts/bh/data.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "bh"

from . import bhdata_io, check, decontam, harvest, render  # noqa: E402

SPIKE = render.MUGGE / "spike"  # public MuggeBench example: smoke data only, never training data


@dataclass
class DataConfig:
    work: Path  # W; everything goes under W/data
    sandbox: str = "local"  # local (stages/sandbox.py isolation) | docker | podman (the bench's images)
    workers: int = 0  # sandbox runs at once; 0 = stages/sandbox.py's default (physical cores - 2)
    timeout: float = check.TIMEOUT  # seconds per check command
    max_fix: int = 2  # fix rows per ticket (distinct prompts)
    write_from_fix: bool = True
    extra_require_check: bool = False  # drop free rows that name no ticket to check them against
    benches: list[str] | None = None  # MuggeBench dirs for decontam; None = decontam.default_benches()
    seed: int = 0

    @property
    def d(self) -> Path:
        return Path(self.work) / "data"

    @property
    def export_dir(self) -> Path:
        return self.d / "export"

    def bench(self, which: str) -> Path:
        return bhdata_io.layout(self.export_dir).train_bench if which == "train" else bhdata_io.layout(self.export_dir).heldout_bench

    def rows(self, name: str) -> Path:
        return self.d / "rows" / f"{name}.jsonl"

    def stats(self, name: str) -> Path:
        return self.d / "stats" / f"{name}.json"


def _dump(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(p)


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _index(cfg: DataConfig) -> decontam.Index:
    return decontam.index(cfg.benches)


# ---------- 1-2: export and check ----------

def export(cfg: DataConfig) -> bhdata_io.Export:
    e = bhdata_io.export(cfg.export_dir)
    _dump(cfg.stats("export"), {k: str(v) for k, v in asdict(e).items()})
    return e


def check_benches(cfg: DataConfig, which: tuple[str, ...] = ("train", "heldout")) -> dict:
    """Reference answers and decontam for each bench: W/data/stats/check-<which>.json with
    {"tickets", "passed", "broken": {key: why}, "excluded": {key: why}}. Raises when most of
    a bench is excluded for the canary, which means the export stamped MuggeBench's canary on
    practice tickets."""
    ix = _index(cfg)
    out = {}
    for w in which:
        b = cfg.bench(w)
        rep = check.check_references(b, workers=cfg.workers or None, timeout=cfg.timeout, kind=cfg.sandbox)
        rep["excluded"] = decontam.contaminated_tickets(b, ix)
        canary = sum(1 for v in rep["excluded"].values() if v == "canary")
        if rep["tickets"] and canary > rep["tickets"] / 2:
            raise RuntimeError(f"{canary} of {rep['tickets']} {w} tickets carry the MUGGE-BENCH- canary: the export "
                               f"must not stamp practice kits with MuggeBench's canary")
        _dump(cfg.stats(f"check-{w}"), rep)
        print(f"[check] {w}: {rep['passed']}/{rep['tickets']} reference answers pass; {len(rep['broken'])} broken, "
              f"{len(rep['excluded'])} excluded by decontam", flush=True)
        for k, why in list(rep["broken"].items())[:20]:
            print(f"  broken {k}: {why.splitlines()[0]}", flush=True)
        out[w] = rep
    return out


def _guards(cfg: DataConfig, which: str) -> tuple[set[str], set[str]]:
    rep = _load(cfg.stats(f"check-{which}"))
    if not rep:
        raise FileNotFoundError(f"no {cfg.stats(f'check-{which}')}: run `data.py check` first")
    return set(rep["broken"]), set(rep["excluded"])


# ---------- 3: the teacher's rows ----------

def teacher_rows(cfg: DataConfig, runs: list[str | Path]) -> Path:
    broken, excluded = _guards(cfg, "train")
    rows, stats = harvest.harvest(cfg.bench("train"), runs, source="teacher", broken=broken, excluded=excluded,
                                  max_fix=cfg.max_fix, write_from_fix=cfg.write_from_fix, sandbox=cfg.sandbox,
                                  workers=cfg.workers or None, timeout=cfg.timeout)
    harvest.write_jsonl(cfg.rows("teacher"), rows)
    _dump(cfg.stats("teacher"), stats.as_dict())
    return cfg.rows("teacher")


# ---------- 4: bhdata's free rows ----------

OWNS_LINE = re.compile(r"^Files you own \(write each in full\): (.*)$", re.M)


def extra_rows(cfg: DataConfig, path: Path | None = None) -> Path:
    """bhdata's free rows (mutation fix, test, review), held to the engine's wire: the system
    prompt must be the engine's, the user message a write or fix message, the answer every
    owned file, each path alone on its line before its fence (an answer whose paths the
    parser had to guess is dropped), re-rendered through the engine when only its layout
    differs (fence language, blank lines). A row naming its ticket ("meta": {"project": <bench project dir>, "ticket": id})
    is checked in the sandbox; review rows are not worker calls and go to extra-review.jsonl
    for BH0.2."""
    path = path or bhdata_io.layout(cfg.export_dir).extra_rows
    rows = harvest.read_jsonl(path)
    system = render.system_prompt()
    review = bhdata_io.review_system()
    counts: Counter = Counter(total=len(rows))
    shaped, reviews = [], []
    for r in rows:
        m = r.get("messages") or []
        if len(m) != 3 or [x.get("role") for x in m] != ["system", "user", "assistant"]:
            counts["not_one_call"] += 1
        elif m[0]["content"] != system:
            if (review and m[0]["content"] == review) or (r.get("meta") or {}).get("kind") == "review":
                reviews.append({**r, "kind": "review", "source": "extra"})
                counts["review"] += 1
            else:
                counts["system_drift"] += 1
        elif not (m[1]["content"].startswith("Ticket: ") and render.PACK_END in m[1]["content"]
                  and OWNS_LINE.search(m[1]["content"])):
            counts["not_a_worker_message"] += 1
        else:
            shaped.append(r)
    owns = [OWNS_LINE.search(r["messages"][1]["content"]).group(1).split(", ") for r in shaped]
    canon = render.call([{"op": "canonical", "answer": r["messages"][2]["content"], "owns": o} for r, o in zip(shaped, owns)])
    ready = []
    for r, o, x in zip(shaped, owns, canon):
        if x["parsed"]["missing"] or x["parsed"]["extra"]:
            counts["parser"] += 1
            continue
        if x["parsed"]["loose"]:  # the parser had to guess a path: the row would teach the wrong shape
            counts["loose"] += 1
            continue
        if x["content"] != r["messages"][2]["content"]:
            counts["recanonicalised"] += 1
        meta = r.get("meta") or {}
        kind = render.kind_of(r["messages"][1]["content"])
        msgs = [*r["messages"][:2], {"role": "assistant", "content": x["content"]}]
        tid = re.search(r"^Ticket: (\S+)$", msgs[1]["content"], re.M)
        ready.append({"id": hashlib.sha256((msgs[1]["content"] + "\0" + msgs[2]["content"]).encode()).hexdigest()[:24],
                      "kind": kind, "source": "extra", "project": meta.get("project", ""),
                      "ticket": meta.get("ticket") or (tid.group(1) if tid else ""), "kit": meta.get("kit", ""),
                      "messages": msgs, "completion_tokens": None, "cut": False, "forced": False,
                      "meta": {**meta, "kind": kind, "source": "extra"}, "_files": x["parsed"]["files"]})
    # sandbox check where the row names its ticket
    named = [r for r in ready if r["meta"].get("project") and r["meta"].get("ticket")]
    tks = {}
    for proj in sorted({r["meta"]["project"] for r in named}):
        tks.update({f"{t['dir']}::{t['id']}": t for t in render.tickets(proj).values()})
    items = []
    for r in named:
        t = tks.get(f"{Path(r['meta']['project']).resolve()}::{r['meta']['ticket']}")
        if t is None:
            r["meta"]["checked"] = False
            continue
        items.append((r, check.Item(t["dir"], t["id"], r["_files"], key=t["key"])))
    res = check.check_many([it for _, it in items], {it.key: tks[f"{it.project}::{it.ticket}"] for _, it in items},
                           workers=cfg.workers or None, timeout=cfg.timeout, kind=cfg.sandbox)
    failed = set()
    for (r, _), x in zip(items, res):
        r["meta"]["checked"] = x.ok
        if not x.ok:
            failed.add(id(r))
            counts[f"sandbox_{x.stage}"] += 1
    out = []
    for r in ready:
        if id(r) in failed:
            continue
        r.setdefault("meta", {}).setdefault("checked", False)
        if not r["meta"]["checked"]:
            counts["unchecked"] += 1
            if cfg.extra_require_check:
                continue
        r.pop("_files", None)
        out.append(r)
    out, dropped = decontam.filter_rows(out, _index(cfg))
    out, bh_dropped = bhdata_io.bhdata_decontam(out)
    counts.update({f"decontam_{k}": v for k, v in dropped.items()})
    counts["decontam_bhdata"] += bh_dropped
    counts["rows"] = len(out)
    if counts["unchecked"]:
        print(f"[extra] WARNING: {counts['unchecked']} free rows name no ticket, so their answers were not run in the "
              f"sandbox{' and were dropped' if cfg.extra_require_check else ''}", flush=True)
    bad = counts["parser"] + counts["system_drift"] + counts["not_a_worker_message"] + counts["not_one_call"]
    if rows and bad / len(rows) > harvest.DROP_WARN:
        print(f"[extra] WARNING: {bad} of {len(rows)} free rows do not have the engine's wire shape", flush=True)
    harvest.write_jsonl(cfg.rows("extra"), out)
    harvest.write_jsonl(cfg.rows("extra-review"), reviews)
    _dump(cfg.stats("extra"), dict(counts))
    print(f"[extra] {len(out)} of {len(rows)} free rows kept; {dict(counts)}", flush=True)
    return cfg.rows("extra")


# ---------- round 2 ----------

def repair_requests(cfg: DataConfig, runs: list[str | Path]) -> Path:
    broken, excluded = _guards(cfg, "heldout")
    reqs, counts = harvest.repair_requests(cfg.bench("heldout"), runs, broken=broken, excluded=excluded,
                                           sandbox=cfg.sandbox, workers=cfg.workers or None, timeout=cfg.timeout)
    p = cfg.d / "round2" / "repair-requests.jsonl"
    harvest.write_jsonl(p, reqs)
    _dump(cfg.stats("repair-requests"), counts)
    return p


def answer_requests(requests: Path, out: Path, *, url: str, model: str, samples: int = 2, max_tokens: int = 4096,
                    temperature: float = 0.7, top_p: float = 0.95, top_k: int = 20, thinking: bool = False,
                    concurrency: int = 64, timeout: float = 3600.0) -> Path:
    """The teacher's answers to a requests file ([{"id", "messages"}]), `samples` per request,
    from any OpenAI-compatible server: [{"id", "answers": [{"content", "cut", "completionTokens",
    "maxTokens"}]}]. Thinking off by default and top_k always finite (run 1's top_k 0 was one of
    its three bugs); a long read timeout, since a big prefill can take minutes."""
    reqs = harvest.read_jsonl(requests)
    if top_k <= 0:
        raise ValueError("top_k must be finite (>0)")
    lock = threading.Lock()
    done = {r["id"] for r in harvest.read_jsonl(out)}  # resumable
    out.parent.mkdir(parents=True, exist_ok=True)

    def one(q: dict) -> None:
        answers = []
        for s in range(samples):
            body = {"model": model, "messages": q["messages"], "max_tokens": max_tokens, "temperature": temperature,
                    "top_p": top_p, "top_k": top_k, "seed": s, "chat_template_kwargs": {"enable_thinking": thinking}}
            req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                         headers={"content-type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read())
            except Exception as e:  # noqa: BLE001 - one failed call is one missing sample
                print(f"[answer] {q['id']} sample {s}: {e}", flush=True)
                continue
            ch = data["choices"][0]
            answers.append({"content": ch["message"].get("content") or "", "cut": ch.get("finish_reason") == "length",
                            "completionTokens": (data.get("usage") or {}).get("completion_tokens"), "maxTokens": max_tokens})
        with lock, out.open("a") as f:
            f.write(json.dumps({"id": q["id"], "answers": answers}, ensure_ascii=False) + "\n")

    todo = [q for q in reqs if q["id"] not in done]
    print(f"[answer] {len(todo)} requests × {samples} to {url} ({model}), thinking {'on' if thinking else 'off'}", flush=True)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(one, todo))
    return out


def repair(cfg: DataConfig, answers: Path, requests: Path | None = None) -> Path:
    requests = requests or cfg.d / "round2" / "repair-requests.jsonl"
    rows, counts = harvest.repair_rows(cfg.bench("heldout"), harvest.read_jsonl(requests), harvest.read_jsonl(answers),
                                       sandbox=cfg.sandbox, workers=cfg.workers or None, timeout=cfg.timeout)
    harvest.write_jsonl(cfg.rows("own-repair"), rows)
    _dump(cfg.stats("own-repair"), counts)
    return cfg.rows("own-repair")


def rft_rows(cfg: DataConfig, runs: list[str | Path]) -> Path:
    broken, excluded = _guards(cfg, "heldout")
    rows, stats = harvest.harvest(cfg.bench("heldout"), runs, source="rft", broken=broken, excluded=excluded,
                                  max_fix=cfg.max_fix, write_from_fix=cfg.write_from_fix, sandbox=cfg.sandbox,
                                  workers=cfg.workers or None, timeout=cfg.timeout)
    harvest.write_jsonl(cfg.rows("rft"), rows)
    _dump(cfg.stats("rft"), stats.as_dict())
    return cfg.rows("rft")


# ---------- 5: build ----------

ROUND_PARTS = {1: ("teacher", "extra"), 2: ("teacher", "extra", "own-repair", "rft")}


def build(cfg: DataConfig, round_: int = 1, *, parts: tuple[str, ...] | None = None, tokenizer: str | None = None,
          out: Path | None = None, allow_smoke: bool = False) -> Path:
    """The SFT file: the parts' rows, decontaminated again, deduplicated by (user, assistant),
    shuffled with a fixed seed. Smoke rows (spike) are refused unless allow_smoke."""
    parts = parts or ROUND_PARTS[round_]
    rows, counts = [], Counter()
    for name in parts:
        got = harvest.read_jsonl(cfg.rows(name))
        counts[f"in_{name}"] = len(got)
        rows += got
    smoke = [r for r in rows if (r.get("meta") or {}).get("smoke")]
    if smoke and not allow_smoke:
        raise RuntimeError(f"{len(smoke)} rows are smoke rows from the public spike; they are never training data")
    if not allow_smoke:
        rows, dropped = decontam.filter_rows(rows, _index(cfg))
        counts.update({f"decontam_{k}": v for k, v in dropped.items()})
    seen, uniq = set(), []
    for r in rows:
        h = hashlib.sha256((r["messages"][1]["content"] + "\0" + r["messages"][2]["content"]).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            uniq.append(r)
    counts["duplicates"] = len(rows) - len(uniq)
    random.Random(cfg.seed).shuffle(uniq)
    out = out or cfg.d / ("train.jsonl" if round_ == 1 else f"train-round{round_}.jsonl")
    harvest.write_jsonl(out, uniq)
    counts["rows"] = len(uniq)
    counts.update({f"kind_{k}": v for k, v in Counter((r.get("meta") or {}).get("kind") for r in uniq).items()})
    if tokenizer and uniq:
        counts["tokenizer_check"] = tokenizer_roundtrip(uniq[0], tokenizer)
    _dump(cfg.stats(out.stem), dict(counts))
    print(f"[build] {len(uniq)} rows -> {out}; {dict(counts)}", flush=True)
    return out


def tokenizer_roundtrip(row: dict, tokenizer) -> str:
    """One real row through the student's tokenizer and chat template: the rendered text
    decodes back unchanged, the answer is in it verbatim, and the prompt the server will
    send (template with the generation prompt, thinking off) is a prefix of the training
    text, so masking the prompt tokens masks exactly the prompt. tokenizer: a dir or an HF
    tokenizer object. Raises on a mismatch; returns "ok"."""
    if isinstance(tokenizer, (str, Path)):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer))
    msgs = row["messages"]
    kw = {"enable_thinking": False}
    full = tokenizer.apply_chat_template(msgs, tokenize=False, **kw)
    prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, **kw)
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    back = tokenizer.decode(ids)
    problems = []
    if back != full:
        problems.append("decode(encode(text)) != text")
    if msgs[-1]["content"] not in full:
        problems.append("the answer is not verbatim in the rendered text")
    if not full.startswith(prompt):
        problems.append("the served prompt (generation prompt, thinking off) is not a prefix of the training text")
    else:
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if ids[:len(p_ids)] != p_ids:
            problems.append("the prompt's tokens are not a prefix of the row's tokens (prompt masking would be off)")
    for m in msgs[:2]:
        if m["content"] not in full:
            problems.append(f"the {m['role']} message is not verbatim in the rendered text")
    if problems:
        raise RuntimeError("tokenizer round trip failed: " + "; ".join(problems))
    print(f"[tokenizer] round trip ok on one real row ({len(ids)} tokens)", flush=True)
    return "ok"


# ---------- smoke: the dry run on the public spike ----------

def _serve(answer_for) -> tuple[str, callable]:
    """A tiny OpenAI-compatible server answering each chat call with answer_for(prompt)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
            prompt = body["messages"][-1]["content"]
            content = answer_for(prompt)
            data = json.dumps({"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                               "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(content) // 4}}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}/v1", srv.shutdown


def _bench_run(bench: Path, out: Path, *args: str) -> None:
    cmd = [render.bun(), "bench/run.ts", "--bench", str(bench), "--out", str(out), "--transcripts",
           "--sandbox", "local", "--concurrency", "8", *args]
    print(f"[smoke] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=render.MUGGE, check=True, stdout=subprocess.DEVNULL)


def smoke(cfg: DataConfig) -> dict:
    """The whole data path on the 15 public spike tickets, with no GPU: the fake model stands in
    for the teacher (reference answers; each ticket's first call empty, so real fix calls
    happen), a scripted 'student' writes half-broken files over HTTP, and the reference
    stands in for the teacher's repairs. Rows land in W/data/smoke/ marked smoke: spike
    tickets are smoke data only, never training data, and build() refuses them."""
    sm = DataConfig(**{**asdict(cfg), "work": Path(cfg.work) / "data" / "smoke-work"})
    root = Path(cfg.work) / "data" / "smoke"
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(sm.work, ignore_errors=True)
    bench = sm.export_dir / "train-bench" / "spike"
    bench.mkdir(parents=True)
    for part in ("plan.json", "project", "solutions", "hidden"):  # a bench project, not the spike's own scripts
        (shutil.copytree if (SPIKE / part).is_dir() else shutil.copyfile)(SPIKE / part, bench / part)
    shutil.copytree(bench, sm.export_dir / "heldout-bench" / "spike")
    t0 = time.time()
    report: dict = {}

    # 2: reference answers (decontam is skipped: the spike IS MuggeBench)
    for w in ("train", "heldout"):
        rep = check.check_references(sm.bench(w), workers=cfg.workers or None, timeout=cfg.timeout, kind=cfg.sandbox)
        rep["excluded"] = {}
        _dump(sm.stats(f"check-{w}"), rep)
    report["references"] = f"{rep['passed']}/{rep['tickets']}"

    # 3: the 'teacher' (fake model) through the engine's loop, 2 seeds, then harvest
    _bench_run(sm.bench("train"), root / "teacher-run", "--fake", "--fake-flaky", "--seeds", "2")
    teacher_rows(sm, [root / "teacher-run"])
    report["teacher"] = _load(sm.stats("teacher"))

    # round 2: a scripted student over HTTP; one ticket in three right, the rest half-written
    tks = render.tickets(sm.bench("heldout"))
    refs = {t["id"]: json.loads(Path(t["dir"], "solutions", f"{t['id']}.json").read_text())["files"] for t in tks.values()}

    def student(prompt: str) -> str:
        tid = re.search(r"^Ticket: (\S+)$", prompt, re.M).group(1)
        files = refs[tid]
        if int(hashlib.md5(tid.encode()).hexdigest(), 16) % 3:
            files = {p: "\n".join(c.splitlines()[: max(1, len(c.splitlines()) // 2)]) + "\n" for p, c in files.items()}
        return render.one({"op": "answer", "files": files, "note": "student"})["content"]

    url, stop = _serve(student)
    try:
        _bench_run(sm.bench("heldout"), root / "student-run", "--url", url, "--model", "student", "--seeds", "1",
                   "--thinking", "off", "--top-k", "20")
    finally:
        stop()
    req_path = repair_requests(sm, [root / "student-run"])

    def teacher_fix(prompt: str) -> str:
        tid = re.search(r"^Ticket: (\S+)$", prompt, re.M).group(1)
        return render.one({"op": "answer", "files": refs[tid], "note": "reference stands in for the teacher"})["content"]

    url, stop = _serve(teacher_fix)
    try:
        ans = answer_requests(req_path, sm.d / "round2" / "repair-answers.jsonl", url=url, model="teacher", samples=2,
                              concurrency=8)
    finally:
        stop()
    repair(sm, ans)
    rft_rows(sm, [root / "student-run"])
    report["repair_requests"] = _load(sm.stats("repair-requests"))
    report["own_repair"] = _load(sm.stats("own-repair"))
    report["rft"] = _load(sm.stats("rft"))

    # build, with every row marked smoke
    for name in ("teacher", "own-repair", "rft"):
        rows = harvest.read_jsonl(sm.rows(name))
        for r in rows:
            r["meta"]["smoke"] = True
        harvest.write_jsonl(sm.rows(name), rows)
    out = build(sm, 2, parts=("teacher", "own-repair", "rft"), out=root / "train-smoke.jsonl", allow_smoke=True)
    try:
        build(sm, 2, parts=("teacher",), out=root / "must-not-exist.jsonl")
        report["smoke_guard"] = "FAILED: build accepted smoke rows"
    except RuntimeError:
        report["smoke_guard"] = "ok: build refuses smoke rows"
    rows = harvest.read_jsonl(out)
    report["rows"] = len(rows)
    report["by_kind"] = dict(Counter(f"{r['meta']['source']}/{r['meta']['kind']}" for r in rows))
    # every row is the engine's own call: re-render each write row from its ticket and compare
    writes = [r for r in rows if r["meta"]["kind"] in ("write", "write-from-fix")]
    again = render.call([{"op": "write", "project": tks[r["meta"]["key"]]["dir"], "ticket": tks[r["meta"]["key"]]["id"],
                          "answer": r["messages"][2]["content"]} for r in writes])
    report["write_rows_rerendered_identical"] = f"{sum(a['messages'] == r['messages'] for a, r in zip(again, writes))}/{len(writes)}"
    report["seconds"] = round(time.time() - t0, 1)
    _dump(root / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


# ---------- row-file commands (scripts/bh_run.py's interface) ----------
# Each reads and writes files only and writes <out>.stats.json beside its output.

def _stats_path(out: Path) -> Path:
    return out.with_name(out.name + ".stats.json")


def export_kits(kits: Path | None, out: Path, *, limit_kits: int | None = None, heldout_share: float | None = None,
                seed: int | None = None) -> dict:
    """bhdata's export into out (train-bench/, heldout-bench/, extra_rows.jsonl) and
    out/export.json with its counts. canary_hits: practice tickets carrying MuggeBench's canary."""
    args = [*(["--kits", str(kits)] if kits else []), *(["--limit-kits", str(limit_kits)] if limit_kits else []),
            *(["--heldout-share", str(heldout_share)] if heldout_share is not None else []),
            *(["--seed", str(seed)] if seed is not None else [])]
    e = bhdata_io.export(out, args=args + shlex.split(os.environ.get("BHDATA_EXPORT_ARGS", "")))
    st = {"kits": 0, "extra_rows": len(harvest.read_jsonl(e.extra_rows)), "canary_hits": 0}
    for name, b in (("train", e.train_bench), ("heldout", e.heldout_bench)):
        tks = render.tickets(b)
        st[f"{name}_projects"] = len({t["project"] for t in tks.values()})
        st[f"{name}_tickets"] = len(tks)
        st["kits"] += len({t.get("kit") or t["project"] for t in tks.values()})  # split by kit: no kit is in both
        st["canary_hits"] += sum(1 for v in decontam.contaminated_tickets(b, decontam.Index()).values() if v == "canary")
    _dump(out / "export.json", st)
    return st


def harvest_files(transcripts: list[Path], bench: Path, out: Path, *, calls_logs: list[Path] = (), source: str = "teacher",
                  cfg: DataConfig | None = None) -> dict:
    """harvest() over the bench runs whose transcripts.jsonl these are (results.jsonl beside each)."""
    cfg = cfg or DataConfig(work=out.parent)
    rows, st = harvest.harvest(bench, sorted({Path(t).parent for t in transcripts}), source=source,
                               max_fix=cfg.max_fix, write_from_fix=cfg.write_from_fix, sandbox=cfg.sandbox,
                               workers=cfg.workers or None, timeout=cfg.timeout, calls_logs=calls_logs)
    harvest.write_jsonl(out, rows)
    d = st.dropped
    stats = {"calls": st.answers, "tickets": len({r["key"] for r, _ in harvest.load_runs(sorted({Path(t).parent for t in transcripts}))}),
             "tickets_solved": st.tickets_with_rows, "rows": len(rows),
             "write_rows": sum(1 for r in rows if r["kind"] == "write"), "fix_rows": sum(1 for r in rows if r["kind"] == "fix"),
             "dropped": {"length": d["cut"], "parse": d["parser"], "forced": d["forced"],
                         "failed": sum(v for k, v in d.items() if k.startswith("sandbox_")),
                         "dupe": 0, **{k: v for k, v in d.items() if k not in ("cut", "parser", "forced") and not k.startswith("sandbox_")}},
             "parser_drops_all_calls": st.parser, "cut_all_calls": st.cut, "drop_share": round(st.drop_share(), 4),
             "warnings": st.warnings}
    _dump(_stats_path(out), stats)
    return stats


def check_rows(rows_path: Path, bench: Path, out: Path, *, cfg: DataConfig | None = None) -> dict:
    """The rows whose answers pass their ticket's acceptance and hidden tests (no network)."""
    cfg = cfg or DataConfig(work=out.parent)
    rows = harvest.read_jsonl(rows_path)
    tks = render.tickets(bench)
    by_pt = {(t["project"], t["id"]): t for t in tks.values()}
    keyed, errors = [], 0
    for r in rows:
        t = tks.get((r.get("meta") or {}).get("key", "")) or by_pt.get((r.get("project"), r.get("ticket")))
        if t is None:
            errors += 1
        else:
            keyed.append((r, t))
    canon = render.call([{"op": "canonical", "answer": r["messages"][-1]["content"], "owns": t["owns"]} for r, t in keyed])
    items = []
    for (r, t), x in zip(keyed, canon):
        if x["parsed"]["missing"] or x["parsed"]["extra"]:
            errors += 1
        else:
            items.append((r, check.Item(t["dir"], t["id"], x["parsed"]["files"], key=t["key"])))
    res = check.check_many([it for _, it in items], tks, workers=cfg.workers or None, timeout=cfg.timeout, kind=cfg.sandbox)
    kept = [r for (r, _), x in zip(items, res) if x.ok]
    harvest.write_jsonl(out, kept)
    st = {"rows_in": len(rows), "passed": len(kept), "failed": len(items) - len(kept), "errors": errors}
    _dump(_stats_path(out), st)
    return st


def decontam_rows(rows_path: Path, out: Path, benches: list[str] | None = None) -> dict:
    rows = harvest.read_jsonl(rows_path)
    kept, dropped = decontam.filter_rows(rows, decontam.index(benches))
    kept, bh = bhdata_io.bhdata_decontam(kept)
    harvest.write_jsonl(out, kept)
    st = {"rows_in": len(rows), "kept": len(kept), "dropped_canary": dropped.get("canary", 0),
          "dropped_overlap": dropped.get("overlap", 0), "dropped_bhdata": bh}
    _dump(_stats_path(out), st)
    return st


def render_calls(calls_path: Path, out: Path) -> dict:
    """Call specs {id, kind, source, project, ticket, kit, bench, current{}, failed{command, output},
    answer_files{}, note} rendered into rows by the engine (prompts.ts, files.ts)."""
    specs = harvest.read_jsonl(calls_path)
    tks_by_bench = {b: render.tickets(b) for b in sorted({s["bench"] for s in specs})}
    answers = render.call([{"op": "answer", "files": s.get("answer_files") or {}, "note": s.get("note") or ""} for s in specs])
    reqs, keep = [], []
    for s, a in zip(specs, answers):
        t = tks_by_bench[s["bench"]].get(f"{s['project']}/{s['ticket']}")
        if t is None:
            continue
        req = {"op": "fix" if s.get("kind") == "fix" else "write", "project": t["dir"], "ticket": t["id"],
               "current": s.get("current") or None, "answer": a["content"]}
        if req["op"] == "fix":
            req["failed"] = s["failed"]
        reqs.append(req)
        keep.append((s, t))
    rendered = render.call(reqs, strict=False)
    rows, errors = [], len(specs) - len(keep)
    for (s, t), r in zip(keep, rendered):
        if r.get("error") or r["parsed"]["missing"]:
            errors += 1
            continue
        row = harvest.make_row(t, r["messages"], kind=s.get("kind", "write"), source=s.get("source", "extra"))
        row["id"] = s.get("id") or row["id"]
        row["kit"] = s.get("kit") or row["kit"]
        rows.append(row)
    harvest.write_jsonl(out, rows)
    st = {"calls": len(specs), "rows": len(rows), "errors": errors}
    _dump(_stats_path(out), st)
    return st


def repairs_files(transcripts: list[Path], bench: Path, out: Path, *, calls_logs: list[Path] = (),
                  cfg: DataConfig | None = None) -> dict:
    cfg = cfg or DataConfig(work=out.parent)
    runs = sorted({Path(t).parent for t in transcripts})
    reqs, counts = harvest.repair_requests(bench, runs, sandbox=cfg.sandbox, workers=cfg.workers or None,
                                           timeout=cfg.timeout, calls_logs=calls_logs)
    harvest.write_jsonl(out, reqs)
    pairs = harvest.load_runs(runs)
    st = {"tickets": len({r["key"] for r, _ in pairs}), "failed_tickets": len({r["key"] for r, _ in pairs if not r.get("solved")}),
          "prompts": len(reqs), **counts}
    _dump(_stats_path(out), st)
    return st


# ---------- command line ----------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["export", "check", "teacher", "extra", "repair-requests", "answer", "repair", "rft",
                                    "build", "smoke", "harvest", "decontam", "render", "repairs"])
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("BH_WORK", "bh-work")))
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--sandbox", default="local")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=check.TIMEOUT)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--tokenizer")
    ap.add_argument("--requests", type=Path)
    ap.add_argument("--answers", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--url")
    ap.add_argument("--model")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    # the row-file commands
    ap.add_argument("--kits", type=Path)
    ap.add_argument("--limit-kits", type=int)
    ap.add_argument("--heldout-share", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--transcripts", nargs="*", type=Path, default=[])
    ap.add_argument("--calls-log", nargs="*", type=Path, default=[])
    ap.add_argument("--bench", nargs="*", type=Path, default=[])
    ap.add_argument("--rows", type=Path)
    ap.add_argument("--calls", type=Path)
    ap.add_argument("--source", default="teacher")
    ap.add_argument("--keep", default="shortest", choices=["shortest"])
    a = ap.parse_args(argv)
    cfg = DataConfig(work=a.work, sandbox=a.sandbox, workers=a.workers, timeout=a.timeout)
    bench = a.bench[0] if a.bench else None
    if a.cmd == "export":
        if a.out:
            export_kits(a.kits, a.out, limit_kits=a.limit_kits, heldout_share=a.heldout_share, seed=a.seed)
        else:
            export(cfg)
    elif a.cmd == "check":
        if a.rows:
            check_rows(a.rows, bench, a.out, cfg=cfg)
        else:
            check_benches(cfg)
    elif a.cmd == "harvest":
        harvest_files(a.transcripts, bench, a.out, calls_logs=a.calls_log, source=a.source, cfg=cfg)
    elif a.cmd == "decontam":
        decontam_rows(a.rows, a.out, [str(b) for b in a.bench] or None)
    elif a.cmd == "render":
        render_calls(a.calls, a.out)
    elif a.cmd == "repairs":
        repairs_files(a.transcripts, bench, a.out, calls_logs=a.calls_log, cfg=cfg)
    elif a.cmd == "teacher":
        teacher_rows(cfg, a.runs)
    elif a.cmd == "extra":
        extra_rows(cfg)
    elif a.cmd == "repair-requests":
        repair_requests(cfg, a.runs)
    elif a.cmd == "answer":
        answer_requests(a.requests or cfg.d / "round2" / "repair-requests.jsonl",
                        a.out or cfg.d / "round2" / "repair-answers.jsonl", url=a.url, model=a.model,
                        samples=a.samples, max_tokens=a.max_tokens, top_k=a.top_k, thinking=a.thinking == "on")
    elif a.cmd == "repair":
        repair(cfg, a.answers or cfg.d / "round2" / "repair-answers.jsonl", a.requests)
    elif a.cmd == "rft":
        rft_rows(cfg, a.runs)
    elif a.cmd == "build":
        build(cfg, a.round, tokenizer=a.tokenizer)
    elif a.cmd == "smoke":
        rep = smoke(cfg)
        return 0 if rep.get("smoke_guard", "").startswith("ok") and rep["rows"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
