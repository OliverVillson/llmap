"""The dry run's stand-ins (bh_run.py --dry-run DIR): a fake OpenAI-style server that
answers every chat call with the ticket's files (and top-k log-probs when asked), and one
mock per command kind that writes what the real command writes, in its real shape.

What stays real in a dry run: bh_run.py's plan, paths, restores and uploads (to a local
bucket folder), every guard and threshold, the recording proxy (each mock bench call goes
through it to the fake server), the bash syntax of every command, and the summary. What
is mocked: GPUs, models, vLLM, llama.cpp, bun and the data path's commands.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bh.exec import Cmd, dump, load, read_jsonl, write_jsonl
from bh.wire import SYSTEM

PROJECTS = 6  # per bench split
TICKETS = 4  # per project


def h(*xs) -> int:
    return int(hashlib.sha256("\x00".join(map(str, xs)).encode()).hexdigest()[:8], 16)


def owned(user: str) -> list[str]:
    for line in user.splitlines():
        if line.startswith("Files you own (write each in full): "):
            return [p.strip() for p in line.split(": ", 1)[1].split(",") if p.strip()]
    return ["main.py"]


def answer_for(user: str, seed) -> str:
    files = "\n\n".join(f"{p}\n```python\ndef f():\n    return {h(user, seed, p) % 97}\n```" for p in owned(user))
    return files + "\n\nNote: done"


class FakeUpstream:
    """vLLM/llama-server as far as the harness talks to them: /health, chat completions
    (with logprobs as token ids on request; every 23rd answer cut at the cap),
    /tokenize and /apply-template."""

    def __init__(self):
        up = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"status": "ok"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
                if self.path.endswith("/chat/completions"):
                    self._send(200, up.chat(body))
                elif self.path.endswith("/tokenize"):
                    self._send(200, {"tokens": [ord(c) % 1000 for c in json.dumps(body.get("messages"))][:64]})
                elif self.path.endswith("/apply-template"):
                    self._send(200, {"prompt": json.dumps(body.get("messages"))})
                else:
                    self._send(404, {"error": self.path})

            def log_message(self, *a):
                pass

        self.requests: list[dict] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def chat(self, body: dict) -> dict:
        self.requests.append({k: body.get(k) for k in ("top_k", "top_p", "min_p", "temperature", "max_tokens",
                                                        "chat_template_kwargs", "logprobs", "top_logprobs")})
        user = body["messages"][-1]["content"]
        answer = answer_for(user, body.get("seed"))
        cut = h(user, body.get("seed"), "cut") % 23 == 0
        ids = [ord(c) % 5000 for c in answer]
        choice = {"index": 0, "message": {"role": "assistant", "content": answer},
                  "finish_reason": "length" if cut else "stop"}
        if body.get("logprobs"):
            k = int(body.get("top_logprobs") or 1)
            choice["logprobs"] = {"content": [
                {"token": f"token_id:{t}", "logprob": -0.1,
                 "top_logprobs": [{"token": f"token_id:{(t + j) % 5000}", "logprob": -0.1 - j} for j in range(k)]}
                for t in ids]}
        return {"choices": [choice], "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": len(ids)}}

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def plan(project: str, n: int, canary: bool = False) -> dict:
    return {"bench": {"split": "train", **({"canary": "MUGGE-BENCH-x"} if canary else {})}, "interfaces": [],
            "tickets": [{"id": f"{project}-t{i}", "title": f"ticket {i}", "role": "implement", "owns": [f"src/m{i}.py"],
                         "reads": [], "context": f"Implement m{i}.", "acceptance": [f"python -m pytest tests/test_m{i}.py"],
                         "depends_on": [], "image": "python", "budget": {"attempts": 4, "max_tokens": 60000},
                         "bench": {"theme": "backend", "language": "python", "role": "implement", "hidden": []}}
                        for i in range(n)]}


def pack(project: str, t: dict, fix: str = "") -> str:
    text = (f"Ticket: {t['id']}\nTitle: {t['title']}\nRole: {t['role']}\n\n{t['context']}\n\n"
            f"Files you own (write each in full): {', '.join(t['owns'])}\n"
            f"Done when these commands pass: {' ; '.join(t['acceptance'])}\n\n## Interfaces\n\n## Files to read\n\n"
            f"## Your files as they are now\n" + "\n".join(f"--- {p}\n" for p in t["owns"]))
    return text + (f"\n\n## Fix\nThis command failed:\n$ {t['acceptance'][0]}\n{fix}\n\nReturn your files fixed so it passes."
                   if fix else "")


def row(kind: str, source: str, key: str, user: str, answer: str, **meta) -> dict:
    """A row as data.py writes it."""
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user},
                         {"role": "assistant", "content": answer}],
            "meta": {"key": key, "split": "train", "theme": "backend", "language": "python", "role": "implement",
                     "kind": kind, "source": source, **meta}}


def tickets_of(bench: Path) -> dict:
    out = {}
    for p in sorted(Path(bench).iterdir()) if Path(bench).exists() else []:
        for t in (load(p / "plan.json") or {}).get("tickets", []):
            out[f"{p.name}/{t['id']}"] = (p.name, t)
    return out


class Mock:
    def __init__(self, cfg, upstream: FakeUpstream):
        self.cfg, self.up = cfg, upstream
        self.run = None  # bh_run.Run, set by main
        self.fail = set(cfg.extra.get("dry_fail", ()))  # command kinds to fail, for the guard tests

    def close(self):
        self.up.close()

    def __call__(self, cmd: Cmd) -> None:
        if cmd.kind in self.fail:
            raise RuntimeError(f"(dry run) {cmd.kind} fails as asked")
        fn = getattr(self, "k_" + cmd.kind.replace("-", "_"), None)
        if fn is None:
            raise AssertionError(f"no mock for command kind {cmd.kind!r}: {cmd.sh}")
        fn(cmd.args, cmd)

    # ---- nothing to write
    def k_gpu_free(self, a, c): pass
    def k_serve_vllm(self, a, c):
        if a["name"] == "student" and self.cfg.extra.get("dry_vllm_student_fail"):
            raise RuntimeError("(dry run) vLLM does not know this architecture")
    def k_serve_llama(self, a, c): pass
    def k_setup_vm(self, a, c): pass
    def k_setup_muggebench(self, a, c): pass
    def k_versions(self, a, c): pass
    def k_vllm_register(self, a, c): pass
    def k_vocab_register(self, a, c): pass
    def k_tokcheck(self, a, c): pass
    def k_roundtrip(self, a, c): pass

    def k_vllm_check(self, a, c):
        dump(Path(a["out"]), {"ok": True, "vllm": self.cfg.vllm, "models": [{"model": m, "ok": True, "problems": []}
                                                                            for m in a["models"]]})

    # ---- data path: data.py's commands, each writing what it writes under W/data
    def d(self, a) -> Path:
        return Path(a["work"]) / "data"

    def k_data_export(self, a, c):
        out = self.d(a) / "export"
        for split in ("train", "heldout"):
            for i in range(PROJECTS):
                name = f"kit{i:02d}-{split}"
                dump(out / f"{split}-bench" / name / "plan.json", plan(name, TICKETS))
        t = plan("kitx", 1)["tickets"][0]
        extra = [row("fix", "bhdata", f"kitx/x{i}", pack("kitx", t, f"err {i}"), answer_for(f"x{i}", 0)) for i in range(5)]
        extra.append(row("review", "bhdata", "kitx/r1", "review this", "LGTM"))
        write_jsonl(out / "extra_rows.jsonl", extra)
        dump(self.d(a) / "stats" / "export.json", {k: str(out / v) for k, v in (
            ("train_bench", "train-bench"), ("heldout_bench", "heldout-bench"), ("extra_rows", "extra_rows.jsonl"))})

    def k_data_check(self, a, c):
        for w in ("train", "heldout"):
            tks = tickets_of(self.d(a) / "export" / f"{w}-bench")
            broken = {k: "acceptance: exit 1" for k in list(tks)[:1]}
            dump(self.d(a) / "stats" / f"check-{w}.json", {"tickets": len(tks), "passed": len(tks) - len(broken),
                                                          "broken": broken, "excluded": {}})

    def k_bench(self, a, c):
        """bench/run.ts: each ticket and seed as one or two calls through the proxy (a write,
        and a fix when the write 'fails'), results, transcripts and report as run.ts writes them."""
        bench, out, url = Path(a["bench"]), Path(a["out"]), a["url"]
        tickets = []
        for p in sorted(bench.iterdir()):
            pj = load(p / "plan.json")
            if pj:
                tickets += [(p.name, t) for t in pj["tickets"]]
        if a.get("tickets"):
            tickets = tickets[:a["tickets"]]
        rows, trans = [], []
        served = (self.run.servers.current() or {}).get("model", "") if self.run else ""
        for seed in range(a["seeds"]):
            for project, t in tickets:
                calls = []
                user = pack(project, t)
                ans = self.call(url, user, seed, c)
                calls.append({"prompt": user, "answer": ans})
                first = h(a["name"], project, t["id"], seed, "w") % 3 != 0
                solved = first
                if not first:
                    user = pack(project, t, f"AssertionError seed {seed}")
                    ans = self.call(url, user, seed, c)
                    calls.append({"prompt": user, "answer": ans})
                    solved = h(a["name"], project, t["id"], seed, "f") % 2 == 0
                if a["name"] == "student" and h(project, t["id"], seed) % 4 == 0:
                    solved = False
                if a["name"] == "teacher" and any(x in served for x in self.cfg.extra.get("dry_g2_fail", ())):
                    solved = False
                key = f"{project}/{t['id']}"
                rows.append({"project": project, "ticket": t["id"], "key": key, "seed": seed, "solved": solved,
                             "visible": solved, "hidden": solved, "attempts": len(calls), "firstTry": first and solved,
                             "calls": len(calls), "emptyAnswers": 0, "refused": 0, "promptTokens": 1000,
                             "completionTokens": sum(len(x["answer"]) // 4 for x in calls)})
                trans.append({"key": key, "seed": seed, "calls": calls})
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / "results.jsonl", rows)
        if a["transcripts"]:
            write_jsonl(out / "transcripts.jsonl", trans)
        calls = sum(r["calls"] for r in rows)
        dump(out / "report.json", {"score": round(100 * sum(r["solved"] for r in rows) / max(1, len(rows)), 1),
                                   "tickets": len(tickets), "seeds": a["seeds"], "wallMs": max(1, calls) * 120})

    def call(self, url: str, user: str, seed: int, c: Cmd) -> str:
        """One engine call as bench/run.ts sends it, from the command's own flags; the
        engine asks a call for what is left of the ticket's 60K budget (the proxy caps it)."""
        w = shell_words(c)
        flag = lambda f: w[w.index(f) + 1]
        body = {"model": flag("--model"), "messages": [{"role": "system", "content": SYSTEM},
                                                       {"role": "user", "content": user}],
                "max_tokens": 60000, "temperature": float(flag("--temperature")), "seed": seed,
                "top_p": float(flag("--top-p")), "top_k": int(flag("--top-k")),
                "chat_template_kwargs": {"enable_thinking": flag("--thinking") != "off"}}
        req = urllib.request.Request(f"{url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def harvest(self, a, which: str, source: str) -> tuple[list[dict], dict]:
        """harvest.harvest as far as the harness sees it: per ticket the shortest passing
        last call (write or fix), write-from-fix when no seed passed first time; cut calls
        (the proxy log) and broken tickets never rows; Stats.as_dict's counts."""
        from bh.wire import call_key
        broken = set((load(self.d(a) / "stats" / f"check-{which}.json") or {}).get("broken", {}))
        st = {"runs": 0, "answers": 0, "cut_or_forced": 0, "parser_drops": 0, "solved_runs": 0, "candidates": {},
              "dropped": {}, "rows": {}, "tickets_with_rows": 0, "warnings": []}
        best: dict = {}
        for run in map(Path, a["runs"]):
            cut = {e["key"] for e in read_jsonl(run / "proxy" / "calls.jsonl") if e["cut"]}
            res = {(r["key"], r["seed"]): r for r in read_jsonl(run / "results.jsonl")}
            for t in read_jsonl(run / "transcripts.jsonl"):
                st["runs"] += 1
                st["answers"] += len(t["calls"])
                st["cut_or_forced"] += sum(call_key(x["prompt"], x["answer"]) in cut for x in t["calls"])
                if not res.get((t["key"], t["seed"]), {}).get("solved"):
                    continue
                st["solved_runs"] += 1
                last = t["calls"][-1]
                if t["key"] in broken:
                    st["dropped"]["broken"] = st["dropped"].get("broken", 0) + 1
                    continue
                if call_key(last["prompt"], last["answer"]) in cut:
                    st["dropped"]["cut"] = st["dropped"].get("cut", 0) + 1
                    continue
                kind = "write" if len(t["calls"]) == 1 else "fix"
                cands = [(kind, last["prompt"])] + ([("write-from-fix", t["calls"][0]["prompt"])] if kind == "fix" else [])
                for kd, prompt in cands:
                    r = row(kd, source, t["key"], prompt, last["answer"], seed=t["seed"], split=which)
                    k = (t["key"], kd)
                    if k not in best or len(r["messages"][2]["content"]) < len(best[k]["messages"][2]["content"]):
                        best[k] = r
        rows = [r for (key, kind), r in best.items() if kind != "write-from-fix" or (key, "write") not in best]
        for r in rows:
            k = r["meta"]["kind"]
            st["rows"][k] = st["rows"].get(k, 0) + 1
        st["tickets_with_rows"] = len({r["meta"]["key"] for r in rows})
        st["drop_share"] = round(st["cut_or_forced"] / max(1, st["answers"]), 4)
        return rows, st

    def k_data_teacher(self, a, c):
        rows, st = self.harvest(a, "train", "teacher")
        write_jsonl(self.d(a) / "rows" / "teacher.jsonl", rows)
        dump(self.d(a) / "stats" / "teacher.json", st)

    def k_data_rft(self, a, c):
        rows, st = self.harvest(a, "heldout", "student-rft")
        write_jsonl(self.d(a) / "rows" / "rft.jsonl", rows)
        dump(self.d(a) / "stats" / "rft.json", st)

    def k_data_extra(self, a, c):
        rows = read_jsonl(self.d(a) / "export" / "extra_rows.jsonl")
        worker = [{**r, "meta": {**r["meta"], "source": "bhdata-extra", "checked": False}} for r in rows
                  if r["meta"]["kind"] != "review"]
        write_jsonl(self.d(a) / "rows" / "extra.jsonl", worker)
        write_jsonl(self.d(a) / "rows" / "extra-review.jsonl", [r for r in rows if r["meta"]["kind"] == "review"])
        dump(self.d(a) / "stats" / "extra.json", {"total": len(rows), "review": len(rows) - len(worker),
                                                 "unchecked": len(worker), "rows": len(worker)})

    def k_data_build(self, a, c):
        parts = ("teacher", "extra") if a["round"] == 1 else ("teacher", "extra", "own-repair", "rft")
        rows, seen = [], set()
        for n in parts:
            for r in read_jsonl(self.d(a) / "rows" / f"{n}.jsonl"):
                k = (r["messages"][1]["content"], r["messages"][2]["content"])
                if k not in seen:
                    seen.add(k)
                    rows.append(r)
        name = "train.jsonl" if a["round"] == 1 else f"train-round{a['round']}.jsonl"
        write_jsonl(self.d(a) / name, rows)
        dump(self.d(a) / "stats" / f"{Path(name).stem}.json", {"rows": len(rows), "tokenizer_check": "ok"})

    def k_data_repair_requests(self, a, c):
        tks = tickets_of(self.d(a) / "export" / "heldout-bench")
        out = []
        for run in map(Path, a["runs"]):
            for r in read_jsonl(run / "results.jsonl"):
                if r["solved"] or r["key"] not in tks:
                    continue
                project, t = tks[r["key"]]
                out.append({"id": f"{r['key']}#s{r['seed']}", "key": r["key"], "seed": r["seed"],
                            "messages": [{"role": "system", "content": SYSTEM},
                                         {"role": "user", "content": pack(project, t, "student broke it")}]})
        write_jsonl(self.d(a) / "round2" / "repair-requests.jsonl", out)
        dump(self.d(a) / "stats" / "repair-requests.json", {"requests": len(out)})

    def k_data_answer(self, a, c):
        """The real data.answer_requests, through the proxy to the fake server."""
        from bh import data as D
        w = shell_words(c)
        D.answer_requests(Path(a["requests"]), Path(a["out"]), url=a["url"], model=w[w.index("--model") + 1],
                          samples=a["samples"], max_tokens=int(w[w.index("--max-tokens") + 1]),
                          top_k=int(w[w.index("--top-k") + 1]), thinking=w[w.index("--thinking") + 1] == "on",
                          concurrency=8)

    def k_data_repair(self, a, c):
        reqs = {q["id"]: q for q in read_jsonl(self.d(a) / "round2" / "repair-requests.jsonl")}
        rows, n = [], 0
        for x in read_jsonl(Path(shell_words(c)[shell_words(c).index("--answers") + 1])):
            ok = [y for y in x["answers"] if not y["cut"] and y["content"]]
            n += len(x["answers"])
            if ok and x["id"] in reqs:
                q = reqs[x["id"]]
                best = min(ok, key=lambda y: len(y["content"]))
                rows.append({"messages": [*q["messages"], {"role": "assistant", "content": best["content"]}],
                             "meta": {"key": q["key"], "split": "heldout", "kind": "fix", "source": "own-repair",
                                      "seed": q["seed"]}})
        write_jsonl(self.d(a) / "rows" / "own-repair.jsonl", rows)
        dump(self.d(a) / "stats" / "own-repair.json", {"answers": n, "rows": len(rows)})

    # ---- training and export
    def k_sft(self, a, c):
        out = Path(a["out"])
        b = self.cfg.base_model
        dump(out / "config.json", {"architectures": [b.archs[0]]})
        (out / "tokenizer.json").write_text("{}")
        rows = sum(len(read_jsonl(Path(r))) for r in a["rows"])
        dump(out / "sft.json", {"base": a["base"], "arch": b.archs[0], "mode": a["mode"], "updates": 10, "planned": 10,
                                "stopped": "", "ce": 0.5, "kl": 0.1 if a["kd"] else 0.0, "skipped": 0, "minutes": 0.1,
                                "rows": {"rows_in": rows, "rows": rows, "dropped": {}},
                                "examples": {"too_long": 0, "kd_rows": rows if a["kd"] else 0, "teacher_id_rows": 0},
                                "lora": {}, "roundtrip": {"ok": True}, "kd_weight": 0.5 if a["kd"] else 0.0})

    def k_traces(self, a, c):
        write_jsonl(Path(a["out"]), [])

    def k_vocab_run(self, a, c):
        out = Path(a["out"])
        dump(out / "config.json", {"architectures": [self.cfg.base_model.archs[0]]})
        dump(out / "vocab_index.json", {"kept": [0, 1, 2]})

    def k_gguf_convert(self, a, c):
        Path(a["out"]).parent.mkdir(parents=True, exist_ok=True)
        Path(a["out"]).write_bytes(b"GGUF" * 1000)

    def k_calib(self, a, c):
        Path(a["out"]).write_text("calibration text\n")

    def k_imatrix(self, a, c):
        Path(a["out"]).write_bytes(b"IMAT")

    def k_quantize(self, a, c):
        Path(a["out"]).write_bytes(b"Q" * (100 + 10 * len(a["type"])))

    def k_footprint(self, a, c):
        """Sizes as the plan estimates them: Qwen3.5-2B fits from Q6_K down (1.3 GB file
        after the vocabulary cut), MiniCPM5-2B from Q4_K_S, the 0.8B at Q8_0."""
        from bh.footprint import name
        qt = Path(a["gguf"]).stem
        fp = {"Q8_0": 2.35e9, "Q6_K": 1.62e9, "Q5_K_M": 1.45e9, "Q4_K_M": 1.30e9, "Q4_K_S": 1.85e9, "IQ4_XS": 1.75e9,
              "Q5_K_M@llama": 2.25e9}.get(qt, 1.5e9)
        if self.cfg.base_model.family == "llama":
            fp = {"Q5_K_M": 2.25e9, "Q4_K_M": 2.05e9, "Q4_K_S": 1.92e9, "IQ4_XS": 1.86e9}.get(qt, fp)
        if self.cfg.base == "qwen3.5-0.8b":
            fp = 0.95e9 if qt == "Q8_0" else 0.6e9
        fp = int(fp)
        dump(Path(a["out"]), {"gguf": a["gguf"], "file_bytes": int(fp * 0.8), "footprint": fp, "runs": [],
                              "source": "llama.cpp buffers", "name": name(a["version"], fp)})

    def k_muggebench_sh(self, a, c):
        Path(a["out"], "muggebench.txt").write_text("score 51.0% [39.0, 63.0] footprint 1.62 GB TOO CLOSE\n")


def shell_words(cmd: Cmd) -> list[str]:
    """The command's words, after any `cd dir &&`."""
    return shlex.split(cmd.sh.splitlines()[-1].split(" || ")[0].split(" && ")[-1])
