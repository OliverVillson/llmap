#!/usr/bin/env python3
"""The Centipede train harness: BH<version>-<GB>, a small coding model for the Centipede
crew, taught by a large teacher on the engine's own calls. Forked from centipede.py (run
1: Granite 3B from 30B): one config, one command per stage, resumable, results uploaded
after every step.

    python3 scripts/bh_run.py g0                 # the G0 smoke: end to end on the 4060
    python3 scripts/bh_run.py s1                 # B200 session 1: teacher, G2, answers, SFT, smoke
    python3 scripts/bh_run.py laptop             # export, footprint, student samples
    python3 scripts/bh_run.py eval               # MuggeBench score (a second laptop, at the same time)
    python3 scripts/bh_run.py s2                 # B200 session 2: the own-mistakes round
    python3 scripts/bh_run.py s1 sft..smoke      # a span of the plan (also: data.., ..sft, one stage)
    python3 scripts/bh_run.py s1 status          # the plan, what is done, the results so far
    python3 scripts/bh_run.py g0 start           # detached: survives logging out; `watch` and `stop` it
    options: --set key=value  --dry-run DIR  --force  --shutdown   (run sheet: scripts/bh/README.md)

Stages, in order (each runs alone and is skipped when its output exists; inputs another
machine made are fetched from the bucket):

  setup            the VM (scripts/setup_vm.sh, pinned: vLLM, transformers major, llama.cpp,
                   UV_EXCLUDE_NEWER), MuggeBench's sandbox images, the teacher's boot dry-run
  data             data.py export (bhdata kits -> train-bench / heldout-bench / free rows) and
                   check (reference answers in the sandbox, MuggeBench decontam)
  teacher-serve    vLLM with the NVFP4 teacher, thinking off; G2 in its first minutes: >= 55%
                   of 200 kit tickets and >= 250 answers/min, else the next teacher
  teacher-answers  the teacher works train-bench through bench/run.ts (the engine's write ->
                   check -> fix loop), 2-4 seeds; data.py teacher / extra / build -> rows;
                   its top-20 log-probs for the logit KL (Qwen path)
  sft              full-FT SFT of the base on the rows, prompts masked (scripts/bh/sft.py)
  smoke            the bf16 student on the public spike through bench/run.ts, ~10 minutes
  export           (laptop) vocab cut, BF16 GGUF, imatrix on training-shaped tickets, quant ladder
  footprint        (laptop) each quant's footprint; the best that fits 2 GB ships as BH<v>-<GB>
  score            (eval) MuggeBench: the dev set via muggebench.sh, else bench/run.ts
  round2-sample    the student answers the held-out tickets (laptop GGUF or B200 bf16)
  round2-repair    data.py repair-requests / answer (teacher, via the proxy) / repair / rft /
                   build --round 2: own-repair and the student's own passes join round 1's rows
  round2-rft       SFT again with the student's own passes and the repairs, then its smoke

After every step (and when the run ends, however it ends) everything small and the logs go
to the bucket, and the big files once each.

Every bench run goes through the recording proxy (scripts/bh/proxy.py): finite top_k,
thinking off, the per-call cap, and a log of every call's finish (so cut answers never
become rows). Guards from past runs: drop counts over 20% warned, forced/cut/thinking rows
never trained on, the GPU-busy check in each long step's first minutes, one server per
GPU, the tokenizer round-trip on a real row, optional steps that cannot fail the job.

--dry-run DIR runs every stage on the CPU with mocks (scripts/bh/mock.py): DIR/<profile> is
the machine, DIR/bucket the bucket, so s1, laptop and s2 dry-run in turn with one DIR hand
their results over as the real machines do. The commands (bash syntax), paths, proxy,
guards, restores and uploads are real, and so is data.py answer (through the proxy); GPUs,
models and data.py's other commands (bhdata, the sandbox) are not: the data thread's own
smoke (`data.py smoke`) runs those on the public spike.

Everything data.py writes is under W/data (W = Config.work); the guards' output, the SFT's
input, is W/rows/train.jsonl (round 1) and W/rows/train-round2.jsonl.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import REPO, Done, Skip, Step, run_plan  # noqa: E402

from bh import bench as B  # noqa: E402
from bh import config as C  # noqa: E402
from bh import live  # noqa: E402
from bh import export as X  # noqa: E402
from bh import rows as R  # noqa: E402
from bh import sampling as S  # noqa: E402
from bh.exec import Bucket, Cmd, Exec, StepTimeout, dump, load, read_jsonl, write_jsonl  # noqa: E402
from bh.proxy import recording  # noqa: E402
from bh.servers import Servers, llama_spec, vllm_spec  # noqa: E402

STAGES = ("setup", "data", "teacher-serve", "teacher-answers", "sft", "smoke", "export", "footprint", "score",
          "round2-sample", "round2-repair", "round2-rft")
ALIASES = {"round2": ("round2-sample", "round2-repair", "round2-rft")}
PLAN = {
    "g0": STAGES,
    "s1": ("setup", "data", "teacher-serve", "teacher-answers", "sft", "smoke"),
    "laptop": ("export", "footprint", "round2-sample"),
    "eval": ("score",),
    "s2": ("setup", "data", "teacher-serve", "round2-sample", "round2-repair", "round2-rft"),
}
SANDBOX_IMAGES = ("c", "node", "python", "web", "dotnet")
LLAMA_BINS = ("llama-server", "llama-quantize", "llama-imatrix", "llama-tokenize")
DATA_PY = "python scripts/bh/data.py"  # the venv python: build loads the student tokenizer


def q(x) -> str:
    return X.q(x)


class Run:
    def __init__(self, cfg: C.Config, ex: Exec, *, force: bool = False, url: str | None = None):
        self.cfg, self.ex, self.force = cfg, ex, force
        self.W = Path(cfg.work)
        self.servers = Servers(cfg, ex)
        self.local_url = url or self.servers.url  # where this machine's servers answer (the dry run's stand-in)
        self.url = self.local_url  # where the current role's model answers: here, or another machine
        self.model = ""  # the served name on another machine, when it is not the role's own
        self.bucket = Bucket(cfg.bucket, cfg.run)
        self.nvme = Path(cfg.nvme)
        self.llama = self.nvme / "llama.cpp"
        self.bins = self.llama / "build" / "bin"
        self.t0 = time.monotonic()
        self.stage_keys: list[str] = []
        self.uploaded = False  # the run's last upload went through, every hand-over file included
        self.live = False  # this process runs stages (status only reads what earlier runs left)

    # ---------- paths ----------
    def p(self, *parts) -> Path:
        return self.W.joinpath(*parts)

    @property
    def data(self) -> Path:
        return self.p("data")

    @property
    def rows(self) -> Path:
        return self.p("rows")

    def model_dir(self, name: str) -> Path:
        return self.p("sft", name, "model")

    def base_dir(self) -> Path:
        b = self.cfg.base_model
        d = self.nvme / "models" / b.hf.split("/")[-1]
        return d if (d / "config.json").exists() or self.ex.dry else Path(b.hf)

    def teacher_dir(self, key: str) -> str:
        t = C.TEACHERS[key]
        d = self.nvme / "models" / t.hf.split("/")[-1]
        return str(d) if (d / "config.json").exists() else t.hf

    # ---------- session clock ----------
    def elapsed_min(self) -> float:
        """Minutes this machine has been billed: its uptime on the B200 (billed from
        boot), this command's run time elsewhere."""
        if self.cfg.where == "b200" and not self.ex.dry:
            try:
                return float(Path("/proc/uptime").read_text().split()[0]) / 60
            except OSError:
                pass
        return (time.monotonic() - self.t0) / 60

    def cap(self, wanted: float, after: float) -> float:
        """A time cap for a step: wanted, less what the session cannot spare once `after`
        minutes are kept for the steps that must follow. 0 when there is no session clock."""
        if not self.cfg.session_minutes:
            return wanted
        left = self.cfg.session_minutes - self.elapsed_min() - after
        if left < 1:
            raise Skip(f"the session's {self.cfg.session_minutes:g} min leave no time for it "
                       f"({after:g} min kept for what follows)")
        return min(wanted, left)

    # ---------- restore ----------
    def need(self, name: str, path: Path, *, untar: Path | None = None) -> bool:
        """path exists here, fetched from the bucket (and unpacked into untar) when not."""
        if path.exists():
            return True
        if not self.bucket.ok():
            return False
        if untar is None:
            if not self.bucket.get(name, path):
                return False
            self.mark(path, name)  # it is in the bucket already: never sent back
            return True
        tar = untar.with_name(untar.name + ".tar")
        if not self.bucket.get(name, tar):
            return False
        untar.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "xf", str(tar), "-C", str(untar)], check=True)
        tar.unlink()
        return path.exists()

    def wait_for(self, name: str, path: Path, *, untar: Path | None = None) -> bool:
        """need(), polling the bucket for up to wait_minutes while another machine is still
        making it (the eval laptop waits for the ship)."""
        minutes, every = (min(self.cfg.wait_minutes, 2), 1) if self.ex.dry else (self.cfg.wait_minutes, 60)
        deadline = time.monotonic() + 60 * minutes
        said = False
        while not self.need(name, path, untar=untar):
            if time.monotonic() >= deadline:
                return False
            if not said:
                print(f"waiting for {name} from another machine (up to {minutes:g} min)", flush=True)
                said = True
            time.sleep(every)
        return True

    # ---------- machines ----------
    def serve(self, step, role: str, start) -> dict:
        """The role's model (teacher, student, eval): on the machine config.backends names for
        it, which serves it already, else started here by start()."""
        b = C.backend(self.cfg, role)
        if b is None:
            self.url, self.model = self.local_url, ""
            return start()
        self.url, self.model = b["url"].rstrip("/").removesuffix("/v1"), b.get("model", "")
        from bh.servers import get_json
        try:
            get_json(f"{self.url}/v1/models", timeout=30)
        except OSError as e:
            raise RuntimeError(f"{b['name']} ({b['url']}) serves {role} but does not answer: {e}") from e
        print(f"{role}: served by {b['name']} at {b['url']}", flush=True)
        return {"remote": b["name"], "url": b["url"]}

    # ---------- the proxy ----------
    def recorder(self, out: Path, *, logprobs: bool = False):
        b = self.cfg.base_model
        return recording(self.url, out, top_k=self.cfg.top_k, top_p=self.cfg.top_p, temperature=self.cfg.temperature,
                         thinking=self.cfg.thinking, max_tokens=self.cfg.max_call_tokens,
                         min_p=S.MIN_P if b.min_p is None else b.min_p,
                         logprobs=self.cfg.teacher_logprobs if logprobs and self.cfg.kd else 0)

    def bench(self, step, bench: Path, out: Path, name: str, *, seeds: int, concurrency: int, logprobs=False,
              transcripts=False, tickets=0, minutes: float = 0, partial_ok: bool = False) -> dict:
        """One bench run through the recording proxy; its stats and the proxy's counts.
        minutes: stopped there (the session clock's share); with partial_ok what finished
        by then counts, else StepTimeout."""
        out.mkdir(parents=True, exist_ok=True)
        stopped = None
        with self.recorder(out / "proxy", logprobs=logprobs) as (url, rec):
            cmd = B.command(self.cfg, bench, out, url, name, model=self.model, seeds=seeds, concurrency=concurrency,
                            transcripts=transcripts, tickets=tickets)
            cmd.timeout_min = minutes
            try:
                self.ex.run(step, cmd)
            except StepTimeout as e:
                if not partial_ok:
                    raise
                stopped = str(e)
        st = {**B.stats(out), "proxy": dict(rec.counts)}
        if stopped:
            st["stopped"] = stopped
            self.ex.warn(f"{name}: {stopped}; {st['runs']} runs finished by then count")
        if st["proxy"]["think"]:
            self.ex.warn(f"{name}: {st['proxy']['think']} of {st['proxy']['calls']} answers opened a thinking block "
                         "with thinking off")
        if st["proxy"]["top_k_set"] or st["proxy"]["thinking_set"]:
            self.ex.warn(f"{name}: the proxy set top_k on {st['proxy']['top_k_set']} and thinking on "
                         f"{st['proxy']['thinking_set']} calls the caller sent otherwise")
        if st["proxy"]["calls"] and st["proxy"]["cut"] / st["proxy"]["calls"] > self.cfg.drop_warn:
            self.ex.warn(f"{name}: {st['proxy']['cut']} of {st['proxy']['calls']} answers cut at the "
                         f"{self.cfg.max_call_tokens}-token cap")
        dump(out / "stats.json", st)
        return st

    # ================= stages =================

    # ---------- setup ----------
    def setup_missing(self) -> list[str]:
        out = [] if (REPO / ".env.vm").exists() else [".env.vm (the venvs)"]
        out += [b for b in LLAMA_BINS if not (self.bins / b).exists()]
        if not shutil.which("bun") and not (Path.home() / ".bun" / "bin" / "bun").exists():
            out.append("bun")
        return out

    def setup_done(self) -> bool:
        return not self.ex.dry and not self.setup_missing() and self.p("setup.json").exists()

    def run_setup(self, step: Step) -> None:
        cfg = self.cfg
        probs = C.pins_problems(cfg)
        if probs and cfg.where == "b200":
            raise RuntimeError("refusing a paid session without exact pins: " + "; ".join(probs))
        if cfg.where == "b200":
            teacher = C.TEACHERS[cfg.teachers[0]].hf
            self.ex.run(step, Cmd(f"NVME={q(cfg.nvme)} TEACHER={q(teacher)} STUDENT={q(cfg.base_model.hf)} "
                                  f"LLAMA_CPP_SHA={cfg.llama_cpp_sha} UV_EXCLUDE_NEWER={cfg.uv_exclude_newer} "
                                  f"SKIP_POLYGLOT=1 bash scripts/setup_vm.sh", "setup-vm"))  # setup_vm.sh pins vllm==0.31.0 itself
            self.ex.run(step, Cmd(f"NVME={q(cfg.nvme)} bash scripts/setup_muggebench.sh {' '.join(SANDBOX_IMAGES)}",
                                  "setup-muggebench"))
        missing = [] if self.ex.dry else self.setup_missing()
        if missing:
            raise RuntimeError("missing here: " + ", ".join(missing) + ". On a Linux GPU box run "
                               "`NVME=<dir> SKIP_POLYGLOT=1 bash scripts/setup_vm.sh` (the 4060 under WSL2 too); "
                               "on the Mac use muggebench.sh for scoring")
        # versions: exact vLLM, transformers' major (5.x renamed the dtype kwarg and the tokenizer backends)
        self.ex.run(step, Cmd(
            f'"${{LOBBOT_VLLM_PY:-python}}" -c "import vllm,sys; v=vllm.__version__; print(\'vllm\', v); '
            f"sys.exit(0 if {q(cfg.vllm)}.endswith(\'==\'+v) or \'@\' in {q(cfg.vllm)} else 1)\" && "
            f'python -c "import transformers,sys; v=transformers.__version__; print(\'transformers\', v); '
            f"sys.exit(0 if int(v.split(\'.\')[0])=={cfg.transformers_major} else 1)\"", "versions"))
        # the boot dry-run: the teacher ladder resolves in this vLLM, no GPU needed
        models = " ".join(q(self.teacher_dir(t)) for t in cfg.teachers)
        boot = self.p("teacher", "boot.json")
        boot.parent.mkdir(parents=True, exist_ok=True)
        self.ex.run(step, Cmd(f'"${{LOBBOT_VLLM_PY:-python}}" scripts/bh/vllm_check.py --pin {q(cfg.vllm)} '
                              f"--out {q(boot)} {models}", "vllm-check", {"out": str(boot), "models": list(cfg.teachers)}))
        dump(self.p("setup.json"), {"time": time.strftime("%F %T"), "pins": {
            "vllm": cfg.vllm, "llama_cpp": cfg.llama_cpp_sha, "uv_exclude_newer": cfg.uv_exclude_newer,
            "transformers_major": cfg.transformers_major}, "boot": load(boot)})

    # ---------- data ----------
    def bench_dir(self, which: str) -> Path:
        """data.py's exported bench projects: train or heldout."""
        return self.data / "export" / f"{which}-bench"

    def data_cmd(self, sub: str, kind: str, extra: str = "", **args) -> Cmd:
        """One data.py command (scripts/bh/data.py, the data thread's); everything it
        writes is under W/data."""
        env = {"NVME": self.cfg.nvme, **({"BHDATA": self.cfg.kits} if self.cfg.kits else {})}  # NVME: decontam finds the dev set
        return Cmd(f"{DATA_PY} {sub} --work {q(self.W)} --sandbox {self.cfg.sandbox} {extra}".rstrip(), kind,
                   {"work": str(self.W), **args}, env=env)

    DATA_TAR = ("export", "stats/export.json", "stats/check-train.json", "stats/check-heldout.json")

    def need_data(self) -> None:
        """The exported, checked bench projects here, from the bucket when another machine
        made them."""
        if not self.need("data.tar", self.data / "stats" / "check-heldout.json", untar=self.data):
            raise RuntimeError("no exported data here or in the bucket: run the data stage first")

    def data_done(self) -> bool:
        return all((self.data / "stats" / f"check-{w}.json").exists() for w in ("train", "heldout"))

    def run_data(self, step: Step) -> None:
        """data.py export (bhdata's kits as MuggeBench-format projects, split by kit, and its
        free rows), then data.py check (every reference answer in the sandbox; MuggeBench
        decontam). G0 keeps the first limit_kits kits of each split (data.py --limit-kits)."""
        if self.need("data.tar", self.data / "stats" / "check-heldout.json", untar=self.data):
            raise Done("restored from the bucket")
        limit = f"--limit-kits {self.cfg.limit_kits}" if self.cfg.limit_kits else ""
        self.ex.run(step, self.data_cmd("export", "data-export", limit, limit_kits=self.cfg.limit_kits))
        if not B.projects(self.bench_dir("train")):
            raise RuntimeError(f"no bench project in {self.bench_dir('train')}")
        self.ex.run(step, self.data_cmd("check", "data-check"))
        for w in ("train", "heldout"):
            rep = load(self.data / "stats" / f"check-{w}.json") or {}
            n, bad = rep.get("tickets") or 0, len(rep.get("broken") or {}) + len(rep.get("excluded") or {})
            if n and bad / n > 0.1:
                self.ex.warn(f"data: {len(rep.get('broken') or {})} broken and {len(rep.get('excluded') or {})} "
                             f"decontam-excluded of {n} {w} tickets; none of them become rows")
        self.tar_up(self.data, "data.tar", self.DATA_TAR)

    # ---------- the teacher ----------
    def teacher_json(self) -> Path:
        return self.p("teacher", "teacher.json")

    def teacher_done(self) -> bool:
        """A teacher passed G2 in this run, here or on an earlier session (restored): session
        2 boots it when a stage needs it instead of paying for G2 again."""
        if not self.teacher_json().exists():
            self.need("teacher.json", self.teacher_json())
        t = load(self.teacher_json())
        return bool(t and t.get("chosen"))

    def serve_teacher(self, step, key: str) -> dict:
        t = C.TEACHERS[key]
        spec = vllm_spec(self.cfg, self.teacher_dir(key), "teacher", extra=t.vllm_args)
        return self.serve(step, "teacher", lambda: self.servers.ensure(step, spec, health_s=1800))

    def run_teacher_serve(self, step: Step) -> None:
        """Boot each teacher of the ladder in turn until one passes G2: >= g2_min_pass % of
        g2_tickets kit tickets solved and >= g2_min_apm answers a minute, measured on the
        first G2 tickets of train-bench (thinking off). G2's transcripts are kept: they
        are training tickets, so harvest uses them too."""
        cfg = self.cfg
        self.need_data()
        out = self.p("teacher")
        tried = (load(self.teacher_json()) or {}).get("tried", [])
        done_keys = {x["teacher"] for x in tried}
        for key in cfg.teachers:
            if key in done_keys:
                continue
            g2 = out / "g2" / key
            rec = {"teacher": key, "hf": C.TEACHERS[key].hf, "time": time.strftime("%F %T")}
            try:
                self.serve_teacher(step, key)
                minutes = self.cap(cfg.g2_minutes, after=0)
                st = self.bench(step, self.bench_dir("train"), g2, "teacher", seeds=1,
                                concurrency=cfg.bench_concurrency, logprobs=True, transcripts=True,
                                tickets=cfg.g2_tickets, minutes=minutes)
                rec.update(g2=st, minutes=minutes)
                ok_pass = (st["score"] or 0) >= cfg.g2_min_pass
                ok_apm = (st["answers_per_min"] or 0) >= cfg.g2_min_apm
                rec["pass"] = ok_pass and ok_apm
                if not rec["pass"]:
                    rec["why"] = (f"{st['score']}% solved (need {cfg.g2_min_pass}), {st['answers_per_min']} answers/min "
                                  f"(need {cfg.g2_min_apm})")
            except Skip:
                raise
            except Exception as e:  # noqa: BLE001  (a teacher that does not boot: the next one)
                rec.update(**{"pass": False, "why": f"{type(e).__name__}: {e}"[:600]})
            tried.append(rec)
            dump(self.teacher_json(), {"tried": tried, "chosen": key if rec["pass"] else None,
                                       "vllm": cfg.vllm, "image": cfg.vllm_image})
            if rec["pass"]:
                print(f"G2: {key} passes: {rec['g2']['score']}% of {rec['g2']['tickets']} tickets, "
                      f"{rec['g2']['answers_per_min']} answers/min", flush=True)
                return
            self.ex.warn(f"G2: {key} fails ({rec['why']}); trying the next teacher")
            self.servers.stop(step)
        raise RuntimeError("no teacher of the ladder passed G2: " + "; ".join(f"{x['teacher']}: {x.get('why')}"
                                                                             for x in tried))

    def chosen_teacher(self) -> str:
        t = load(self.teacher_json()) or {}
        if not t.get("chosen"):
            if not self.need("teacher.json", self.teacher_json()) or not (load(self.teacher_json()) or {}).get("chosen"):
                raise RuntimeError("no teacher has passed G2 yet: run teacher-serve first")
            t = load(self.teacher_json())
        return t["chosen"]

    # ---------- teacher answers ----------
    def answers_done(self) -> bool:
        return (self.rows / "train.jsonl").exists() and (self.rows / "rows.json").exists()

    def run_teacher_answers(self, step: Step) -> None:
        """The teacher works train-bench through the engine's loop (bench/run.ts, shards,
        transcripts, its top-k log-probs logged by the proxy), then data.py turns the runs
        into rows: teacher (harvest and sandbox re-check), extra (bhdata's free rows), build
        (round 1: decontam, dedupe, shuffle, the tokenizer round trip on the student's
        tokenizer). The guards here run last; W/rows/train.jsonl is what the SFT reads."""
        cfg = self.cfg
        if self.need("train-rows.jsonl", self.rows / "train.jsonl"):
            for name in ("rows.json", "topk.jsonl.gz", "probe.json"):
                self.need({"rows.json": "rows-stats.json"}.get(name, name), self.rows / name)
            self.restore_data_rows()
            if (self.rows / "rows.json").exists():
                raise Done("restored from the bucket")
        self.need_data()
        key = self.chosen_teacher()
        self.serve_teacher(step, key)
        minutes = self.cap(cfg.answer_minutes, after=cfg.sft_minutes + cfg.smoke_minutes + 15)
        shards = B.shard(self.bench_dir("train"), self.p("answers", "bench"), cfg.shard_projects)
        t0 = time.monotonic()
        outs = [self.p("teacher", "g2", key)]
        for i, sh in enumerate(shards):
            out = self.p("answers", sh.name)
            outs.append(out)
            if (out / "stats.json").exists():
                continue
            left = minutes - (time.monotonic() - t0) / 60
            if left < 1:
                self.ex.warn(f"teacher answers: the {minutes:.0f}-minute cap left {len(shards) - i} of "
                             f"{len(shards)} shards unanswered; training on what is in")
                break
            st = self.bench(step, sh, out, "teacher", seeds=cfg.samples, concurrency=cfg.bench_concurrency,
                            logprobs=True, transcripts=True, minutes=left, partial_ok=True)
            if st.get("stopped"):
                (out / "stats.json").unlink(missing_ok=True)  # a resume answers this shard again
                break
        outs = [o for o in outs if (o / "transcripts.jsonl").exists()]
        if not outs:
            raise RuntimeError("no teacher transcripts to harvest")
        self.ex.run(step, self.data_cmd("teacher", "data-teacher", self.runs_args(outs), runs=[str(o) for o in outs]))
        extra = False
        if cfg.extra_rows and (self.data / "export" / "extra_rows.jsonl").exists():
            try:  # optional: bhdata's free rows are a bonus on top of the teacher's
                self.ex.run(step, self.data_cmd("extra", "data-extra"))
                extra = True
            except Exception as e:  # noqa: BLE001
                self.ex.warn(f"data.py extra failed ({str(e)[:300]}); training on the teacher's rows alone")
        if not extra:
            write_jsonl(self.data / "rows" / "extra.jsonl", [])
        st = self.build_rows(step, 1)
        hv = load(self.data / "stats" / "teacher.json") or {}
        w = R.drop_warning(hv, cfg.drop_warn)
        if w:
            self.ex.warn(f"teacher harvest: {w}")
        rows_st = {**st, "harvest": hv, "extra": load(self.data / "stats" / "extra.json")}
        rows_st.update(self.topk(outs))
        self.probe(step)
        self.upload_once(self.rows / "train.jsonl", "train-rows.jsonl")
        for n in ("teacher", "extra"):
            self.upload_once(self.data / "rows" / f"{n}.jsonl", f"data-rows-{n}.jsonl")
        dump(self.rows / "rows.json", rows_st)  # last: it marks the stage done

    @staticmethod
    def runs_args(runs: list[Path]) -> str:
        """--runs for data.py, and the proxy's per-call logs beside them (cut calls never rows)."""
        logs = [r / "proxy" / "calls.jsonl" for r in runs if (r / "proxy" / "calls.jsonl").exists()]
        return "--runs " + " ".join(q(r) for r in runs) + ("".join(f" --calls-log {q(x)}" for x in logs[:1])
                                                           + "".join(f" {q(x)}" for x in logs[1:]))

    def restore_data_rows(self) -> None:
        """data.py's round-1 row files (round 2's build adds to them), from the bucket."""
        for n in ("teacher", "extra"):
            self.need(f"data-rows-{n}.jsonl", self.data / "rows" / f"{n}.jsonl")

    def build_rows(self, step, round_: int) -> dict:
        """data.py build for the round, then the guards here: W/rows/train.jsonl (round 1)
        or train-round2.jsonl, and their counts."""
        self.ex.run(step, self.data_cmd("build", "data-build", f"--round {round_} --tokenizer {q(self.base_dir())}",
                                        round=round_))
        name = "train.jsonl" if round_ == 1 else f"train-round{round_}.jsonl"
        kept, dropped = R.filter_rows(read_jsonl(self.data / name), self.cfg.thinking)
        write_jsonl(self.rows / name, kept)
        if dropped:
            self.ex.warn(f"round {round_} rows: the last guards dropped {dropped} "
                         "(shape, system, kind, cut, forced, thinking, parse, canary, dupe)")
        build = load(self.data / "stats" / f"{Path(name).stem}.json") or {}
        if build.get("tokenizer_check") not in (None, "ok"):
            self.ex.warn(f"round {round_}: the tokenizer round trip said {build.get('tokenizer_check')}")
        st = {"rows": len(kept), "guards": dropped, "build": build,
              "kinds": dict(Counter(R.meta(r).get("kind") for r in kept)),
              "sources": dict(Counter(R.meta(r).get("source") for r in kept))}
        print(f"round {round_} rows: {len(kept)} {st['sources']} {st['kinds']}", flush=True)
        if not kept:
            raise RuntimeError(f"no round {round_} rows survived; see {self.data / 'stats'}")
        return st

    def topk(self, outs: list[Path]) -> dict:
        """The teacher's top-k log-probs of the kept rows only, from the proxy logs, as one
        gzipped file (the SFT's logit KL; the rest of the log stays on this disk). A row
        matches its call by (user, answer): harvest renders answers in the canonical form,
        so a teacher answer laid out otherwise, and every write-from-fix row (its answer was
        written for the fix prompt), trains on CE alone."""
        if not self.cfg.kd:
            return {}
        from bh.wire import call_key
        teacher = [r for r in read_jsonl(self.rows / "train.jsonl") if R.meta(r).get("source") == "teacher"]
        keys = {call_key(r["messages"][1]["content"], r["messages"][2]["content"]) for r in teacher}
        out, n = self.rows / "topk.jsonl.gz", 0
        with gzip.open(out.with_name(out.name + ".tmp"), "wt") as f:
            for o in outs:
                p = o / "proxy" / "topk.jsonl"
                if not p.exists():
                    continue
                for line in open(p):
                    k = line[9:33] if line.startswith('{"key": "') else json.loads(line)["key"]
                    if k in keys:
                        keys.discard(k)
                        f.write(line)
                        n += 1
        out.with_name(out.name + ".tmp").replace(out)
        if n < 0.5 * len(teacher):
            self.ex.warn(f"top-k log-probs for only {n} of {len(teacher)} teacher rows: the logit KL covers few rows")
        return {"topk_rows": n, "teacher_rows": len(teacher)}

    def probe(self, step) -> None:
        """One real row and the teacher server's own prompt ids for it (vLLM /tokenize with
        the same chat_template_kwargs): the SFT's round-trip check compares the student's
        tokenization with what was served (same tokenizer on the Qwen path)."""
        rows = read_jsonl(self.rows / "train.jsonl")
        if not rows:
            return
        row = next((r for r in rows if R.meta(r).get("source") == "teacher"), rows[0])
        rec = {"row": row, "thinking": self.cfg.thinking,
               "tokenizer": "same" if self.cfg.base_model.teacher_tokenizer else "other"}
        if self.ex.dry:
            rec["prompt_ids"] = None
        else:
            from bh.servers import post_json
            try:
                r = post_json(f"{self.url}/tokenize", {"model": self.model or "teacher", "messages": row["messages"][:-1],
                                                       "add_generation_prompt": True,
                                                       "chat_template_kwargs": {"enable_thinking": self.cfg.thinking}},
                              timeout=120)
                rec["prompt_ids"] = r.get("tokens")
            except Exception as e:  # noqa: BLE001  (optional: the SFT's round trip runs without the served ids)
                self.ex.warn(f"probe: the teacher server's /tokenize failed ({e}); the SFT's round-trip runs without it")
        dump(self.rows / "probe.json", rec)

    # ---------- SFT ----------
    def sft_done(self, name: str = "sft") -> bool:
        return (self.model_dir(name) / "sft.json").exists()

    def train_rows(self, name: str = "sft") -> Path:
        """The SFT's rows for a model (round 1's for sft, round 2's for round2), here or
        from the bucket."""
        f = "train.jsonl" if name == "sft" else "train-round2.jsonl"
        if not self.need({"train.jsonl": "train-rows.jsonl", "train-round2.jsonl": "round2-rows.jsonl"}[f], self.rows / f):
            raise RuntimeError(f"no {self.rows / f}: run {'teacher-answers' if name == 'sft' else 'round2-repair'} first")
        return self.rows / f

    def run_sft(self, step: Step, name: str = "sft") -> None:
        cfg = self.cfg
        if self.need(f"{name}-model.tar", self.model_dir(name) / "sft.json", untar=self.model_dir(name)):
            raise Done("restored from the bucket")
        self.servers.stop(step)  # the trainer gets the whole GPU
        rows = [self.train_rows(name)]
        for n in ("topk.jsonl.gz", "probe.json"):
            self.need(n, self.rows / n)
        topk = self.rows / "topk.jsonl"
        if cfg.kd and (self.rows / "topk.jsonl.gz").exists() and not topk.exists():
            with gzip.open(self.rows / "topk.jsonl.gz", "rb") as src, open(topk.with_suffix(".part"), "wb") as dst:
                shutil.copyfileobj(src, dst)
            topk.with_suffix(".part").replace(topk)
        minutes = self.cap(cfg.sft_minutes if name == "sft" else cfg.rft_minutes, after=cfg.smoke_minutes + 10)
        full = cfg.sft_mode == "full"
        args = [f"--base {q(self.base_dir())}", *(f"--rows {q(r)}" for r in rows), f"--out {q(self.model_dir(name))}",
                f"--mode {cfg.sft_mode}", f"--epochs {cfg.epochs}", f"--lr {cfg.lr if full else cfg.lora_lr}",
                f"--lora-r {cfg.lora_r}", f"--max-len {cfg.max_len}", f"--batch-tokens {cfg.batch_tokens}",
                f"--max-minutes {minutes:.1f}", f"--save-minutes {cfg.save_minutes}"]
        if cfg.kd and topk.exists():
            args += [f"--topk {q(topk)}", f"--kd-weight {cfg.kd_weight}"]
        if (self.rows / "probe.json").exists():
            args.append(f"--probe {q(self.rows / 'probe.json')}")
        if cfg.thinking:
            args.append("--thinking")
        self.ex.run(step, Cmd("python scripts/bh/sft.py " + " ".join(args), "sft",
                              {"out": str(self.model_dir(name)), "rows": [str(r) for r in rows], "base": str(self.base_dir()),
                               "mode": cfg.sft_mode, "kd": cfg.kd and topk.exists()}, gpu=True, watch_from_event=True))
        rep = load(self.model_dir(name) / "sft.json") or {}
        if rep.get("arch") not in cfg.base_model.archs:
            raise RuntimeError(f"the SFT model is {rep.get('arch')}, not one of {cfg.base_model.archs}")
        if rep.get("stopped"):
            self.ex.warn(f"SFT {name}: {rep['stopped']}")
        self.tar_up(self.model_dir(name), f"{name}-model.tar", None)

    # ---------- smoke ----------
    def smoke_done(self, name: str = "sft") -> bool:
        return self.p("smoke", name, "smoke.json").exists()

    def run_smoke(self, step: Step, name: str = "sft") -> None:
        """The bf16 student on the public spike (15 tickets) through the engine's loop,
        ~10 minutes. Served by vLLM; a student vLLM cannot load (a text-only hybrid
        checkpoint, say) is converted to a BF16 GGUF and served by llama-server instead.
        It never fails the job: it says loudly whether the student writes files at all."""
        cfg = self.cfg
        model = self.model_dir(name)
        if not self.need(f"{name}-model.tar", model / "sft.json", untar=model):
            raise RuntimeError(f"no {name} model at {model}")
        out = self.p("smoke", name)
        served = "vllm"

        def start():
            nonlocal served
            try:
                return self.servers.ensure(step, vllm_spec(cfg, str(model), "student", seqs=64), health_s=900)
            except Exception as e:  # noqa: BLE001
                self.ex.warn(f"smoke: vLLM would not serve the student ({str(e)[:200]}); trying llama-server on a "
                             "BF16 GGUF")
                gguf = out / "student-bf16.gguf"
                self.servers.stop(step)
                self.ex.run(step, X.convert_cmd(str(self.llama), model, gguf))
                served = "llama-server"
                return self.servers.ensure(step, llama_spec(cfg, str(gguf), "student", slots=16), health_s=900)

        try:
            minutes = self.cap(cfg.smoke_minutes, after=5)
            if self.serve(step, "student", start).get("remote"):
                served = C.backend(cfg, "student")["name"]
            spike = self.p("smoke", "spike")
            spike.mkdir(parents=True, exist_ok=True)
            if not (spike / "spike").exists():
                os.symlink(REPO / "mugge" / "spike", spike / "spike")
            st = self.bench(step, spike, out, "student", seeds=cfg.smoke_seeds, concurrency=16,
                            tickets=cfg.smoke_tickets, minutes=minutes, partial_ok=True)
        except Skip:
            raise
        except Exception as e:  # noqa: BLE001  (optional: reported, never fatal)
            st = {"error": f"{type(e).__name__}: {e}"[:500]}
        broken = not st.get("solved") and (st.get("empty_share") or 0) > 0.5
        st.update(served=served, model=name,
                  verdict=("BROKEN: no ticket solved and most answers wrote no file; do not export" if broken
                           else "ok" if st.get("solved") else "weak: nothing solved"))
        dump(out / "smoke.json", st)
        print(f"SMOKE {name}: {st.get('solved')} of {st.get('runs')} spike runs solved, "
              f"{st.get('empty_answers')} of {st.get('calls')} answers without a file: {st['verdict']}", flush=True)
        if broken:
            self.ex.warn(f"smoke {name}: {st['verdict']}")

    # ---------- export (laptop) ----------
    def export_dir(self) -> Path:
        return self.p("export", self.student)

    @property
    def student(self) -> str:
        """The model the laptop stages export, measure and score: sft, or round2 after
        session 2 (--set student=round2)."""
        return self.cfg.student

    def export_done(self) -> bool:
        return (self.export_dir() / "export.json").exists()

    def run_export(self, step: Step) -> None:
        cfg, name = self.cfg, self.student
        model = self.model_dir(name)
        if not self.need(f"{name}-model.tar", model / "sft.json", untar=model):
            raise RuntimeError(f"no {name} model here or in the bucket")
        sm = load(self.p("smoke", name, "smoke.json")) or {}
        if self.need(f"smoke-{name}.json", self.p("smoke", name, "smoke.json")):
            sm = load(self.p("smoke", name, "smoke.json")) or {}
        if str(sm.get("verdict", "")).startswith("BROKEN") and not self.force:
            raise RuntimeError(f"the {name} smoke says {sm['verdict']}; --force exports anyway")
        rows = self.train_rows(name)
        self.servers.stop(step)  # imatrix and the vocab trim need the GPU (a smoke's server may still be up)
        d = self.export_dir()
        d.mkdir(parents=True, exist_ok=True)
        hf = model
        # the vocabulary cut: optional, gated (stages/vocab.py); any failure keeps the full vocab
        if cfg.base_model.vocab_cut:
            try:
                traces = d / "traces.jsonl"
                self.ex.run(step, X.traces_cmd(model, rows, cfg.max_len, traces))
                for c in X.vocab_cmds(str(self.llama), model, d / "vocab", traces, cfg.codebench):
                    self.ex.run(step, c)
                if (d / "vocab" / "vocab_index.json").exists():
                    hf = d / "vocab"
            except Exception as e:  # noqa: BLE001  (optional)
                self.ex.warn(f"export: vocab cut skipped ({str(e)[:300]}); the full vocab ships")
        bf16 = d / "bf16.gguf"
        if not bf16.exists():  # written as bf16.part.gguf, so it is whole when it is there
            self.ex.run(step, X.convert_cmd(str(self.llama), hf, bf16))
        try:  # llama-tokenize == HF on rows and edge texts (optional, reported)
            self.ex.run(step, X.tokcheck_cmd(str(self.llama), hf, bf16))
        except Exception as e:  # noqa: BLE001
            self.ex.warn(f"export: llama-tokenize != HF tokenization ({str(e)[:300]})")
        calib = d / "calib.txt"
        self.ex.run(step, X.calib_cmd(hf, [rows], calib, 400))
        imat = d / "imatrix.gguf"
        if not imat.exists():
            self.ex.run(step, X.imatrix_cmd(self.bins, bf16, calib, imat, cfg.imatrix_chunks))
        ladder = []
        for qt in cfg.base_model.quant_ladder:
            out = d / f"{qt}.gguf"
            if not out.exists():
                self.ex.run(step, X.quant_cmd(self.bins, bf16, imat, out, qt))
            ladder.append({"type": qt, "path": str(out), "bytes": out.stat().st_size})
        dump(d / "export.json", {"model": name, "hf": str(hf), "vocab_cut": hf != model, "bf16": str(bf16),
                                 "imatrix": str(imat), "ladder": ladder})

    # ---------- footprint (laptop) ----------
    def footprint_json(self) -> Path:
        return self.p("footprint", self.student, "footprint.json")

    def footprint_done(self) -> bool:
        return self.footprint_json().exists()

    def run_footprint(self, step: Step) -> None:
        """Each quant, best first, measured as Oliver defined the footprint; the first at or
        under the limit ships as BH<version>-<GB>.gguf (round 2: BH<version>.1-<GB>.gguf)."""
        cfg = self.cfg
        ex = load(self.export_dir() / "export.json")
        if not ex:
            raise RuntimeError("no export: run export first")
        self.servers.stop(step)  # each quant is measured alone on the GPU
        calib = self.export_dir() / "calib.txt"
        results, ship = [], None
        for rung in ex["ladder"]:
            out = self.p("footprint", self.student, f"{rung['type']}.json")
            if load(out) is None:  # missing, or cut short by a stopped run
                self.ex.run(step, Cmd(f"python3 scripts/bh/footprint.py --gguf {q(rung['path'])} --llama-cpp {q(self.llama)} "
                                      f"--text {q(calib)} --out {q(out)} --prompt-tokens {cfg.footprint_prompt} "
                                      f"--version {cfg.ship_version(self.student)}" + (f" --footprint-sh {q(cfg.footprint_sh)}"
                                                                    if cfg.footprint_sh else ""),
                                      "footprint", {"gguf": rung["path"], "out": str(out), "version": cfg.ship_version(self.student)}))
            r = load(out) or {}
            results.append({"type": rung["type"], **{k: r.get(k) for k in ("footprint", "file_bytes", "name", "source")}})
            if r.get("footprint") and r["footprint"] <= cfg.footprint_limit:
                ship = {"type": rung["type"], "gguf": rung["path"], **results[-1]}
                break
        if not ship:
            raise RuntimeError(f"no quant of the ladder fits {cfg.footprint_limit / 1e9:g} GB: "
                               + ", ".join(f"{r['type']} {r['footprint']}" for r in results))
        dst = self.ship_path(ship["name"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ship["gguf"], dst.with_suffix(".part"))
        dst.with_suffix(".part").replace(dst)
        ship["shipped"] = str(dst)
        dump(self.footprint_json(), {"model": self.student, "ladder": results, "ship": ship,
                                     "limit": cfg.footprint_limit})
        # the GGUF first: footprint-<student>.json tells the eval laptop it is there
        self.upload_once(dst, self.ship_key(dst.name))
        self.upload_once(self.footprint_json(), f"footprint-{self.student}.json")
        print(f"ships as {ship['name']}: {ship['type']}, footprint {ship['footprint'] / 1e9:.3f} GB "
              f"(file {ship['file_bytes'] / 1e9:.3f} GB)", flush=True)

    def ship_path(self, name: str) -> Path:
        """This machine's copy of the shipped GGUF. Round 1 and round 2 can ship under the
        same name (same version, same footprint), so each model has its own folder."""
        return self.p("ship", self.student, f"{name}.gguf")

    def ship_key(self, filename: str) -> str:
        return f"ship/{self.student}/{filename}"

    def shipped(self) -> Path:
        """The shipped GGUF, here or from the bucket (waiting for it on the eval laptop)."""
        if not self.wait_for(f"footprint-{self.student}.json", self.footprint_json()):
            raise RuntimeError(f"nothing shipped yet: run footprint first (waited {self.cfg.wait_minutes:g} min)")
        ship = (load(self.footprint_json()) or {})["ship"]
        p = self.ship_path(ship["name"])
        if not p.exists() and not self.need(self.ship_key(p.name), p):
            raise RuntimeError(f"the shipped {p.name} is neither here nor in the bucket")
        return p

    # ---------- score (laptop) ----------
    def score_json(self) -> Path:
        return self.p("score", self.student, "score.json")

    def score_done(self) -> bool:
        """Scored, and for the GGUF that ships now (a new export and footprint re-score)."""
        sc, fp = load(self.score_json()), load(self.footprint_json())
        if not sc:
            return False
        return not fp or Path(sc.get("gguf", "")).name == f"{fp['ship']['name']}.gguf"

    def run_score(self, step: Step) -> None:
        """MuggeBench on the shipped GGUF: BlackHole's muggebench.sh when it is here (the
        dev set, its verdict), else bench/run.ts against one llama-server (16K slots, q8
        KV, finite top_k, thinking off) on score_bench or the spike."""
        cfg = self.cfg
        gguf = self.shipped()
        out = self.p("score", self.student)
        out.mkdir(parents=True, exist_ok=True)
        if cfg.muggebench_sh:
            self.servers.stop(step)
            self.ex.run(step, Cmd(f"( set -o pipefail; bash {q(cfg.muggebench_sh)} {q(gguf)} | tee {q(out / 'muggebench.txt')} )",
                                  "muggebench-sh", {"gguf": str(gguf), "out": str(out)}, gpu=True))
            dump(self.score_json(), {"gguf": str(gguf), "via": "muggebench.sh",
                                     "report": (out / "muggebench.txt").read_text()[-3000:]})
            return
        self.serve(step, "eval", lambda: self.servers.ensure(step, llama_spec(cfg, str(gguf), "student", slots=cfg.slots)))
        self.template_check(step, out)
        bench = Path(cfg.score_bench) if cfg.score_bench else self.p("score", "spike")
        if not cfg.score_bench and not (bench / "spike").exists():
            bench.mkdir(parents=True, exist_ok=True)
            os.symlink(REPO / "mugge" / "spike", bench / "spike")
        st = self.bench(step, bench, out / "bench", "student", seeds=cfg.score_seeds, concurrency=cfg.slots)
        st.update(gguf=str(gguf), bench=str(bench), bar=cfg.score_bar,
                  verdict=B.verdict(st["low"], st["high"], cfg.score_bar),
                  point_meets=st["score"] is not None and st["score"] >= cfg.score_bar,
                  footprint=(load(self.footprint_json()) or {}).get("ship"))
        prev = load(self.score_json()) or {}
        dump(self.score_json(), {**prev, **st})
        print(f"SCORE {gguf.name}: {st['score']}% [{st['low']}, {st['high']}] on {st['tickets']} tickets x "
              f"{cfg.score_seeds} seeds: {st['verdict']} (bar {cfg.score_bar})", flush=True)

    def template_check(self, step, out: Path) -> None:
        """The GGUF's own chat template (llama-server /apply-template, thinking off) against
        the HF template on one real row: a mismatch means training and serving differ."""
        if not self.need("probe.json", self.rows / "probe.json"):
            return
        p = load(self.rows / "probe.json")
        rec = {"row": p["row"], "thinking": self.cfg.thinking, "prompt_text": None}
        if not self.ex.dry:
            from bh.servers import post_json
            try:
                rec["prompt_text"] = post_json(f"{self.url}/apply-template", {
                    "messages": p["row"]["messages"][:-1],
                    "chat_template_kwargs": {"enable_thinking": self.cfg.thinking}}, timeout=60).get("prompt")
            except OSError as e:
                self.ex.warn(f"template check: /apply-template failed ({e})")
                return
        dump(out / "probe.json", rec)
        hf = self.model_dir(self.student)
        if not self.ex.dry and not (hf / "config.json").exists():
            self.ex.warn("template check skipped: the HF model is not on this machine (it runs where export ran)")
            return
        try:
            self.ex.run(step, Cmd(f"python scripts/bh/export.py roundtrip --hf {q(hf)} --probe {q(out / 'probe.json')}",
                                  "roundtrip", {"probe": str(out / "probe.json")}))
        except Exception as e:  # noqa: BLE001
            self.ex.warn(f"the shipped GGUF's chat template does not render the training prompt: {str(e)[:300]}")

    # ---------- round 2 ----------
    def r2(self, *parts) -> Path:
        return self.p("round2", *parts)

    def sample_done(self) -> bool:
        return (self.r2("samples", "stats.json")).exists()

    def run_round2_sample(self, step: Step) -> None:
        """The student answers the held-out tickets through the engine's loop, transcripts
        kept: on a laptop from the shipped GGUF (llama-server), on the B200 from the bf16
        model (vLLM) when the laptops did not make them."""
        cfg = self.cfg
        if self.need("round2-samples.tar", self.r2("samples", "stats.json"), untar=self.r2("samples")):
            raise Done("restored from the bucket (the laptops' samples)")
        self.need_data()
        out = self.r2("samples")
        if cfg.where == "laptop":
            gguf = self.shipped()
            self.serve(step, "student", lambda: self.servers.ensure(step, llama_spec(cfg, str(gguf), "student",
                                                                                    slots=cfg.slots)))
            conc = cfg.slots
        else:
            model = self.model_dir("sft")
            if not self.need("sft-model.tar", model / "sft.json", untar=model):
                raise RuntimeError("no sft model to sample")
            self.serve(step, "student", lambda: self.servers.ensure(step, vllm_spec(cfg, str(model), "student")))
            conc = cfg.bench_concurrency
        minutes = self.cap(cfg.round2_minutes, after=cfg.rft_minutes + 20)
        self.bench(step, self.bench_dir("heldout"), out, "student", seeds=cfg.round2_seeds, concurrency=conc,
                   transcripts=True, tickets=cfg.round2_tickets, minutes=minutes if cfg.session_minutes else 0,
                   partial_ok=True)
        self.tar_up(out, "round2-samples.tar", None)

    def repair_done(self) -> bool:
        return (self.rows / "train-round2.jsonl").exists() and self.r2("round2.json").exists()

    def run_round2_repair(self, step: Step) -> None:
        """data.py repair-requests renders the fix call the engine would make next on each of
        the student's failed held-out runs; the teacher answers each repair_samples times
        (data.py answer, through the proxy: its sampling, cap and finish log); data.py repair
        keeps the shortest passing answer per failure (own-repair rows) and data.py rft the
        student's own passes; build --round 2 joins them with round 1's teacher and extra
        rows. The guards here run last; W/rows/train-round2.jsonl is what round2-rft reads."""
        cfg = self.cfg
        if self.need("round2-rows.jsonl", self.rows / "train-round2.jsonl"):
            self.need("round2.json", self.r2("round2.json"))
            if self.r2("round2.json").exists():
                raise Done("restored from the bucket")
        samples = self.r2("samples")
        if not self.need("round2-samples.tar", samples / "stats.json", untar=samples):
            raise RuntimeError("no student samples: run round2-sample first")
        self.need_data()
        self.restore_data_rows()
        if not (self.data / "rows" / "teacher.jsonl").exists():
            raise RuntimeError("no round-1 teacher rows (data-rows-teacher.jsonl): run teacher-answers first")
        self.ex.run(step, self.data_cmd("repair-requests", "data-repair-requests", self.runs_args([samples]),
                                        runs=[str(samples)]))
        reqs = self.data / "round2" / "repair-requests.jsonl"
        answers = self.data / "round2" / "repair-answers.jsonl"
        if read_jsonl(reqs):
            self.serve_teacher(step, self.chosen_teacher())
            minutes = self.cap(cfg.round2_minutes, after=cfg.rft_minutes + 15)
            with self.recorder(self.r2("repairs", "proxy")) as (url, rec):
                cmd = self.data_cmd(
                    "answer", "data-answer",
                    f"--requests {q(reqs)} --out {q(answers)} --url {q(url)} --model {q(self.model or 'teacher')} "
                    f"--samples {cfg.repair_samples} --max-tokens {cfg.max_call_tokens} --top-k {cfg.top_k} "
                    f"--thinking {'on' if cfg.thinking else 'off'}",
                    requests=str(reqs), out=str(answers), url=url, samples=cfg.repair_samples)
                cmd.timeout_min = minutes if cfg.session_minutes else 0
                try:
                    self.ex.run(step, cmd)
                except StepTimeout as e:
                    self.ex.warn(f"round 2: teacher repairs {e}; the answers in by then count")
            pc = dict(rec.counts)
            if pc["calls"] and pc["cut"] / pc["calls"] > cfg.drop_warn:
                self.ex.warn(f"round 2: {pc['cut']} of {pc['calls']} teacher repairs cut at the "
                             f"{cfg.max_call_tokens}-token cap")
        else:
            self.ex.warn("round 2: the student's failures gave no fix call to repair (none failed acceptance)")
            write_jsonl(answers, [])
            pc = {}
        self.ex.run(step, self.data_cmd("repair", "data-repair", f"--answers {q(answers)}"))
        self.ex.run(step, self.data_cmd("rft", "data-rft", self.runs_args([samples]), runs=[str(samples)]))
        st = self.build_rows(step, 2)
        sd = self.data / "stats"
        dump(self.r2("round2.json"), {**st, "requests": load(sd / "repair-requests.json"), "proxy": pc,
                                      "own_repair": load(sd / "own-repair.json"), "rft": load(sd / "rft.json")})
        self.upload_once(self.rows / "train-round2.jsonl", "round2-rows.jsonl")
        self.upload_once(self.r2("round2.json"), "round2.json")

    def rft_done(self) -> bool:
        return self.sft_done("round2") and self.smoke_done("round2")

    def run_round2_rft(self, step: Step) -> None:
        """Round 2's SFT from the base on teacher + extra + own-repair + rft rows, then its smoke."""
        self.train_rows("round2")
        if not self.sft_done("round2"):
            try:
                self.run_sft(step, "round2")
            except Done:  # restored from the bucket: its smoke still runs
                pass
        self.run_smoke(step, "round2")

    # ---------- upload ----------
    def tar_up(self, d: Path, name: str, members) -> None:
        """d (or its members) as name.tar in the bucket; kept here until it is sent."""
        tar = d.with_name(d.name + ".up.tar")
        ms = [m for m in (members or [p.name for p in d.iterdir()]) if (d / m).exists()]
        subprocess.run(["tar", "cf", str(tar), "-C", str(d), *ms], check=True)
        if self.upload_once(tar, name):
            tar.unlink(missing_ok=True)

    # A hand-over file (the data, a model, rows, the ship) that fails to upload stays on the
    # pending list, and every later upload sends the list first, in order, so a file that
    # says another is there (footprint-*.json after its GGUF) never arrives before it.
    def pending(self) -> list[list[str]]:
        return (load(self.p(".uploaded", "pending.json")) or {}).get("files", [])

    def set_pending(self, files: list[list[str]]) -> None:
        dump(self.p(".uploaded", "pending.json"), {"files": files})

    @staticmethod
    def signature(src: Path) -> str:
        st = src.stat()
        return f"{st.st_size}:{st.st_mtime_ns}"

    def mark(self, src: Path, name: str) -> None:
        m = self.p(".uploaded", name.replace("/", "_"))
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(self.signature(src))

    def upload_once(self, src: Path, name: str) -> bool:
        """src as name in the bucket unless this very file was sent already; False (and the
        file on the pending list) when it could not be sent now."""
        if not src.exists():
            return True
        m = self.p(".uploaded", name.replace("/", "_"))
        if m.exists() and m.read_text() == self.signature(src):
            return True
        pend = [x for x in self.pending() if x[1] != name]
        if not pend and self.bucket.ok() and self.bucket.put(src, name):
            self.mark(src, name)
            if len(pend) != len(self.pending()):
                self.set_pending(pend)
            return True
        self.set_pending(pend + [[str(src), name]])
        self.ex.warn(f"upload of {src} to {self.bucket.key(name)} " + ("waits for an earlier one" if pend else "failed")
                     + "; it is retried after every step")
        return False

    def send_pending(self) -> bool:
        """The pending hand-over files, in order, stopping at the first that still fails."""
        files = self.pending()
        while files:
            src, name = Path(files[0][0]), files[0][1]
            if src.exists() and not (self.bucket.ok() and self.bucket.put(src, name)):
                self.set_pending(files)
                return False
            if src.exists():
                self.mark(src, name)
                if src.name.endswith(".up.tar"):
                    src.unlink(missing_ok=True)
            files = files[1:]
        self.set_pending([])
        return True

    def upload(self, step: Step | None, final: bool) -> None:
        """After every step: every small result and the logs as results-<profile>.tar, and
        the files later stages or machines need, once each."""
        self.uploaded = False
        dump(self.p("state.json"), {"time": time.strftime("%F %T"), "step": step.name if step else None,
                                    "state": step.state if step else "finished", "final": final,
                                    "log": str(step.log) if step and step.log else None, "warnings": self.ex.notes})
        if not self.bucket.ok():
            raise RuntimeError("no bucket login: nothing uploaded (evroc login --oidc-port 8765)")
        if not self.send_pending():
            raise RuntimeError("a hand-over file is still not in the bucket: " + " ".join(n for _, n in self.pending()))
        self.W.mkdir(parents=True, exist_ok=True)
        tar = self.W.with_name(f"{self.W.name}-results.tar")
        big = ("*.safetensors", "*.gguf", "*.pt", "*.bin", "*.part", "*.tmp", "*.tar", "topk.jsonl")
        names = " ".join(f"! -name '{g}'" for g in big)
        r = subprocess.run(["bash", "-c", f"cd {q(self.W.parent)} && {{ find {q(self.W.name)} -type f -size -300M {names} "
                                          f"-print0; find {q(self.cfg.logs)} -type f -print0 2>/dev/null; }} "
                                          f"| tar cf {q(tar)}.part --null -T - 2>/dev/null; mv {q(tar)}.part {q(tar)}"],
                           capture_output=True, text=True)
        if r.returncode == 0 and tar.exists():
            if not self.bucket.put(tar, f"results-{self.cfg.profile}.tar"):
                raise RuntimeError("upload of the results failed")
        for src, name in ((self.teacher_json(), "teacher.json"), (self.rows / "rows.json", "rows-stats.json"),
                          (self.rows / "topk.jsonl.gz", "topk.jsonl.gz"),
                          (self.rows / "probe.json", "probe.json"), (self.p("smoke", "sft", "smoke.json"), "smoke-sft.json"),
                          (self.p("smoke", "round2", "smoke.json"), "smoke-round2.json")):
            if src.exists():
                self.upload_once(src, name)
        if self.pending():
            raise RuntimeError("a hand-over file is still not in the bucket: " + " ".join(n for _, n in self.pending()))
        dump(self.p(".uploaded", "last.json"), {"time": time.strftime("%F %T"), "step": step.name if step else None,
                                                "final": final})
        self.uploaded = True

    # ---------- summary ----------
    def summary(self) -> str:
        out = [f"BH{self.cfg.version} train harness, profile {self.cfg.profile}, base {self.cfg.base}, run {self.cfg.run}"]
        t = load(self.teacher_json()) or {}
        for x in t.get("tried", []):
            g = x.get("g2") or {}
            out.append(f"teacher {x['teacher']}: G2 {'pass' if x.get('pass') else 'FAIL'} "
                       f"({g.get('score')}% of {g.get('tickets')} tickets, {g.get('answers_per_min')} answers/min)"
                       + (f": {x.get('why')}" if not x.get("pass") else ""))
        rs = load(self.rows / "rows.json") or {}
        if rs:
            hv = rs.get("harvest") or {}
            out.append(f"round 1 rows: {rs.get('rows')} {rs.get('sources')} {rs.get('kinds')}; harvest: "
                       f"{hv.get('solved_runs')} solved runs, {hv.get('answers')} answers, drop share "
                       f"{hv.get('drop_share')}, dropped {hv.get('dropped')}; top-k on {rs.get('topk_rows', 0)} of "
                       f"{rs.get('teacher_rows', 0)} teacher rows")
        for name in ("sft", "round2"):
            s = load(self.model_dir(name) / "sft.json")
            if s:
                out.append(f"SFT {name}: {s['rows']['rows']} rows, {s['updates']}/{s['planned']} updates, ce {s['ce']:.3f}"
                           f", KL rows {s['examples']['kd_rows']}, {s['minutes']} min" + (f" ({s['stopped']})" if s["stopped"] else ""))
            sm = load(self.p("smoke", name, "smoke.json"))
            if sm:
                out.append(f"smoke {name}: {sm.get('solved')}/{sm.get('runs')} spike runs, empty {sm.get('empty_answers')}"
                           f"/{sm.get('calls')}: {sm.get('verdict')}")
        for name in ("sft", "round2"):
            f = load(self.p("footprint", name, "footprint.json"))
            if f:
                s = f["ship"]
                out.append(f"ship {name}: {s['name']} ({s['type']}, footprint {s['footprint'] / 1e9:.3f} GB, "
                           f"file {s['file_bytes'] / 1e9:.3f} GB)")
            sc = load(self.p("score", name, "score.json"))
            if sc and sc.get("score") is not None:
                out.append(f"score {name}: {sc['score']}% [{sc['low']}, {sc['high']}] {sc['verdict']} (bar {sc['bar']})")
        r2 = load(self.r2("round2.json"))
        if r2:
            out.append(f"round 2 rows: {r2['rows']} {r2['sources']}; {(r2.get('requests') or {}).get('requests', 0)} "
                       f"repair requests, {(r2.get('own_repair') or {}).get('rows', 0)} own-repair rows, "
                       f"{sum(((r2.get('rft') or {}).get('rows') or {}).values())} rft rows")
        notes = self.ex.notes
        last = None if self.live else load(self.p("state.json"))
        if last:
            how = {"failed": "stopped at", "stopped": "interrupted during", "finished": "finished"}.get(last["state"],
                                                                                                   "last step")
            out.append(f"last run ({last['time']}): {how}" + (f" {last['step']}" if last["state"] != "finished" else "")
                       + (f"; log: {last['log']}" if last["state"] in ("failed", "stopped") and last.get("log") else ""))
            notes = last.get("warnings") or []
        pend = self.pending()
        if pend:
            out.append("not in the bucket yet (sent by the next run): " + " ".join(n for _, n in pend))
        if notes:
            out += ["", "Warnings:"] + [f"  {n}" for n in notes]
        return "\n".join(out)

    # ---------- the plan ----------
    def steps(self, names: list[str]) -> list[Step]:
        cfg = self.cfg
        table = {
            "setup": ("setup: VM, pins, sandbox images, boot dry-run", {"": 25}, self.run_setup, self.setup_done),
            "data": ("data: kits to bench projects and rows", {"": 5}, self.run_data, self.data_done),
            "teacher-serve": ("teacher: boot and G2", {"": cfg.g2_minutes + 10}, self.run_teacher_serve, self.teacher_done),
            "teacher-answers": ("teacher answers, harvest, check", {"": cfg.answer_minutes + 10},
                                self.run_teacher_answers, self.answers_done),
            "sft": ("SFT", {"": cfg.sft_minutes + 5}, self.run_sft, self.sft_done),
            "smoke": ("spike smoke of the bf16 student", {"": cfg.smoke_minutes + 3}, self.run_smoke, self.smoke_done),
            "export": ("export: vocab cut, GGUF, imatrix, quant ladder", {"": 40}, self.run_export, self.export_done),
            "footprint": ("footprint and the shipped name", {"": 15}, self.run_footprint, self.footprint_done),
            "score": ("MuggeBench score", {"": 60}, self.run_score, self.score_done),
            "round2-sample": ("round 2: student samples held-out", {"": cfg.round2_minutes}, self.run_round2_sample,
                              self.sample_done),
            "round2-repair": ("round 2: teacher repairs", {"": cfg.round2_minutes}, self.run_round2_repair,
                              self.repair_done),
            "round2-rft": ("round 2: RFT and its smoke", {"": cfg.rft_minutes + cfg.smoke_minutes}, self.run_round2_rft,
                           self.rft_done),
        }
        allowed = C.WHERE[cfg.where]
        out = []
        self.stage_keys = list(names)
        for n in names:
            label, mins, run, done = table[n]
            if n not in allowed:
                def refuse(st, n=n):
                    raise Skip(f"{n} does not run on a {cfg.where} ({cfg.profile} profile)")
                run = refuse
            out.append(Step(label, mins, self.tracked(n, run), done))
        return out

    def tracked(self, name: str, fn):
        """fn, recording the stage it is in for `watch` (bh/live.py)."""
        def run(st: Step) -> None:
            stages = live.read_state(self.cfg).get("stages") or {}
            live.record(self.cfg, stage=name, stage_started=time.time(), stage_minutes=st.total,
                        stage_log=str(st.log) if st.log else None, stages={**stages, name: "running"})
            end = "failed"
            try:
                fn(st)
                end = "done"
            except Done:
                end = "done"
                raise
            except Skip:
                end = "skipped"
                raise
            except KeyboardInterrupt:
                end = "stopped"
                raise
            finally:
                stages = live.read_state(self.cfg).get("stages") or {}
                live.record(self.cfg, stages={**stages, name: end})
        return run


# ---------- the command line ----------

USAGE = """usage: bh_run.py PROFILE [STAGES] [options]

  PROFILE   g0 | s1 | laptop | eval | s2   (where this runs; scripts/bh/README.md has the run sheet)
  STAGES    none: the profile's whole plan, resuming (finished stages are skipped)
            one or more stages, or a span of the plan: sft..smoke, data.., ..sft
            round2 = round2-sample round2-repair round2-rft
            status: the plan with what is done, and the results so far (one shot)
  start [STAGES]   the same, detached: it keeps going when you log out (one log per run)
  watch            the live view of this profile's run: stage, clock, gates, tok/s, log tail
  stop             end the run cleanly (as Ctrl-C: uploads, then exits; rerun to resume)
  --set key=value  override a config field (repeatable; scripts/bh/config.py)
  --dry-run DIR    on the CPU with mocks: DIR/<profile> is the machine, DIR/bucket the bucket
  --force          export even when the smoke said BROKEN
  --shutdown       power the VM off once everything is uploaded

stages: """ + " ".join(STAGES)


def parse(argv: list[str]) -> dict:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        raise SystemExit(0 if argv else 2)
    a = {"profile": argv[0], "stages": [], "sets": [], "dry": None, "force": False, "shutdown": False,
         "cmd": "run", "argv": [x for x in argv if x != "start"]}
    if a["profile"] not in PLAN:
        raise SystemExit(f"unknown profile {a['profile']!r}\n\n{USAGE}")
    it = iter(argv[1:])
    for x in it:
        if x == "--set":
            a["sets"].append(next(it, ""))
        elif x == "--dry-run":
            a["dry"] = Path(next(it, "."))
        elif x == "--force":
            a["force"] = True
        elif x == "--shutdown":
            a["shutdown"] = True
        elif x in ("status", "start", "watch", "stop") and not a["stages"]:
            a["cmd"] = x
        elif x.startswith("-"):
            raise SystemExit(f"unknown option {x}\n\n{USAGE}")
        else:
            a["stages"].append(x)
    return a


def stage_names(asked: list[str], profile: str) -> list[str]:
    """The stages to run: the plan, named stages, or a span of the plan (a..b, a.., ..b)."""
    plan = list(PLAN[profile])
    if not asked:
        return plan
    out = []
    for s in asked:
        if ".." in s:
            lo, hi = s.split("..", 1)
            for n in (lo, hi):
                if n and n not in plan:
                    raise SystemExit(f"{n!r} is not in the {profile} plan: {' '.join(plan)}")
            out += plan[plan.index(lo) if lo else 0:plan.index(hi) + 1 if hi else len(plan)]
            continue
        for n in ALIASES.get(s, (s,)):
            if n not in STAGES:
                raise SystemExit(f"unknown stage {n!r}; stages: {' '.join(STAGES)}, round2")
            out.append(n)
    return list(dict.fromkeys(out))


def main(argv: list[str] | None = None) -> int:
    a = parse(sys.argv[1:] if argv is None else argv)
    env = dict(os.environ)
    mock = None
    if a["dry"]:
        # one machine per profile under DIR, one bucket for all: s1, laptop and s2 dry-run in
        # turn with the same DIR hand their results over the way the real machines do
        d = a["dry"].resolve()
        m = d / a["profile"]
        env.update(NVME=str(m / "nvme"), BH_WORK=str(m / "work"), BH_LOGS=str(m / "logs"), BH_BUCKET=str(d / "bucket"))
    cfg = C.load(a["profile"], env, None, a["sets"])
    if a["dry"] and a["cmd"] == "run":
        from bh import mock as M
        upstream = M.FakeUpstream()
        mock = M.Mock(cfg, upstream)
        ex = Exec(cfg, dry=True, mock=mock, reading=lambda: (90, 40000))
        run = Run(cfg, ex, force=a["force"], url=upstream.url)
        mock.run = run
    else:
        ex = Exec(cfg, dry=bool(a["dry"]))
        run = Run(cfg, ex, force=a["force"])
    plan = run.steps(stage_names(a["stages"], cfg.profile))
    logs = Path(cfg.logs)
    if a["cmd"] == "status":
        run_plan(plan, run.summary, logs, ["--list"])
        print()
        return run_plan(plan, run.summary, logs, ["summary"])
    if a["cmd"] == "start":
        return live.start(cfg, a["argv"])
    if a["cmd"] == "stop":
        return live.stop(cfg)
    if a["cmd"] == "watch":
        return live.watch(cfg, run)
    if a["shutdown"] and not run.bucket.ok():
        print("Refusing --shutdown: no bucket login, so nothing could be uploaded before the VM powers off.")
        return 2
    other = live.running(cfg)
    if other and other != os.getpid():
        print(f"a {cfg.profile} run is already going (pid {other}): bh_run.py {cfg.profile} watch, or stop it first")
        return 2
    live.record(cfg, state="running", pid=os.getpid(), argv=a["argv"], log=os.environ.get("BH_RUN_LOG"),
                started=time.time(), stage=None, exit=None, stopping=False, warnings=[],
                stages={k: "done" if s.done() else "waiting" for k, s in zip(run.stage_keys, plan)})
    ex.on_warn = lambda m: live.record(cfg, warnings=ex.notes[-20:])

    def stop(signum, frame):  # ssh dropped or killed: the run ends the way Ctrl-C ends it (upload, server stop)
        raise KeyboardInterrupt(signal.Signals(signum).name)

    handlers = {}
    if threading.current_thread() is threading.main_thread():
        handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGHUP)}
    run.live = True
    os.environ["PATH"] = f"{Path.home() / '.bun' / 'bin'}:{os.environ['PATH']}"
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # centipede.py's: no FlashInfer JIT on the VMs
    logs.mkdir(parents=True, exist_ok=True)
    run.W.mkdir(parents=True, exist_ok=True)
    dump(run.W / "config.json", cfg.to_json())
    code = 1
    try:
        code = run_plan(plan, run.summary, logs, ["run"], after_step=run.upload)
    finally:
        live.record(cfg, state={0: "finished", 130: "stopped"}.get(code, "failed"), exit=code, stage=None)
        try:
            run.servers.stop()
        except Exception as e:  # noqa: BLE001
            print(f"stopping the server: {e}")
        if mock:
            mock.close()
        for sig, h in handlers.items():
            signal.signal(sig, h)
    if a["shutdown"]:
        if code == 0 and run.uploaded:
            print("Results uploaded; powering off in 1 minute (sudo shutdown -c cancels).")
            subprocess.run(["sudo", "shutdown", "-h", "+1"], check=False)
        else:
            print("Not powering off: the run did not finish with every result uploaded.")
    return code


if __name__ == "__main__":
    sys.exit(main())
