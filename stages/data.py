"""Stage 1: teacher-generated task data.

Reads taskspec.json and uses the teacher (vLLM) to:
  1. brainstorm scenarios the task's inputs come from (diversity),
  2. write new task inputs per scenario + style hint, deduplicated,
  3. answer every input (the distillation targets),
then drops bad answers (empty, truncated, invalid JSON when the output format
is JSON) and splits off a held-out set. Code TaskSpecs (task_type "code") differ:
the teacher writes each new input together with its tests, and an answer is kept only
when it builds and passes those tests in stages/sandbox.py. Seeds must pass their own
tests first. Code specs can add contest rows (Config.data_contest_rows): LiveCodeBench
problems released before data_contest_before, so older than the ones the eval scores
(lcb_since), each answered data_contest_samples times by the teacher; the shortest answer
that passes every test becomes a train row (never held out). Code specs can also add ready
rows from a file (Config.data_extra_rows), train only. Writes:
  data/train.jsonl    chat-format SFT data (also REAP calibration data); seeds included
  data/heldout.jsonl  held-out inputs with teacher reference answers (inputs written
                      by Gemini when Config.testgen_model and GEMINI_API_KEY are set;
                      code specs: always teacher inputs, rows add "tests" and "language")
  data/calib.txt      plain text for llama.cpp imatrix
  data/stats.json     counts, drop reasons, timings
  work/data_scenarios.json, work/data_inputs.jsonl   resume cache for a crashed run

Knobs (env vars, defaults in brackets): LOBBOT_DATA_SCENARIOS [40],
LOBBOT_DATA_PER_PROMPT [8], LOBBOT_DATA_MAX_LEN [8192], LOBBOT_DATA_GPU_UTIL [0.85],
LOBBOT_DATA_TP [1], LOBBOT_DATA_BATCH [256], LOBBOT_DATA_ANSWER_MAX_TOKENS [1536]
(per job: Config data_answer_max_tokens and data_max_len win over the env vars)
(answers that hit it are dropped as truncated; raise it for code or other long
outputs, together with LOBBOT_DATA_MAX_LEN and heal's max_len). Sizes come from Config
(n_generate, n_heldout).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
import zlib
from dataclasses import dataclass

from common.progress import emit
from stages import codebench, harness, sandbox, testgen
from stages._util import DRY_RUN, Job, read_jsonl, write_jsonl
from stages.taskdata import GEMMA4_THINK, think_tags

STAGE = "data"
N_SCENARIOS = int(os.environ.get("LOBBOT_DATA_SCENARIOS", 40))
PER_PROMPT = int(os.environ.get("LOBBOT_DATA_PER_PROMPT", 8))
MAX_MODEL_LEN = int(os.environ.get("LOBBOT_DATA_MAX_LEN", 8192))
GPU_UTIL = float(os.environ.get("LOBBOT_DATA_GPU_UTIL", 0.85))
TP = int(os.environ.get("LOBBOT_DATA_TP", 1))
BATCH = int(os.environ.get("LOBBOT_DATA_BATCH", 256))
# FlashInfer's B200 decode kernels are JIT-built at startup and need ninja plus a matching nvcc.
# FLASH_ATTN ships prebuilt. Empty string lets vLLM choose.
ATTN_BACKEND = os.environ.get("LOBBOT_DATA_ATTN_BACKEND", "FLASH_ATTN")
# FlashInfer's top-p/top-k sampler is JIT-built the same way during warmup.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
OVERSHOOT = 1.3      # extra inputs requested to cover dedupe and answer filtering
CONTEST_OVERSHOOT = 1.5  # contest problems asked per wanted row; the teacher solves maybe 60-80%
# Contest answers go to vLLM in chunks this big, not BATCH: each chunk waits for its slowest
# answer, and with long thinking answers small chunks leave the GPU mostly idle at their end.
CONTEST_BATCH = 2048
MAX_ROUNDS = 6
MAX_INPUT_CHARS = 6000  # keeps every answer prompt well inside MAX_MODEL_LEN
FEWSHOT = 6          # seed examples shown to the teacher when answering
SEED_REPEAT = 2      # human-written seeds appear this many times in train
ANSWER_MAX_TOKENS = int(os.environ.get("LOBBOT_DATA_ANSWER_MAX_TOKENS", 1536))


def answer_max_tokens(cfg) -> int:
    """Per-job Config value first, then the env knob / default above."""
    return int(getattr(cfg, "data_answer_max_tokens", None) or ANSWER_MAX_TOKENS)


def max_model_len(cfg) -> int:
    return int(getattr(cfg, "data_max_len", None) or MAX_MODEL_LEN)

STYLE_HINTS = [
    "short and terse", "long and detailed", "messy, with typos and informal language",
    "an unusual edge case", "ambiguous or tricky to handle", "from an expert user",
    "from a first-time user", "with irrelevant extra details", "emotional, in a hurry",
    "very polite and formal", "full of numbers, dates or identifiers",
    "mixing several requests or topics",
]


# --------------------------------------------------------------------------- prompts

LANGUAGE_NAMES = {"c": "C", "javascript": "JavaScript", "typescript": "TypeScript", "python": "Python",
                  "asm": "x86-64 assembly (GNU as, System V ABI)"}
# How tests reach the answer (stages/sandbox.py); assembly is linked against C tests.
TESTS_JOIN = {"asm": "The tests are a separate C file with main() that declares the assembly functions it "
                     "calls and is linked with the answer; it must exit non-zero on failure (use assert.h), "
                     "like the examples."}


def system_prompt(spec) -> str:
    """The task prompt the student is trained with. package.py puts it in the Modelfile."""
    if spec.is_code:
        lang = LANGUAGE_NAMES[spec.language]
        return (
            f"You are an expert {lang} programmer. Task: {spec.description}\n"
            f"Input format: {spec.input_format}\nOutput format: {spec.output_format}\n"
            f"Answer with the {lang} code only: no explanation, no tests"
            + (", no main()." if spec.language in ("c", "asm") else ".")
        )
    return (
        f"You are an expert at this task: {spec.description}\n"
        f"Input format: {spec.input_format}\nOutput format: {spec.output_format}\n"
        "Answer with the output only, no preamble."
    )


def scenarios_prompt(spec, n: int) -> list[dict]:
    shown = "\n".join(f"- {_trunc(e.input, 300)}" for e in spec.seed_examples[:8])
    return [{"role": "user", "content": (
        f"Task: {spec.description}\nInput format: {spec.input_format}\n"
        f"Example inputs:\n{shown}\n\n"
        f"List {n} distinct real-world scenarios, sub-topics or situations that inputs for this task "
        "come from. Cover common cases, rare cases and edge cases. Each item is a short phrase. "
        "Return a JSON array of strings and nothing else."
    )}]


def gen_inputs_prompt(spec, rng: random.Random, scenario: str, style: str) -> list[dict]:
    shots = rng.sample(spec.seed_examples, k=min(3, len(spec.seed_examples)))
    shown = "\n\n".join(f"<input>\n{e.input}\n</input>" for e in shots)
    return [{"role": "user", "content": (
        f"Task: {spec.description}\nInput format: {spec.input_format}\n\n"
        f"Here are example inputs:\n\n{shown}\n\n"
        f"Write exactly {PER_PROMPT} NEW, realistic inputs for this task.\n"
        f"Scenario: {scenario}\nStyle: {style}\n"
        "Make them differ from the examples and from each other (length, wording, names, details). "
        "Write only the inputs, not the answers. Return a JSON array of strings and nothing else."
    )}]


def gen_code_inputs_prompt(spec, rng: random.Random, scenario: str, style: str) -> list[dict]:
    """Code specs: each new input comes with the tests its answer must pass."""
    shots = rng.sample(spec.seed_examples, k=min(3, len(spec.seed_examples)))
    shown = "\n\n".join(json.dumps({"input": e.input, "tests": e.tests}, ensure_ascii=False) for e in shots)
    lang = LANGUAGE_NAMES[spec.language]
    return [{"role": "user", "content": (
        f"Task: {spec.description}\nLanguage: {lang}\nInput format: {spec.input_format}\n\n"
        f"Here are example inputs, each with the tests its answer must pass:\n\n{shown}\n\n"
        f"Write exactly {PER_PROMPT} NEW, realistic inputs for this task, each with its tests.\n"
        f"Scenario: {scenario}\nStyle: {style}\n"
        + TESTS_JOIN.get(spec.language, "The tests are appended after the answer code in the same file "
                         "and must exit non-zero on failure, like the examples.")
        + " Each input must name every function, type and signature its "
        "tests use, so a correct answer can be written from the input alone. Test behaviour, "
        "including edge cases, not implementation details. Make the inputs differ from the examples "
        "and from each other. Write only inputs and tests, not the answers. Return a JSON array of "
        'objects {"input": "...", "tests": "..."} and nothing else.'
    )}]


def answer_messages(spec, shots, text: str) -> list[dict]:
    sys = system_prompt(spec)
    if spec.eval_criteria:
        sys += f"\nA good answer satisfies: {spec.eval_criteria}"
    msgs = [{"role": "system", "content": sys}]
    for e in shots:
        msgs += [{"role": "user", "content": e.input}, {"role": "assistant", "content": e.output}]
    msgs.append({"role": "user", "content": text})
    return msgs


# --------------------------------------------------------------------------- parsing & filters

_THINK = re.compile(r"(<think>.*?</think>|<\|channel>.*?<channel\|>)\s*", re.S)
_FENCE = re.compile(r"^```[\w-]*\s*\n?|\n?```\s*$")


def split_think(text: str) -> tuple[str, str]:
    """(thinking, answer) from a raw generation. The thinking ends at the last
    </think>; templates that open <think> in the prompt leave only the closing tag.
    Gemma 4 thinks in a thought channel: <|channel>thought, the thinking, <channel|>."""
    if "</think>" in text:
        head, tail = text.rsplit("</think>", 1)
        return head.replace("<think>", "", 1).strip(), tail
    if "<channel|>" in text:
        head, tail = text.rsplit("<channel|>", 1)
        return head.split("<|channel>", 1)[-1].strip().removeprefix("thought\n").strip(), tail
    return "", text


def clean(text: str) -> str:
    return _FENCE.sub("", _THINK.sub("", text).strip()).strip()


def parse_array(text: str) -> list[str]:
    text = clean(text)
    a, b = text.find("["), text.rfind("]")
    if a < 0 or b <= a:
        return []
    try:
        items = json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict):  # [{"input": "..."}] style
            it = it.get("input") or it.get("text") or json.dumps(it, ensure_ascii=False)
        if isinstance(it, str) and len(it.strip()) > 10:
            out.append(it.strip())
    return out


def parse_code_inputs(text: str) -> list[tuple[str, str]]:
    """[(input, tests)] from a JSON array of {"input", "tests"} objects; rows without tests are skipped."""
    text = clean(text)
    a, b = text.find("["), text.rfind("]")
    if a < 0 or b <= a:
        return []
    try:
        items = json.loads(text[a:b + 1])
    except json.JSONDecodeError:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict) and isinstance(it.get("input"), str) and isinstance(it.get("tests"), str):
            if len(it["input"].strip()) > 10 and it["tests"].strip():
                out.append((it["input"].strip(), it["tests"].strip()))
    return out


def expects_json(spec) -> bool:
    return not spec.is_code and "json" in f"{spec.output_format} {spec.eval_criteria}".lower()


_CODE_BLOCK = re.compile(r"```[\w+#-]*[ \t]*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    """The code in an answer: the longest fenced block if there is one ("Here it is: ```c ...```"),
    else the cleaned text."""
    text = _THINK.sub("", text).strip()
    blocks = _CODE_BLOCK.findall(text)
    return max(blocks, key=len).strip() if blocks else clean(text)


def check_answer(text: str, finished: bool, want_json: bool, want_code: bool = False) -> tuple[str | None, str]:
    """(cleaned answer, "") or (None, drop reason)."""
    if not finished:
        return None, "truncated"
    ans = extract_code(text) if want_code else clean(text)
    if not ans:
        return None, "empty"
    if want_json:
        try:
            json.loads(ans)
        except json.JSONDecodeError:
            m = re.search(r"(\{.*\}|\[.*\])", ans, re.S)  # salvage "Here it is: {...}"
            try:
                json.loads(m.group(1)) if m else None
            except json.JSONDecodeError:
                m = None
            if not m:
                return None, "invalid_json"
            ans = m.group(1)
    return ans, ""


def norm_key(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()[:160]


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "..."


# --------------------------------------------------------------------------- teacher

@dataclass
class Gen:
    text: str
    finished: bool  # False if it hit max_tokens


class Teacher:
    def __init__(self, model_path: str, max_len: int = MAX_MODEL_LEN, top_k: int = 20):
        from vllm import LLM, SamplingParams

        self.SamplingParams, self.top_k = SamplingParams, top_k
        self.llm = LLM(model=model_path, max_model_len=max_len, gpu_memory_utilization=GPU_UTIL,
                       tensor_parallel_size=TP, seed=0,
                       **({"attention_backend": ATTN_BACKEND} if ATTN_BACKEND else {}))
        # Gemma 4's thought channel opens and closes with special tokens, which the
        # decoded text drops unless told to keep them; split_think needs them.
        template = getattr(self.llm.get_tokenizer(), "chat_template", None)
        self.keep_special = isinstance(template, str) and think_tags(template) == GEMMA4_THINK

    def chat(self, convs: list[list[dict]], temperature: float, max_tokens: int, thinking: bool = False) -> list[Gen]:
        sp = self.SamplingParams(temperature=temperature, top_p=0.95 if temperature > 0.5 else 0.9,
                                 max_tokens=max_tokens, **({"top_k": self.top_k} if thinking else {}),
                                 **({"skip_special_tokens": False} if thinking and self.keep_special else {}))
        try:  # Qwen3 hybrid checkpoints: no <think> unless asked; Instruct-2507 ignores the flag
            res = self.llm.chat(convs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": thinking})
        except TypeError:
            res = self.llm.chat(convs, sp, use_tqdm=False)
        return [Gen(r.outputs[0].text, r.outputs[0].finish_reason != "length") for r in res]


def chat_batched(teacher, convs, temperature, max_tokens, lo, hi, label, thinking: bool = False,
                 batch: int = BATCH) -> list[Gen]:
    out: list[Gen] = []
    for i in range(0, len(convs), batch):
        out += teacher.chat(convs[i:i + batch], temperature, max_tokens, thinking)
        emit(STAGE, pct=lo + (hi - lo) * len(out) / len(convs), msg=f"{label} {len(out)}/{len(convs)}")
    return out


class DryRunTeacher:
    """No model: seed variations and seed answers, a few broken, for CPU tests."""

    def __init__(self, spec, rng: random.Random):
        self.spec, self.rng = spec, rng

    def chat(self, convs, temperature, max_tokens, thinking=False):
        out = [self._one(conv) for conv in convs]
        if thinking:  # an open <think> in the prompt, as Qwen3.6's template does; a few never close
            out = [Gen(g.text if self.rng.random() < 0.05 else f"Let me work this out first.\n</think>\n\n{g.text}",
                       g.finished) for g in out]
        return out

    def _one(self, conv) -> Gen:
        prompt = conv[-1]["content"]
        if "real-world scenarios" in prompt:
            return Gen(json.dumps([f"scenario number {i}" for i in range(N_SCENARIOS)]), True)
        if conv[0]["content"] == codebench.SYSTEM or prompt.startswith("Ticket: livecodebench-"):
            return self._contest(prompt)
        if self.spec.is_code:
            return self._code(prompt)
        if "NEW, realistic inputs" in prompt:
            base = self.rng.choice(self.spec.seed_examples).input
            return Gen("```json\n" + json.dumps(
                [f"{base} (case {self.rng.randrange(10**9)})" for _ in range(PER_PROMPT)]) + "\n```", True)
        r = self.rng.random()
        ex = self.spec.seed_examples[len(prompt) % len(self.spec.seed_examples)].output
        return Gen("{broken" if r < 0.03 else ex, r > 0.01)

    def _code(self, prompt: str) -> Gen:
        """Inputs reuse a seed's tests; answers are that seed's code, sometimes a wrong seed's
        (fails its tests) or garbage (does not build)."""
        seeds = self.spec.seed_examples
        if "NEW, realistic inputs" in prompt:
            picks = [self.rng.choice(seeds) for _ in range(PER_PROMPT)]
            return Gen("```json\n" + json.dumps([
                {"input": f"{e.input} (case {self.rng.randrange(10**9)})", "tests": e.tests} for e in picks
            ]) + "\n```", True)
        if prompt.startswith("Ticket: "):  # a harness fix call: the task's seed code, as a file
            own = max((e for e in seeds if e.input in prompt), key=lambda e: len(e.input), default=seeds[0])
            src = re.search(r"^Files you own \(write each in full\): (\S+)", prompt, re.M).group(1)
            code = own.output if self.rng.random() < 0.8 else next(e for e in seeds if e is not own).output
            return Gen(harness.render_files({src: code}, "fixed"), True)
        own = max((e for e in seeds if prompt.startswith(e.input)), key=lambda e: len(e.input), default=seeds[0])
        r = self.rng.random()
        if r < 0.05:
            return Gen("{broken", True)
        if r < 0.15:
            return Gen(next(e for e in seeds if e is not own).output, True)
        return Gen(f"Here is the code:\n```\n{own.output}\n```", True)

    def _contest(self, prompt: str) -> Gen:
        """LiveCodeBench problems (contest_rows). The dry-run ones (tests/test_data.py) all add
        two integers, read from stdin or passed to a Solution method; answers are right, some
        with an extra comment line, wrong (they subtract) or truncated."""
        m = re.search(r"def (\w+)\(self", prompt)
        r = self.rng.random()
        op = "-" if r < 0.15 else "+"
        code = (f"class Solution:\n    def {m.group(1)}(self, a, b):\n        return a {op} b" if m
                else f"a, b = map(int, input().split())\nprint(a {op} b)")
        if r > 0.7:
            code = "# add the two numbers\n" + code
        if prompt.startswith("Ticket: "):
            src = re.search(r"^Files you own \(write each in full\): (\S+)", prompt, re.M).group(1)
            return Gen(harness.render_files({src: code}, "adds them"), not 0.15 <= r < 0.25)
        return Gen(f"Here it is:\n```python\n{code}\n```", not 0.15 <= r < 0.25)


# --------------------------------------------------------------------------- stage

def run_stage(job: Job) -> None:
    spec, cfg = job.spec, job.config
    rng = random.Random(0)
    t0 = time.monotonic()
    n_held = cfg.n_heldout
    target = int((cfg.n_generate + n_held) * OVERSHOOT)
    want_json = expects_json(spec)
    stats: dict = {"seeds": len(spec.seed_examples), "expects_json": want_json, "dry_run": DRY_RUN}
    if spec.is_code:
        stats.update(task_type="code", language=spec.language)
        check_seeds(spec)
    # contest problems load (and their date guard runs) before the teacher does, extra rows too
    contest = contest_problems(cfg) if spec.is_code and cfg.data_contest_rows > 0 else []
    extra = extra_rows(job, stats) if spec.is_code and cfg.data_extra_rows else []

    if DRY_RUN:
        teacher = DryRunTeacher(spec, rng)
    else:
        model = job.model_path(cfg.teacher)
        emit(STAGE, pct=1, msg=f"loading teacher {model}")
        teacher = Teacher(model, max_model_len(cfg), cfg.thinking_top_k)
    stats["load_s"] = round(time.monotonic() - t0, 1)
    emit(STAGE, pct=10, msg="teacher loaded")

    # 1) scenarios (cached for resume)
    scen_path = job.path("work", "data_scenarios.json")
    if scen_path.exists():
        scenarios = json.loads(scen_path.read_text())
    else:
        g = teacher.chat([scenarios_prompt(spec, N_SCENARIOS)], 0.8, 2048)[0]
        scenarios = parse_array(g.text)[:N_SCENARIOS] or ["a typical everyday case"]
        scen_path.write_text(json.dumps(scenarios, indent=1))
    print(f"[data] {len(scenarios)} scenarios: {scenarios[:5]}", flush=True)
    emit(STAGE, pct=12, msg=f"{len(scenarios)} scenarios")

    # 2) inputs (cached for resume)
    inputs_path = job.path("work", "data_inputs.jsonl")
    seen = {norm_key(e.input) for e in spec.seed_examples}
    inputs: list[str] = []
    tests_of: dict[str, str] = {}  # code specs: input -> its tests
    for r in read_jsonl(inputs_path) if inputs_path.exists() else []:
        if spec.is_code and not r.get("tests"):
            continue
        if norm_key(r["input"]) not in seen:
            seen.add(norm_key(r["input"]))
            inputs.append(r["input"])
            if spec.is_code:
                tests_of[r["input"]] = r["tests"]
    rounds = 0
    while len(inputs) < target and rounds < MAX_ROUNDS:
        rounds += 1
        n_prompts = -(-(target - len(inputs)) // PER_PROMPT) + 2
        make = gen_code_inputs_prompt if spec.is_code else gen_inputs_prompt
        convs = [make(spec, rng, rng.choice(scenarios), rng.choice(STYLE_HINTS)) for _ in range(n_prompts)]
        lo = 12 + 33 * len(inputs) / target
        # code inputs carry their tests, so they need more room
        gens = chat_batched(teacher, convs, 1.0, 6144 if spec.is_code else 4096, lo, 45,
                            f"writing inputs (round {rounds})")
        added = 0
        for g in gens:
            pairs = parse_code_inputs(g.text) if spec.is_code else [(s, "") for s in parse_array(g.text)]
            for s, tests in pairs:
                k = norm_key(s)
                if k not in seen and len(s) + len(tests) <= MAX_INPUT_CHARS:
                    seen.add(k)
                    inputs.append(s)
                    added += 1
                    if spec.is_code:
                        tests_of[s] = tests
        write_jsonl(inputs_path, [{"input": s, "tests": tests_of[s]} if spec.is_code else {"input": s}
                                  for s in inputs])
        print(f"[data] round {rounds}: +{added} unique inputs, {len(inputs)}/{target}", flush=True)
        if added == 0:
            break
    inputs = inputs[:target]
    stats.update(input_rounds=rounds, inputs=len(inputs))
    if len(inputs) < 2 * n_held:
        raise RuntimeError(f"teacher produced only {len(inputs)} usable inputs (need {2 * n_held}+); "
                           "check the TaskSpec seeds and the teacher model")
    emit(STAGE, pct=45, msg=f"{len(inputs)} unique inputs")

    # 2b) optional: Gemini writes the held-out test inputs (stages/testgen.py); the teacher answers them
    ext: list[str] = []
    testgen_path = job.path("work", "data_testgen.jsonl")  # cached so a rerun keeps the same test set
    if testgen_path.exists() and not spec.is_code:
        ext = [r["input"] for r in read_jsonl(testgen_path) if norm_key(r["input"]) not in seen]
        print(f"[data] {len(ext)} held-out inputs from cache {testgen_path.name}", flush=True)
    elif not DRY_RUN and testgen.enabled(cfg) and not spec.is_code:  # Gemini inputs come without tests
        emit(STAGE, pct=45, msg=f"{cfg.testgen_model} writing {n_held} held-out test inputs")
        ext = testgen.held_out_inputs(spec, cfg, n_held, seen, norm_key, parse_array, MAX_INPUT_CHARS)
        if ext:
            write_jsonl(testgen_path, [{"input": s} for s in ext])
        emit(STAGE, pct=46, msg=f"{len(ext)} held-out test inputs from {cfg.testgen_model}")

    # 3) answers
    shots = rng.sample(spec.seed_examples, k=min(FEWSHOT, len(spec.seed_examples)))
    convs = [answer_messages(spec, shots, s) for s in inputs + ext]
    think = cfg.data_thinking
    # Thinking samples at Qwen's recommended 0.6; greedy-ish decoding makes it loop.
    gens = chat_batched(teacher, convs, cfg.data_thinking_temperature if think else 0.3, answer_max_tokens(cfg), 45, 95,
                        "teacher answering" + (" (thinking)" if think else ""), think)
    sys = system_prompt(spec)
    rows, ext_rows, drops, ext_drops = [], [], {}, {}
    candidates = []  # code specs: (input, answer, thinking) waiting for the sandbox
    for i, (s, g) in enumerate(zip(inputs + ext, gens)):
        reasoning, text = split_think(g.text) if think else ("", g.text)
        ans, why = check_answer(text, g.finished, want_json, want_code=spec.is_code)
        if ans is not None and think and not reasoning:
            ans, why = None, "no_thinking_end"  # the thinking never closed; nothing to learn it from
        is_ext = i >= len(inputs)
        if ans is None:
            d = ext_drops if is_ext else drops
            d[why] = d.get(why, 0) + 1
        elif spec.is_code:
            candidates.append((s, ans, reasoning))
        else:
            (ext_rows if is_ext else rows).append(_row(sys, s, ans, reasoning))
    failed = []  # code specs: (input, answer, sandbox result) that did not pass, drafts for fix rows
    if spec.is_code:
        emit(STAGE, pct=95, msg=f"running tests for {len(candidates)} answers")
        results = sandbox.run_many([(spec.language, a, tests_of[s]) for s, a, _ in candidates])
        for (s, a, reasoning), res in zip(candidates, results):
            if res.passed:
                rows.append(_row(sys, s, a, reasoning))
            else:
                drops[res.reason] = drops.get(res.reason, 0) + 1
                failed.append((s, a, res))
        n_pass = len(rows)
        stats.update(sandbox_runs=len(candidates),
                     sandbox_pass_rate=round(n_pass / len(candidates), 3) if candidates else 0.0)
        print(f"[data] sandbox: {n_pass}/{len(candidates)} answers passed their tests", flush=True)
    stats.update(answered=len(rows), dropped=drops)
    if ext:
        stats.update(heldout_answered=len(ext_rows), heldout_dropped=ext_drops)
    print(f"[data] kept {len(rows)}/{len(inputs)} answers, dropped {drops}", flush=True)
    if len(rows) < 2 * n_held:
        raise RuntimeError(f"only {len(rows)} usable answers after filtering ({drops})")

    # 4) split: held-out from teacher rows only; human seeds always go to train
    rng.shuffle(rows)
    if ext_rows and len(ext_rows) >= n_held // 2:  # Gemini-written tests; all teacher rows can train
        heldout, train = ext_rows[:n_held], rows[:cfg.n_generate]
        stats["heldout_source"] = cfg.testgen_model
    else:
        n_held = min(n_held, len(rows) // 5)
        heldout, train = rows[:n_held], rows[n_held:n_held + cfg.n_generate]
        stats["heldout_source"] = "teacher"
    if spec.is_code and (cfg.data_harness_share > 0 or cfg.data_fix_rows > 0):
        held = {r["messages"][1]["content"] for r in heldout}
        train = harness_rows(spec, cfg, teacher, rng, train, failed, tests_of, held, stats)
    if contest:  # train only: the held-out set stays the task's own
        train += contest_rows(cfg, teacher, rng, contest, stats)
    train += extra  # train only, like contest rows
    if not think:  # the human seeds have no thinking, and every row of a thinking model thinks
        train += [_row(sys, e.input, e.output) for e in spec.seed_examples] * SEED_REPEAT
    if think:
        stats["thinking_chars_median"] = statistics.median(
            len(r["messages"][-1].get("reasoning_content", "")) for r in train) if train else 0
    rng.shuffle(train)
    write_jsonl(job.path("data", "train.jsonl"), train)
    write_jsonl(job.path("data", "heldout.jsonl"), [_heldout_row(spec, r, tests_of) for r in heldout])
    job.path("data", "calib.txt").write_text(  # the last exchange: extra rows can be whole chats
        "\n\n".join(r["messages"][-2]["content"] + "\n" + _with_thinking(r["messages"][-1]) for r in train[:1000])
    )
    stats.update(train=len(train), heldout=len(heldout), elapsed_s=round(time.monotonic() - t0, 1))
    job.path("data", "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[data] stats {json.dumps(stats)}", flush=True)
    job.mark_done(STAGE, {"train": len(train), "heldout": len(heldout)})
    emit(STAGE, "done", 100, f"{len(train)} train, {len(heldout)} held out")


def task_ticket(spec, task: str, code: str = "") -> harness.Ticket:
    """A code task as a Mugge ticket: the task is the context, the answer file is the one
    owned file and the sandbox's build and run are the acceptance commands."""
    src, cmds, where = sandbox.layout(spec.language)
    title = task.strip().splitlines()[0][:80] if task.strip() else "Write the code"
    return harness.Ticket(id=f"T{zlib.crc32(norm_key(task).encode()) % 100000:05d}", title=title,
                          context=f"{task.strip()}\n\n{where}", owns=[src], acceptance=cmds,
                          current={src: code} if code else {})


def harness_rows(spec, cfg, teacher, rng, train, failed, tests_of, held, stats) -> list[dict]:
    """Mugge-shaped rows (stages/harness.py). data_harness_share of the plain train rows become
    write calls with the same answer, as a file. Then up to data_fix_rows fix rows: a failed
    draft and the failing command's output in, the teacher's fix out when it passes the tests.
    Drafts are the teacher's failed answers, topped up with answers sampled at temperature 1.0
    (thinking off). Held-out inputs never become drafts."""
    src, cmds, _ = sandbox.layout(spec.language)
    think = cfg.data_thinking
    out, n_write = [], 0
    for r in train:
        m = r["messages"]
        if rng.random() < cfg.data_harness_share:
            msgs = harness.write_messages(task_ticket(spec, m[1]["content"]))
            out.append(_chat_row(msgs, harness.render_files({src: m[2]["content"]}), m[2].get("reasoning_content", "")))
            n_write += 1
        else:
            out.append(r)
    stats["harness_write_rows"] = n_write
    if cfg.data_fix_rows <= 0:
        return out

    pool = [d for d in failed if d[0] not in held and d[2].reason in ("compile_error", "test_failed", "timeout")]
    need = int(cfg.data_fix_rows * 1.5)  # room for fixes that fail too
    if len(pool) < need:
        pick = [s for s in tests_of if s not in held]
        rng.shuffle(pick)
        pick = pick[:min(len(pick), 4 * (need - len(pool)))]  # ~15-25% of quick drafts fail (a guess)
        shots = rng.sample(spec.seed_examples, k=min(FEWSHOT, len(spec.seed_examples)))
        gens = chat_batched(teacher, [answer_messages(spec, shots, s) for s in pick], 1.0, answer_max_tokens(cfg),
                            95, 96, "quick drafts to fix")
        drafts = [(s, a) for s, g in zip(pick, gens) for a, _ in [check_answer(g.text, g.finished, False, True)] if a]
        res = sandbox.run_many([(spec.language, a, tests_of[s]) for s, a in drafts])
        pool += [(s, a, r) for (s, a), r in zip(drafts, res)
                 if not r.passed and r.reason in ("compile_error", "test_failed", "timeout")]
    rng.shuffle(pool)
    pool = pool[:need]
    convs = []
    for s, a, r in pool:
        cmd = cmds[0] if r.reason == "compile_error" and len(cmds) > 1 else cmds[-1]
        out_text = harness.tail(r.output, 2000) or ("(timed out)" if r.reason == "timeout" else "(no output)")
        convs.append(harness.fix_messages(task_ticket(spec, s, a), cmd, out_text))
    gens = chat_batched(teacher, convs, cfg.data_thinking_temperature if think else 0.3, answer_max_tokens(cfg), 96, 98,
                        "teacher fixing drafts" + (" (thinking)" if think else ""), think)
    drops, fixes = {}, []
    for (s, _, _), conv, g in zip(pool, convs, gens):
        reasoning, text = split_think(g.text) if think else ("", g.text)
        files, note = harness.parse_files(text, [src])
        why = ("truncated" if not g.finished else "no_thinking_end" if think and not reasoning
               else "no_file" if not files.get(src, "").strip() else "")
        if why:
            drops[why] = drops.get(why, 0) + 1
        else:
            fixes.append((s, conv, files[src], note, reasoning))
    res = sandbox.run_many([(spec.language, code, tests_of[s]) for s, _, code, _, _ in fixes])
    n_fix = 0
    for (s, conv, code, note, reasoning), r in zip(fixes, res):
        if not r.passed:
            drops[r.reason] = drops.get(r.reason, 0) + 1
        elif n_fix < cfg.data_fix_rows:
            out.append(_chat_row(conv, harness.render_files({src: code}, note), reasoning))
            n_fix += 1
    stats.update(fix_pool=len(pool), fix_rows=n_fix, fix_dropped=drops)
    print(f"[data] harness: {n_write} write rows, {n_fix} fix rows from {len(pool)} drafts, dropped {drops}", flush=True)
    return out


def contest_problems(cfg) -> list[dict]:
    """LiveCodeBench problems (every release, from code_eval_dir) released before
    data_contest_before. The eval scores the ones released on or after lcb_since and the
    undated ones, so a cut-off after lcb_since is refused and both are dropped here."""
    before, since = cfg.data_contest_before, cfg.lcb_since
    if before > since:
        raise ValueError(f"data_contest_before {before} is after lcb_since {since}: contest rows would "
                         "train on LiveCodeBench problems the eval scores")
    rows = codebench.load_suite("livecodebench", cfg.code_eval_dir)
    # long statements are dropped like long task inputs, so every prompt fits the teacher's context
    out = [p for p in rows if p.get("date") and p["date"][:10] < before and p["date"][:10] < since
           and len(p["prompt"]) <= MAX_INPUT_CHARS]
    if not out:
        raise RuntimeError(f"no LiveCodeBench problems released before {before} in {cfg.code_eval_dir}")
    return out


def contest_rows(cfg, teacher, rng, problems, stats) -> list[dict]:
    """Up to data_contest_rows train rows from contest problems (contest_problems), so heal and
    REAP calibration see contest reasoning. Each problem is asked data_contest_samples times,
    as a plain prompt or, with probability data_contest_harness_share, as a Mugge ticket, and
    keeps its shortest answer (thinking plus code) that passes every test. The row's answer is
    that code rendered canonically (one fenced block, or the ticket's file), not the teacher's
    prose; its thinking goes in reasoning_content."""
    think, n = cfg.data_thinking, max(1, cfg.data_contest_samples)
    n_avail = len(problems)
    rng.shuffle(problems)
    asks = []  # (problem, messages, asked as a ticket)
    for p in problems[:int(cfg.data_contest_rows * CONTEST_OVERSHOOT)]:
        mugge = rng.random() < cfg.data_contest_harness_share
        msgs = (harness.write_messages(codebench.ticket(p)) if mugge else
                [{"role": "system", "content": codebench.SYSTEM},
                 {"role": "user", "content": codebench.build_prompt(p)}])
        asks.append((p, msgs, mugge))
    gens = chat_batched(teacher, [m for _, m, _ in asks for _ in range(n)], cfg.data_thinking_temperature if think else 0.3,
                        answer_max_tokens(cfg), 98, 99, "teacher answering contest problems"
                        + (" (thinking)" if think else ""), think, batch=CONTEST_BATCH)
    drops, cands, items = {}, [], []  # cands: (problem index, code, note, thinking) for the sandbox
    for k, g in enumerate(gens):
        p, _, mugge = asks[k // n]
        reasoning, text = split_think(g.text) if think else ("", g.text)
        code = codebench.harness_code(p, text) if mugge else codebench.extract_code(text)
        why = ("truncated" if not g.finished else "no_thinking_end" if think and not reasoning
               else "empty" if not code.strip() else "")
        if why:
            drops[why] = drops.get(why, 0) + 1
            continue
        note = harness.parse_files(text, [sandbox.layout(p["language"])[0]])[1] if mugge else ""
        cands.append((k // n, code, note, reasoning))
        items.append((p["language"], *codebench.assemble(p, text, code), codebench.time_limit(p)))
    emit(STAGE, pct=99, msg=f"running tests for {len(items)} contest answers")
    best: dict[int, tuple] = {}  # problem index -> (code, note, thinking) of its shortest passing answer
    for (i, code, note, reasoning), r in zip(cands, sandbox.run_many(items)):
        if not r.passed:
            drops[r.reason] = drops.get(r.reason, 0) + 1
        elif i not in best or len(code) + len(reasoning) < len(best[i][0]) + len(best[i][2]):
            best[i] = (code, note, reasoning)
    out, n_ticket = [], 0
    for i in sorted(best)[:cfg.data_contest_rows]:
        (p, msgs, mugge), (code, note, reasoning) = asks[i], best[i]
        lang = p["language"]
        answer = (harness.render_files({sandbox.layout(lang)[0]: code}, note) if mugge
                  else f"```{codebench.FENCE[lang]}\n{code.rstrip()}\n```")
        out.append(_chat_row(msgs, answer, reasoning))
        n_ticket += mugge
    stats.update(contest_available=n_avail, contest_problems=len(asks), contest_answers=len(gens),
                 contest_passed_any=len(best), contest_rows=len(out), contest_ticket_rows=n_ticket,
                 contest_dropped=drops)
    if think:
        stats["contest_thinking_chars_median"] = statistics.median(
            len(r["messages"][-1].get("reasoning_content", "")) for r in out) if out else 0
    print(f"[data] contest: {len(out)} rows ({n_ticket} as tickets) from {len(asks)} LiveCodeBench problems "
          f"released before {cfg.data_contest_before}, {len(best)} solved, dropped {drops}", flush=True)
    return out


def extra_rows(job: Job, stats: dict) -> list[dict]:
    """The ready chat rows of Config.data_extra_rows, as they are: each a {"messages": [...]}
    ending in the assistant's answer, e.g. teacher transcripts in aider's format."""
    path = job.root / job.config.data_extra_rows  # an absolute path stays as is
    rows = read_jsonl(path)
    for i, r in enumerate(rows, 1):
        if (r.get("messages") or [{}])[-1].get("role") != "assistant":
            raise ValueError(f"{path} line {i}: not a chat row ending in an assistant message")
    stats["extra_rows"] = len(rows)
    print(f"[data] {len(rows)} extra rows from {path}", flush=True)
    return rows


def _chat_row(msgs: list[dict], answer: str, reasoning: str = "") -> dict:
    return {"messages": [*msgs, {"role": "assistant", "content": answer,
                                 **({"reasoning_content": reasoning} if reasoning else {})}]}


def check_seeds(spec) -> None:
    """Code specs: every seed answer must pass its own tests, or the teacher learns from broken shots."""
    res = sandbox.run_many([(spec.language, e.output, e.tests) for e in spec.seed_examples])
    bad = [(i, r) for i, r in enumerate(res) if not r.passed]
    if bad:
        i, r = bad[0]
        raise RuntimeError(f"{len(bad)} seed example(s) fail their own tests; seed {i}: {r.reason}\n{r.output}")
    print(f"[data] all {len(res)} seed examples pass their tests ({spec.language})", flush=True)


def _heldout_row(spec, r: dict, tests_of: dict[str, str]) -> dict:
    inp = r["messages"][1]["content"]
    row = {"input": inp, "reference": r["messages"][2]["content"]}
    if spec.is_code:
        row.update(tests=tests_of[inp], language=spec.language, system=system_prompt(spec))
    return row


def _row(sys: str, inp: str, out: str, reasoning: str = "") -> dict:
    """A chat row. The teacher's thinking, when kept, goes in reasoning_content (the
    field Qwen's chat templates render inside <think>, Gemma 4's in its thought
    channel); taskdata trains on it."""
    return {"messages": [
        {"role": "system", "content": sys},
        {"role": "user", "content": inp},
        {"role": "assistant", "content": out, **({"reasoning_content": reasoning} if reasoning else {})},
    ]}


def _with_thinking(msg: dict) -> str:
    return (f"<think>\n{msg['reasoning_content']}\n</think>\n\n" if msg.get("reasoning_content") else "") + msg["content"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    run_stage(Job(ap.parse_args().job))
