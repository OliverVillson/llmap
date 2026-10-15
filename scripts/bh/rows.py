"""Training rows: which ones may be trained on, and how one becomes token ids exactly as
the model meets it at inference.

A row is one engine call, {"messages": [system, user, assistant], "meta": {...}} as
data.py writes it. The guards here are the last line before training, whatever the data
path already did:

  - the system message is the engine's (stages/harness.py SYSTEM, held equal to
    prompts.ts by tests/test_harness.py), and there are exactly system, user, assistant;
  - meta.kind is write, fix or write-from-fix (BH0.1 trains the worker only);
  - nothing cut at the token cap, nothing forced after a thinking cap, no file whose path
    the parser had to guess (loose), no thinking in a thinking-off answer;
  - the answer parses back into at least one file with the engine's parser
    (stages/harness.py parse_files, files.ts's twin);
  - no MuggeBench canary anywhere.

Tokenizing (tokenize_row) renders the prompt with the model's own chat template, with
add_generation_prompt and enable_thinking=false, which is what vLLM and llama-server
render for the engine's request; the answer follows as the model would write it, then
the template's end of turn. The prompt is tokenized on its own, as the server does.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stages.harness import parse_files  # noqa: E402

from bh.wire import SYSTEM  # noqa: E402

CANARY = "MUGGE-BENCH-"
KINDS = ("write", "fix", "write-from-fix")


def meta(row: dict) -> dict:
    return row.get("meta") or {}


def row_key(row: dict) -> str:
    from bh.wire import call_key
    m = row["messages"]
    return call_key(m[1]["content"], m[2]["content"])


def drop_reason(row: dict, thinking: bool = False) -> str | None:
    msgs = row.get("messages") or []
    if [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
        return "shape"
    if msgs[0].get("content") != SYSTEM:
        return "system"
    mt = meta(row)
    if mt.get("kind") not in KINDS:
        return "kind"
    if row.get("cut") or mt.get("cut"):
        return "cut"
    if row.get("forced") or mt.get("forced"):
        return "forced"
    if row.get("loose") or mt.get("loose"):
        return "loose"
    answer = msgs[2].get("content") or ""
    if not thinking and ("<think>" in answer or "</think>" in answer or msgs[2].get("reasoning_content")):
        return "thinking"
    if not answer.strip() or not parse_files(answer)[0]:
        return "parse"
    if any(CANARY in (m.get("content") or "") for m in msgs):
        return "canary"
    return None


def filter_rows(rows: list[dict], thinking: bool = False) -> tuple[list[dict], dict]:
    """(rows to train on, counts of the dropped by reason)."""
    keep, dropped = [], Counter()
    seen = set()
    for r in rows:
        why = drop_reason(r, thinking)
        if why is None and row_key(r) in seen:
            why = "dupe"
        if why:
            dropped[why] += 1
        else:
            seen.add(row_key(r))
            keep.append(r)
    return keep, dict(dropped)


def drop_warning(stats: dict, limit: float) -> str | None:
    """The run-1/exp02 guard on harvest's counts (data.py's stats/teacher.json): more than
    `limit` of the answers dropped for length or because the parser could not read them
    (670 of 900 teacher fix answers once), loose paths (run 1's prog.c) included."""
    calls = stats.get("answers") or 0
    cut, parse, loose = (int(stats.get(k) or 0) for k in ("cut_or_forced", "parser_drops", "loose"))
    bad = cut + parse + loose
    if calls and bad / calls > limit:
        return (f"{bad} of {calls} answers ({100 * bad / calls:.0f}%) dropped for length or parsing "
                f"(cut or forced {cut}, parser {parse}, loose {loose}); over {100 * limit:.0f}%: "
                "check the token cap and that the parser and the training format agree")
    return None


# ---------- tokens (in the training venv: tok is a transformers tokenizer) ----------

def render_prompt(tok, messages: list[dict], thinking: bool = False) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def end_of_turn(tok, messages: list[dict], thinking: bool = False) -> str:
    """What the template puts after an assistant answer, up to and including its end of
    turn token (Qwen's <|im_end|>, Llama's <|eot_id|>): the full chat rendered, minus the
    prompt and the answer. Falls back to the EOS token."""
    prompt = render_prompt(tok, messages[:-1], thinking)
    full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=thinking)
    answer = messages[-1]["content"]
    if full.startswith(prompt):
        rest = full[len(prompt):]
        i = rest.find(answer)
        if i >= 0:
            tail = rest[i + len(answer):]
            eos = [t for t in (getattr(tok, "eos_token", None), "<|im_end|>", "<|eot_id|>", "<|end_of_text|>") if t]
            cut = min((tail.find(t) + len(t) for t in eos if t in tail), default=-1)
            if cut > 0:
                return tail[:cut]
    added = set(tok.get_added_vocab())
    for t in ("<|im_end|>", "<|eot_id|>", "<|end|>"):  # the turn's end marker when the template has one
        if t in added and t in (tok.chat_template or ""):
            return t
    return tok.eos_token or ""


def tokenize_row(tok, messages: list[dict], *, thinking: bool = False, answer_ids: list[int] | None = None) -> dict:
    """{"prompt": ids, "answer": ids (the answer then the end of turn), "n_answer": the
    answer's own ids before the end of turn}. answer_ids, when given (the teacher's own
    tokens from the proxy's log, same tokenizer), are used as the answer when they decode
    to it, so the stored top-k log-probs line up position by position."""
    prompt_text = render_prompt(tok, messages[:-1], thinking)
    prompt = tok(prompt_text, add_special_tokens=False)["input_ids"]
    answer = messages[-1]["content"]
    own = None
    if answer_ids is not None and tok.decode(answer_ids, skip_special_tokens=False) == answer:
        own = list(answer_ids)
    ans = own if own is not None else tok(answer, add_special_tokens=False)["input_ids"]
    eot = tok(end_of_turn(tok, messages, thinking), add_special_tokens=False)["input_ids"]
    return {"prompt": prompt, "answer": ans + eot, "n_answer": len(ans), "teacher_ids": own is not None}


def roundtrip(tok, row: dict, thinking: bool = False, served_prompt_ids: list[int] | None = None) -> dict:
    """The tokenizer round-trip check on one real row: the ids decode back to exactly the
    rendered text, the prompt carries the system and user messages verbatim, the answer
    ids decode to the answer, and (when the server's own tokenization of the same
    request is given) the prompt ids equal the server's. Raises on any mismatch."""
    msgs = row["messages"]
    t = tokenize_row(tok, msgs, thinking=thinking)
    prompt_text = render_prompt(tok, msgs[:-1], thinking)
    problems = []
    if tok.decode(t["prompt"], skip_special_tokens=False) != prompt_text:
        problems.append("prompt ids do not decode to the rendered prompt")
    for m in msgs[:-1]:
        if m["content"] not in prompt_text:
            problems.append(f"the {m['role']} message is not verbatim in the rendered prompt")
    if tok.decode(t["answer"][:t["n_answer"]], skip_special_tokens=False) != msgs[-1]["content"]:
        problems.append("answer ids do not decode to the answer")
    if not thinking and "<think>" in prompt_text and "</think>" not in prompt_text.rsplit("<think>", 1)[-1]:
        problems.append("thinking-off prompt leaves a <think> block open")
    if served_prompt_ids is not None and list(served_prompt_ids) != t["prompt"]:
        n = next((i for i, (a, b) in enumerate(zip(served_prompt_ids, t["prompt"])) if a != b),
                 min(len(served_prompt_ids), len(t["prompt"])))
        problems.append(f"prompt ids differ from the server's at token {n} "
                        f"({len(t['prompt'])} here, {len(served_prompt_ids)} served)")
    if problems:
        raise ValueError("tokenizer round-trip: " + "; ".join(problems))
    return {"prompt_tokens": len(t["prompt"]), "answer_tokens": t["n_answer"], "ok": True}
