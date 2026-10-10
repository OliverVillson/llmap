"""Task data helpers for the REAP, heal and quantize stages: read <job>/data/train.jsonl
into chat messages, pick calibration rows, and tokenize them with the prompt masked out
of the loss."""
from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from pathlib import Path


def load_examples(path: Path) -> list[dict]:
    """Read a jsonl file into a list of {"messages": [...]} where the
    last message is the assistant target, plus the row's "meta" if it has one
    (training ignores it). Accepts prompt/response, input/output,
    instruction/output and messages rows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"task data not found: {path}")
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = to_messages(row)
            if msgs:
                out.append({"messages": msgs, **({"meta": row["meta"]} if "meta" in row else {})})
    if not out:
        raise ValueError(f"no usable examples in {path}")
    return out


def is_extra(ex: dict) -> bool:
    """A row the data stage took from Config.data_extra_rows."""
    return bool((ex.get("meta") or {}).get("extra"))


def calib_examples(path: Path, n: int, extra_share: float = 0.0) -> list[dict]:
    """n rows of a train.jsonl to calibrate on, in a fixed shuffle. extra_share
    (Config.calib_extra_share) of them are extra rows, each distinct row at most once,
    taken from each language (meta.language) in turn; the rest are the other rows, then
    any extra rows left. 0 takes the first n of the shuffle, copies and all."""
    if not 0 <= extra_share <= 1:
        raise ValueError(f"calib_extra_share must be between 0 and 1, not {extra_share}")
    examples = load_examples(path)
    rng = random.Random(0)
    rng.shuffle(examples)
    if not extra_share:
        return examples[:n]
    by_lang, other, seen = {}, [], set()
    for ex in examples:
        if not is_extra(ex):
            other.append(ex)
        elif (key := json.dumps(ex["messages"], sort_keys=True)) not in seen:
            seen.add(key)
            by_lang.setdefault(str(ex["meta"].get("language") or ""), []).append(ex)
    extra = [ex for turn in itertools.zip_longest(*(by_lang[k] for k in sorted(by_lang))) for ex in turn if ex]
    k = min(round(n * extra_share), len(extra))
    out = (extra[:k] + other + extra[k:])[:n]
    rng.shuffle(out)  # REAP measures layer importance on the first rows
    langs = Counter(str(ex["meta"].get("language") or "?") for ex in out if is_extra(ex))
    print(f"calibration: {sum(langs.values())} of {len(out)} rows are extra rows {dict(sorted(langs.items()))}",
          flush=True)
    return out


def to_messages(row: dict) -> list[dict] | None:
    if "messages" in row:
        msgs = [m for m in row["messages"] if m.get("content")]
        if len(msgs) >= 2 and msgs[-1]["role"] == "assistant":
            return msgs
        return None
    for pk, rk in (("prompt", "response"), ("input", "output"),
                   ("instruction", "output"), ("question", "answer")):
        if pk in row and rk in row:
            msgs = []
            if row.get("system"):
                msgs.append({"role": "system", "content": row["system"]})
            msgs.append({"role": "user", "content": str(row[pk])})
            msgs.append({"role": "assistant", "content": str(row[rk])})
            return msgs
    return None


# How a generation marks its thinking: (opening, closing). Qwen wraps it in
# <think> tags. Gemma 4 (turns in <|turn>...<turn|>) writes it in a thought
# channel, which the model opens itself after the <|think|> switch its template
# puts in the system turn when enable_thinking is on; vLLM's "gemma4" reasoning
# parser splits it off.
QWEN_THINK = ("<think>\n", "\n</think>\n\n")
GEMMA4_THINK = ("<|channel>thought\n", "\n<channel|>")


def think_tags(template: str | None) -> tuple[str, str]:
    """The thinking markers of a chat template, or of a prompt rendered with one
    (by its last turn marker: a message may quote the other format)."""
    t = template or ""
    return GEMMA4_THINK if t.rfind("<|turn>") > t.rfind("<|im_start|>") else QWEN_THINK


def _template(tok, msgs, thinking: bool = False, **kw):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, enable_thinking=thinking, **kw)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, **kw)


def tokenize_example(tok, msgs: list[dict], max_len: int) -> dict:
    """input_ids plus labels with the prompt masked to -100, so loss is only on
    the teacher's answer. The prompt is rendered exactly as at inference
    (add_generation_prompt=True), which matters for templates whose generation
    prompt differs from a rendered past turn (Gemma 4 adds an empty thought
    channel). The answer is followed by the template's end-of-turn marker.

    An answer with reasoning_content (data_thinking) is trained as a thinking
    turn: the prompt is rendered with thinking on and the target is the thinking,
    </think>, then the answer, so the model learns to think and to close it.
    Gemma 4's target is its thought channel (<|channel>thought, the thinking,
    <channel|>) and the answer."""
    answer = msgs[-1]["content"]
    reasoning = (msgs[-1].get("reasoning_content") or "").strip()
    if tok.chat_template:
        prompt = _template(tok, msgs[:-1], thinking=bool(reasoning), add_generation_prompt=True)
        full = _template(tok, [*msgs[:-1], {"role": "assistant", "content": answer}])
        at = full.rfind(answer)
        suffix = full[at + len(answer):] if at >= 0 else (tok.eos_token or "")
        suffix = suffix.rstrip("\n")  # keep e.g. <|im_end|> / <turn|> as a target, not the newline
    else:  # bare tokenizer (tests): plain concatenation
        prompt = "".join(m["content"] + "\n" for m in msgs[:-1])
        suffix = tok.eos_token or ""
    if reasoning:
        opening, closing = think_tags(tok.chat_template)
        if opening == GEMMA4_THINK[0]:  # trained to open it, as the model does at inference
            reasoning = opening + reasoning
        elif not prompt.rstrip().endswith("<think>"):  # templates that leave opening it to the model
            prompt += opening
        answer = reasoning + closing + answer
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer + suffix, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return {"input_ids": ids, "labels": labels}
