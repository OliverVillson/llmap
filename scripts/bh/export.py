#!/usr/bin/env python3
"""Export helpers that need the training venv (transformers), and the export commands.

    python scripts/bh/export.py calib --hf <dir> --rows rows.jsonl [...] --out calib.txt [--max-rows 400]
        imatrix text shaped like the work: rows rendered as the model meets them (the
        prompt by its own chat template, thinking off, then the answer and end of turn)
    python scripts/bh/export.py roundtrip --hf <dir> --probe probe.json
        the tokenizer round-trip check (rows.roundtrip) on probe.json's row, against the
        server's own prompt ids ("prompt_ids") or rendered text ("prompt_text") when it
        holds them; exits 1 on a mismatch

The commands bh_run.py's export step runs (convert, vocab trim, imatrix, quantize) are
built by the functions below, so the dry run checks them too.
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bh.exec import Cmd  # noqa: E402


def q(x) -> str:
    return shlex.quote(str(x))


def convert_cmd(llama_cpp: str, hf: Path, out: Path) -> Cmd:
    return Cmd(f"python {q(Path(llama_cpp) / 'convert_hf_to_gguf.py')} {q(hf)} --outtype bf16 --outfile {q(out)}",
               "gguf-convert", {"hf": str(hf), "out": str(out)})


def vocab_cmds(llama_cpp: str, hf: Path, out: Path, traces: Path, codebench: str) -> list[Cmd]:
    """stages/vocab.py's gated trim (exit 3: no size passed, keep the full vocab), then the
    converter told the trimmed tokenizer's hash."""
    return [Cmd(f"python -m stages.vocab run --hf {q(hf)} --out {q(out)} --traces {q(traces)} --codebench {q(codebench)} "
                f"--llama-cpp {q(llama_cpp)} || [ $? -eq 3 ]", "vocab-run", {"hf": str(hf), "out": str(out)}, gpu=True),
            Cmd(f"if [ -f {q(out / 'vocab_index.json')} ]; then python -m stages.vocab register --llama-cpp {q(llama_cpp)} "
                f"--hf {q(out)}; fi", "vocab-register", {"out": str(out)})]


def traces_cmd(hf: Path, rows: Path, max_len: int, out: Path) -> Cmd:
    return Cmd(f"python -m stages.agent traces --hf {q(hf)} --rows {q(rows)} --max-len {max_len} --out {q(out)}",
               "traces", {"out": str(out)})


def tokcheck_cmd(llama_cpp: str, hf: Path, gguf: Path) -> Cmd:
    return Cmd(f"python -m stages.vocab tokcheck --hf {q(hf)} --gguf {q(gguf)} --llama-cpp {q(llama_cpp)} --texts 200",
               "tokcheck", {"hf": str(hf), "gguf": str(gguf)})


def calib_cmd(hf: Path, rows: list[Path], out: Path, max_rows: int) -> Cmd:
    rs = " ".join(f"--rows {q(r)}" for r in rows)
    return Cmd(f"python scripts/bh/export.py calib --hf {q(hf)} {rs} --out {q(out)} --max-rows {max_rows}",
               "calib", {"out": str(out)})


def imatrix_cmd(bin_dir: Path, bf16: Path, calib: Path, out: Path, chunks: int) -> Cmd:
    return Cmd(f"{q(bin_dir / 'llama-imatrix')} -m {q(bf16)} -f {q(calib)} -o {q(out)} -c 4096 --chunks {chunks} -ngl 999",
               "imatrix", {"out": str(out)}, gpu=True)


def quant_cmd(bin_dir: Path, bf16: Path, imatrix: Path, out: Path, qtype: str) -> Cmd:
    return Cmd(f"{q(bin_dir / 'llama-quantize')} --imatrix {q(imatrix)} {q(bf16)} {q(out)} {qtype}",
               "quantize", {"out": str(out), "type": qtype})


# ---------- the venv side ----------

def calib(a) -> None:
    from transformers import AutoTokenizer
    from bh import rows as R
    tok = AutoTokenizer.from_pretrained(a.hf)
    rows = [json.loads(x) for p in a.rows for x in Path(p).read_text().splitlines() if x.strip()]
    rows, _ = R.filter_rows(rows)
    random.Random(0).shuffle(rows)
    texts = []
    for r in rows[:a.max_rows]:
        m = r["messages"]
        texts.append(R.render_prompt(tok, m[:-1]) + m[-1]["content"] + R.end_of_turn(tok, m))
    Path(a.out).write_text("\n".join(texts))
    print(json.dumps({"rows": len(texts), "chars": sum(map(len, texts))}))


def roundtrip(a) -> int:
    from transformers import AutoTokenizer
    from bh import rows as R
    tok = AutoTokenizer.from_pretrained(a.hf)
    p = json.loads(Path(a.probe).read_text())
    row, problems = p["row"], []
    try:
        rep = R.roundtrip(tok, row, p.get("thinking", False), p.get("prompt_ids"))
    except ValueError as e:
        problems.append(str(e))
        rep = {}
    if p.get("prompt_text") is not None and p["prompt_text"] != R.render_prompt(tok, row["messages"][:-1]):
        problems.append("the server's chat template renders the prompt differently from the HF template")
    out = {**rep, "ok": not problems, "problems": problems}
    print(json.dumps(out))
    return 0 if not problems else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calib")
    c.add_argument("--hf", required=True)
    c.add_argument("--rows", action="append", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--max-rows", type=int, default=400)
    r = sub.add_parser("roundtrip")
    r.add_argument("--hf", required=True)
    r.add_argument("--probe", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "calib":
        calib(a)
        return 0
    return roundtrip(a)


if __name__ == "__main__":
    sys.exit(main())
