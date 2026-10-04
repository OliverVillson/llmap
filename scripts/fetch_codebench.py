"""Download the code eval suites into one directory of JSONL files.

    uv run --with datasets --with huggingface_hub python scripts/fetch_codebench.py --out /mnt/nvme/codebench

Writes <out>/<suite>.jsonl in the problem shape stages/codebench.py reads:
  multipl-e-{py,js,ts,cpp}  nuprl/MultiPL-E HumanEval translations (Python
                            falls back to openai/openai_humaneval)
  livecodebench             livecodebench/code_generation_lite, every release
                            file; stages/eval.py keeps the recent ones (Config.lcb_since)
Run it once per VM; eval reads the files offline.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import pickle
import re
import zlib
from pathlib import Path

MULTIPLE = {"py": "python", "js": "js", "ts": "ts", "cpp": "cpp"}


def multipl_e_row(lang: str, r: dict) -> dict:
    m = re.search(r"candidate\s*=\s*([\w:]+)", r["tests"])
    tests = r["tests"]
    # C++ completions stop before the closing brace, so the tests open with it.
    # Answers here are whole functions; codebench.assemble closes body-only ones.
    if tests.lstrip().startswith("}"):
        tests = tests.lstrip()[1:]
    return {"id": r["name"], "language": lang, "prompt": r["prompt"], "stub": r["prompt"],
            "entry": m.group(1) if m else "", "tests": tests}


def fetch_multipl_e(out: Path) -> None:
    from datasets import load_dataset

    for short, lang in MULTIPLE.items():
        try:
            ds = load_dataset("nuprl/MultiPL-E", f"humaneval-{short}", split="test")
            rows = [multipl_e_row(lang, r) for r in ds]
        except Exception as e:
            if short != "py":
                raise
            print(f"MultiPL-E humaneval-py unavailable ({e}); using openai/openai_humaneval")
            ds = load_dataset("openai/openai_humaneval", split="test")
            rows = [{"id": r["task_id"], "language": "python", "prompt": r["prompt"], "stub": r["prompt"],
                     "entry": r["entry_point"], "tests": r["test"] + f"\n\ncheck({r['entry_point']})\n"} for r in ds]
        write(out / f"multipl-e-{short}.jsonl", rows)


class _StrOnly(pickle.Unpickler):
    """LiveCodeBench pickles a plain JSON string; refuse anything else."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing {module}.{name}")


def decode_cases(s: str) -> list[dict]:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        raw = _StrOnly(io.BytesIO(zlib.decompress(base64.b64decode(s)))).load()
        return json.loads(raw)


def functional_tests(func: str, cases: list[dict]) -> str:
    data = json.dumps([[c["input"], c["output"]] for c in cases])
    return (
        "import json as _json\n"
        f"_cases = _json.loads({data!r})\n"
        "_f = getattr(Solution(), " + repr(func) + ")\n"
        "for _i, (_inp, _out) in enumerate(_cases):\n"
        "    _got = _f(*[_json.loads(_l) for _l in _inp.split('\\n') if _l.strip()])\n"
        "    _got = list(_got) if isinstance(_got, tuple) else _got\n"
        "    _want = _json.loads(_out)\n"
        "    assert _got == _want, f'case {_i}: expected {_want!r}, got {_got!r}'\n")


def lcb_row(r: dict, max_cases: int) -> dict | None:
    cases = decode_cases(r["public_test_cases"]) + decode_cases(r["private_test_cases"])
    if not cases:
        return None
    cases = cases[:max_cases]
    prompt = r["question_content"].strip()
    if cases[0].get("testtype") == "functional":
        func = json.loads(r.get("metadata") or "{}").get("func_name")
        if not func:
            return None
        prompt += f"\n\nUse this starter code:\n```python\n{r['starter_code'].rstrip()}\n```"
        tests = functional_tests(func, cases)
    else:
        tests = [{"input": c["input"], "output": c["output"]} for c in cases]
    return {"id": f"{r['platform']}/{r['question_id']}", "language": "python", "prompt": prompt, "stub": "",
            "entry": "", "tests": tests, "date": str(r["contest_date"])[:10], "difficulty": r.get("difficulty")}


def fetch_livecodebench(out: Path, max_cases: int) -> None:
    from huggingface_hub import hf_hub_download, list_repo_files

    repo = "livecodebench/code_generation_lite"
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset") if re.fullmatch(r"test\d*\.jsonl", f))
    rows, seen = [], set()
    for f in files:
        path = hf_hub_download(repo, f, repo_type="dataset")
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = lcb_row(json.loads(line), max_cases)
            if row and row["id"] not in seen:
                seen.add(row["id"])
                rows.append(row)
    rows.sort(key=lambda r: r["date"])
    write(out / "livecodebench.jsonl", rows)
    if rows:
        print(f"livecodebench: {rows[0]['date']} .. {rows[-1]['date']}")


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{path}: {len(rows)} problems")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=["multipl-e", "livecodebench"])
    ap.add_argument("--max-cases", type=int, default=50, help="test cases kept per LiveCodeBench problem")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "multipl-e"):
        fetch_multipl_e(out)
    if args.only in (None, "livecodebench"):
        fetch_livecodebench(out, args.max_cases)


if __name__ == "__main__":
    main()
