"""Never train on MuggeBench: drop any row or practice ticket that carries the
`MUGGE-BENCH-` canary or shares text with a MuggeBench ticket (the public spike, the private
dev set when it is on this machine, any other bench dir given).

The index holds, for every bench ticket, its context, its reference answer's files, its
hidden test files and the scaffold's files, as 13-token shingles (whitespace tokens, the
usual contamination n-gram). A text is contaminated when it holds the canary, or shares at
least MIN_HITS distinct shingles with the index (one shared shingle is often boilerplate:
a license line, an import block). Stdlib only. bhdata has its own decontam
(bhdata.decontam); bhdata_io.py runs it too when bhdata is reachable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

CANARY = "MUGGE-BENCH-"
N = 13
MIN_HITS = 3
_TOKEN = re.compile(r"\S+")
TEXT_SUFFIXES = {".py", ".c", ".h", ".ts", ".js", ".tsx", ".jsx", ".cs", ".json", ".md", ".txt", ".toml", ".yaml",
                 ".yml", ".sh", ".html", ".css", ".cpp", ".hpp", ".go", ".rs", ".java", ".sql", "", ".cfg", ".ini"}


def shingles(text: str, n: int = N) -> set[int]:
    toks = _TOKEN.findall(text.lower())
    out = set()
    for i in range(max(0, len(toks) - n + 1)):
        out.add(int.from_bytes(hashlib.blake2b(" ".join(toks[i:i + n]).encode(), digest_size=8).digest(), "big"))
    return out


def _texts(root: Path) -> list[str]:
    out = []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules", "build", "__pycache__", "bin", "obj")]
        for f in files:
            p = Path(dirpath, f)
            if p.suffix.lower() in TEXT_SUFFIXES and p.stat().st_size < 1 << 20:
                try:
                    out.append(p.read_text())
                except (UnicodeDecodeError, OSError):
                    pass
    return out


def bench_texts(bench: str | Path) -> list[str]:
    """Every text a bench dir's projects hold: plan contexts, references, hidden tests, scaffold."""
    bench = Path(bench)
    projects = [bench] if (bench / "plan.json").exists() else [p for p in sorted(bench.iterdir()) if (p / "plan.json").exists()]
    out = []
    for p in projects:
        plan = json.loads((p / "plan.json").read_text())
        out += [t.get("context", "") for t in plan.get("tickets", [])]
        for s in sorted((p / "solutions").glob("*.json")):
            out += [str(v) for v in json.loads(s.read_text()).get("files", {}).values()]
        out += _texts(p / "hidden") + _texts(p / "project")
    return out


@dataclass
class Index:
    shingles: set[int] = field(default_factory=set)
    sources: list[str] = field(default_factory=list)

    def add_bench(self, bench: str | Path) -> None:
        if not Path(bench).is_dir():
            raise FileNotFoundError(f"no MuggeBench dir {bench} to decontaminate against")
        for t in bench_texts(bench):
            self.shingles |= shingles(t)
        self.sources.append(str(bench))

    def hits(self, text: str) -> int:
        return len(shingles(text) & self.shingles)

    def why(self, text: str) -> str:
        """"" when clean, else the reason it is not."""
        if CANARY in text:
            return "canary"
        h = self.hits(text)
        return f"overlap:{h}" if h >= MIN_HITS else ""


def default_benches() -> list[Path]:
    """The MuggeBench dirs on this machine: the public spike, plus the dev set where the run
    sheets put it (MUGGE_BENCH_DIRS, colon separated, adds more)."""
    from .render import MUGGE
    dirs = [MUGGE / "spike", Path(os.environ.get("NVME", "/mnt/nvme")) / "muggebench"]
    dirs += [Path(d) for d in os.environ.get("MUGGE_BENCH_DIRS", "").split(":") if d]
    return [d for d in dirs if d.exists()]


def index(benches: list[str | Path] | None = None) -> Index:
    ix = Index()
    for b in default_benches() if benches is None else benches:
        ix.add_bench(b)
    return ix


def row_text(row: dict) -> str:
    return "\n".join(str(m.get("content", "")) + "\n" + str(m.get("reasoning_content", "")) for m in row.get("messages", []))


def filter_rows(rows: list[dict], ix: Index) -> tuple[list[dict], dict[str, int]]:
    """(rows that are clean, counts of dropped rows by reason)."""
    kept, dropped = [], {}
    for r in rows:
        why = ix.why(row_text(r))
        if why:
            k = why.split(":")[0]
            dropped[k] = dropped.get(k, 0) + 1
        else:
            kept.append(r)
    return kept, dropped


def contaminated_tickets(bench: str | Path, ix: Index) -> dict[str, str]:
    """Practice tickets (a bhdata export) that must not be trained on: {key: reason}. A
    ticket is judged by its context and its reference answer; a project whose plan or
    scaffold carries the canary is dropped whole."""
    bench = Path(bench)
    out = {}
    projects = [bench] if (bench / "plan.json").exists() else [p for p in sorted(bench.iterdir()) if (p / "plan.json").exists()]
    for p in projects:
        plan_text = (p / "plan.json").read_text()
        plan = json.loads(plan_text)
        whole = "canary" if CANARY in plan_text or any(CANARY in t for t in _texts(p / "project")) else ""
        for t in plan.get("tickets", []):
            key = f"{p.name}/{t['id']}"
            if whole:
                out[key] = whole
                continue
            sol = p / "solutions" / f"{t['id']}.json"
            ref = "\n".join(map(str, json.loads(sol.read_text()).get("files", {}).values())) if sol.exists() else ""
            why = ix.why(t.get("context", "") + "\n" + ref)
            if why:
                out[key] = why
    return out
