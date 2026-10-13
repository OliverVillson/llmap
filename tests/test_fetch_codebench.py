"""scripts/fetch_codebench.py (pinned revisions, every case, window id hashes, CodeContests
and APPS deduped against LiveCodeBench, APPS forms checked by reference solutions) and
scripts/setup_vm.sh (the llama.cpp pin, the suites it fetches, SKIP_EXTRAS, the order of
its checks). Hugging Face is faked; the sandbox and git are real."""

import base64
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import types
import zlib
from pathlib import Path

import pytest

from stages import codebench as cb
from stages import sandbox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_codebench as fc  # noqa: E402

SETUP = ROOT / "scripts" / "setup_vm.sh"
SHA = "0123456789abcdef0123456789abcdef01234567"


def fake_hub(monkeypatch, tmp_path, files: dict[str, dict[str, str]], sha: str = SHA) -> list:
    """Stands in for huggingface_hub: files[repo][name] is the text of that dataset file.
    Returns the (call, repo, revision) log."""
    calls = []
    store = tmp_path / "hub"

    class Api:
        def dataset_info(self, repo, revision=None):
            calls.append(("info", repo, revision))
            return types.SimpleNamespace(sha=sha)

    def list_repo_files(repo, repo_type=None, revision=None):
        assert repo_type == "dataset"
        calls.append(("list", repo, revision))
        return sorted(files[repo])

    def hf_hub_download(repo, name, repo_type=None, revision=None):
        assert repo_type == "dataset"
        calls.append(("download", repo, revision))
        p = store / repo / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(files[repo][name])
        return str(p)

    mod = types.ModuleType("huggingface_hub")
    mod.HfApi, mod.list_repo_files, mod.hf_hub_download = Api, list_repo_files, hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return calls


def jsonl(rows) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def packed(cases) -> str:
    """LiveCodeBench's private case encoding: base64 of zlib of a pickled JSON string."""
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(cases)))).decode()


def lcb_raw(qid, date, cases, platform="atcoder", func=None, words="Add the two numbers"):
    kind = "functional" if func else "stdin"
    cases = [{**c, "testtype": kind} for c in cases]
    return {"question_title": qid, "question_content": f"{words} ({qid}).", "platform": platform, "question_id": qid,
            "contest_id": "c", "contest_date": f"{date}T00:00:00", "difficulty": "easy",
            "starter_code": "class Solution:\n    def f(self, x: int) -> int:\n" if func else "",
            "metadata": json.dumps({"func_name": func} if func else {}),
            "public_test_cases": json.dumps(cases[:1]), "private_test_cases": packed(cases[1:])}


# ---------- LiveCodeBench ----------


def test_lcb_row_keeps_every_case_unless_capped():
    many = [{"input": f"{i}\n", "output": f"{i}\n"} for i in range(80)]
    r = lcb_raw("abc1", "2025-01-02", many)
    assert len(fc.lcb_row(r)["tests"]) == 80 and fc.lcb_row(r)["public"] == 1
    assert len(fc.lcb_row(r, 50)["tests"]) == 50  # the old cap, now only on request
    assert fc.lcb_row(r, 0) == fc.lcb_row(r)


def test_codecontests_row_zero_means_every_case():
    t = lambda n: {"input": [f"{i}\n" for i in range(n)], "output": [f"{i}\n" for i in range(n)]}
    r = {"name": "1_A. X", "description": "d", "source": 2, "public_tests": t(1), "private_tests": t(30),
         "generated_tests": t(40)}
    assert len(fc.codecontests_row(r, 0)["tests"]) == 71 and len(fc.codecontests_row(r, 50)["tests"]) == 50


def test_ids_sha256_ignores_order():
    assert fc.ids_sha256(["b", "a"]) == fc.ids_sha256(iter(["a", "b"])) == hashlib.sha256(b"a\nb").hexdigest()
    assert fc.ids_sha256(["a"]) != fc.ids_sha256(["a", "b"])


def lcb_files() -> dict[str, str]:
    stdin = lambda n: [{"input": f"{i} {i}\n", "output": f"{2 * i}\n"} for i in range(n)]
    func = [{"input": f"{i}", "output": f"{i}"} for i in range(3)]
    files = {f: "" for f in fc.LCB_FILES}
    files["test.jsonl"] = jsonl([lcb_raw("abc390_a", "2025-01-25", stdin(61)),   # test window, 61 cases
                                 lcb_raw("2010A", "2024-07-30", stdin(2), platform="codeforces")])  # before both
    files["test5.jsonl"] = jsonl([lcb_raw("3456", "2024-09-10", func, platform="leetcode", func="f"),  # dev
                                  lcb_raw("abc380_b", "2024-11-16", stdin(3)),  # dev
                                  {**lcb_raw("empty", "2024-10-01", []), "public_test_cases": "[]",
                                   "private_test_cases": "[]"}])  # no cases: dropped
    files["test6.jsonl"] = jsonl([lcb_raw("abc402_c", "2025-05-03", stdin(2)),  # after both windows
                                  lcb_raw("abc390_a", "2025-01-25", stdin(2)),  # repeated: the first is kept
                                  lcb_raw("3500", "2025-04-30", func, platform="leetcode", func="f")])  # last test day
    files["test7.jsonl"] = jsonl([lcb_raw("abc999_z", "2025-06-01", stdin(2))])  # a later release: not read
    return files


def test_fetch_livecodebench_pins_one_commit_keeps_every_case_and_hashes_the_windows(monkeypatch, tmp_path):
    calls = fake_hub(monkeypatch, tmp_path, {fc.LCB_REPO: lcb_files()})
    out = tmp_path / "codebench"
    out.mkdir()
    meta = fc.fetch_livecodebench(out)

    assert calls[0] == ("info", fc.LCB_REPO, fc.REVISIONS[fc.LCB_REPO])
    assert all(rev == SHA for kind, _, rev in calls[1:]) and len(calls) == 1 + 1 + len(fc.LCB_FILES)
    text = (out / "livecodebench.jsonl").read_text()
    rows = [json.loads(l) for l in text.splitlines()]
    assert [r["date"] for r in rows] == sorted(r["date"] for r in rows)
    assert [r["id"] for r in rows] == ["codeforces/2010A", "leetcode/3456", "atcoder/abc380_b", "atcoder/abc390_a",
                                       "leetcode/3500", "atcoder/abc402_c"]
    big = next(r for r in rows if r["id"] == "atcoder/abc390_a")
    assert len(big["tests"]) == 61 and big["public"] == 1
    assert rows[1]["tests"] == {"func": "f", "cases": [[str(i), str(i)] for i in range(3)]}
    assert not list(out.glob("*.part"))

    saved = json.loads((out / "livecodebench.meta.json").read_text())
    assert saved == {**meta, "suite": "livecodebench", "fetched": saved["fetched"]}
    assert meta["sha"] == SHA and meta["revision"] == "main" and meta["max_cases"] == 0
    assert meta["files"] == list(fc.LCB_FILES)
    assert meta["rows"] == 6 and meta["bytes"] == len(text.encode())
    assert meta["file_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert meta["ids_sha256"] == fc.ids_sha256(r["id"] for r in rows)
    assert meta["read"] == 8 and meta["dropped"] == {"no_cases_or_func": 1, "duplicate": 1}
    dev, test = meta["windows"]["dev"], meta["windows"]["test"]
    assert (dev["since"], dev["until"], test["since"], test["until"]) == ("2024-08-01", "2024-12-31",
                                                                          "2025-01-01", "2025-04-30")
    assert dev["ids"] == ["atcoder/abc380_b", "leetcode/3456"] and dev["n"] == 2 and dev["functional"] == 1
    assert test["ids"] == ["atcoder/abc390_a", "leetcode/3500"] and test["functional"] == 1
    assert test["ids_sha256"] == fc.ids_sha256(test["ids"])

    # the windows as the eval selects them (load_suite since, then until)
    loaded = cb.load_suite("livecodebench", out, since=test["since"])
    assert sorted(p["id"] for p in loaded if p["date"] <= test["until"]) == test["ids"]


def test_fetch_livecodebench_caps_on_request_and_refuses_a_commit_without_the_release(monkeypatch, tmp_path):
    fake_hub(monkeypatch, tmp_path, {fc.LCB_REPO: lcb_files()})
    meta = fc.fetch_livecodebench(tmp_path, max_cases=5)
    big = next(json.loads(l) for l in (tmp_path / "livecodebench.jsonl").read_text().splitlines() if "abc390_a" in l)
    assert len(big["tests"]) == 5 and meta["max_cases"] == 5

    files = lcb_files()
    del files["test4.jsonl"]
    fake_hub(monkeypatch, tmp_path, {fc.LCB_REPO: files})
    with pytest.raises(RuntimeError, match="test4.jsonl"):
        fc.fetch_livecodebench(tmp_path)
    assert not (tmp_path / "livecodebench.meta.json").exists()  # the old meta no longer vouches for the file


# ---------- CodeContests ----------


LCB_TEXT = ("Takahashi has a sequence of N integers and wants to know how many pairs of adjacent elements "
            "have an odd sum after every one of Q updates")


def write_lcb(out: Path) -> None:
    rows = [{"id": "atcoder/abc1", "language": "python", "prompt": LCB_TEXT, "stub": "", "entry": "",
             "tests": [{"input": "1\n", "output": "1\n"}], "date": "2025-01-01", "public": 1}]
    (out / "livecodebench.jsonl").write_text(jsonl(rows))


def cc_raw(name, description, n=3):
    t = lambda k: {"input": [f"{i}\n" for i in range(k)], "output": [f"{i}\n" for i in range(k)]}
    return {"name": name, "description": description, "source": 2, "difficulty": 7, "public_tests": t(1),
            "private_tests": t(n), "generated_tests": t(0)}


def test_fetch_codecontests_pins_the_commit_and_drops_problems_sharing_13_words_with_livecodebench(monkeypatch, tmp_path):
    seen = {}

    def load_dataset(repo, split=None, streaming=None, revision=None):
        seen.update(repo=repo, split=split, streaming=streaming, revision=revision)
        return iter([cc_raw("1_A. Odd", f"Note. {LCB_TEXT}, then print it."),  # 13+ words in common
                     cc_raw("2_B. Sum", "Print the sum of the numbers you are given."),
                     cc_raw("2_B. Sum again", "The same id twice."),
                     cc_raw("3_C. Big", "Print the numbers back.", n=80)])

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    fake_hub(monkeypatch, tmp_path, {})
    with pytest.raises(FileNotFoundError, match="livecodebench first"):
        fc.fetch_codecontests(tmp_path)
    write_lcb(tmp_path)
    meta = fc.fetch_codecontests(tmp_path, max_cases=50)
    assert seen == {"repo": fc.CC_REPO, "split": "train", "streaming": True, "revision": SHA}
    rows = [json.loads(l) for l in (tmp_path / "codecontests.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["codeforces/2_B", "codeforces/3_C"]
    assert len(rows[1]["tests"]) == 50
    assert meta["dropped"] == {"lcb_overlap": 1} and meta["sha"] == SHA and meta["rows"] == 2
    assert meta["ids_sha256"] == fc.ids_sha256(["codeforces/2_B", "codeforces/3_C"])
    assert (tmp_path / "codecontests.meta.json").exists()


# ---------- APPS ----------


def apps_raw(pid, question, fn, starter, inputs, outputs, solutions):
    return {"problem_id": pid, "question": question, "starter_code": starter, "difficulty": "introductory",
            "url": f"https://www.codewars.com/kata/{pid}",
            "input_output": json.dumps({"fn_name": fn, "inputs": inputs, "outputs": outputs} if fn else
                                       {"inputs": inputs, "outputs": outputs}),
            "solutions": json.dumps(solutions)}


ADD = "Write a function that adds two numbers a and b and returns their sum to the caller."
TWO_SUM = "Given an array of integers nums and an integer target, return indices of the two numbers that add up to target."
FIRST = "Return a list holding only the first element of the given list of numbers, keeping its type."


def apps_files() -> dict[str, str]:
    add = "def add(a, b):\n    return a + b\n"
    two_sum = ("class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
               "        seen = {}\n        for i, x in enumerate(nums):\n            if target - x in seen:\n"
               "                return [seen[target - x], i]\n            seen[x] = i\n")
    train = [
        apps_raw(1, ADD, "add", "def add(a, b):", [[1, 2], [3, 4]], [[3], [7]], [add]),
        apps_raw(2, TWO_SUM, "twoSum", "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n        ",
                 [[[2, 7, 11], 9], [[3, 2, 4], 6]], [[[0, 1]], [[1, 2]]], [two_sum.replace("[seen[target - x], i]", "[0, 1]"), two_sum]),
        # expected values stored bare, each a one-item list: only the as-given form passes
        apps_raw(3, FIRST, "first", "def first(xs):", [[[5, 6]], [[7]]], [[5], [7]], ["def first(xs):\n    return [xs[0]]\n"]),
        apps_raw(4, "Read two numbers and print their sum on one line please.", None, "", ["1 2\n"], ["3\n"], ["print(3)"]),
        apps_raw(5, f"Kata: {LCB_TEXT}.", "odd", "def odd(xs):", [[[1]]], [[0]], ["def odd(xs):\n    return 0\n"]),
        apps_raw(6, "Multiply the two numbers a and b together and return the product of them.", "mul",
                 "def mul(a, b):", [[2, 3]], [[6]], ["def mul(a, b):\n    return a + b\n"]),  # wrong reference
        apps_raw(7, "Return the number one, whatever the argument is, every single time it is called.", "one",
                 "def one(x):", [[1]], [[1]], []),  # no reference solution
        apps_raw(8, "A function named other is wanted here but the starter defines another name.", "other",
                 "def different(x):", [[1]], [[1]], ["def other(x):\n    return 1\n"]),
    ]
    test = [apps_raw(1, ADD.upper(), "add", "def add(a, b):", [[1, 1]], [[2]], [add])]  # train's problem 1 again
    return {"train.jsonl": jsonl(train), "test.jsonl": jsonl(test), "README.md": "not data"}


def test_apps_rows_take_values_out_of_one_item_lists_first():
    r = json.loads(apps_files()["train.jsonl"].splitlines()[2])
    unwrapped, as_given = fc.apps_rows(r, "train")
    assert unwrapped["tests"] == {"func": "first", "cases": [["[5, 6]", "5"], ["[7]", "7"]]}
    assert as_given["tests"] == {"func": "first", "cases": [["[5, 6]", "[5]"], ["[7]", "[7]"]]}
    assert unwrapped["id"] == "apps/train/3" and unwrapped["prompt"].endswith("```python\ndef first(xs):\n```")
    r2 = {**r, "input_output": json.dumps({"fn_name": "first", "inputs": [[[5, 6]]], "outputs": [[5, 6]]})}
    assert [row["tests"]["cases"] for row in fc.apps_rows(r2, "train")] == [[["[5, 6]", "[5, 6]"]]]
    assert fc.apps_rows({**r, "input_output": json.dumps({"fn_name": "first", "inputs": ["5"], "outputs": [[5]]})},
                        "train") == []  # an input that is not an argument list


def test_fetch_apps_keeps_problems_a_reference_solution_passes_in_the_form_it_passes(monkeypatch, tmp_path):
    calls = fake_hub(monkeypatch, tmp_path, {fc.APPS_REPO: apps_files()})
    with pytest.raises(FileNotFoundError, match="livecodebench first"):
        fc.fetch_apps(tmp_path)
    write_lcb(tmp_path)
    meta = fc.fetch_apps(tmp_path, max_cases=50)
    assert {rev for kind, _, rev in calls if kind != "info"} == {SHA} and meta["sha"] == SHA
    assert meta["files"] == ["test.jsonl", "train.jsonl"]
    rows = {r["id"]: r for r in map(json.loads, (tmp_path / "apps.jsonl").read_text().splitlines())}
    # test.jsonl is read first, so its copy of problem 1 is the one kept
    assert sorted(rows) == ["apps/test/1", "apps/train/2", "apps/train/3"]
    assert rows["apps/test/1"]["tests"] == {"func": "add", "cases": [["1\n1", "2"]]}
    assert rows["apps/train/2"]["tests"]["cases"] == [["[2, 7, 11]\n9", "[0, 1]"], ["[3, 2, 4]\n6", "[1, 2]"]]
    assert rows["apps/train/3"]["tests"]["cases"] == [["[5, 6]", "[5]"], ["[7]", "[7]"]]
    assert meta["outputs_as_given"] == 1
    assert meta["dropped"] == {"call_based": 7, "duplicate": 1, "lcb_overlap": 1, "no_solution": 1,
                               "reference_failed": 1}
    assert meta["rows"] == 3 and meta["ids_sha256"] == fc.ids_sha256(rows)

    # rows read like LiveCodeBench's functional ones and are graded by the same harness
    p = {**rows["apps/train/2"], "suite": "apps"}
    assert "Use this starter code" in cb.build_prompt(p) and "stdin" not in cb.build_prompt(p)
    good = ("```python\nclass Solution:\n    def twoSum(self, nums, target):\n"
            "        return next([i, j] for i in range(len(nums)) for j in range(i + 1, len(nums))"
            " if nums[i] + nums[j] == target)\n```")
    bad = "```python\nclass Solution:\n    def twoSum(self, nums, target):\n        return [0, 1]\n```"
    got = sandbox.run_many([cb.job(p, good), cb.job(p, bad), cb.job(rows["apps/test/1"], "```python\nadd = lambda a, b: a + b\n```")])
    assert [r.passed for r in got] == [True, False, True]


def test_fetch_apps_refuses_to_write_nothing(monkeypatch, tmp_path):
    files = {"train.jsonl": jsonl([apps_raw(6, "Multiply the two numbers a and b and return the product of both.",
                                            "mul", "def mul(a, b):", [[2, 3]], [[6]], ["def mul(a, b):\n    return 5\n"])])}
    fake_hub(monkeypatch, tmp_path, {fc.APPS_REPO: files})
    write_lcb(tmp_path)
    with pytest.raises(RuntimeError, match="reference"):
        fc.fetch_apps(tmp_path)
    assert not (tmp_path / "apps.jsonl").exists() and not (tmp_path / "apps.meta.json").exists()


def test_main_fetches_the_named_suites_livecodebench_first_with_every_lcb_case(monkeypatch, tmp_path):
    got = []
    for name in ("multipl_e", "livecodebench", "codecontests", "apps", "general_calib"):
        monkeypatch.setattr(fc, f"fetch_{name}", lambda out, *a, _n=name: got.append((_n, *a)))
    fc.main(["--out", str(tmp_path / "cb"), "--only", "apps", "codecontests", "livecodebench"])
    assert got == [("livecodebench", 0), ("codecontests", 50), ("apps", 50)]
    got.clear()
    fc.main(["--out", str(tmp_path / "cb"), "--max-cases", "7", "--train-max-cases", "0"])
    assert got == [("multipl_e",), ("livecodebench", 7), ("codecontests", 0), ("apps", 0), ("general_calib",)]


# ---------- setup_vm.sh ----------


def bash(script: str, **env) -> subprocess.CompletedProcess:
    """Runs script in bash with setup_vm.sh sourced for its functions only."""
    return subprocess.run(["bash", "-c", f'source "{SETUP}"\n{script}'], capture_output=True, text=True,
                          env={**os.environ, "SETUP_VM_LIB": "1", **env}, timeout=120)


def test_setup_vm_parses_and_sources_without_side_effects(tmp_path):
    assert subprocess.run(["bash", "-n", str(SETUP)], capture_output=True).returncode == 0
    r = bash('echo "$LLAMA_CPP_SHA $REPO"', NVME=str(tmp_path / "nvme"))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["af50f2f248b475d9760f8e67503d01c0f194f5f4", str(ROOT)]
    assert not (tmp_path / "nvme").exists()


def git(*args, cwd=None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def upstream(tmp_path):
    """A llama.cpp stand-in with two commits that serves fetches by commit id."""
    up = tmp_path / "upstream"
    up.mkdir()
    git("init", "-q", cwd=up)
    git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=up)
    shas = []
    for i in range(2):
        (up / "file.txt").write_text(f"version {i}\n")
        git("add", "file.txt", cwd=up)
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", f"v{i}", cwd=up)
        shas.append(git("rev-parse", "HEAD", cwd=up))
    return up, shas


@pytest.mark.skipif(not shutil.which("git"), reason="needs git")
def test_llama_cpp_checkout_fetches_only_the_pinned_commit_and_verifies_an_existing_one(tmp_path, upstream):
    up, (old, new) = upstream
    dest = tmp_path / "llama.cpp"
    env = {"LLAMA_CPP_REPO": f"file://{up}", "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig")}

    r = bash(f'llama_cpp_checkout "{dest}"', LLAMA_CPP_SHA=old, **env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert git("rev-parse", "HEAD", cwd=dest) == old and (dest / "file.txt").read_text() == "version 0\n"
    assert git("rev-parse", "--is-shallow-repository", cwd=dest) == "true"
    assert git("rev-list", "--count", "HEAD", cwd=dest) == "1"

    # at the pin already: checked, not fetched (the upstream is gone)
    (dest / "build").mkdir()
    r = bash(f'llama_cpp_checkout "{dest}"', LLAMA_CPP_SHA=old, LLAMA_CPP_REPO="file:///nonexistent")
    assert r.returncode == 0 and f"llama.cpp at {old}" in r.stdout

    # a clean checkout at another commit moves to the pin; untracked build output stays
    r = bash(f'llama_cpp_checkout "{dest}"', LLAMA_CPP_SHA=new, **env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert git("rev-parse", "HEAD", cwd=dest) == new and (dest / "build").is_dir()

    # local changes to tracked files are never thrown away
    (dest / "file.txt").write_text("edited\n")
    r = bash(f'llama_cpp_checkout "{dest}"', LLAMA_CPP_SHA=old, **env)
    assert r.returncode != 0 and "local changes" in r.stdout
    assert (dest / "file.txt").read_text() == "edited\n"
    # ... but a pinned checkout with them passes (a registered tokenizer hash, say)
    assert bash(f'llama_cpp_checkout "{dest}"', LLAMA_CPP_SHA=new, **env).returncode == 0

    assert "full 40-character" in bash(f'llama_cpp_checkout "{dest}"', LLAMA_CPP_SHA=new[:12], **env).stdout
    other = tmp_path / "not-git"
    other.mkdir()
    (other / "x").write_text("")
    r = bash(f'llama_cpp_checkout "{other}"', LLAMA_CPP_SHA=new, **env)
    assert r.returncode != 0 and "not a git checkout" in r.stdout


def test_missing_suites_counts_a_suite_once_its_meta_is_written(tmp_path):
    (tmp_path / "livecodebench.jsonl").write_text("")  # a partial or old-format fetch: no meta
    (tmp_path / "codecontests.jsonl").write_text("")
    (tmp_path / "codecontests.meta.json").write_text("{}")
    (tmp_path / "multipl-e-py.jsonl").write_text("")
    r = bash(f'missing_suites "{tmp_path}" multipl-e livecodebench codecontests apps general-calib')
    assert r.returncode == 0 and r.stdout.split() == ["livecodebench", "apps", "general-calib"]


def test_setup_vm_extras_pins_builds_and_the_cuda_check_order():
    text = SETUP.read_text()
    lines = text.splitlines()
    # extras sit inside `if [ -z "$SKIP_EXTRAS" ]` blocks; the suites under SKIP_EXTRAS are the three Event Horizon uses
    blocks, depth, inside = [], 0, False
    for line in lines:
        if re.match(r'\s*if \[ -z "\$SKIP_EXTRAS" \]; then', line):
            inside, depth = True, 0
        if inside:
            blocks.append(line)
            depth += line.strip().startswith("if ") - (line.strip() == "fi")
            if depth == 0:
                inside = False
    extras = "\n".join(blocks)
    for marker in ("typescript", "llmcompressor==", "polyglot.py\" setup"):
        assert marker in extras and text.count(marker) == extras.count(marker), marker
    assert 'SUITES="livecodebench codecontests apps"' in text
    assert "export LOBBOT_CODEBENCH=$NVME/codebench" in text and "LOBBOT_CODEBENCH" not in extras
    # the build has the KL, tokenizer and slot-sizing tools
    build = text[text.index("cmake --build"):].split("\n\n")[0]
    for target in ("llama-quantize", "llama-perplexity", "llama-tokenize", "llama-batched-bench"):
        assert target in build
    # gguf from the checkout, not PyPI; the CUDA check runs after the last install into venv-train
    first = text[text.index('"torch>=2.7"'):]
    assert "gguf" not in first[:first.index("\n\n")].split("--no-deps")[0]
    assert '"$NVME/llama.cpp/gguf-py"' in text
    installs = [i for i, l in enumerate(lines) if "uv pip install" in l and "venv-train" in l]
    check = next(i for i, l in enumerate(lines) if "torch.cuda.is_available(), torch" in l)
    assert installs and max(installs) < check
    assert text.count("torch.cuda.is_available(), torch") == 1
    assert 'uv pip freeze --python "$NVME/venv-train/bin/python"' in text
