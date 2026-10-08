"""The code10x recipes on a dense model (experiment 03, Qwen3.8-27B).

A dense model has no experts, so REAP has nothing to prune and heal nothing to
repair: a recipe comes down to its bits.

  q4    r25q4's bits: every feed-forward tensor q4_k, attention q5_k, output head
        q6_k, token embedding q4_k, the rest (Gated DeltaNet) llama.cpp's Q4_K_M mix
  w18s  r50w95s's bits: each layer's feed-forward tensors get 2 to 8 bits by
        saliency (stages/bits.py), averaging FFN_BITS like r50w95s's experts did,
        and every other tensor is q8_0

Subcommands, each skipping work already on disk:
  gguf      the HF model to a BF16 GGUF and an 8-bit copy. Text only: llama.cpp
            writes a vision tower to a separate mmproj file, which is never made
  calib     the 8-bit copy answers code problems with thinking on (MultiPL-E and
            the C set, never LiveCodeBench); its thinking and answers, in the chat
            format, are the calibration text
  imatrix   the BF16 model, its layers split between GPU and RAM, measures the
            importance matrix on that text
  quantize  one recipe into <job>/out/model.gguf, plus work/allocation.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from common.progress import emit
from stages import bits, codebench, quantize
from stages._util import DRY_RUN, Job, run, write_jsonl

STAGE = "quantize"
# r50w95s's experts averaged ~3.35 bits with every other tensor at q8_0
# (/mnt/project-files/plan/model-rebuilds.md); w18s gives the FFN the same.
FFN_BITS = 3.35
FFN = {"ffn_gate": "gate", "ffn_up": "up", "ffn_down": "down"}
Q4_ARGS = ["--output-tensor-type", "q6_k", "--token-embedding-type", "q4_k",
           "--tensor-type", r"attn_(q|k|v|output)\.weight=q5_k",
           "--tensor-type", r"ffn_(gate|up|down)\.weight=q4_k"]
RECIPES = ("q4", "w18s")
# Calibration problems per suite, spread over each suite; LiveCodeBench is the test.
CALIB_SUITES = {"multipl-e-py": 10, "multipl-e-js": 10, "multipl-e-ts": 10, "multipl-e-cpp": 10, "c-set": 8}
CALIB_MAX_TOKENS = 6144
IMATRIX_CHUNKS = 400  # x 512 tokens
END_OF_TURN = "<|im_end|>\n"  # Qwen's chat format


@dataclass
class Tensor:
    name: str
    n: int  # elements
    dims: int
    shape: tuple[int, ...] = ()  # (out, in) for a 2D weight


def ffn_slot(name: str) -> tuple[int, str] | None:
    """(layer, gate|up|down) for a dense feed-forward weight."""
    m = re.fullmatch(r"blk\.(\d+)\.(ffn_gate|ffn_up|ffn_down)\.weight", name)
    return (int(m[1]), FFN[m[2]]) if m else None


# ---------- the model's tensors ----------

def read_tensors(job: Job, path: Path) -> tuple[list[Tensor], int]:
    """The GGUF's tensors and its number of main layers (without the MTP layer)."""
    gguf = quantize.gguf_lib(job)
    r = gguf.GGUFReader(str(path))
    arch = str(bytes(r.fields["general.architecture"].parts[-1]), "utf-8")

    def kv(key: str) -> int:
        f = r.fields.get(f"{arch}.{key}")
        return int(f.parts[-1][0]) if f else 0

    tensors = [Tensor(t.name, int(t.n_elements), len(t.shape), tuple(int(x) for x in reversed(t.shape.tolist())))
               for t in r.tensors]
    return tensors, kv("block_count") - kv("nextn_predict_layers")


def synthetic_tensors(layers: int = 64, hidden: int = 5120, ffn: int = 17408, vocab: int = 248320) -> list[Tensor]:
    """Qwen3.8-27B's tensors from its model card: per block of four layers, three Gated
    DeltaNet (16 QK heads, 48 V heads of 128) and one gated attention (24 Q heads,
    4 KV heads of 256). For dry runs and tests."""
    out = [Tensor("token_embd.weight", vocab * hidden, 2, (vocab, hidden)),
           Tensor("output.weight", vocab * hidden, 2, (vocab, hidden)),
           Tensor("output_norm.weight", hidden, 1)]

    def w(name: str, o: int, i: int) -> None:
        out.append(Tensor(name, o * i, 2, (o, i)))

    for l in range(layers):
        b = f"blk.{l}."
        w(b + "ffn_gate.weight", ffn, hidden)
        w(b + "ffn_up.weight", ffn, hidden)
        w(b + "ffn_down.weight", hidden, ffn)
        out += [Tensor(b + "attn_norm.weight", hidden, 1), Tensor(b + "post_attention_norm.weight", hidden, 1)]
        if l % 4 == 3:
            w(b + "attn_q.weight", 2 * 24 * 256, hidden)  # queries and their output gate
            w(b + "attn_k.weight", 4 * 256, hidden)
            w(b + "attn_v.weight", 4 * 256, hidden)
            w(b + "attn_output.weight", hidden, 24 * 256)
        else:
            w(b + "attn_qkv.weight", 2 * 16 * 128 + 48 * 128, hidden)
            w(b + "attn_gate.weight", 48 * 128, hidden)
            w(b + "ssm_ba.weight", 2 * 48, hidden)
            w(b + "ssm_out.weight", hidden, 48 * 128)
            out += [Tensor(b + "ssm_a", 48, 1), Tensor(b + "ssm_dt.bias", 48, 1), Tensor(b + "ssm_norm.weight", 128, 1)]
    return out


def tensor_gb(t: Tensor, qtype: str) -> float:
    """Size at a llama.cpp type; 1D tensors (norms, biases) stay F32."""
    return t.n * (32 if t.dims < 2 else bits.BPW[qtype]) / 8e9


def ffn_params(tensors: list[Tensor], layers: int) -> int:
    return sum(t.n for t in tensors if (s := ffn_slot(t.name)) and s[0] < layers)


def static_gb(tensors: list[Tensor], layers: int, qtype: str = "q8_0") -> float:
    """Everything outside the main layers' feed-forward tensors, at one type."""
    return sum(tensor_gb(t, qtype) for t in tensors if not ((s := ffn_slot(t.name)) and s[0] < layers))


def shape_of(tensors: list[Tensor], layers: int) -> bits.MoEShape:
    """The dense model as bits.MoEShape sees it: one always-active "expert" per layer.
    Only the feed-forward sizes matter; the size of the rest is passed as static_gb."""
    gate = next(t for t in tensors if t.name == "blk.0.ffn_gate.weight")
    vocab = next((t.shape[0] for t in tensors if t.name == "token_embd.weight"), 0)
    return bits.MoEShape(n_layers=layers, hidden=gate.shape[1], moe_intermediate=gate.shape[0], n_experts=1,
                         n_experts_active=1, vocab=vocab, n_heads=1, n_kv_heads=1, head_dim=1)


def allocate_w(tensors: list[Tensor], layers: int, energy: dict[str, list[float]] | None,
               ffn_bits: float = FFN_BITS) -> tuple[list[bits.LayerBits], float, float]:
    """w18s: (per-layer FFN types, estimated size in GB, FFN average bits)."""
    fixed = static_gb(tensors, layers)
    n_ffn = ffn_params(tensors, layers)
    shape = shape_of(tensors, layers)
    plan = bits.allocate(shape, None, fixed + n_ffn * ffn_bits / 8e9, "q2_k", "q8_0", proj_energy=energy,
                         static=bits.static_types("q8_0"), static_gb=fixed)
    proj = shape.expert_params_per_proj
    ffn_gb = sum(proj * (2 * bits.BPW[l.gate_up] + bits.BPW[l.down]) / 8e9 for l in plan)
    return plan, fixed + ffn_gb, ffn_gb * 8e9 / n_ffn


def w_args(plan: list[bits.LayerBits], qtype: str = "q8_0") -> list[str]:
    """llama-quantize flags for w18s; the base type (qtype) covers every tensor not named."""
    args = ["--output-tensor-type", qtype, "--token-embedding-type", qtype]
    for l in plan:
        args += ["--tensor-type", rf"blk\.{l.layer}\.ffn_(gate|up)\.weight={l.gate_up}",
                 "--tensor-type", rf"blk\.{l.layer}\.ffn_down\.weight={l.down}"]
    return args


def imatrix_energy(job: Job, bf16: Path, imatrix: Path, layers: int) -> dict[str, list[float]]:
    """Expected output energy of each feed-forward tensor on the calibration text,
    sum_j E[x_j^2] * ||W[:, j]||^2 (the dense form of quantize.imatrix_energy)."""
    import numpy as np

    gguf = quantize.gguf_lib(job)
    im = {t.name: t for t in gguf.GGUFReader(str(imatrix)).tensors}
    energy = {k: [float("nan")] * layers for k in FFN.values()}
    for t in gguf.GGUFReader(str(bf16)).tensors:
        slot = ffn_slot(t.name)
        s2 = im.get(t.name + ".in_sum2")
        if not slot or slot[0] >= layers or s2 is None:
            continue
        n_in, n_out = (int(x) for x in t.shape.tolist()[:2])
        col2 = (quantize._f32(gguf, t.tensor_type, t.data).reshape(n_out, n_in) ** 2).sum(axis=0)
        energy[slot[1]][slot[0]] = float(np.asarray(s2.data, dtype=np.float32).reshape(-1)[:n_in] @ col2)
    if all(x != x for v in energy.values() for x in v):
        raise RuntimeError("the imatrix has no feed-forward tensors")
    return energy


def ffn_report(job: Job, path: Path, layers: int) -> dict:
    """Size and the feed-forward bits actually written."""
    gguf = quantize.gguf_lib(job)
    widths, total, n = [], 0.0, 0
    for t in gguf.GGUFReader(str(path)).tensors:
        slot = ffn_slot(t.name)
        if not slot or slot[0] >= layers:
            continue
        block, size = gguf.GGML_QUANT_SIZES[t.tensor_type]
        b = size * 8 / block
        widths.append({"layer": slot[0], "tensor": t.name.split(".")[2], "type": t.tensor_type.name.lower(),
                       "bits": round(b, 3)})
        total, n = total + b * int(t.n_elements), n + int(t.n_elements)
    widths.sort(key=lambda w: (w["layer"], w["tensor"]))
    return {"size_gb": round(path.stat().st_size / 1e9, 2), "ffn_bits": round(total / n, 2) if n else None,
            "bit_widths": widths}


# ---------- subcommands ----------

def make_ggufs(job: Job, hf_dir: Path, bf16: Path, q8: Path | None) -> None:
    """The BF16 GGUF, and the 8-bit copy that writes the calibration answers when the
    BF16 model doesn't fit the GPU (q8 is None when it does)."""
    stage = "gguf"
    lc = Path(job.config.llama_cpp)
    bf16.parent.mkdir(parents=True, exist_ok=True)
    if not bf16.exists():
        emit(stage, pct=5, msg=f"converting {hf_dir.name} to a BF16 GGUF")
        tmp = bf16.with_name(bf16.name + ".part")
        if DRY_RUN:
            tmp.write_bytes(b"GGUF dry run")
        else:
            from stages.moe_utils import gguf_convert_flags  # imports torch

            run([sys.executable, str(lc / "convert_hf_to_gguf.py"), str(hf_dir), "--outtype", "bf16",
                 "--outfile", str(tmp), *gguf_convert_flags(hf_dir)], stage)
        tmp.rename(bf16)
    if q8 and not q8.exists():
        emit(stage, pct=70, msg="the 8-bit copy")
        tmp = q8.with_name("tmp-" + q8.name)
        if DRY_RUN:
            tmp.write_bytes(b"GGUF dry run")
        else:
            run([quantize.bin_path(job, "llama-quantize"), str(bf16), str(tmp), "Q8_0"], stage)
        tmp.rename(q8)
    emit(stage, "done", 100, f"{bf16.name}, {q8.name}" if q8 else bf16.name)


def calib_problems(bench_dir: str) -> list[dict]:
    """Spread over each suite, and interleaved, so the imatrix's first chunks already
    cover every language when the text is longer than IMATRIX_CHUNKS."""
    picks = []
    for suite, n in CALIB_SUITES.items():
        rows = codebench.load_suite(suite, bench_dir)
        picks.append(rows[:: max(1, len(rows) // n)][:n])
    return [p for i in range(max(map(len, picks))) for rows in picks if i < len(rows) for p in [rows[i]]]


def chat_text(prompt: str, reasoning: str, answer: str) -> str:
    """One conversation as the model sees it: the template's prompt (thinking opened),
    the thinking, and the answer."""
    return f"{prompt}{reasoning.strip()}\n</think>\n\n{answer.strip()}{END_OF_TURN}"


def calibration(job: Job, model: Path) -> Path:
    from stages import eval as ev

    stage = "calib"
    out = job.path("work", "calib-chat.txt")
    if out.exists():
        emit(stage, "done", 100, f"{out} exists")
        return out
    cfg = job.config
    probs = calib_problems(cfg.code_eval_dir)
    emit(stage, msg=f"{len(probs)} problems, thinking on, up to {CALIB_MAX_TOKENS} tokens each", answered=0, total=len(probs))
    if DRY_RUN:
        out.write_text("".join(chat_text("<|im_start|>user\n" + codebench.build_prompt(p) + "<|im_end|>\n"
                                         "<|im_start|>assistant\n<think>\n", "Let me think.", "```\npass\n```")
                               for p in probs))
        emit(stage, "done", 100, f"{len(probs)} conversations")
        return out
    done, lock = 0, threading.Lock()

    def answered():
        nonlocal done
        with lock:
            done += 1
            emit(stage, msg=f"{done}/{len(probs)} answers", answered=done, total=len(probs))

    proc = ev.serve(job, str(model), "calib")
    try:
        infos: list[dict] = []
        answers, tps = ev.generate(codebench.SYSTEM, [codebench.build_prompt(p) for p in probs], CALIB_MAX_TOKENS,
                                   cfg.code_eval_thinking_temperature, 0, True, answered, infos=infos,
                                   keep_reasoning=True)
        texts = [chat_text(ev.thinking_prompt([{"role": "system", "content": codebench.SYSTEM},
                                               {"role": "user", "content": codebench.build_prompt(p)}]),
                           info.get("reasoning", ""), a)
                 for p, a, info in zip(probs, answers, infos)]
    finally:
        ev.stop(proc)
    write_jsonl(job.path("work", "calib-answers.jsonl"),
                [{"suite": p["suite"], "id": p["id"], "answer": a, **i} for p, a, i in zip(probs, answers, infos)])
    out.write_text("".join(texts))
    emit(stage, "done", 100, f"{len(texts)} conversations, {sum(map(len, texts)) // 1000}k characters, {tps} tok/s")
    return out


def free_vram_gb() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0]) / 1024


def gpu_layers(tensors: list[Tensor], free_gb: float, reserve_gb: float = 6.0) -> int:
    """How many BF16 layers fit on the GPU; llama.cpp runs the rest from RAM. 999
    when the whole model fits, so the output head is on the GPU too."""
    per: dict[int, float] = {}
    rest = 0.0
    for t in tensors:
        gb = t.n * (2 if t.dims > 1 else 4) / 1e9
        m = re.match(r"blk\.(\d+)\.", t.name)
        if m:
            per[int(m[1])] = per.get(int(m[1]), 0.0) + gb
        else:
            rest += gb
    if not per:
        return 0
    if sum(per.values()) + rest <= free_gb - reserve_gb:
        return 999
    layer = max(per.values())
    return max(0, min(len(per), int((free_gb - reserve_gb) / layer)))


def importance(job: Job, bf16: Path) -> Path:
    stage = "imatrix"
    out = job.path("work", "imatrix.gguf")
    if out.exists():
        emit(stage, "done", 100, f"{out} exists")
        return out
    calib = job.path("work", "calib-chat.txt")
    if DRY_RUN:
        out.write_bytes(b"GGUF dry run")
        emit(stage, "done", 100, "dry run")
        return out
    tensors, _ = read_tensors(job, bf16)
    ngl = gpu_layers(tensors, free_vram_gb())
    where = "the whole model on the GPU" if ngl == 999 else f"{ngl} layers on the GPU, the rest in RAM"
    emit(stage, pct=5, msg=f"{where}; {IMATRIX_CHUNKS} chunks of 512 tokens")
    tmp = out.with_name("tmp-" + out.name)  # imatrix wants a .gguf suffix
    proc = subprocess.Popen([quantize.bin_path(job, "llama-imatrix"), "-m", str(bf16), "-f", str(calib),
                             "-o", str(tmp), "-ngl", str(ngl), "-c", "512", "--chunks", str(IMATRIX_CHUNKS),
                             "--parse-special", "--no-ppl"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(f"[{stage}] {line}")
        m = re.search(r"\[(\d+)\]", line)  # llama-imatrix prints [chunk] as it goes
        if m and "compute_imatrix" not in line:
            emit(stage, pct=5 + 90 * min(1.0, int(m[1]) / IMATRIX_CHUNKS))
    if proc.wait() != 0:
        raise RuntimeError(f"llama-imatrix exited with {proc.returncode}")
    tmp.rename(out)
    emit(stage, "done", 100, "imatrix, all on the GPU" if ngl == 999 else f"imatrix from {ngl} GPU layers")
    return out


def quantize_recipe(job: Job, recipe: str, bf16: Path, imatrix: Path) -> dict:
    out = job.path("out", "model.gguf")
    alloc = job.path("work", "allocation.json")
    if out.exists() and alloc.exists():
        emit(STAGE, "done", 100, f"{out} exists")
        return json.loads(alloc.read_text())
    if DRY_RUN:
        tensors, layers = synthetic_tensors(), 64
    else:
        tensors, layers = read_tensors(job, bf16)
    report: dict = {"recipe": recipe, "layers": None}
    if recipe == "q4":
        args, base = Q4_ARGS, "Q4_K_M"
    else:
        energy = None
        if not DRY_RUN:
            emit(STAGE, pct=5, msg="feed-forward sensitivity from the imatrix")
            energy = imatrix_energy(job, bf16, imatrix, layers)
        plan, est, ffn_bits = allocate_w(tensors, layers, energy)
        args, base = w_args(plan), "Q8_0"
        report.update(layers=[asdict(l) for l in plan], est_gb=round(est, 2), est_ffn_bits=round(ffn_bits, 2),
                      sensitivity=bits.sensitivities(layers, None, energy))
        emit(STAGE, pct=20, msg=f"w18s plan: ~{est:.1f} GB, feed-forward ~{ffn_bits:.2f} bits")
    emit(STAGE, pct=25, msg=f"llama-quantize {recipe}")
    if DRY_RUN:
        out.write_bytes(b"GGUF dry run")
        report.update(size_gb=report.get("est_gb", 16.0), ffn_bits=report.get("est_ffn_bits", 4.5), bit_widths=[])
    else:
        quantize.quantize(job, bf16, imatrix, out, args, base)
        report.update(ffn_report(job, out, layers))
    alloc.write_text(json.dumps(report, indent=2))
    job.mark_done(STAGE, {"size_gb": report["size_gb"], "ffn_bits": report["ffn_bits"]})
    emit(STAGE, "done", 100, f"{recipe}: {report['size_gb']} GB, feed-forward {report['ffn_bits']} bits")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gguf")
    g.add_argument("--job", required=True)
    g.add_argument("--hf", required=True)
    g.add_argument("--bf16", required=True)
    g.add_argument("--q8", help="also make this 8-bit copy, for calibration when BF16 doesn't fit the GPU")
    c = sub.add_parser("calib")
    c.add_argument("--job", required=True)
    c.add_argument("--model", required=True)
    i = sub.add_parser("imatrix")
    i.add_argument("--job", required=True)
    i.add_argument("--bf16", required=True)
    q = sub.add_parser("quantize")
    q.add_argument("--job", required=True)
    q.add_argument("--recipe", choices=RECIPES, required=True)
    q.add_argument("--bf16", required=True)
    q.add_argument("--imatrix", required=True)
    a = ap.parse_args()
    job = Job(a.job)
    if a.cmd == "gguf":
        make_ggufs(job, Path(a.hf), Path(a.bf16), Path(a.q8) if a.q8 else None)
    elif a.cmd == "calib":
        calibration(job, Path(a.model))
    elif a.cmd == "imatrix":
        importance(job, Path(a.bf16))
    else:
        quantize_recipe(job, a.recipe, Path(a.bf16), Path(a.imatrix))


if __name__ == "__main__":
    main()
