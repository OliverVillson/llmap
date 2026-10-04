# Experiment 01: does ~10x compression keep coding skill?

**Why:** LobBot's 10x results are on narrow tasks (99% kept on email→JSON, 91% on English→shell). The REAP paper reports near-lossless code generation with 50% of experts pruned on Qwen3-Coder-480B and Kimi-K2, but nobody has measured *our* recipe on code. Mugge's model choice rests on this number, so we measure it first, cheaply.

**Model:** Qwen3.6-35B-A3B. It is Apache 2.0, has 256 experts with 8 routed and 1 shared, and uses hybrid Gated DeltaNet attention. It is the Mugge-Small base and the closest relative of LobBot's current Qwen3 teacher.

**Hardware:** the existing single-B200 evroc VM (`scripts/setup_vm.sh`).

## Variants

| Id | Recipe | Approx. size | Compression |
|---|---|---|---|
| `ref` | BF16, unchanged | ~70 GB | 1x |
| `fp8` | FP8 weights | ~35 GB | 2x |
| `r25q4` | REAP 25% (code calibration) + heal + 4-bit | ~14 GB | ~5x |
| `r50mix` | REAP 50% (code calibration) + heal + mixed ~3-bit | ~7 GB | **~10x** |
| `r50mix-gen` | Same as `r50mix`, but REAP calibrated on general text | ~7 GB | ~10x (ablation) |

The `r50mix-gen` ablation tests how much code-specific calibration matters.

## Data

- **Calibration (REAP):** permissively licensed code in C, JavaScript, TypeScript, Python and HTML/CSS, plus prompts shaped like Mugge's harness. A harness prompt is a ticket and its files going in, and file contents coming out.
- **Heal (LoRA):** the `ref` model answers about 2k harness-shaped coding tasks that come with tests. **Only answers that compile and pass their tests are kept.** This replaces the Gemini judge for code.

## Eval

- **Per-language pass@1:** MultiPL-E for JS, TS, Python and C++, plus a small C set we write with unit tests.
- **LiveCodeBench:** recent problems only, so the questions are newer than the base model.
- **Mugge harness tickets:** the ~15 tickets from the engine spike project (C library, Python API, TS CLI), scored as passing their acceptance commands within 4 attempts.
- **Serving:** throughput in vLLM with 32 and 128 concurrent sequences; tok/s per stream and in total.

Every score is reported as a share of `ref`.

## Pass bar (proposed)

- `r50mix` keeps at least **90%** of `ref` on average across languages, and no language drops below 80%.
- `r25q4` keeps at least **95%**.
- If `r50mix` misses the bar but `r25q4` passes, Mugge-Big targets ~5x instead (about 220 GB for DeepSeek V4.1 Flash). That would mean 2× B200 or a smaller base.

## Code changes needed (from LobBot)

1. **`stages/moe_utils.py`:** support Qwen3.6's MoE block, which has a shared expert next to the routed ones, and load only the text part of the multimodal checkpoint.
2. **`common/taskspec.py`:** a code task type with tests per example and a target language.
3. **`stages/data.py`:** generate answers with vLLM, run each one's tests in a sandbox, and keep only the passing ones.
4. **`stages/eval.py`:** execution-based pass@k next to (not instead of) the judge.
5. **`stages/quantize.py`:** keep the GGUF path (llama.cpp) for this experiment. A vLLM-servable mixed-bit format is a follow-up, because per-expert bit widths are llama.cpp-specific today.
6. **Configs:** a `code10x` job config per variant.

## What it decides

- Whether ~10x is the right target for Mugge-Big on DeepSeek V4.1 Flash, or whether it should be ~5x.
- Whether code-specific REAP calibration is worth building per language.
