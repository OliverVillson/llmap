# Mugge: which big model to compress (research, 2026-10-04)

**Question (Oliver):** the agents run on cloud VMs, not laptops, so start from a much bigger model and use LobBot's ~10x compression on it. The model library can be huge and live on an evroc disk. Which model?

**Short answer:** **DeepSeek V4.1 Flash** for the B200 tier, compressed about 10x to roughly 110 to 130 GB, plus **Qwen3.6-35B-A3B** for the L40S and DGX Spark tier and the engine spike. **GLM-5.2** is the fallback if V4.1 Flash's new architecture blocks LobBot's tooling.

Numbers below come from model cards and vendor reports (sources at the end). Benchmark scores are vendor-reported unless noted; the size math is mine.

---

## 1. What "10x" means, and what LobBot has actually shown

LobBot's 61 GB → 6.5 GB came from two steps:
- REAP removed 50% of experts, about 2x.
- Mixed 2 to 6-bit quantization averaging about 3 bits, about 5x from BF16.

That recipe kept 99% of the teacher's score on email→JSON and 91% on English→shell. Both are narrow, single-answer tasks.

Code is broader, so expect lower retention, but there is outside evidence that it holds up:
- The REAP paper reports near-lossless code generation and tool calling on Qwen3-Coder-480B and Kimi-K2 at 50% of experts pruned.
- It also flags domain-specific calibration as important. LobBot already does this, since it calibrates REAP on task data.

**So the recipe carries over. We should still measure code retention ourselves before betting the flagship on it** (section 5).

## 2. What matters for Mugge specifically

1. **Active parameters.** With dozens of agents decoding at once, throughput is limited by compute per token. Fewer active parameters means more agents per GPU.
2. **KV cache per token.** Every agent holds its own context. A small KV cache means more agents fit at the same time.
3. **Many experts per layer.** More experts give REAP finer choices about what to drop for code.
4. **License.** We will modify and redistribute derived weights inside a paid product, so MIT or Apache is the target.
5. **Tooling risk.** LobBot's REAP, heal and quantize stages are written for a standard Qwen3 MoE (`stages/moe_utils.py`). Every new architecture costs porting work.

## 3. Candidates

| Model | Total / active | Experts | KV cache | License | Coding (vendor) | ~10x size | Fits |
|---|---|---|---|---|---|---|---|
| **DeepSeek V4.1 Flash** (Sep 2026) | 552B backbone / 8B prefill, 16B decode | 384 routed + 1 shared, 6 active | **890 bytes/token** (FP4) | MIT | Terminal-Bench 2.1 90.6, DeepSWE 74.2% | **~110–130 GB** + Engram tables | 1× B200 |
| GLM-5.2 (Jun 2026) | 744B / 40B | 256 | standard | MIT | SWE-bench Pro 62.1%, ~78.7% Verified (Epoch) | ~150 GB | 1× B200, tight |
| DeepSeek V4-Pro (Apr 2026) | 1.6T / 49B | n/a | 10% of V3.2 | MIT | SWE-bench Verified 80.6% | ~320 GB | 2× B200 |
| MiniMax M3 (Jun 2026) | ~428B / ~23B | 128, 4 active | sparse attention | MiniMax Community (terms unclear) | "frontier agentic" (no card number) | ~86 GB | 1× B200 |
| Kimi K3 (Jul 2026) | 2.8T / ~50B | n/a | n/a | MIT-style with a $20M revenue clause | conflicting reports (76.8% to 93.4%) | ~560 GB | multi-GPU |
| **Qwen3.6-35B-A3B** (Apr 2026) | 35B / 3B | 256, 8 + 1 shared | hybrid Gated DeltaNet, small | Apache 2.0 | SWE-bench Verified 73.4%, Terminal-Bench 2.0 51.5% | **~7–10 GB** | L40S, Spark, anything |
| Qwen3-Coder-Next | 80B / ~3B | n/a | hybrid | Apache 2.0 | SWE-bench Verified 70.6% | ~16 GB | L40S |

## 4. Recommendation

### Flagship tier (B200 VMs): DeepSeek V4.1 Flash → "Mugge-Big"

Why it fits Mugge better than anything else:
- **Only 8 to 16B active parameters**, so it costs roughly a third as much compute per token as GLM-5.2 (40B) or V4-Pro (49B). That means about three times as many agents for the same GPU.
- **A 890-byte-per-token KV cache.** After about 120 GB of weights on a 192 GB B200, ~60 GB is left, which holds tens of millions of tokens of context. Memory stops being the limit; compute is.
- **384 experts per layer** give REAP a lot of room to keep the code experts and drop the rest.
- **MIT license.** It's open weights, and the community has already REAP-pruned it to 256 experts and served it in vLLM.

The recipe:
1. Run REAP with about 50% of experts pruned (384 → ~192), calibrated on code in all five Mugge languages.
2. Heal with a LoRA on execution-filtered data.
3. Quantize to a mixed ~3 to 3.5 bits, with FP8 for attention and shared parts.
4. Result: about 110 to 130 GB, served by vLLM on one B200, with role and language LoRAs on top.

Risks to check first:
- The architecture is new: a causal encoder-decoder, CSA2 attention, and Engram memory tables (~189 GiB on disk). LobBot's stages will need porting.
- I couldn't confirm whether the Engram tables can live in host RAM instead of GPU memory.
- The community REAP card ran no coding benchmarks.

### Small tier (L40S, DGX Spark, and the engine spike): Qwen3.6-35B-A3B → "Mugge-Small"

- It's the closest relative of LobBot's current Qwen3 teacher, so the least porting.
- Apache 2.0, and 73.4% SWE-bench Verified.
- The ~10x version is under 10 GB, which leaves an L40S almost all of its memory for parallel agents.
- **Replaces Qwen3-Coder-30B in the engine spike.**

### Fallback flagship: GLM-5.2

- A standard MoE with 256 experts, and the community has already REAP-pruned it.
- About 150 GB at 10x.
- The cost is 40B active parameters, so fewer agents per GPU.

## 5. Proposed path

1. **Prove 10x on code, cheaply.** Run the LobBot recipe on Qwen3.6-35B-A3B on the existing single-B200 setup, calibrated on code. Measure retention on LiveCodeBench plus the engine spike's tickets. This turns "10x keeps the skill" from a narrow-task result into a code result.
2. **Port LobBot to V4.1 Flash.** REAP, heal and quantize all need an 8×B200 node: the unpruned model is ~475 GiB plus Engram tables. This is a one-time factory cost per base, not a cost on every Mugge VM.
3. **Build the model library on the evroc disk.** Unpruned bases for retraining (~0.7 TB for V4.1 Flash, ~70 GB for Qwen3.6), compressed tiers (~120 GB and ~10 GB), and LoRAs per role and language (hundreds of MB each). That comes to roughly 1 to 1.5 TB in total. A project VM loads only its tier plus the LoRAs it needs.
4. **Later option: per-language variants.** A C-only REAP calibration keeps different experts than a TypeScript one, so a variant for a single language family could be pruned harder (around 75%). This only pays off if one project's languages fit in one variant.

## Sources

- [DeepSeek-V4.1-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), [DataCamp on V4.1 Flash](https://www.datacamp.com/blog/deepseek-v4-1-flash), [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash), [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [Community REAP of V4.1 Flash (384 → 256 experts, vLLM)](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E), [GLM-5.2 REAP 504B](https://huggingface.co/0xSero/GLM-5.2-REAP-504B-GGUF)
- [REAP paper](https://arxiv.org/abs/2510.13999), [REAP code and checkpoints](https://github.com/CerebrasResearch/reap)
- [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [MiniMax-M3](https://huggingface.co/unsloth/MiniMax-M3)
- Overviews: [Thunder Compute, Oct 2026](https://www.thundercompute.com/blog/best-open-source-llms), [Morph, best open-source coding model 2026](https://www.morphllm.com/best-open-source-coding-model-2026), [Digital Applied, self-host hardware match](https://www.digitalapplied.com/blog/best-open-weight-coding-models-self-host-hardware-match-2026)
