# llmap

**Large language model augmentation pipeline.** llmap builds the models behind Mugge, a terminal tool where many specialized coding agents work in parallel on a cloud GPU VM. It takes a large open-weight model, compresses it about 10x with expert pruning, healing and mixed-bit quantization, and trains small role and language specialists on top.

llmap grows out of [LobBot](https://github.com/OliverVillson/LobBot), and its history is kept here. LobBot showed the recipe on narrow tasks: a 61 GB Qwen3-30B-A3B became a 6.5 GB model that kept 99% of the teacher's score on email→JSON. llmap points the same recipe at code and at bigger bases.

## Where it is going

| | LobBot (today's code) | llmap (target) |
|---|---|---|
| Base | Qwen3-30B-A3B-Instruct | Qwen3.6-35B-A3B (small tier), DeepSeek V4.1 Flash (B200 tier); see [docs/base-models.md](docs/base-models.md) |
| Job | one narrow task per model | code: implement, test, fix, review, architect, plan, across C, JS, TS, Python and frontend |
| Training data | teacher answers, judged by Gemini | teacher outputs **filtered by execution**: kept only if they compile and pass tests |
| Output | one GGUF for a 16 GB laptop | a compressed base for vLLM plus one LoRA per role and language |
| Eval | judge score vs. teacher | pass@k vs. the unpruned base, and tok/s with many sequences at once |
| Runs on | one B200 on evroc | the same, plus an 8×B200 node for the big base |

## First experiment

[docs/exp-01-code10x.md](docs/exp-01-code10x.md): does ~10x compression keep coding skill? It uses Qwen3.6-35B-A3B on the existing single-B200 evroc setup, before any money goes into the big base.

## The code today

The code is LobBot's and works as described in [docs/lobbot-readme.md](docs/lobbot-readme.md): `pipeline.py` with stages data → reap → heal → quantize → eval → package, an HTTP job API (`agent/server.py`), and the `lobbot` CLI. Renaming to `llmap` comes with the first code change.

```bash
LOBBOT_DRY_RUN=1 python pipeline.py --job /tmp/llmap-job   # every stage without a GPU (copy a taskspec.json in first)
python -m pytest
```
