"""Gemma 4 26B-A4B as a base, on a tiny random lookalike (CPU): text-only loading,
routing, pruning, the reap stage end to end, and one heal step."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.gemma4")

from stages import moe_utils as mu  # noqa: E402
from stages.sft import load_causal_lm  # noqa: E402

E, TOP_K, KEEP = 8, 2, 5
SPECIAL = ["<pad>", "<eos>", "<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|think|>"]
# The parts of Gemma 4's chat template the pipeline relies on (as in vLLM's
# examples/tool_chat_template_gemma4.jinja): <|think|> opens the system turn when
# thinking, the generation prompt closes an empty thought channel when not, and a
# model turn's reasoning_content is rendered as its thought channel.
TEMPLATE = (
    "{%- set enable_thinking = enable_thinking | default(false) -%}{{- bos_token -}}"
    "{%- set msgs = messages -%}"
    "{%- if enable_thinking or messages[0]['role'] == 'system' -%}{{- '<|turn>system\\n' -}}"
    "{%- if enable_thinking -%}{{- '<|think|>\\n' -}}{%- endif -%}"
    "{%- if messages[0]['role'] == 'system' -%}{{- messages[0]['content'] | trim -}}{%- set msgs = messages[1:] -%}"
    "{%- endif -%}{{- '<turn|>\\n' -}}{%- endif -%}"
    "{%- for m in msgs -%}{%- set role = 'model' if m['role'] == 'assistant' else m['role'] -%}"
    "{{- '<|turn>' + role + '\\n' -}}"
    "{%- if role == 'model' and not enable_thinking and not m.get('reasoning_content') -%}"
    "{{- '<|channel>thought\\n<channel|>' -}}{%- endif -%}"
    "{%- if m.get('reasoning_content') -%}{{- '<|channel>thought\\n' + m['reasoning_content'] + '\\n<channel|>' -}}"
    "{%- endif -%}{{- m['content'] | trim -}}{{- '<turn|>\\n' -}}{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|turn>model\\n' -}}"
    "{%- if not enable_thinking -%}{{- '<|channel>thought\\n<channel|>' -}}{%- endif -%}{%- endif -%}")


def build_tokenizer(out):
    """Byte-level tokenizer with Gemma 4's turn and thought-channel special tokens."""
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {c: i for i, c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in SPECIAL])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>")
    tok.chat_template = TEMPLATE
    tok.save_pretrained(out)
    return tok


def build_gemma4_moe(out, vocab_size: int):
    """Gemma 4 26B-A4B's structure at toy size, saved the way it ships (as
    Gemma4ForConditionalGeneration with a vision tower): five sliding-window layers
    and a full-attention one (wider heads, one KV head, K = V), each with a dense MLP
    beside 8 fused experts, top-2. The router's input and per-expert scales are
    randomised so routing that ignores them shows."""
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    text = dict(vocab_size=vocab_size, hidden_size=64, intermediate_size=48, num_hidden_layers=6,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
                num_global_key_value_heads=1, attention_k_eq_v=True, hidden_size_per_layer_input=0,
                sliding_window=8, enable_moe_block=True, num_experts=E, top_k_experts=TOP_K,
                moe_intermediate_size=24, final_logit_softcapping=30.0, max_position_embeddings=4096,
                tie_word_embeddings=True)
    vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                  num_key_value_heads=2, head_dim=16)
    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration(Gemma4Config(text_config=text, vision_config=vision, audio_config=None))
    with torch.no_grad():
        for layer in model.model.language_model.layers:
            layer.router.scale.uniform_(0.5, 2.0)
            layer.router.per_expert_scale.uniform_(0.5, 2.0)
    model.save_pretrained(out)
    return model


@pytest.fixture(scope="module")
def teacher(tmp_path_factory):
    d = tmp_path_factory.mktemp("gemma4-moe")
    tok = build_tokenizer(d)
    build_gemma4_moe(d, len(tok))
    return d


def ids(n=2, length=24):
    return torch.randint(0, 256, (n, length), generator=torch.Generator().manual_seed(0))


def masked(model, kept: dict[int, list[int]]):
    """The model with every pruned expert's router logit at -inf: what pruning must equal."""
    for li, blk in mu.find_moe_blocks(model):
        drop = torch.ones(mu.num_experts(blk), dtype=torch.bool)
        drop[kept[li]] = False

        def hook(_mod, _args, out, drop=drop):
            out = out.clone()
            out[..., drop] = float("-inf")
            return out
        blk.router.proj.register_forward_hook(hook)
    return model


def test_loads_the_text_decoder_only(teacher):
    from transformers import Gemma4ForConditionalGeneration

    full = Gemma4ForConditionalGeneration.from_pretrained(teacher, dtype=torch.float32).eval()
    text = load_causal_lm(str(teacher)).eval()
    assert type(text).__name__ == "Gemma4ForCausalLM"
    assert not any("vision" in n for n, _ in text.named_parameters())
    with torch.no_grad():
        assert torch.equal(text(ids()).logits, full(ids()).logits)
    blocks = mu.find_moe_blocks(text)
    assert [li for li, _ in blocks] == list(range(6))
    assert (mu.num_experts(blocks[0][1]), mu.top_k(blocks[0][1], text.config)) == (E, TOP_K)


def test_route_and_experts_rebuild_the_layer(teacher):
    """route() gives the weights the layer applies (renormalised top-k times
    per_expert_scale), and the experts' weighted outputs plus the dense MLP, each
    post-normed, are the layer's forward."""
    model = load_causal_lm(str(teacher)).eval()
    cfg = model.config
    for li in (0, 5):  # a sliding-window layer and the full-attention one
        layer = dict(mu.find_moe_blocks(model))[li]
        seen = {}
        hooks = [layer.router.register_forward_hook(lambda m, a, o: seen.update(residual=a[0])),
                 mu.moe_module(layer).register_forward_hook(lambda m, a, o: seen.update(args=a, experts=o)),
                 layer.register_forward_hook(lambda m, a, o: seen.update(out=o))]
        with torch.no_grad():
            model(ids())
            residual = seen["residual"]
            w, idx = mu.route(layer, cfg, residual)
            x, w2, idx2 = mu.routed(layer, cfg, seen["args"])
            assert torch.equal(w, w2) and torch.equal(idx, idx2)
            assert torch.allclose(x, layer.pre_feedforward_layernorm_2(residual))
            assert torch.allclose((w / layer.router.per_expert_scale[idx]).sum(-1), torch.ones(len(w)))
            moe = torch.zeros_like(x)
            for slot in range(TOP_K):
                for j in idx[:, slot].unique().tolist():
                    rows = idx[:, slot] == j
                    moe[rows] += w[rows, slot, None] * mu.expert_forward(layer, cfg, j, x[rows])
            assert torch.allclose(moe, seen["experts"], atol=1e-6)
            r = residual.reshape(seen["out"].shape)
            dense = layer.post_feedforward_layernorm_1(layer.mlp(layer.pre_feedforward_layernorm(r)))
            sparse = layer.post_feedforward_layernorm_2(moe.reshape(r.shape))
            out = (r + layer.post_feedforward_layernorm(dense + sparse)) * layer.layer_scalar
            assert torch.allclose(out, seen["out"], atol=1e-5)
        for h in hooks:
            h.remove()


def test_prune_keeps_the_kept_experts_and_reloads(teacher, tmp_path):
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    model = load_causal_lm(str(teacher)).eval()
    keep = [0, 2, 3, 5, 7]
    x = torch.randn(5, 64)
    blocks = mu.find_moe_blocks(model)
    blk = blocks[0][1]
    with torch.no_grad():
        before = {j: mu.expert_forward(blk, model.config, j, x) for j in keep}
        scale = blk.router.per_expert_scale.clone()
        for _, b in blocks:
            mu.prune_block(b, keep)
        mu.set_config_experts(model.config, len(keep))
        for new, j in enumerate(keep):
            assert torch.equal(mu.expert_forward(blk, model.config, new, x), before[j])
        assert torch.equal(blk.router.per_expert_scale, scale[keep]) and blk.router.scale.shape == (64,)
        assert mu.num_experts(blk) == blk.experts.num_experts == model.config.num_experts == len(keep)
        reference = masked(load_causal_lm(str(teacher)).eval(), {li: keep for li in range(6)})
        assert torch.allclose(model(ids()).logits, reference(ids()).logits, atol=1e-5)

    model.save_pretrained(tmp_path)
    mu.fix_saved_config(tmp_path)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert (cfg["architectures"], cfg["model_type"], cfg["num_experts"]) == (["Gemma4ForCausalLM"], "gemma4_text", 5)
    with safe_open(tmp_path / "model.safetensors", "pt") as f:
        names = list(f.keys())
    assert all(n.startswith("model.") and "language_model" not in n and "vision" not in n for n in names)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).eval()
    with torch.no_grad():
        assert torch.equal(reloaded(ids()).logits, model(ids()).logits)


@pytest.fixture(scope="module")
def reaped(teacher, tmp_path_factory):
    """The real reap stage on a tiny Gemma 4 thinking job: 8 -> 5 experts per layer."""
    from stages import reap
    from stages._util import Job

    root = tmp_path_factory.mktemp("job")
    (root / "config.json").write_text(json.dumps({
        "teacher": str(teacher), "reap_sparsity": 0.375, "reap_calib_samples": 12, "reap_max_seq": 128}))
    job = Job(root)
    rows = [{"messages": [{"role": "system", "content": "Write Python."},
                          {"role": "user", "content": f"Add {i} and {i + 1}."},
                          {"role": "assistant", "content": f"print({i} + {i + 1})",
                           "reasoning_content": f"{i} plus {i + 1} is {2 * i + 1}."}]} for i in range(12)]
    job.path("data", "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    reap.run_stage(job)
    return job


def test_reap_stage_prunes_gemma4(reaped, teacher):
    from transformers import AutoModelForCausalLM

    out = reaped.path("work", "reaped")
    cfg = json.loads((out / "config.json").read_text())
    assert (cfg["architectures"], cfg["model_type"], cfg["num_experts"]) == (["Gemma4ForCausalLM"], "gemma4_text", 5)
    sal = json.loads(reaped.path("work", "reap_saliency.json").read_text())
    assert (sal["num_experts_orig"], sal["num_experts_kept"], len(sal["layers"])) == (E, 5, 6)
    assert all(len(L["kept"]) == 5 and sum(L["freq"]) > 0 for L in sal["layers"])
    imp = json.loads(reaped.path("work", "layer_importance.json").read_text())
    assert len(imp) == 6 and all(v > 0 for v in imp)
    pruned = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
    reference = masked(load_causal_lm(str(teacher)).eval(), {L["layer"]: L["kept"] for L in sal["layers"]})
    with torch.no_grad():
        assert torch.allclose(pruned(ids()).logits, reference(ids()).logits, atol=1e-5)


def test_heal_step_trains_router_experts_and_attention(reaped, tmp_path):
    """One LoRA step on thinking rows moves the router (projection and per-expert
    scale), the fused experts, attention and the dense MLP, and the merged model
    saves under the reaped model's tensor names."""
    pytest.importorskip("peft")
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from stages.sft import ATTN, LINEAR_ATTN, train_sft
    from stages.taskdata import load_examples, tokenize_example

    src = reaped.path("work", "reaped")
    examples = load_examples(reaped.path("data", "train.jsonl"))
    tok = AutoTokenizer.from_pretrained(src)
    row = tokenize_example(tok, examples[0]["messages"], 2048)
    trained = [t for t, lab in zip(row["input_ids"], row["labels"]) if lab != -100]
    assert trained[0] == tok.convert_tokens_to_ids("<|channel>")  # it learns to open its thought channel
    assert tok.convert_tokens_to_ids("<channel|>") in trained and trained[-1] == tok.convert_tokens_to_ids("<turn|>")

    out = tmp_path / "healed"
    train_sft(str(src), examples, out, epochs=1.0, lr=1e-3, targets=ATTN + LINEAR_ATTN, max_tokens=1 << 16)
    a, b = load_file(src / "model.safetensors"), load_file(out / "model.safetensors")
    assert set(a) == set(b)
    changed = {k for k in a if not torch.equal(a[k], b[k])}
    for part in ("router.proj.weight", "router.per_expert_scale", "experts.gate_up_proj", "experts.down_proj",
                 "self_attn.q_proj", "mlp.gate_proj"):
        assert any(part in k for k in changed), part
    healed = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    assert type(healed).__name__ == "Gemma4ForCausalLM" and healed.config.num_experts == 5
