"""Exercise 5 -- the LLaMA map lands all 15 mapped tensors, but ok() stays False and a GQA checkpoint is refused.

    Extend `NAME_MAP` to handle the LLaMA naming convention (no biases, RMSNorm, fused qkv layout) and re-run the loader on a stub LLaMA fixture you generate.

Reading of the exercise: the lesson has no `NAME_MAP`; its map is
`make_pretrained_to_local`. `llama_map()` is the extended map for Hugging Face
LLaMA names. It is swapped into the lesson's module for the call (with
`CONV1D_SUFFIXES` emptied, because LLaMA stores `nn.Linear` layout), and the
lesson's own `load_safetensors` runs unchanged. "Fused qkv layout" is the
local side: the lesson's attention has one `qkv` matrix and LLaMA ships
`q_proj`, `k_proj`, `v_proj`, so `fuse_qkv()` concatenates them into one
`self_attn.qkv_proj.weight` as it writes the fixture. `make_llama_stub()`
builds 2 layers at d_model 64 (4 heads, MLP 256, vocab 128, no biases, one
RMSNorm weight per norm, untied `lm_head`), with 4 KV heads (plain
multi-head) and again with 1 (grouped-query); the fixture file goes to a temp
directory. The model is the
lesson's replica with `use_bias=False, weight_tying=False`. No real weights
are downloaded.

**ANSWER: with the extended map, every LLaMA tensor that has a slot lands.**
On the multi-head stub the report is `loaded=15 missing=6 unexpected=2
shape_mismatch=0`. The fused `qkv.weight` is exactly `cat(q, k, v)`, and the
loaded model forwards finite logits. The misses are what the lesson's
architecture lacks, not naming: `pos_embed` (LLaMA uses RoPE) and the 5
LayerNorm `shift`s (RMSNorm has none). The unexpected pair is the SwiGLU
`gate_proj` of each layer, since the lesson's MLP has two matrices, not three.

**FINDING: a loaded RMSNorm weight does not make the model compute RMSNorm.**
The lesson's `LayerNorm` still subtracts the mean. With the loaded scale and a
zero shift it differs from RMSNorm by up to 1.06 on a seeded (1, 8, 64) input
with mean offset 0.5. So `ok()` is False here for a real reason.

**FINDING: "fused qkv" is not the LLaMA layout, and GQA cannot be fused into
this model.** The published TinyLlama header (read 2026-09-29 from
https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors)
has 201 BF16 tensors (3 + 9 x 22 layers), with separate `q_proj` [2048, 2048]
and `k_proj`/`v_proj` [256, 2048] (4 KV heads), no biases and an untied
`lm_head`. On the 1-KV-head stub the fused matrix is (96, 64) against the
lesson's (192, 64): the lesson's shape check refuses both layers
(`shape_mismatch=2`) and assigns nothing.

**FINDING: the lesson's description of its map is not its code.** It says
"The lesson ships it as a dict that the loader iterates". `NAME_MAP` does not
exist, and `load_safetensors` iterates the file's keys, looking each one up in
the map.

Structure: `make_llama_stub()`, `fuse_qkv()`, `llama_map()`, and
`load_llama()`, which swaps the map in and restores it.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

from harness import parity, practice

try:
    import torch
    from safetensors.torch import save_file
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch and safetensors: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "37-loading-pretrained-weights"
D, HEADS, HIDDEN, VOCAB, LAYERS = 64, 4, 256, 128, 2
PROJ = {"q": D, "o": D, "gate": HIDDEN, "up": HIDDEN, "down": D}
LAYER_MAP = {"input_layernorm.weight": "ln1.scale", "post_attention_layernorm.weight": "ln2.scale",
             "self_attn.qkv_proj.weight": "attn.qkv.weight", "self_attn.o_proj.weight": "attn.out_proj.weight",
             "mlp.up_proj.weight": "mlp.fc1.weight", "mlp.down_proj.weight": "mlp.fc2.weight"}


def make_llama_stub(kv_heads, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = {"model.embed_tokens.weight": torch.randn(VOCAB, D, generator=g) * 0.02,
         "model.norm.weight": torch.ones(D), "lm_head.weight": torch.randn(VOCAB, D, generator=g) * 0.02}
    for p in (f"model.layers.{n}" for n in range(LAYERS)):
        t |= {f"{p}.{nm}.weight": 1 + 0.1 * torch.randn(D, generator=g) for nm in ("input_layernorm", "post_attention_layernorm")}
        for proj, rows in {**PROJ, "k": D // HEADS * kv_heads, "v": D // HEADS * kv_heads}.items():
            part = "mlp" if proj in ("gate", "up", "down") else "self_attn"
            t[f"{p}.{part}.{proj}_proj.weight"] = torch.randn(rows, HIDDEN if proj == "down" else D, generator=g) * 0.02
    return t


def fuse_qkv(tensors, dst):
    t = dict(tensors)
    for p in (f"model.layers.{n}.self_attn" for n in range(LAYERS)):
        t[f"{p}.qkv_proj.weight"] = torch.cat([t.pop(f"{p}.{x}_proj.weight") for x in "qkv"])
    save_file(t, str(dst))


def llama_map(num_layers):
    m = {"model.embed_tokens.weight": "tok_embed.weight", "model.norm.weight": "final_ln.scale",
         "lm_head.weight": "lm_head.weight"}
    return m | {f"model.layers.{n}.{k}": f"blocks.{n}.{v}" for n in range(num_layers) for k, v in LAYER_MAP.items()}


def load_llama(ref, model, path):
    saved = ref.make_pretrained_to_local, ref.CONV1D_SUFFIXES
    ref.make_pretrained_to_local, ref.CONV1D_SUFFIXES = llama_map, ()
    try:
        return ref.load_safetensors(model, Path(path), verbose=False)
    finally:
        ref.make_pretrained_to_local, ref.CONV1D_SUFFIXES = saved


def case(ref, cfg, tmp, kv):
    t, fused = make_llama_stub(kv), Path(tmp) / f"llama{kv}.safetensors"
    fuse_qkv(t, fused)
    torch.manual_seed(0)
    model = ref.GPTModel(cfg)
    report = load_llama(ref, model, fused)
    qkv = torch.cat([t[f"model.layers.0.self_attn.{x}_proj.weight"] for x in "qkv"])
    return {"report": report.summary(), "ok": report.ok(), "missing": report.missing,
            "unexpected": report.unexpected, "fused": torch.equal(model.blocks[0].attn.qkv.weight, qkv),
            "finite": bool(torch.isfinite(model(torch.tensor([[1, 2, 3]]))).all()),
            "bad": report.shape_mismatch[:1], "ln1": model.blocks[0].ln1}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cfg = ref.ModelConfig(VOCAB, 32, D, HEADS, LAYERS, HIDDEN // D, 0.0, use_bias=False, weight_tying=False)
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        out = {kv: case(ref, cfg, tmp, kv) for kv in (HEADS, 1)}
        x = torch.randn(1, 8, D, generator=torch.Generator().manual_seed(0)) + 0.5
        ln, _ = out[HEADS].pop("ln1"), out[1].pop("ln1")
        rms = x / x.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt() * ln.scale
        out["norm_gap"] = (ln(x) - rms).abs().max().item()
    out["has_name_map"] = hasattr(ref, "NAME_MAP")
    out["iterates_file"] = "for src_name in pretrained_names" in inspect.getsource(ref.load_safetensors)
    out["doc"] = "ships it as a dict that the loader iterates" in parity.doc_text(PHASE, LESSON)
    return out


def verify(result):
    r, mha, gqa = result, result[HEADS], result[1]
    shifts = ["blocks.0.ln1.shift", "blocks.0.ln2.shift", "blocks.1.ln1.shift", "blocks.1.ln2.shift", "final_ln.shift"]
    return [
        practice.Check(
            "ANSWER: with the extended map, every LLaMA tensor that has a slot lands",
            all([mha["report"] == "loaded=15 missing=6 unexpected=2 shape_mismatch=0", mha["fused"], mha["finite"],
                 sorted(mha["missing"]) == sorted(shifts + ["pos_embed.weight"]),
                 [u.split(".")[-2] for u in mha["unexpected"]] == ["gate_proj"] * 2]),
            f"{mha['report']}; qkv == cat(q,k,v): {mha['fused']}; missing {mha['missing']}; unexpected {mha['unexpected']}",
        ),
        practice.Check(
            "FINDING: a loaded RMSNorm weight does not make the model compute RMSNorm",
            [mha["ok"], 0.5 < r["norm_gap"] < 2.0] == [False, True],
            f"lesson LayerNorm vs RMSNorm with the same loaded scale: max gap {r['norm_gap']:.2f}",
        ),
        practice.Check(
            "FINDING: 'fused qkv' is not the LLaMA layout, and GQA cannot be fused into this model",
            (gqa["report"], gqa["bad"]) == ("loaded=0 missing=21 unexpected=2 shape_mismatch=2",
                                            [("model.layers.0.self_attn.qkv_proj.weight", (96, 64), (192, 64))]),
            f"1 KV head: {gqa['report']}; first mismatch {gqa['bad']}",
        ),
        practice.Check(
            "FINDING: the lesson's description of its map is not its code",
            [r["doc"], r["has_name_map"], r["iterates_file"]] == [True, False, True],
            f"doc says the loader iterates the dict: {r['doc']}; NAME_MAP exists: {r['has_name_map']}; "
            f"loader iterates the file's keys: {r['iterates_file']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
