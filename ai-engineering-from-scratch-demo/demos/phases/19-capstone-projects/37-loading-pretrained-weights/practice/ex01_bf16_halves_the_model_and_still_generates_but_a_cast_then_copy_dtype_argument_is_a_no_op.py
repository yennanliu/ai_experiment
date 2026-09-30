"""Exercise 1 -- bf16 halves the model and still generates, but a cast-then-copy dtype argument is a no-op.

    Add a `dtype` argument to the loader that casts each tensor to a target dtype (`bfloat16`, `float16`, `float32`) during assignment. Confirm a `float32` model can be downcast to `bfloat16` and still generate.

Reading of the exercise: `load(ref, cfg, path, dtype)` wraps the lesson's own
`load_safetensors`. The loader already assigns with
`target.copy_(tensor.to(dtype=target.dtype))`, so the one place a dtype can take
effect is the parameter itself: the wrapper casts the freshly built model to
`dtype` and then lets the unchanged loader cast every tensor during assignment.
The fixture is the lesson's own stub (`make_stub_safetensors`, seed 42) at the
demo's config (vocab 256, d_model 192, 4 layers), written to a temp directory;
no real weights are downloaded. "Still generate" means the lesson's
`quick_generate` returns 32 in-vocab tokens from finite logits.

**ANSWER: yes, for all three dtypes.** Each load reports `loaded=52 missing=0
unexpected=0 shape_mismatch=0`, every parameter ends in the target dtype, the
LM head stays tied, and bf16 and fp16 hold the model in 3,682,560 bytes against
7,365,120 for fp32 (exactly half). The bf16 model generates 32 in-vocab tokens
from finite logits.

**FINDING: casting the tensor before the copy does nothing.** The obvious
dtype argument, `tensor.to(dtype)` before `copy_`, leaves a float32 parameter
float32: `copy_` casts back to the destination. The parameter keeps the
rounded bf16 values in fp32 storage, so the model loses precision and saves
no memory.

**FINDING: on the stub, bf16 greedy output is not the fp32 output.** The stub
is N(0, 0.02) noise, so next-token logits are nearly flat (entropy 5.51 nats of
a possible 5.55) and the gap between the top two logits falls as low as 0.004.
bf16 rounds each weight by up to 2^-8 relative and moves the logits by about
0.01, which is more than that gap, so greedy tokens can flip (the counts are
printed; they depend on the CPU's bf16 kernels, so only the margin-versus-error
property is asserted). fp16, with 3 more mantissa bits, moves them about 0.001.

Structure: `load()` is the dtype-aware loader; `run()` measures one dtype.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "37-loading-pretrained-weights"
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
PROMPT = [[7, 11, 13, 17]]


def load(ref, cfg, path, dtype=torch.float32):
    """The lesson's loader with a dtype argument: every tensor lands in `dtype`."""
    torch.manual_seed(0)
    model = ref.GPTModel(cfg).to(dtype)
    return model, ref.load_safetensors(model, Path(path), verbose=False)


def run(ref, cfg, path, dtype, seq, fp32_logits):
    model, report = load(ref, cfg, path, dtype)
    tokens = ref.quick_generate(model, torch.tensor(PROMPT), n=32)
    with torch.no_grad():
        logits = model(seq).float()
    return {
        "report": report.summary(),
        "dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "tied": model.lm_head.weight.data_ptr() == model.tok_embed.weight.data_ptr(),
        "generated": len(tokens) - 4 == 32 and all(0 <= t < cfg.vocab_size for t in tokens),
        "finite": bool(torch.isfinite(logits).all()),
        "tokens": tokens[4:14],
        "err": (logits - fp32_logits).abs().max().item(),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cfg = ref.ModelConfig(vocab_size=256, context_length=64, d_model=192, num_heads=6, num_layers=4)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gpt2-stub.safetensors"
        ref.make_stub_safetensors(path, cfg, seed=42)
        fp32, _ = load(ref, cfg, path)
        seq = torch.tensor([ref.quick_generate(fp32, torch.tensor(PROMPT), n=32)])
        with torch.no_grad():
            ref_logits = fp32(seq)
            top2 = ref_logits[0, 3:-1].topk(2, dim=-1).values
            param = fp32.final_ln.scale
            param.copy_(torch.full_like(param, 1 / 3).to(torch.bfloat16))
        runs = {name: run(ref, cfg, path, dt, seq, ref_logits) for name, dt in DTYPES.items()}
    w = torch.randn(100_000, generator=torch.Generator().manual_seed(0))
    return {
        "runs": runs,
        "naive": (str(param.dtype), param[0].item(), 1 / 3),
        "margin": (top2[:, 0] - top2[:, 1]).min().item(),
        "entropy": torch.distributions.Categorical(logits=ref_logits[0, 3]).entropy().item(),
        "bf16_rel": ((w.bfloat16().float() - w) / w).abs().max().item(),
    }


def verify(result):
    r, runs = result, result["runs"]
    bf, f16, f32 = runs["bfloat16"], runs["float16"], runs["float32"]
    clean = "loaded=52 missing=0 unexpected=0 shape_mismatch=0"
    return [
        practice.Check(
            "ANSWER: yes, for all three dtypes",
            [(x["report"], x["tied"], x["generated"], x["finite"]) for x in runs.values()] == [(clean, True, True, True)] * 3
            and [x["dtypes"] for x in (bf, f16, f32)] == [["torch.bfloat16"], ["torch.float16"], ["torch.float32"]]
            and (bf["bytes"], f16["bytes"], f32["bytes"]) == (3_682_560, 3_682_560, 7_365_120),
            f"{clean} for each; bytes bf16 {bf['bytes']:,}, fp16 {f16['bytes']:,}, fp32 {f32['bytes']:,}; "
            f"bf16 tokens {bf['tokens']} vs fp32 {f32['tokens']}",
        ),
        practice.Check(
            "FINDING: casting the tensor before the copy does nothing",
            r["naive"][0] == "torch.float32" and r["naive"][1] != r["naive"][2],
            f"after copy_(t.to(bfloat16)) the parameter is {r['naive'][0]} holding "
            f"{r['naive'][1]!r} for 1/3",
        ),
        practice.Check(
            "FINDING: on the stub, bf16 greedy output is not the fp32 output",
            all([r["bf16_rel"] <= 2**-8, 0.003 < r["margin"] < bf["err"] < 0.05,
                 f16["err"] < bf["err"], r["entropy"] > 0.99 * math.log(256)]),
            f"min top-2 gap {r['margin']:.4f}, bf16 logit error {bf['err']:.4f}, fp16 {f16['err']:.4f}; "
            f"bf16 weight rounding {r['bf16_rel']:.5f} <= 2^-8; entropy {r['entropy']:.2f} of "
            f"{math.log(256):.2f} nats",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
