"""Phase 11 / Lesson 08 -- the lesson's LoRALayer vs peft.

The lesson writes LoRA from scratch: `(x @ A @ B) * (alpha / rank)`, base weights
frozen, B initialised to zeros. Every production fine-tune uses `peft` instead.
This demo shows they are the same object, and then shows the one place the
lesson deliberately simplifies.

Weight layout: the lesson keeps `A: [in, rank]` and `B: [rank, out]` and
right-multiplies. peft stores them as `nn.Linear` layers -- `lora_A.weight` is
`[rank, in]`, `lora_B.weight` is `[out, rank]` -- so each goes in transposed.

Run:  uv run demo run phases/11-llm-engineering/08-fine-tuning-lora
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from harness.explain import explain          # noqa: E402
from harness.parity import assert_close, load_reference, report  # noqa: E402

LESSON = "phases/11-llm-engineering/08-fine-tuning-lora/code/lora.py"
SEED = 42
RANK, ALPHA = 8, 16
D_MODEL, HIDDEN, N_CLASSES = 256, 512, 10
TARGETS = ["0", "2", "4"]        # the nn.Linear layers of the lesson's demo model


def peft_lora(base_model, *, rank=RANK, alpha=ALPHA):
    """A peft LoRA model wrapping a copy of `base_model` at this rank/alpha."""
    import copy

    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        copy.deepcopy(base_model),
        LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                   target_modules=TARGETS),
    )


def copy_adapter(lesson_layers, peft_model):
    """Push the lesson's A/B into peft, transposed to peft's Linear layout."""
    import torch

    with torch.no_grad():
        for name, lesson_layer in lesson_layers.items():
            target = peft_model.base_model.model[int(name)]
            target.lora_A["default"].weight.copy_(lesson_layer.lora.A.T)
            target.lora_B["default"].weight.copy_(lesson_layer.lora.B.T)


def main() -> int:
    import copy

    import torch

    ref = load_reference(LESSON)
    torch.manual_seed(SEED)

    base = ref.create_demo_model(D_MODEL, HIDDEN, N_CLASSES)
    pristine = copy.deepcopy(base)
    x = torch.randn(16, D_MODEL)

    lesson_model = copy.deepcopy(base)
    lesson_layers = ref.inject_lora(lesson_model, TARGETS, rank=RANK, alpha=ALPHA)

    checks = []

    # --- a freshly injected adapter is a no-op ----------------------------
    # B starts at zeros in both implementations, so training begins from the
    # base model exactly. This is the claim that makes LoRA safe to attach.
    with torch.no_grad():
        checks.append(
            assert_close(lesson_model(x), pristine(x),
                         label="fresh adapter changes nothing", atol=0.0)
        )

    # Give B real values so the rest of the comparison is not comparing zeros.
    with torch.no_grad():
        for layer in lesson_layers.values():
            layer.lora.B.normal_(0.0, 0.02)

    peft_model = peft_lora(pristine)
    copy_adapter(lesson_layers, peft_model)

    # --- the forward pass -------------------------------------------------
    with torch.no_grad():
        checks.append(
            assert_close(lesson_model(x), peft_model(x),
                         label=f"forward, r={RANK} alpha={ALPHA}", atol=1e-6)
        )

    # The alpha/rank scaling is the parameter people get wrong; sweep it.
    for rank, alpha in ((4, 8), (16, 16), (32, 8)):
        trial = copy.deepcopy(pristine)
        layers = ref.inject_lora(trial, TARGETS, rank=rank, alpha=alpha)
        with torch.no_grad():
            for layer in layers.values():
                layer.lora.B.normal_(0.0, 0.02)
        peft_trial = peft_lora(pristine, rank=rank, alpha=alpha)
        copy_adapter(layers, peft_trial)
        with torch.no_grad():
            checks.append(
                assert_close(trial(x), peft_trial(x),
                             label=f"forward, r={rank} alpha={alpha}", atol=1e-6)
            )

    # --- parameter accounting ---------------------------------------------
    counts = ref.count_parameters(lesson_model)
    peft_trainable, peft_total = peft_model.get_nb_trainable_parameters()
    checks.append(
        assert_close(counts["trainable"], peft_trainable,
                     label="trainable parameter count", atol=0.0)
    )
    checks.append(
        assert_close(counts["total"], peft_total,
                     label="total parameter count", atol=0.0)
    )

    # --- merging the adapter back in --------------------------------------
    merged_lesson = copy.deepcopy(lesson_model)
    ref.merge_lora_weights(merged_lesson)
    merged_peft = peft_model.merge_and_unload()
    with torch.no_grad():
        checks.append(
            assert_close(merged_lesson[0].weight, merged_peft[0].weight,
                         label="merged weights, layer 0", atol=1e-6)
        )
        checks.append(
            assert_close(merged_lesson(x), lesson_model(x),
                         label="merge preserves the forward pass", atol=1e-5)
        )

    report(checks, title="phase 11 / lesson 08: hand-written LoRA vs peft")

    print(f"\n{counts['trainable']:,} trainable of {counts['total']:,} parameters "
          f"({counts['trainable_pct']:.2f}%) -- the whole point of LoRA.")

    # --- where the lesson diverges, stated plainly ------------------------
    # `quantize_to_nf4` in the lesson is block-wise *symmetric* quantisation to
    # the int4 range. Real NF4 uses a 16-value non-uniform codebook fitted to a
    # normal distribution. Same structure, different codebook -- so this is
    # measured and reported, not asserted.
    weight = merged_lesson[0].weight.data
    quantized, scales, shape, pad = ref.quantize_to_nf4(weight)
    restored = ref.dequantize_from_nf4(quantized, scales, shape, pad)
    error = (restored - weight)
    # Relative error is reported against the RMS of the weights, not per element:
    # a weight that happens to sit near zero quantises to zero and would report
    # 100% on its own while contributing nothing to the layer.
    relative = (error.pow(2).mean().sqrt() / weight.pow(2).mean().sqrt()).item()
    print("\nnot parity, a divergence: the lesson's `quantize_to_nf4` is symmetric")
    print("int4-range quantisation, not the NF4 codebook bitsandbytes implements.")
    print(f"  round-trip error on a {tuple(shape)} weight: "
          f"max {error.abs().max().item():.3e}, {relative:.2%} of weight RMS")
    print(f"  memory: {weight.numel() * 4:,} bytes fp32 -> "
          f"{quantized.numel() + scales.numel() * 4:,} bytes int8 + fp32 scales")
    return 0


if __name__ == "__main__":
    if explain(__file__):
        raise SystemExit(0)
    raise SystemExit(main())
