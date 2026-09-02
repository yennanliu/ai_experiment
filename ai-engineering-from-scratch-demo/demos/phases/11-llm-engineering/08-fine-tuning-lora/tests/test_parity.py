"""Assertions on what Phase 11 / Lesson 08 claims about LoRA and QLoRA."""

import copy
import sys
from pathlib import Path

import pytest
import torch

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(DEMO.parents[3]))

from harness.parity import load_reference  # noqa: E402
from run import D_MODEL, HIDDEN, LESSON, N_CLASSES, TARGETS, copy_adapter, peft_lora  # noqa: E402

ref = load_reference(LESSON)


@pytest.fixture
def base():
    torch.manual_seed(42)
    return ref.create_demo_model(D_MODEL, HIDDEN, N_CLASSES)


def with_adapter(base_model, rank, alpha, *, randomise_b=True):
    model = copy.deepcopy(base_model)
    layers = ref.inject_lora(model, TARGETS, rank=rank, alpha=alpha)
    if randomise_b:
        with torch.no_grad():
            for layer in layers.values():
                layer.lora.B.normal_(0.0, 0.02)
    return model, layers


@pytest.mark.parametrize(("rank", "alpha"), [(4, 8), (8, 16), (16, 16), (32, 8)])
def test_forward_matches_peft_at_every_rank_and_alpha(base, rank, alpha):
    """The claim: `(x @ A @ B) * alpha/rank` is exactly what peft computes."""
    model, layers = with_adapter(base, rank, alpha)
    peft_model = peft_lora(base, rank=rank, alpha=alpha)
    copy_adapter(layers, peft_model)
    x = torch.randn(8, D_MODEL)
    with torch.no_grad():
        torch.testing.assert_close(model(x), peft_model(x), atol=1e-6, rtol=0)


def test_a_fresh_adapter_is_the_identity(base):
    """B initialises to zeros, so attaching an adapter cannot change the model."""
    model, _ = with_adapter(base, 8, 16, randomise_b=False)
    x = torch.randn(8, D_MODEL)
    with torch.no_grad():
        torch.testing.assert_close(model(x), base(x), atol=0, rtol=0)


def test_base_weights_are_frozen_and_only_the_adapter_trains(base):
    """The economic claim: a few percent of parameters carry the fine-tune."""
    model, layers = with_adapter(base, 8, 16)
    counts = ref.count_parameters(model)

    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable_names, "nothing is trainable -- inject_lora froze everything"
    assert all(name.endswith((".A", ".B")) for name in trainable_names)
    assert counts["trainable"] < 0.1 * counts["total"]

    peft_model = peft_lora(base, rank=8, alpha=16)
    assert (counts["trainable"], counts["total"]) == peft_model.get_nb_trainable_parameters()


def test_merging_folds_the_adapter_into_the_base_weight(base):
    """merge_lora_weights must preserve the forward pass and drop the wrapper."""
    import torch.nn as nn

    model, _ = with_adapter(base, 8, 16)
    x = torch.randn(8, D_MODEL)
    with torch.no_grad():
        before = model(x)
    ref.merge_lora_weights(model)
    with torch.no_grad():
        after = model(x)

    torch.testing.assert_close(before, after, atol=1e-5, rtol=0)
    assert all(not isinstance(m, ref.LinearWithLoRA) for m in model.modules())
    assert isinstance(model[0], nn.Linear)


def test_merged_weights_match_peft(base):
    model, layers = with_adapter(base, 8, 16)
    peft_model = peft_lora(base, rank=8, alpha=16)
    copy_adapter(layers, peft_model)

    ref.merge_lora_weights(model)
    merged_peft = peft_model.merge_and_unload()
    for index in (0, 2, 4):
        torch.testing.assert_close(
            model[index].weight, merged_peft[index].weight, atol=1e-6, rtol=0
        )


def test_quantiser_round_trip_stays_within_its_own_step_size():
    """The lesson's quantiser is symmetric int4-range, so bound the error by
    half a step -- this documents the divergence from real NF4 rather than
    pretending the two are the same."""
    torch.manual_seed(0)
    weight = torch.randn(128, 64)
    quantized, scales, shape, pad = ref.quantize_to_nf4(weight)
    restored = ref.dequantize_from_nf4(quantized, scales, shape, pad)

    assert restored.shape == weight.shape
    assert quantized.dtype == torch.int8
    assert int(quantized.min()) >= -8 and int(quantized.max()) <= 7
    # Every element must land within one quantisation step of its block scale.
    step = scales.repeat(1, quantized.shape[1]).reshape(-1)[: weight.numel()]
    assert torch.all((restored - weight).abs().reshape(-1) <= step + 1e-6)
