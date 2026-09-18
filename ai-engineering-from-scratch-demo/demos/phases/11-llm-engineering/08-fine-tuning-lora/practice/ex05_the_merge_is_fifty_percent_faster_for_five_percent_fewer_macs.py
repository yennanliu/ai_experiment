"""Exercise 5 — the merge is 1.5x faster for 4.6% fewer multiply-accumulates.

    **Merge vs. unmerged inference.** Compare the output of the LoRA model
    before and after merge_lora_weights on the same 100 inputs. Verify the
    outputs are identical (within floating-point tolerance of 1e-5). Then
    benchmark inference speed for both -- merged should be slightly faster since
    it's a single matrix multiply instead of two.

Reading of the exercise: the 100 inputs are one seeded batch, and "speed" is
measured twice -- as wall clock, which is what a benchmark reports, and as
multiply-accumulates and tensor operations, which are what a benchmark is
supposed to be a proxy for. The model is trained first, because an unmerged
adapter at initialisation has B = 0 and merging it is a no-op.

**ANSWER: the outputs agree to 5.7e-07, comfortably inside 1e-5.** The merge is
exact: `weight += (A @ B).T * scaling` reproduces
`linear(x) + (x @ A @ B) * scaling` up to float32 rounding on a 512-wide
accumulation.

**ANSWER: merged inference is about 1.5x faster, and it is not the arithmetic.**
The unmerged model does 416,848 multiply-accumulates per sample against the
merged model's 398,336 -- 4.6% more. A 4.6% arithmetic saving does not buy a 50%
speedup; the per-operation dispatch does.

**MECHANISM: the tensor operations drop from 9 to 3.** Each of the three
`LinearWithLoRA` layers runs one `nn.Linear` plus two matmuls inside
`LoRALayer.forward`; after the merge each is a bare `nn.Linear`. The exercise's
"a single matrix multiply instead of two" undercounts by one: it is one instead
of three.

**FINDING: the saving is concentrated in the smallest layer.** Layer 4 is
512->10, so its linear is 5,120 MACs and its adapter is 4,176 -- the adapter is
82% as expensive as the layer it adapts. Layers 0 and 2 pay 4.7% and 3.1%. The
overhead of LoRA at inference is a function of how narrow the output is.

**FINDING: merging is not free and not reversible.** `merge_lora_weights` writes
into `linear.weight.data` in place and replaces the wrapper with the bare
linear, so the adapter is gone: the model can no longer be swapped to another
adapter, which is exactly what exercise 4's multi-adapter serving needs. Speed
and swapability are exclusive here, and the exercise asks for both in the same
lesson.

Structure: `macs` counts multiply-accumulates analytically, `tensor_ops` counts
module invocations with forward hooks, and `timed` is the wall clock.
"""

from __future__ import annotations

import time

import torch

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "08-fine-tuning-lora"
TARGETS = ["0", "2", "4"]
SEED, RANK, REPEATS = 42, 8, 200


def prepared(ref):
    torch.manual_seed(SEED)
    model = ref.create_demo_model()
    ref.inject_lora(model, TARGETS, rank=RANK)
    ref.train_lora(model, ref.create_demo_data(500), epochs=5, lr=1e-3, batch_size=32)
    return model


def macs(ref, model):
    """Multiply-accumulates per sample, and the adapter's share per layer."""
    per_layer = []
    for module in model.modules():
        if isinstance(module, ref.LinearWithLoRA):
            linear = module.linear.in_features * module.linear.out_features
            adapter = (module.lora.A.numel() + module.lora.B.numel())
            per_layer.append((linear, adapter))
    base = sum(linear for linear, _ in per_layer)
    return base, base + sum(a for _, a in per_layer), per_layer


def tensor_ops(ref, model, batch):
    """How many Linear and LoRALayer forwards one inference runs."""
    counts = {"linear": 0, "lora": 0}
    handles = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(
                lambda *_: counts.__setitem__("linear", counts["linear"] + 1)))
        elif isinstance(module, ref.LoRALayer):
            handles.append(module.register_forward_hook(
                lambda *_: counts.__setitem__("lora", counts["lora"] + 1)))
    with torch.no_grad():
        model(batch)
    for handle in handles:
        handle.remove()
    return counts["linear"] + 2 * counts["lora"]


def timed(model, batch, repeats=REPEATS):
    with torch.no_grad():
        model(batch)
        start = time.perf_counter()
        for _ in range(repeats):
            model(batch)
        return time.perf_counter() - start


def solve():
    ref = parity.load_reference(PHASE, LESSON, "lora")
    model = prepared(ref)
    batch = torch.randn(100, 256, generator=torch.Generator().manual_seed(3))
    merged_macs, unmerged_macs, per_layer = macs(ref, model)
    ops_before = tensor_ops(ref, model, batch)
    with torch.no_grad():
        before = model(batch).clone()
    unmerged = timed(model, batch)
    ref.merge_lora_weights(model)
    with torch.no_grad():
        after = model(batch).clone()
    return {
        "difference": float((before - after).abs().max()),
        "tolerance": 1e-5,
        "merged_macs": merged_macs, "unmerged_macs": unmerged_macs,
        "mac_overhead": round(100 * (unmerged_macs / merged_macs - 1), 1),
        "per_layer": [round(100 * a / linear, 1) for linear, a in per_layer],
        "ops_before": ops_before, "ops_after": tensor_ops(ref, model, batch),
        "speedup": round(unmerged / timed(model, batch), 2),
        "adapters_left": sum(isinstance(m, ref.LinearWithLoRA) for m in model.modules()),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the outputs agree well inside the 1e-5 tolerance",
            result["difference"] < result["tolerance"],
            f"max absolute difference over 100 inputs is {result['difference']:.2e} against "
            f"a tolerance of {result['tolerance']:.0e}. `weight += (A @ B).T * scaling` "
            "reproduces linear(x) + (x @ A @ B) * scaling up to float32 rounding on a "
            "512-wide accumulation",
        ),
        practice.Check(
            "ANSWER: merged is faster, and it is not the arithmetic",
            all([result["speedup"] > 1.1, result["mac_overhead"] < 10]),
            f"the unmerged model does {result['unmerged_macs']:,} multiply-accumulates per "
            f"sample against the merged model's {result['merged_macs']:,} -- "
            f"{result['mac_overhead']}% more -- while the wall clock shows a "
            f"{result['speedup']}x speedup. A {result['mac_overhead']}% arithmetic saving "
            "does not buy that; per-operation dispatch does",
        ),
        practice.Check(
            "MECHANISM: the tensor operations drop from 9 to 3",
            all([result["ops_before"] == 9, result["ops_after"] == 3]),
            f"each of the three LinearWithLoRA layers runs one nn.Linear plus the two "
            f"matmuls inside LoRALayer.forward: {result['ops_before']} operations before "
            f"the merge and {result['ops_after']} after. The exercise's 'a single matrix "
            "multiply instead of two' undercounts by one -- it is one instead of three",
        ),
        practice.Check(
            "FINDING: the overhead is concentrated in the narrowest layer",
            all([max(result["per_layer"]) > 50, min(result["per_layer"]) < 10]),
            f"the adapter's MACs as a percentage of the layer it adapts: "
            f"{result['per_layer']} for the 256->512, 512->512 and 512->10 layers. On the "
            "output projection the adapter costs 82% of the layer, because the layer is "
            "narrow and the adapter's cost depends on the rank and not on the output width",
        ),
        practice.Check(
            "FINDING: merging is not reversible, and that is the trade",
            result["adapters_left"] == 0,
            f"`merge_lora_weights` writes into linear.weight.data in place and replaces "
            f"the wrapper with the bare linear, leaving {result['adapters_left']} adapters "
            "in the model. The merged model cannot be swapped to another adapter, which is "
            "exactly what exercise 4's multi-adapter serving needs: speed and swapability "
            "are exclusive here",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
