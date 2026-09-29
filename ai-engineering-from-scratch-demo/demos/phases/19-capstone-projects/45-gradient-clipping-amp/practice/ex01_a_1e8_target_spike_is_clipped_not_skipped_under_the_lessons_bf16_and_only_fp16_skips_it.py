"""Exercise 1 — a 1e8 target spike never reaches the skip path under the lesson's BF16 default; clipping absorbs it.

    Replace the synthetic Inf injection with a real loss spike (multiply one batch's target by 1e8) and verify the skip path triggers.

Reading of the exercise: the lesson's own demo is rerun unchanged -- the
seed-7 toy model from `build_toy_model`, `AmpTrainState(lr=1e-2,
max_norm=1.0, device_type="cpu")`, 20 steps -- except that on step 5,
instead of `inject_inf_into_first_grad`, the batch's targets are multiplied
by 1e8. "Verify the skip path triggers" is tested as posed, under the
lesson's CPU default (autocast BF16) and under FP16 autocast, and the factor
is swept to find where each of the lesson's two checks (loss, then grad)
actually fires. CPU only: GradScaler is disabled on CPU by the lesson, so
there is no loss scaling here; on CUDA the FP16 grads would be multiplied by
the scale (65536 at start) before backward, so they would overflow at a
smaller spike than they do here.

**ANSWER: under the lesson's own default the skip path does not trigger.**
CPU autocast runs BF16, whose range reaches 3.4e38. The spiked loss is
9.79e15 and the gradient norm 1.19e8, both finite, so the step is taken and
`clip_global_l2_norm` scales it down to 1.0. Under FP16 autocast the same
spike does skip, as `non_finite_grad`: the backward pass overflows FP16's
65504. The loss check would only fire at a factor of 1e19, where the
FP32 loss itself overflows.

**FINDING: the skip threshold is set by the dtype, not by the spike.** FP16
skips from factor 1e6 up (1e5 still steps), BF16 steps through
everything up to 1e18, and every run skips as `non_finite_loss` at 1e19.

**FINDING: clipping, not skipping, is what keeps this run alive.** With the
BF16 spike clipped to norm 1.0 the run ends at loss 0.0614 after 20 steps
(0.0555 without a spike) and 2.4e-5 after 100. Unclipped (`max_norm=1e30`),
the one 1.19e8 gradient inflates AdamW's second moment and the run ends at
0.475 after 20 steps and 0.677 after 100 -- worse at 100 than at 20.

Expected output: three PASS checks with the table's numbers.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "45-gradient-clipping-amp"
SPIKE_STEP, FACTORS = 5, (1e5, 1e6, 1e8, 1e18, 1e19)
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def run(ref, dtype, factor, max_norm=1.0, steps=20):
    """The lesson's demo, with the Inf injection replaced by a target spike."""
    torch.manual_seed(0)
    model, inputs, targets = ref.build_toy_model()
    state = ref.AmpTrainState(model=model, lr=1e-2, max_norm=max_norm, device_type="cpu", amp_dtype=dtype)
    for index in range(steps):
        state.step(inputs, targets * factor if index == SPIKE_STEP else targets)
    row = state.log[SPIKE_STEP]
    return {"reason": row.skip_reason or "step", "pre": row.grad_l2_pre_clip, "post": row.grad_l2_post_clip,
            "loss": row.loss, "final": state.log[-1].loss, "skips": state.skip_count}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {
        "spike": {name: run(ref, dt, 1e8) for name, dt in DTYPES.items()},
        "sweep": {name: [run(ref, dt, f)["reason"] for f in FACTORS] for name, dt in DTYPES.items()},
        "clean_final": run(ref, torch.bfloat16, 1.0)["final"],
        "clip_100": run(ref, torch.bfloat16, 1e8, steps=100)["final"],
        "noclip_20": run(ref, torch.bfloat16, 1e8, max_norm=1e30)["final"],
        "noclip_100": run(ref, torch.bfloat16, 1e8, max_norm=1e30, steps=100)["final"],
    }


def close(value, want, rel=0.02):
    return abs(value - want) <= rel * abs(want)


def verify(result):
    bf, fp = result["spike"]["bf16"], result["spike"]["fp16"]
    return [
        practice.Check(
            "ANSWER: under the lesson's BF16 default a 1e8 spike is clipped, not skipped; FP16 skips it",
            (bf["reason"], bf["skips"], bf["post"], fp["reason"], fp["skips"]) == ("step", 0, 1.0, "non_finite_grad", 1)
            and close(bf["pre"], 1.1866e8) and close(bf["loss"], 9.786e15),
            f"bf16: loss {bf['loss']:.3g}, grad norm {bf['pre']:.3g} -> {bf['post']}, {bf['skips']} skips; "
            f"fp16: {fp['reason']}, {fp['skips']} skip",
        ),
        practice.Check(
            "FINDING: the skip threshold is set by the dtype -- FP16 skips from 1e6, BF16 only at 1e19",
            result["sweep"] == {"bf16": ["step"] * 4 + ["non_finite_loss"],
                                "fp16": ["step"] + ["non_finite_grad"] * 3 + ["non_finite_loss"]},
            f"factors {FACTORS}: {result['sweep']}",
        ),
        practice.Check(
            "FINDING: clipping keeps the run alive; unclipped, one spike leaves it worse at 100 steps than 20",
            close(result["clean_final"], 0.0555) and close(bf["final"], 0.0614) and result["clip_100"] < 1e-4
            and close(result["noclip_20"], 0.475) and close(result["noclip_100"], 0.677),
            f"clipped: {bf['final']:.4f} at 20, {result['clip_100']:.2g} at 100 (clean {result['clean_final']:.4f}); "
            f"unclipped: {result['noclip_20']:.3f} at 20, {result['noclip_100']:.3f} at 100",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
