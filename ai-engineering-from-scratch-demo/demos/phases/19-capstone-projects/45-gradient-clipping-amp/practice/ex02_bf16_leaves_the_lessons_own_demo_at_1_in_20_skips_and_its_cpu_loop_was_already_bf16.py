"""Exercise 2 — on the lesson's own demo --bf16 leaves the skip rate at 1 in 20; only a real spike drops it to 0.

    Add a `--bf16` mode that switches autocast to BF16 instead of FP16. BF16 has a wider exponent range than FP16 and rarely needs loss scaling; verify the skip rate drops to zero on the same demo.

Reading of the exercise: a small CLI (`parse`) wraps the lesson's
`run_demo` loop -- seed-7 toy model, `AmpTrainState(lr=1e-2, max_norm=1.0,
device_type="cpu")`, 20 steps -- with FP16 autocast by default and `--bf16`
switching it to BF16. `--spike` swaps the demo's Inf injection on step 5 for
exercise 1's real spike (targets x 1e8), and `--scaler` forces an enabled
CPU GradScaler, which is what the lesson's CUDA path does; the lesson itself
disables the scaler on CPU. "The same demo" is run both ways. CPU only: on
CUDA the scaler is on by default and the FP16 grads are scaled by 65536
before backward.

**ANSWER: on the lesson's demo the skip rate does not drop to zero.** It is
1/20 under FP16 and 1/20 under `--bf16`, because the demo's skip comes from
`inject_inf_into_first_grad`, which writes +Inf into a gradient whatever the
dtype. On a real spike the claim holds: FP16 skips 1/20 steps as
`non_finite_grad`, `--bf16` skips 0/20, because BF16's largest finite value
is 3.39e38 against FP16's 65504.

**FINDING: the lesson's CPU loop is already BF16.** `AmpTrainState` with
`device_type="cpu"` defaults `amp_dtype` to `torch.bfloat16`, so the shipped
demo never ran FP16. The work in this exercise is adding the FP16 mode, not
the BF16 one.

**FINDING: on CPU the logged scaling factor is 1.0 on every step.** The
scaler is disabled, so the halving the doc describes never shows. With an
enabled scaler, the FP16 skip halves it from 65536 to 32768. That change
first appears on the next row, because the skip row logs the scale from
before `update()`.

Expected output: three PASS checks.
"""

from __future__ import annotations

import argparse
import sys

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "45-gradient-clipping-amp"
BAD_STEP = 5


def parse(argv):
    cli = argparse.ArgumentParser(description="lesson 45 demo with an FP16/BF16 switch")
    cli.add_argument("--bf16", action="store_true", help="autocast to BF16 instead of FP16")
    cli.add_argument("--spike", action="store_true", help="targets x 1e8 on step 5 instead of an Inf grad")
    cli.add_argument("--scaler", action="store_true", help="force an enabled GradScaler, as on CUDA")
    return cli.parse_args(argv)


def run(ref, argv):
    args = parse(argv)
    model, inputs, targets = ref.build_toy_model()
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    state = ref.AmpTrainState(model=model, lr=1e-2, max_norm=1.0, device_type="cpu", amp_dtype=dtype)
    if args.scaler:
        state.scaler = torch.amp.GradScaler("cpu", enabled=True)
    bad_step = {"inputs": inputs, "targets": targets * 1e8} if args.spike else {
        "inputs": inputs, "targets": targets, "gradient_corruptor": ref.inject_inf_into_first_grad}
    for index in range(20):
        state.step(**bad_step) if index == BAD_STEP else state.step(inputs, targets)
    return summarize(state)


def summarize(state):
    return {"skip_rate": state.skip_count / len(state.log), "reasons": [r.reason for r in state.skip_log],
            "scales": sorted({r.scaler_scale for r in state.log}),
            "scale_around_skip": [state.log[i].scaler_scale for i in (BAD_STEP - 1, BAD_STEP, BAD_STEP + 1)]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    modes = {"fp16": [], "bf16": ["--bf16"]}
    return {
        "default_dtype": str(ref.AmpTrainState(ref.build_toy_model()[0], device_type="cpu").amp_dtype),
        "same_demo": {name: run(ref, flags) for name, flags in modes.items()},
        "spike": {name: run(ref, flags + ["--spike"]) for name, flags in modes.items()},
        "scaler": run(ref, ["--spike", "--scaler"]),
        "max": {name: torch.finfo(dt).max for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16))},
    }


def verify(result):
    same, spike, scaler = result["same_demo"], result["spike"], result["scaler"]
    return [
        practice.Check(
            "ANSWER: --bf16 leaves the lesson's demo at 1/20 skips; on a real spike it drops 1/20 -> 0/20",
            (same["fp16"]["skip_rate"], same["bf16"]["skip_rate"], spike["fp16"]["skip_rate"],
             spike["bf16"]["skip_rate"], spike["fp16"]["reasons"]) == (0.05, 0.05, 0.05, 0.0, ["non_finite_grad"])
            and (result["max"]["fp16"], round(result["max"]["bf16"] / 1e38, 2)) == (65504.0, 3.39),
            f"injected demo fp16 {same['fp16']['skip_rate']} / bf16 {same['bf16']['skip_rate']}; spike fp16 "
            f"{spike['fp16']['skip_rate']} / bf16 {spike['bf16']['skip_rate']}; max finite {result['max']}",
        ),
        practice.Check(
            "FINDING: the lesson's CPU loop is already BF16",
            result["default_dtype"] == "torch.bfloat16",
            f"AmpTrainState(device_type='cpu').amp_dtype = {result['default_dtype']}",
        ),
        practice.Check(
            "FINDING: on CPU the logged scale is 1.0 on every step; an enabled scaler halves it, one row late",
            all(run_["scales"] == [1.0] for run_ in (*same.values(), *spike.values()))
            and scaler["scale_around_skip"] == [65536.0, 65536.0, 32768.0],
            f"lesson CPU scales {same['fp16']['scales']}; enabled scaler steps 4-6: {scaler['scale_around_skip']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(print(run(parity.load_reference(PHASE, LESSON, "main"), sys.argv[1:]))
                     if sys.argv[1:] else practice.selfcheck(globals()))
