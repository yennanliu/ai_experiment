"""Exercise 3 — 6ND at 40% MFU reproduces Llama 2's published hours to 1.4%, and the pipeline's own figure to 369x.

    Implement a cost estimator from first principles. For stage 04
    (pre-training), estimate FLOPs as 6 x params x tokens, assume 40% MFU (model
    FLOPs utilization) on H100 at 989 TFLOPs BF16, at $2.50/GPU-hour. Report the
    estimate for a 7B model trained on 2T tokens. Compare to published Llama 2
    numbers.

Reading of the exercise: the estimator is built with the constants the exercise
gives, and the comparison to Llama 2 is made in Llama 2's units. Meta reports
**184,320 A100-80GB hours** for the 7B, not H100 hours and not dollars, so the
same FLOPs are re-divided by the A100's 312 TFLOPs BF16 before the two numbers
are set beside each other. The pipeline's own figure for the same stage is read
off its `cost_table`.

**ANSWER: $147,455, or 58,982 H100-hours, and the check against Llama 2 passes.**
`6 x 7e9 x 2e12` is `8.400e22` FLOPs; at 989 TFLOPs and 40% MFU that is 58,982
H100-hours. Divided by the A100's 312 TFLOPs at the same MFU it is **186,966
A100-hours** against Meta's published **184,320** -- a ratio of **1.014**. A
back-of-envelope with two constants and one published utilisation figure lands
within 1.4% of a real pre-training run.

**FINDING: the pipeline books the same stage at $400.** `simulate_stage`'s
`cost_table` gives `checkpoint` a flat `(7200, 400)`, so the estimator the
exercise asks you to write disagrees with the pipeline it is being written for by
**369x**. The whole twelve-stage pipeline costs **$1,941** in the manifest and
stage 04 alone costs **$147,455** in reality.

**FINDING: the budget gate is the wrong order of magnitude, in both directions.**
`DEFAULT_GATES["cost_total_usd"]` is `<= 50000` and `Manifest.budget_usd` is
`50000`. The simulated pipeline spends 3.9% of that, so the budget never binds;
the real stage 04 is **2.95x** the entire budget, so it would halt the run at
stage 04 and never reach the gate. The one number in the manifest that is meant
to stop a run is set between the two worlds.

**MECHANISM: 6ND is 2ND forward and 4ND backward, and MFU is where the honesty
is.** The 6 is arithmetic -- one multiply-accumulate per parameter per token,
doubled for the backward pass and again for the gradient. The 40% is not: it is a
measurement of what a real cluster achieves against its spec sheet, and it is the
only term in the estimate that cannot be derived. Getting within 1.4% means
Meta's cluster ran at almost exactly the 40% the exercise assumes.

Structure: `flops` and `gpu_hours` are the estimator; `arm` prices one accelerator
so the H100 estimate and the A100 comparison come from the same FLOP count.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "13-building-complete-llm-pipeline"
PARAMS, TOKENS, MFU, RATE = 7e9, 2e12, 0.40, 2.50
H100_TFLOPS, A100_TFLOPS = 989e12, 312e12
LLAMA2_A100_HOURS = 184_320
STAGE = "04_pretrained_base"


def flops(params=PARAMS, tokens=TOKENS):
    """The exercise's own formula: one multiply-accumulate per parameter per token, x6."""
    return 6 * params * tokens


def gpu_hours(total_flops, peak_tflops, mfu=MFU):
    return total_flops / (peak_tflops * mfu) / 3600


def arm(peak_tflops):
    hours = gpu_hours(flops(), peak_tflops)
    return {"hours": hours, "usd": hours * RATE}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    manifest = ref.run(ref.Manifest(), ref.ArtifactStore())
    booked = next(s for s in manifest.stages if s.name == STAGE)
    h100, a100 = arm(H100_TFLOPS), arm(A100_TFLOPS)
    return {
        "flops": flops(),
        "h100": h100,
        "a100": a100,
        "published": LLAMA2_A100_HOURS,
        "ratio": a100["hours"] / LLAMA2_A100_HOURS,
        "booked": (booked.cost_usd, booked.wall_clock_sec),
        "pipeline_total": manifest.total_cost_usd,
        "budget": manifest.budget_usd,
        "gate": ref.DEFAULT_GATES["cost_total_usd"],
    }


def verify(result):
    h100, a100 = result["h100"], result["a100"]
    booked_usd, booked_wall = result["booked"]
    budget, gate = result["budget"], result["gate"]
    return [
        practice.Check(
            "ANSWER: $147,455 on H100s, and 186,966 A100-hours against Meta's published 184,320",
            0.98 < result["ratio"] < 1.02,
            f"6 x {PARAMS:.0e} x {TOKENS:.0e} is {result['flops']:.3e} FLOPs; at "
            f"{H100_TFLOPS / 1e12:.0f} TFLOPs and {MFU:.0%} MFU that is {h100['hours']:,.0f} "
            f"H100-hours, ${h100['usd']:,.0f} at ${RATE:.2f}/GPU-hour. Divided by the A100's "
            f"{A100_TFLOPS / 1e12:.0f} TFLOPs at the same MFU it is {a100['hours']:,.0f} "
            f"A100-hours against Meta's {result['published']:,} for the Llama 2 7B, a ratio of "
            f"{result['ratio']:.3f}",
        ),
        practice.Check(
            "FINDING: the pipeline books the same stage at $400 -- a factor of 369",
            h100["usd"] > 300 * booked_usd,
            f"simulate_stage's cost_table gives a checkpoint stage a flat "
            f"({booked_wall:.0f}, {booked_usd:.0f}), so the estimator the exercise asks you to "
            f"write disagrees with the pipeline it is written for by "
            f"{h100['usd'] / booked_usd:.0f}x. The whole twelve-stage pipeline costs "
            f"${result['pipeline_total']:,.0f} in the manifest and stage 04 alone costs "
            f"${h100['usd']:,.0f} in reality",
        ),
        practice.Check(
            "FINDING: the budget gate is the wrong order of magnitude in both directions",
            result["pipeline_total"] < 0.1 * budget < budget < h100["usd"],
            f"DEFAULT_GATES['cost_total_usd'] is {gate['op']} {gate['value']:,.0f} and "
            f"Manifest.budget_usd is {budget:,.0f}. The simulated pipeline spends "
            f"{result['pipeline_total'] / budget:.1%} of that, so the budget never binds; the "
            f"real stage 04 is {h100['usd'] / budget:.2f}x the entire budget, so it would halt "
            "the run at stage 04 and never reach the gate. The one number meant to stop a run is "
            "set between the two worlds",
        ),
        practice.Check(
            "MECHANISM: the 6 is arithmetic and the 40% is the only measured term",
            abs(result["flops"] / (PARAMS * TOKENS) - 6) < 1e-9,
            f"{result['flops']:.3e} / (params x tokens) is exactly 6: one multiply-accumulate per "
            "parameter per token, doubled for the backward pass and again for the gradient. The "
            f"{MFU:.0%} is not arithmetic -- it is what a real cluster achieves against its spec "
            f"sheet, and it is the only term here that cannot be derived. Landing within "
            f"{abs(result['ratio'] - 1):.1%} of Meta's figure means their cluster ran at almost "
            f"exactly the {MFU:.0%} the exercise assumes",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
