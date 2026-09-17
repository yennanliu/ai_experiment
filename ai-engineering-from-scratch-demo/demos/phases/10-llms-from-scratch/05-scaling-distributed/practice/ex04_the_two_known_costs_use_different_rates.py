"""Exercise 4 — the estimator matches both GPU-hour figures exactly and neither dollar figure.

    Build a cost estimator. Given a model size, target token count,
    GPU type (A100 at $2/hr, H100 at $3.50/hr), and parallelism strategy,
    estimate the total training cost in dollars. Validate against known costs:
    Llama 3 405B reportedly cost ~$100M, DeepSeek V3 cost ~$5.6M.

Reading of the exercise: the estimator is the lesson's own
`training_cost_estimator`, so "build" is read as *run it against the two
validations the exercise names and find out which of its inputs the answer
actually depends on*. DeepSeek V3 is costed on its **37B active** parameters,
because 6ND counts the parameters a token touches and V3 is a mixture of experts
with 671B total; both readings are run.

**ANSWER: the compute model is exact, once utilisation is not guessed.** The
published runs imply **34.5%** MFU for Llama 3 (30.84M H100-hours) and **33.1%**
for DeepSeek V3 (2.788M H800-hours) at 990 TFLOPS. Feed those back in and the
estimator returns 3.0830e7 and 2.7852e6 GPU-hours -- **within 0.1% of both**.
6ND has one free parameter and it is not the dollar rate.

**FINDING: the lesson's default `utilization=0.4` is the whole error.** At 0.4
the estimator is 13.8% and 17.3% low on hours, in the same direction and by
about the ratio of 0.4 to the achieved 0.33-0.345. It is not a modelling failure;
it is an optimistic constant.

**FINDING: the two "known costs" are not measured the same way.** Llama 3's
~$100M over 30.84M hours implies **$3.24/GPU-hour**; DeepSeek's $5.576M over
2.788M hours is exactly **$2.00** -- their paper's stated rental assumption, for
the final run only. The estimator's hardcoded $3.50 sits above both, so at the
matched utilisation it lands **+7.9%** on Llama 3 and **+74.8%** on DeepSeek. No
single estimator validates against both, because the two numbers are answers to
different questions.

**FINDING: the 17.9x headline is sparsity, not thrift.** It factors exactly as
**11.1x GPU-hours** times **1.62x price**, and those hours are **11.5x FLOPs**
with the two runs within 4% of each other on MFU. Cost DeepSeek V3 at its full
671B parameters and the estimator returns
**$146.3M** -- more than Llama 3's $93.1M. "$5.6M" is a statement about
activating 37B of 671B, and about renting at $2.

Structure: `PUBLISHED` carries the two reference runs; `implied_mfu` inverts 6ND
for the utilisation each achieved; `estimate` is one call to the reference.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "05-scaling-distributed"
TFLOPS, DEFAULT_UTIL, LISTED_RATE = 990e12, 0.40, 3.50
PUBLISHED = {
    "Llama 3 405B": dict(params=405, tokens=15.6, hours=30.84e6, cost=100e6),
    "DeepSeek V3": dict(params=37, tokens=14.8, hours=2.788e6, cost=5.576e6),
}


def flops(params_billions, tokens_trillions):
    """The 6ND law: six FLOPs per parameter per token, forward and backward."""
    return 6 * params_billions * 1e9 * tokens_trillions * 1e12


def implied_mfu(run):
    """The utilisation a published GPU-hour figure actually achieved."""
    return flops(run["params"], run["tokens"]) / (run["hours"] * 3600 * TFLOPS)


def estimate(ref, run, utilization):
    return ref.training_cost_estimator(run["params"], run["tokens"], "h100",
                                       utilization=utilization)


def summary(arms, key, fmt):
    """One `name value` phrase per published run — the detail strings' only formatter."""
    return ", ".join(fmt(name, arm[key]) for name, arm in arms.items())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    arms = {}
    for name, run in PUBLISHED.items():
        mfu = implied_mfu(run)
        default, matched = estimate(ref, run, DEFAULT_UTIL), estimate(ref, run, round(mfu, 3))
        arms[name] = {
            "mfu": mfu,
            "rate": run["cost"] / run["hours"],
            "hours_default": default["total_gpu_hours"] / run["hours"] - 1,
            "hours_matched": matched["total_gpu_hours"] / run["hours"] - 1,
            "cost_matched": matched["estimated_cost"] / run["cost"] - 1,
            "cost_default": default["estimated_cost"],
        }
    dense = estimate(ref, dict(params=671, tokens=14.8), DEFAULT_UTIL)
    llama = arms["Llama 3 405B"]
    return {
        "arms": arms,
        "dense_671b": dense["estimated_cost"],
        "llama_cost": llama["cost_default"],
        "matched_ok": all(abs(a["hours_matched"]) < 0.01 for a in arms.values()),
        "default_low": all(-0.2 < a["hours_default"] < -0.1 for a in arms.values()),
        "mfu_range": (min(a["mfu"] for a in arms.values()),
                      max(a["mfu"] for a in arms.values())),
        "headline": PUBLISHED["Llama 3 405B"]["cost"] / PUBLISHED["DeepSeek V3"]["cost"],
        "hours_ratio": PUBLISHED["Llama 3 405B"]["hours"] / PUBLISHED["DeepSeek V3"]["hours"],
        "flops_ratio": flops(405, 15.6) / flops(37, 14.8),
    }


def verify(result):
    arms = result["arms"]
    llama, deepseek = arms["Llama 3 405B"], arms["DeepSeek V3"]
    price_ratio = llama["rate"] / deepseek["rate"]
    return [
        practice.Check(
            "ANSWER: at the utilisation each run achieved, 6ND reproduces both to within 0.1%",
            result["matched_ok"],
            "the published figures imply "
            + summary(arms, "mfu", lambda k, v: f"{k} {100 * v:.1f}% MFU")
            + f" at {TFLOPS / 1e12:.0f} TFLOPS, and feeding those back gives "
            + summary(arms, "hours_matched", lambda k, v: f"{k} {100 * v:+.1f}%")
            + " against the published GPU-hours. The compute model is exact; it has one free "
            "parameter and that parameter is not the dollar rate",
        ),
        practice.Check(
            f"FINDING: the default utilization={DEFAULT_UTIL} is the whole error",
            result["default_low"],
            "at the lesson's default the estimator is "
            + summary(arms, "hours_default", lambda k, v: f"{100 * v:.1f}%")
            + " low on hours, in the same direction on both runs and by about the ratio of "
            f"{DEFAULT_UTIL} to the {100 * result['mfu_range'][0]:.0f}-"
            f"{100 * result['mfu_range'][1]:.0f}% actually achieved. It is an "
            "optimistic constant, not a modelling failure",
        ),
        practice.Check(
            "FINDING: the two 'known costs' imply $3.24/hour and $2.00/hour",
            abs(deepseek["rate"] - 2.00) < 0.01 and llama["rate"] > 3.0
            and deepseek["cost_matched"] > 5 * llama["cost_matched"] > 0,
            f"Llama 3's ~$100M over {PUBLISHED['Llama 3 405B']['hours']:.3g} hours implies "
            f"${llama['rate']:.2f}/GPU-hour; DeepSeek's $5.576M over "
            f"{PUBLISHED['DeepSeek V3']['hours']:.3g} is exactly ${deepseek['rate']:.2f}, their "
            f"stated rental assumption for the final run only. The estimator's hardcoded "
            f"${LISTED_RATE:.2f} sits above both, so at the matched utilisation it lands "
            f"{100 * llama['cost_matched']:+.1f}% on one and "
            f"{100 * deepseek['cost_matched']:+.1f}% on the other. No single estimator validates "
            "against both, because the two numbers answer different questions",
        ),
        practice.Check(
            "FINDING: the 17.9x headline is 11.5x compute and 1.62x price, not efficiency",
            result["dense_671b"] > result["llama_cost"]
            and abs(result["headline"] - result["hours_ratio"] * price_ratio) < 0.1,
            f"{result['headline']:.1f}x factors exactly as {result['hours_ratio']:.1f}x GPU-hours "
            f"times {price_ratio:.2f}x price -- and the hours are "
            f"{result['flops_ratio']:.1f}x FLOPs, the two runs being within "
            f"{100 * abs(llama['mfu'] / deepseek['mfu'] - 1):.0f}% of each other on MFU. Cost "
            f"DeepSeek V3 at its full 671B rather than its 37B active and the estimator returns "
            f"${result['dense_671b'] / 1e6:.1f}M against Llama 3's "
            f"${result['llama_cost'] / 1e6:.1f}M -- more, not less. '$5.6M' is a statement about "
            "activating 37B of 671B, and about renting at $2",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
