"""Exercise 5 — alpha goes 0.769 to 0.954 as temperature rises, which is the opposite of the prediction.

    Measure failure modes. Run speculative decode at temperature=1.5 (high
    stochasticity). Show α collapses and the algorithm is slower than plain
    decode due to draft overhead.

Reading of the exercise: the module has no temperature -- `make_target` hard-codes
`standard_normal(vocab) * 1.4` -- so one is added the only way it can be, by
dividing the logits, and the draft is then rebuilt from each temperature's target
with the lesson's own `make_draft`. Acceptance is the lesson's own
`measure_alpha`.

**ANSWER: acceptance rises with temperature, monotonically.**

    T     target entropy    alpha    E[tokens] at K=4
    0.5       1.437         0.769         3.17
    1.0       2.697         0.855         3.74
    1.5       3.097         0.896         4.06
    2.0       3.251         0.919         4.25
    4.0       3.409         0.954         4.56

At the 1.5 the exercise names, alpha is **0.896** against 0.855 at T=1. The
algorithm gets faster, not slower.

**MECHANISM: `make_draft` blends the target with a uniform, and temperature
moves the target toward the uniform.** The draft is `a*p + (1-a)*u`, so as `p`
flattens the two converge: at `T = 4` the target's entropy is **3.409** against a
uniform's `ln(32) = 3.466`, and any blend of two nearly identical distributions
is nearly identical to both. The construction ties the draft's quality to the
target's flatness, in the direction that flatters it.

**FINDING: the real effect needs a draft that is *independently* trained, and it
is at the other end of the sweep.** Holding the draft fixed at its `T = 1` form
while the target's temperature moves gives **0.490, 0.855, 0.917, 0.873, 0.789**
-- non-monotone, peaking at T=1.5 because `make_draft` already flattens toward
uniform, and collapsing **0.427** on the cold side. Cooling the target is what
breaks a fixed draft; heating it is what the exercise looks at.

**FINDING: the module has no way to be slower than plain decode.** There is no
cost term anywhere, so `expected_tokens` reports 4.06 at T=1.5 and 3.17 at
T=0.5, both above 1. "Slower than plain decode due to draft overhead" is a claim
about a quantity that has to be supplied from outside, the same gap Exercise 2
finds.

Structure: `target_at` rebuilds the lesson's target at one temperature;
`coupled` rebuilds the draft with it and `frozen` keeps the T=1 draft.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "25-speculative-decoding"
VOCAB, HINT, SEED, SAMPLES, DRAFT_K = 32, 0.7, 2, 20_000, 4
TEMPERATURES = (0.5, 1.0, 1.5, 2.0, 4.0)


def target_at(temperature, vocab=VOCAB, seed=SEED):
    """`make_target`'s own construction with the logits divided by a temperature."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal(vocab) * 1.4 / temperature
    shifted = np.exp(logits - logits.max())
    return shifted / shifted.sum()


def entropy(distribution):
    return float(-(distribution * np.log(distribution)).sum())


def coupled(ref, temperature):
    """Draft rebuilt from this temperature's target: what the lesson's code does."""
    rng = np.random.default_rng(3)
    target = target_at(temperature)
    draft = ref.make_draft(target, HINT, rng)
    alpha = ref.measure_alpha(target, draft, SAMPLES, rng)
    return {"entropy": entropy(target), "alpha": alpha,
            "tokens": ref.expected_tokens(alpha, DRAFT_K)}


def frozen(ref, temperature):
    """Draft fixed at its T=1 form while the target's temperature moves."""
    rng = np.random.default_rng(3)
    draft = ref.make_draft(target_at(1.0), HINT, rng)
    target = target_at(temperature)
    return ref.measure_alpha(target, draft, SAMPLES, rng)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {temperature: coupled(ref, temperature) for temperature in TEMPERATURES}
    fixed = {temperature: frozen(ref, temperature) for temperature in TEMPERATURES}
    return {
        "rows": rows,
        "frozen": fixed,
        "uniform_entropy": float(np.log(VOCAB)),
        "monotone": all(rows[a]["alpha"] < rows[b]["alpha"]
                        for a, b in zip(TEMPERATURES, TEMPERATURES[1:])),
        "frozen_peak": max(fixed, key=fixed.get),
        "frozen_cold_drop": max(fixed.values()) - fixed[0.5],
        "frozen_hot_drop": max(fixed.values()) - fixed[4.0],
        "cost_names": [name for name in dir(ref)
                       if any(word in name.lower() for word in ("cost", "latency", "speed"))],
    }


def column(rows, key, fmt):
    return ", ".join(f"T={t} {format(row[key], fmt)}" for t, row in rows.items())


def verify(result):
    rows, fixed = result["rows"], result["frozen"]
    return [
        practice.Check(
            "ANSWER: acceptance rises with temperature, monotonically",
            result["monotone"] and rows[1.5]["alpha"] > rows[1.0]["alpha"],
            "the acceptance rate is " + column(rows, "alpha", ".3f")
            + " at target entropies of " + column(rows, "entropy", ".3f")
            + f". At the {1.5} the exercise names, alpha is {rows[1.5]['alpha']:.3f} against "
            f"{rows[1.0]['alpha']:.3f} at T=1, and expected tokens at K={DRAFT_K} goes "
            f"{rows[1.0]['tokens']:.2f} to {rows[1.5]['tokens']:.2f}. The algorithm gets faster, "
            "not slower",
        ),
        practice.Check(
            "MECHANISM: the draft is a blend with a uniform, and temperature flattens the target",
            rows[4.0]["entropy"] > 0.97 * result["uniform_entropy"],
            f"make_draft returns a*p + (1-a)*u, so as p flattens the two converge: at T=4 the "
            f"target's entropy is {rows[4.0]['entropy']:.3f} against a uniform's "
            f"ln({VOCAB}) = {result['uniform_entropy']:.3f}, and any blend of two nearly "
            "identical distributions is nearly identical to both. The construction ties the "
            "draft's quality to the target's flatness, in the direction that flatters it",
        ),
        practice.Check(
            "FINDING: freeze the draft and alpha peaks at 1.5, collapsing 0.427 on the cold side",
            result["frozen_cold_drop"] > 0.3 and result["frozen_hot_drop"] > 0.1,
            "a draft distilled at one temperature and deployed at another has a fixed "
            "distribution while the target moves. Holding the draft at its T=1 form gives "
            + ", ".join(f"T={temp} {value:.3f}" for temp, value in fixed.items())
            + f" -- non-monotone, peaking at T={result['frozen_peak']} and falling "
            f"{result['frozen_cold_drop']:.3f} on the cold side and "
            f"{result['frozen_hot_drop']:.3f} on the hot one. That cold-side collapse is the "
            "failure mode the exercise describes, at the opposite end of the sweep from where it "
            "looks for it",
        ),
        practice.Check(
            "FINDING: the module has no way to be slower than plain decode",
            not result["cost_names"] and min(row["tokens"] for row in rows.values()) > 1,
            f"there is no cost term anywhere -- no name mentioning cost, latency or speed, "
            f"{result['cost_names']} -- so expected_tokens reports "
            + column(rows, "tokens", ".2f")
            + " and every one is above 1. 'Slower than plain decode due to draft overhead' is a "
            "claim about a quantity that has to be supplied from outside the module, which is the "
            "same gap Exercise 2 finds",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
