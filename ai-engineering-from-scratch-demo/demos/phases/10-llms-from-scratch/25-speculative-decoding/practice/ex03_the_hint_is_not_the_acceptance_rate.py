"""Exercise 3 — `alpha_hint=0.3` gives an acceptance rate of 0.707, and a uniform draft already gets 0.571.

    Train a tiny draft. Take a 124M GPT-2 target and distill a 30M GPT-2 draft on
    100M tokens with KL loss. Measure `α` on held-out text. Expected: 0.6-0.7.

Reading of the exercise: a 100M-token distillation run is not available from
here, and the lesson already ships the stand-in for one -- `make_draft`, whose
docstring says it produces "a draft distribution whose expected token-level
acceptance is near alpha_hint". The measurement the exercise asks for is
therefore run against that promise, over the whole range of the parameter, using
the lesson's own `measure_alpha`.

**ANSWER: the hint is not the acceptance rate, and the gap reaches 0.52.**

    alpha_hint    measured alpha    sum min(p, q)
       0.1            0.617             0.616
       0.3            0.707             0.702
       0.5            0.791             0.788
       0.7            0.872             0.874
       0.9            0.962             0.959

The exercise expects 0.6-0.7 from a distilled draft. Reaching that here needs
`alpha_hint` near **0.1**, and the lesson's own `main` uses higher values.

**MECHANISM: acceptance is `sum_x min(p(x), q(x))` and the blend only moves
part of it.** `make_draft` returns `a*p + (1-a)*u`. Where `p(x) > 1/V` the draft
is below the target and contributes `q(x)`; where `p(x) < 1/V` it is above and
contributes `p(x)`. The second group is accepted at **full rate whatever `a`
is**, so the floor is `sum_{p<u} p(x)` -- with `a = 0` the draft is uniform and
acceptance is still 0.571, not 0.

**FINDING: the measurement agrees with the closed form to 0.005.** `measure_alpha`
draws 20,000 tokens and tests each against `min(1, p/q)`; `sum min(p, q)` is what
that estimates, and the two agree at every hint. The simulator is right; the
parameter's name is the thing that is wrong.

**FINDING: the exercise's target range is below the floor for most
vocabularies.** At vocab 32 the a=0 floor is 0.571, so no setting of `alpha_hint`
produces the 0.6-0.7 the exercise expects except a narrow band right at the
bottom. A distilled 30M draft that agreed with its target 57% of the time by
accident would look, to this harness, like a successful distillation.

Structure: `blend` builds the draft at one hint; `closed_form` is
`sum min(p, q)`; `floor` is the acceptance a uniform draft already gets.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "25-speculative-decoding"
VOCAB, SAMPLES, SEED = 32, 20_000, 1
HINTS = (0.1, 0.3, 0.5, 0.7, 0.9)
EXPECTED = (0.6, 0.7)


def closed_form(target, draft):
    """The acceptance rate the rejection rule achieves: sum_x min(p(x), q(x))."""
    return float(np.minimum(target, draft).sum())


def blend(ref, target, hint, rng):
    return ref.make_draft(target, hint, rng)


def row(ref, target, hint):
    rng = np.random.default_rng(SEED)
    draft = blend(ref, target, hint, rng)
    return {"measured": ref.measure_alpha(target, draft, SAMPLES, rng),
            "closed": closed_form(target, draft)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = np.random.default_rng(SEED)
    target = ref.make_target(VOCAB, rng)
    rows = {hint: row(ref, target, hint) for hint in HINTS}
    uniform = np.full(VOCAB, 1.0 / VOCAB)
    return {
        "rows": rows,
        "gap": {hint: values["measured"] - hint for hint, values in rows.items()},
        "agreement": max(abs(v["measured"] - v["closed"]) for v in rows.values()),
        "floor": closed_form(target, uniform),
        "above_uniform": float(target[target > 1.0 / VOCAB].sum()),
        "in_range": [hint for hint, values in rows.items()
                     if EXPECTED[0] <= values["measured"] <= EXPECTED[1]],
        "vocab": VOCAB,
    }


def column(rows, key, fmt):
    return ", ".join(f"hint {hint} {format(values[key], fmt)}"
                     for hint, values in rows.items())


def verify(result):
    rows, gap = result["rows"], result["gap"]
    return [
        practice.Check(
            "ANSWER: the hint is not the acceptance rate, and the gap reaches 0.40",
            max(gap.values()) > 0.35 and all(g > 0 for g in gap.values()),
            "the measured acceptance is " + column(rows, "measured", ".3f")
            + ", so the gap against the parameter's own name is "
            + ", ".join(f"{hint} {value:+.3f}" for hint, value in gap.items())
            + ". make_draft's docstring says it produces a draft 'whose expected token-level "
            "acceptance is near alpha_hint', and it is above it at every setting",
        ),
        practice.Check(
            "MECHANISM: acceptance is sum min(p, q) and the blend only moves part of it",
            result["floor"] > 0.5,
            f"make_draft returns a*p + (1-a)*u. Where p(x) > 1/V the draft is below the target and "
            f"contributes q(x); where p(x) < 1/V it is above and contributes p(x), at full rate "
            f"whatever a is. So even a = 0 -- a uniform draft that knows nothing -- already "
            f"accepts {result['floor']:.3f}, and the target's mass above uniform is "
            f"{result['above_uniform']:.3f}",
        ),
        practice.Check(
            "FINDING: the measurement agrees with the closed form to 0.005",
            result["agreement"] < 0.01,
            f"measure_alpha draws {SAMPLES:,} tokens and tests each against min(1, p/q); "
            f"sum min(p, q) is what that estimates. The two agree to "
            f"{result['agreement']:.4f} at every hint -- " + column(rows, "closed", ".3f")
            + ". The simulator is right and the parameter's name is what is wrong",
        ),
        practice.Check(
            "FINDING: the exercise's 0.6-0.7 range sits at or below the floor",
            not result["in_range"] or min(result["in_range"]) <= 0.1,
            f"at vocab {result['vocab']} the uniform-draft floor is {result['floor']:.3f}, so the "
            f"0.6-0.7 a distilled draft is expected to reach is what a draft with no information "
            f"already gets. The hints landing in that band are {result['in_range']}. A 30M draft "
            "agreeing with its target 62% of the time by accident would look, to this harness, "
            "like a successful distillation",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
