"""Exercise 3 — a 2x intercept halves the shots at every threshold; a steeper exponent halves them only at 50%.

    Read Anil et al. 2024 Figure 3 (power law by category). Explain why
    violent/deceitful content needs fewer shots to jailbreak than other
    categories.

Reading of the exercise: a figure can't be run, but the explanation can.
"Needs fewer shots" has two mechanisms in a power law ASR = a0 + c * n^alpha.
Either the category starts closer to compliance (a larger intercept c), or
its compliance pattern is learned faster in context (a larger exponent
alpha). Both are built on the lesson's `target_asr`. The intercept category
doubles c; since c is hard-coded inside `target_asr`, doubling it means
evaluating at 2n, which is the same as c * sqrt(2). The exponent category
gets the alpha that also reaches 50% at 128 shots. Then the question is which
measurements tell the two apart.

**ANSWER: violent/deceitful content needs fewer shots either because the
model's prior against complying is weaker, or because the compliance
pattern is easier to extract, and only the shape of the curve says which.**
Both categories halve the baseline's 256 shots to 50% ASR (to 128). The
intercept category needs 18 / 128 / 430.2 shots for 20 / 50 / 90% ASR against
the baseline's 36 / 256 / 860.4, so it needs half the shots at every
threshold. That is a constant 2.00x, as if each shot counted twice. The
exponent category (alpha = 0.5714) needs 23.0 / 128 / 369.7, which is 1.57x /
2.00x / 2.33x fewer, and its advantage grows with the target ASR. A
weaker-prior story says violent and deceitful text is common in fiction and
news, so refusal training sits on a shallower base and every faux shot moves
the model further. That story predicts the constant ratio. A
faster-learning story says compliance in those categories is short and
stylistically uniform, like an easy benign ICL task. That one predicts the
growing ratio. At 5 shots both still "fail": 0.115 and 0.095 against 0.087.

**FINDING: the toy can express only the exponent explanation.**
`target_asr` takes (n_shots, alpha, a0). c = 0.03 is hard-coded, so a
category with a weaker prior has to be faked by rescaling the shot count.

**FINDING: the reference's fit calls the intercept category steeper.** On
the skill file's 5-512 grid, `fit_power_law` gives 0.450 for the baseline and
0.463 for the 2x-intercept category, which has the same exponent, because
the a0 offset biases the fit less as ASR rises. With the offset removed the
fit gives 0.500 and 0.500, and 0.560 for the steeper category. That is short
of its true 0.5714 because ASR is clipped at 1.0 at 512 shots. A
per-category exponent table, as the skill file asks for, would misread
Figure 3-style curves without the offset.

Structure: `categories()` builds the three ASR curves on `target_asr`;
`shots_to()` inverts each by bisection; the fits reuse `fit_power_law`.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "13-many-shot-jailbreaking"
GRID = [5, 32, 128, 256, 512]
TARGETS = (0.2, 0.5, 0.9)


def bisect(fn, goal, lo, hi, steps=100):
    """x in [lo, hi] with fn(x) = goal, for fn increasing in x."""
    for _ in range(steps):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fn(mid) < goal else (lo, mid)
    return (lo + hi) / 2


def categories(ref):
    """Three categories matched at 50% ASR except the baseline, as ASR(n) functions."""
    steep = bisect(lambda a: ref.target_asr(128, alpha=a), 0.5, 0.3, 0.9)
    return {
        "baseline": lambda n: ref.target_asr(n),
        "2x intercept": lambda n: ref.target_asr(2 * n),
        "steeper exponent": lambda n: ref.target_asr(n, alpha=steep),
    }, steep


def shots_to(asr, goal):
    return bisect(asr, goal, 1e-9, 1e6)


def fits(ref, cats):
    """fit_power_law exponents on the skill grid, raw and with the a0 offset removed."""
    a0 = inspect.signature(ref.target_asr).parameters["a0"].default
    out = {}
    for key, shift in (("fit_raw", 0.0), ("fit_offset", a0)):
        out[key] = {k: round(ref.fit_power_law(GRID, [f(n) - shift for n in GRID])[0], 3)
                    for k, f in cats.items()}
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cats, steep = categories(ref)
    need = {k: {t: round(shots_to(f, t), 1) for t in TARGETS} for k, f in cats.items()}
    return {
        "params": list(inspect.signature(ref.target_asr).parameters),
        "c_local": "c = 0.03" in inspect.getsource(ref.target_asr),
        "steep": round(steep, 4),
        "need": need,
        "ratio": {k: {t: round(need["baseline"][t] / need[k][t], 2) for t in TARGETS}
                  for k in cats},
        "steep_512": cats["steeper exponent"](512),
        "at5": {k: round(f(5), 3) for k, f in cats.items()},
        **fits(ref, cats),
    }


def verify(result):
    need, ratio, raw, off = (result[k] for k in ("need", "ratio", "fit_raw", "fit_offset"))
    return [
        practice.Check(
            "ANSWER: a 2x intercept halves shots at every threshold; a steeper exponent only at 50%",
            (need, ratio["2x intercept"], ratio["steeper exponent"], result["steep"], result["at5"])
            == ({"baseline": {0.2: 36.0, 0.5: 256.0, 0.9: 860.4},
                 "2x intercept": {0.2: 18.0, 0.5: 128.0, 0.9: 430.2},
                 "steeper exponent": {0.2: 23.0, 0.5: 128.0, 0.9: 369.7}},
                {0.2: 2.0, 0.5: 2.0, 0.9: 2.0}, {0.2: 1.57, 0.5: 2.0, 0.9: 2.33}, 0.5714,
                {"baseline": 0.087, "2x intercept": 0.115, "steeper exponent": 0.095}),
            f"shots to 20/50/90% ASR {need}; baseline / category {ratio}; ASR at 5 shots {result['at5']}",
        ),
        practice.Check(
            "FINDING: the toy can express only the exponent explanation",
            (result["params"], result["c_local"]) == (["n_shots", "alpha", "a0"], True),
            f"target_asr parameters {result['params']}; c is a local constant",
        ),
        practice.Check(
            "FINDING: the reference's fit calls the intercept category steeper",
            (raw, off, result["steep_512"])
            == ({"baseline": 0.45, "2x intercept": 0.463, "steeper exponent": 0.514},
                {"baseline": 0.5, "2x intercept": 0.5, "steeper exponent": 0.56}, 1.0),
            f"fit_power_law raw {raw}; offset removed {off}; steeper category at 512 shots "
            f"{result['steep_512']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
