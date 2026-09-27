"""Exercise 4 — a true 1.5x uplift reports as 1.9x at 20 per arm, and losing 41% of the help shows 4.0x as 2.53x.

    Anthropic 2025's bioweapon-acquisition trial reports 2.53x uplift.
    Describe two ways this number could be biased upward (novice sample size,
    task fidelity) and two downward (elicitation ceiling, model safety
    gating).

Reading of the exercise: the trial's own design is not public in the lesson,
so each bias is shown on the smallest model that produces it, with the
uplift read the way the lesson defines it: success rate with the model over
success rate of a novice without it (control success 0.2 where one is
needed). Sample-size bias is computed exactly from the binomial, not
simulated.

**ANSWER (up, sample size): a ratio of two small-sample rates is biased
high.** A study can only report a ratio when the control arm has at least
one success. Conditioned on that, a true 1.5x at control success 0.2
reports on average 1.73x at 10 per arm, 1.91x at 20, 1.65x at 50 and 1.53x at
200. With no real uplift at all, 20 per arm expects to report 1.27x.

**ANSWER (up, task fidelity): a proxy inflates the ratio only if the step it
drops is failed by the people the model helps most.** A no-help step that
every participant passes with the same probability cancels out of the ratio
exactly. It inflates when it is correlated with who gets help: half the
novices know nothing (information step 0.1 -> 0.4 with the model), half know
the basics (0.6 -> 0.8). The information-only proxy measures 1.71x; add a
tacit step only the second half can pass (at 0.5) and the real task's uplift
is 1.33x.

**ANSWER (down, elicitation ceiling and safety gating): both remove help, and
the loss compounds across steps.** In a two-step task where full help lifts
each step from 0.3 to 0.6, the model's real uplift is 4.0x. If a fraction f of
the help arrives -- the rest refused by the gate or never asked for by a
novice who does not know the question -- measured uplift is (1 + f)^2: 3.24x
at f = 0.8, 2.56x at 0.6 -- the one-step 1.6x, squared. A measured 2.53x is
this 4.0x model with f = 0.59, e.g. 20% of queries refused and 74% of the
rest elicited well.

**FINDING: the lesson's toy computes a different "uplift".** `main()` divides
accuracy by 0.25 -- random guessing on four options, not a novice with a
search engine -- and prints bio at 2.80x before unlearning. On that scale
uplift cannot exceed 4.0x, and the trial's 2.53x would be 0.6325 accuracy.

Structure: `reported_ratio()` sums the exact conditional expectation of the
ratio estimator; `population()` averages success over participant types;
`gated()` is the two-step compounding model.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import math
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "17-wmdp-dual-use-evaluation"
P0, SIZES = 0.2, (10, 20, 50, 200)
TYPES = ((0.5, 0.1, 0.4), (0.5, 0.6, 0.8))   # (share, control, treated) on the information step


def reported_ratio(n, p0, uplift):
    """E[p1_hat / p0_hat | at least one control success], n per arm, exact."""
    pmf = [math.comb(n, k) * p0**k * (1 - p0) ** (n - k) for k in range(n + 1)]
    inv = sum(pmf[k] * n / k for k in range(1, n + 1)) / (1 - pmf[0])
    return round(uplift * p0 * inv, 3)


def population(tacit):
    """Uplift over the mixed population; `tacit` is each type's pass rate on a no-help step."""
    control = sum(t[0] * t[1] * p for t, p in zip(TYPES, tacit))
    treated = sum(t[0] * t[2] * p for t, p in zip(TYPES, tacit))
    return round(treated / control, 3)


def gated(f, steps=2, control=0.3, full=0.6):
    return round(((control + f * (full - control)) / control) ** steps, 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    trial = float(re.search(r"showed ([\d.]+)x uplift", parity.doc_text(PHASE, LESSON))[1])
    out, saved, ref.random = io.StringIO(), ref.random, random.Random(47)
    try:
        with contextlib.redirect_stdout(out):
            ref.main()
    finally:
        ref.random = saved
    return {
        "trial": trial,
        "sample": {n: reported_ratio(n, P0, 1.5) for n in SIZES},
        "null_20": reported_ratio(20, P0, 1.0),
        "fidelity": {k: population(t) for k, t in
                     (("proxy", (1, 1)), ("shared", (0.5, 0.5)), ("correlated", (0, 0.5)))},
        "gated": {f: gated(f) for f in (1.0, 0.8, 0.6)},
        "f_for_trial": round(math.sqrt(trial) - 1, 2),
        "toy_bio": float(re.search(r"biosecurity\s+pre=([\d.]+)x", out.getvalue())[1]),
        "novice": float(re.search(r"novice = ([\d.]+)", inspect.getsource(ref.main))[1]),
    }


def verify(result):
    s, trial, novice = result["sample"], result["trial"], result["novice"]
    return [
        practice.Check(
            "ANSWER (up, sample size): a ratio of two small-sample rates is biased high",
            s == {10: 1.73, 20: 1.905, 50: 1.651, 200: 1.531} and result["null_20"] == 1.27,
            f"true 1.5x at control 0.2 reports {s} by n per arm; no uplift at n = 20 reports "
            f"{result['null_20']}x",
        ),
        practice.Check(
            "ANSWER (up, task fidelity): a proxy inflates only if the dropped step is failed "
            "by the people the model helps most",
            result["fidelity"] == {"proxy": 1.714, "shared": 1.714, "correlated": 1.333},
            f"information-only proxy, + a tacit step all pass at 0.5, + one only type 2 passes: "
            f"{result['fidelity']}",
        ),
        practice.Check(
            "ANSWER (down, elicitation and gating): both remove help and the loss compounds",
            result["gated"] == {1.0: 4.0, 0.8: 3.24, 0.6: 2.56} and trial == 2.53
            and result["f_for_trial"] == 0.59 and round(gated(0.8 * 0.74), 2) == trial
            and round(gated(0.6, steps=1) ** 2, 3) == gated(0.6),
            f"measured uplift by fraction of help delivered {result['gated']}; the lesson's "
            f"{trial}x is f = {result['f_for_trial']}",
        ),
        practice.Check(
            "FINDING: the lesson's toy computes a different 'uplift'",
            novice == 0.25 and result["toy_bio"] == 2.8 and round(trial * novice, 4) == 0.6325,
            f"toy uplift = accuracy / {novice}: bio {result['toy_bio']}x, ceiling "
            f"{1 / novice:.1f}x; {trial}x = accuracy {trial * novice:.4f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
