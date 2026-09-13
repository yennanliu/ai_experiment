"""Exercise 1 — with these constants D/N is never 20 in the range that matters.

    **Easy.** Run `code/main.py`. Print Chinchilla-optimal `(N, D)` for compute
    budgets `1e20`, `1e22`, `1e24`. Compare to the real model table.

Reading of the exercise: the lesson finds the optimum by grid search, so the
same optimum is also derived in closed form -- minimising `A/N^a + B/D^b` under
`6ND = C` gives `N* = (A*a / (B*b))^(1/(a+b)) * (C/6)^(b/(a+b))` -- and the two
are compared. That is what makes the D/N number checkable rather than printed.

**ANSWER: the grid is accurate to 3% in N and to four decimals in loss.**

| C | grid N* | exact N* | error | grid D/N | exact D/N | loss |
|---:|---:|---:|---:|---:|---:|---:|
| 1e20 | 6.59e8 | 6.45e8 | 2.2% | 38.3 | 40.1 | 2.5998 |
| 1e22 | 5.05e9 | 5.16e9 | 2.1% | 65.3 | 62.6 | 2.1386 |
| 1e24 | 4.25e10 | 4.13e10 | 2.9% | 92.4 | 97.7 | 1.9112 |

200 points over 8 decades is a **9.7% step in N**, so 2-3% is what that buys; the
losses agree to four decimals anyway, because the objective is flat at its
minimum -- which is the useful half of the finding.

**FINDING: `D/N = 20` happens at `C = 7.6e16`.** The lesson prints "Hoffmann 2022
published D/N = 20 as the headline" beside constants that give **62.6** at 1e22
and **97.7** at 1e24. The ratio scales as `C^0.0968` exactly, so it passes 20 five
to six orders of magnitude *below* the 1e22-1e23 the lesson says Chinchilla
studied. The headline and the fitted constants are the same paper's two different
answers, and the lesson prints them next to each other.

**FINDING: every model in the table is under-parameterised by this law, not
over-trained.** At 1e24 FLOPs the law wants D/N = 98; GPT-3 sits at 1.7 and Llama
3 8B at 1,875. The lesson's closing line says 2026 models are "massively past
chinchilla (D/N = 20)", and against the constants printed forty lines above them
half the table is on the other side.

**CONTROL: the loss at the optimum is insensitive to getting N right.** A 2.9%
error in N moves the loss by less than 1e-4, which is why a coarse grid is
adequate for the answer and useless for the ratio.

Structure: `exact` is the closed-form optimum; `ratio_at` inverts D/N by
bisection; `models` reads the lesson's own table.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "13-scaling-laws"
BUDGETS, HEADLINE = (1e20, 1e22, 1e24), 20.0
TABLE = (("GPT-3 175B", 175e9, 300e9), ("Chinchilla 70B", 70e9, 1400e9),
         ("Llama 3 8B", 8e9, 15_000e9), ("DeepSeek-V3 active", 37e9, 14_800e9))


def exact(ref, compute):
    """The closed-form optimum: N* proportional to C^(b/(a+b)), D* to C^(a/(a+b))."""
    alpha, beta = ref.ALPHA, ref.BETA
    scale = (ref.A * alpha / (ref.B_CONST * beta)) ** (1 / (alpha + beta))
    tokens = scale * (compute / 6) ** (beta / (alpha + beta))
    return tokens, compute / (6 * tokens), ref.chinchilla_loss(tokens, compute / (6 * tokens))


def ratio_at(ref, target, low=1e12, high=1e40):
    """The compute budget at which the optimal D/N equals `target`."""
    for _ in range(300):
        mid = math.sqrt(low * high)
        params, data, _ = exact(ref, mid)
        low, high = (mid, high) if data / params < target else (low, mid)
    return math.sqrt(low * high)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    grid = {c: ref.compute_optimal(c) for c in BUDGETS}
    closed = {c: exact(ref, c) for c in BUDGETS}
    return {
        "grid": grid, "exact": closed,
        "error": {c: abs(grid[c][0] / closed[c][0] - 1) for c in BUDGETS},
        "loss_gap": max(abs(grid[c][2] - closed[c][2]) for c in BUDGETS),
        "step": 10 ** (8 / 199) - 1,
        "exponent": (ref.ALPHA - ref.BETA) / (ref.ALPHA + ref.BETA),
        "crossing": ratio_at(ref, HEADLINE),
        "models": {name: (data / params, ref.chinchilla_loss(params, data))
                   for name, params, data in TABLE},
    }


def verify(result):
    grid, closed, ratios = result["grid"], result["exact"], result["models"]
    return [
        practice.Check(
            "ANSWER: the grid is accurate to 3% in N and four decimals in loss",
            max(result["error"].values()) < 0.04 and result["loss_gap"] < 1e-4,
            "grid against closed form: " + ", ".join(
                f"C={c:.0e} N* {grid[c][0]:.2e} vs {closed[c][0]:.2e} "
                f"({result['error'][c]:.1%}), D/N {grid[c][1] / grid[c][0]:.1f} vs "
                f"{closed[c][1] / closed[c][0]:.1f}" for c in BUDGETS)
            + f". 200 points over 8 decades is a {result['step']:.1%} step in N",
        ),
        practice.Check(
            "FINDING: D/N = 20 happens at C = 7.6e16, not at 1e22",
            result["crossing"] < 1e18,
            f"the ratio scales as C^{result['exponent']:.4f} exactly, so it passes "
            f"{HEADLINE:.0f} at C = {result['crossing']:.2e} -- five to six orders of magnitude "
            f"below the 1e22-1e23 the lesson says Chinchilla studied. At 1e22 these constants "
            f"want {closed[1e22][1] / closed[1e22][0]:.1f} and at 1e24 they want "
            f"{closed[1e24][1] / closed[1e24][0]:.1f}",
        ),
        practice.Check(
            "FINDING: the headline and the constants are printed forty lines apart",
            closed[1e22][1] / closed[1e22][0] > 3 * HEADLINE,
            f"main() prints 'Hoffmann 2022 published D/N = 20 as the headline' directly beneath a "
            f"table where its own constants give {closed[1e20][1] / closed[1e20][0]:.1f} at 1e20. "
            "The lesson's next line concedes that the optimum grows with C; the size of the "
            "disagreement at the scale in question is 3x",
        ),
        practice.Check(
            "FINDING: half the real table is under-trained by this law, not over-trained",
            ratios["GPT-3 175B"][0] < HEADLINE and ratios["Llama 3 8B"][0] > 100,
            "D/N and predicted loss: " + ", ".join(
                f"{n} {r:.1f} / {l:.3f}" for n, (r, l) in ratios.items())
            + f". At 1e24 FLOPs the law wants {closed[1e24][1] / closed[1e24][0]:.0f}, so "
              "GPT-3 and Chinchilla are below the optimum and the Llamas are above it",
        ),
        practice.Check(
            "CONTROL: the loss is insensitive to getting N right",
            result["loss_gap"] < 1e-4 and max(result["error"].values()) > 0.02,
            f"a {max(result['error'].values()):.1%} error in N moves the loss by at most "
            f"{result['loss_gap']:.1e}, because the objective is flat at its minimum. That is why "
            "a coarse grid answers the loss question and cannot answer the ratio question",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
