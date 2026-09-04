"""Exercise 4 — MAP with a Beta(2,2) prior against the MLE, 7 heads in 10.

    **MAP by hand.** Given observed data (7 heads in 10 coin flips), compute the
    MAP estimate of the bias using a Beta(2,2) prior. Compare it to the MLE
    estimate (7/10).

Reading of the exercise: "by hand" means the closed form, derived here rather
than searched for — the Beta posterior is Beta(α+h, β+t), whose mode is
(α+h−1)/(α+β+n−2). The comparison is the point, so check 4 makes it quantitative
instead of observational: the MAP is a weighted average of the MLE and the prior
mean, with weights that are exactly the counts, and check 5 shows the pull
vanishing as data accumulates.

By hand: Beta(2,2) + 7 heads, 3 tails → Beta(9,5); mode = 8/12 = 0.6667.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "07-bayes-theorem"
ALPHA, BETA = 2, 2
HEADS, TAILS = 7, 3
TOL = 1e-12


def map_estimate(alpha, beta, heads, tails):
    """Mode of Beta(α+h, β+t); defined when both posterior parameters exceed 1."""
    a, b = alpha + heads, beta + tails
    if a <= 1 or b <= 1:
        raise ValueError(f"Beta({a},{b}) has no interior mode")
    return (a - 1) / (a + b - 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "bayes")
    posterior = ref.beta_update(ALPHA, BETA, HEADS, TAILS)
    estimate = map_estimate(ALPHA, BETA, HEADS, TAILS)
    mle = HEADS / (HEADS + TAILS)
    prior_mean = ALPHA / (ALPHA + BETA)
    posterior_mean = posterior[0] / (posterior[0] + posterior[1])
    scaled = {}
    for factor in (1, 10, 100, 1000):
        scaled[factor] = map_estimate(ALPHA, BETA, HEADS * factor, TAILS * factor)
    # a uniform Beta(1,1) prior should reproduce the MLE exactly
    flat = map_estimate(1, 1, HEADS, TAILS)
    return {"posterior": posterior, "map": estimate, "mle": mle,
            "prior_mean": prior_mean, "posterior_mean": posterior_mean,
            "scaled": scaled, "flat": flat}


def verify(result):
    n = HEADS + TAILS
    # MAP = (n/(n+α+β−2))·MLE + ((α+β−2)/(n+α+β−2))·prior_mean
    pseudo = ALPHA + BETA - 2
    blended = (n * result["mle"] + pseudo * result["prior_mean"]) / (n + pseudo)
    return [
        practice.Check("the posterior is Beta(9, 5)",
                       tuple(result["posterior"]) == (9, 5),
                       f"beta_update({ALPHA}, {BETA}, {HEADS}, {TAILS}) -> "
                       f"{tuple(result['posterior'])}"),
        practice.Check("MAP = 8/12 = 0.6667, below the MLE of 0.7",
                       abs(result["map"] - 8 / 12) <= TOL and result["map"] < result["mle"],
                       f"MAP {result['map']:.6f} vs MLE {result['mle']:.6f} — the prior pulls "
                       f"toward 0.5 by {result['mle'] - result['map']:.6f}"),
        practice.Check("a uniform Beta(1,1) prior reproduces the MLE exactly",
                       abs(result["flat"] - result["mle"]) <= TOL,
                       f"MAP under Beta(1,1) = {result['flat']:.6f} — MLE is the MAP of a "
                       f"flat prior, which is why the two coincide so often"),
        practice.Check("MAP is exactly the count-weighted blend of MLE and prior mean",
                       abs(result["map"] - blended) <= TOL,
                       f"({n}·{result['mle']:.4f} + {pseudo}·{result['prior_mean']}) / "
                       f"{n + pseudo} = {blended:.6f} — Beta(2,2) is worth exactly "
                       f"{pseudo} pseudo-observations"),
        practice.Check("the prior's influence vanishes as data accumulates",
                       abs(result["scaled"][1000] - result["mle"])
                       < abs(result["scaled"][1] - result["mle"]) / 100,
                       ", ".join(f"n={10 * f}: {result['scaled'][f]:.6f}"
                                 for f in (1, 10, 100, 1000))
                       + f" → MLE {result['mle']}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
