"""Exercise 3 — the lesson's DAG has A cause Y, so a predictor on x0 and x1 - 0.5A still flips 8%, and 0% once that edge is cut.

    Read Kusner et al. 2017. Construct a simple two-feature causal DAG for
    resume scoring and identify the counterfactual-fairness condition it
    implies.

Reading of the exercise: `gen()` already is a two-feature structural causal
model, so it is used as the resume DAG rather than inventing a new one: A is
the protected group, Y "qualified" (base rate 0.3 vs 0.6, so A -> Y), x0 a
work-sample score (Y -> x0) and x1 a proxy such as school prestige (A -> x1),
with A itself fed to the classifier. `scm()` redraws it keeping every
exogenous noise term (U_Y, e0, e1), and a check confirms it reproduces
`gen()` bit for bit on the same seed. Kusner's condition is then measured
directly: flip A for the same person (same noise) and count changed decisions,
on 20,000 people.

**ANSWER: the condition is that the score may depend only on
non-descendants of A. In this DAG that means x0 and the abducted residual
x1 - 0.5*A, never A or raw x1, and only if Y is not a descendant of A.** A
predictor trained on (x0, x1 - 0.5A) flips 0.0% of decisions when the
A -> Y edge is cut (one base rate, 0.45) and is 64.2% accurate.

**FINDING: the lesson's own DAG makes the label itself counterfactually
unfair.** Because A -> Y, 30.1% of people change qualification when A is
flipped, x0 inherits that, and the same (x0, x1 - 0.5A) predictor flips 8.2%
of decisions. Strictly, the only non-descendant left is e1 (x1 - 0.5A + 0.3),
which is independent of Y, so the best CF-fair score is the majority guess:
55.0% accurate, against the baseline's 69.9%.

**FINDING: demographic parity is not counterfactual fairness.** Under the
lesson's DAG the baseline flips 48.2% of decisions; the DP-reweighted model,
whose DP gap is 0.012 in exercise 1, still flips 9.6%.

Structure: `scm()` draws people as noise vectors; `world()` evaluates the
structural equations for any A; `rates()` is Kusner's test; models come
from the reference's own `train()` on seeded data (A-input set to 0 where the
predictor must not read it, which keeps its weight at 0).
"""

from __future__ import annotations

import contextlib
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "21-fairness-criteria-group-individual-counterfactual"
POP, SHARED = 20000, 0.45


def scm(n, rng):
    """gen()'s draws in gen()'s order, kept as exogenous noise."""
    people = []
    for _ in range(n):
        a, u = rng.choice([0, 1]), rng.random()
        y = 1 if u < (0.3 if a == 0 else 0.6) else 0
        x0, x1 = rng.gauss(0.8 * y, 1.0), rng.gauss(-0.3 + a * 0.5, 1.0)
        people.append({"a": a, "u": u, "e0": x0 - 0.8 * y, "e1": x1 - (-0.3 + 0.5 * a)})
    return people


def world(p, a, cut=False):
    """The structural equations with A set to `a`; cut=True removes the A -> Y edge."""
    y = 1 if p["u"] < (SHARED if cut else (0.3 if a == 0 else 0.6)) else 0
    return [0.8 * y + p["e0"], -0.3 + 0.5 * a + p["e1"], float(a)], y, a


def cf_input(x):
    """What Kusner's condition allows: x0 and the abducted residual of x1; no A."""
    return [x[0], x[1] - 0.5 * x[2], 0.0]


def decide(model, x):
    return int(model[3] + sum(w * v for w, v in zip(model[:3], x)) > 0)


def rates(model, pop, cut=False, view=list):
    """(share of decisions that flip when A is flipped for the same person, accuracy)."""
    flips = hits = 0
    for p in pop:
        (x, y, _), (x_cf, _, _) = world(p, p["a"], cut), world(p, 1 - p["a"], cut)
        flips += decide(model, view(x)) != decide(model, view(x_cf))
        hits += decide(model, view(x)) == y
    return round(flips / len(pop), 3), round(hits / len(pop), 3)


@contextlib.contextmanager
def seeded(ref, seed):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        yield ref.random
    finally:
        ref.random = saved


def fit(ref, seed, cut=False, view=list):
    with seeded(ref, seed) as rng:
        return ref.train([(view(x), y, a) for x, y, a in (world(p, p["a"], cut) for p in scm(1000, rng))])


def lesson_models(ref, seed=53):
    """main()'s baseline and DP-reweighted models, on its shipped seed."""
    with seeded(ref, seed):
        train, _ = ref.gen(1000), ref.gen(500)  # main() draws the test set before training
        weights = [{(0, 1): 2.0, (1, 1): 0.5}.get((a, y), 1.0) for _, y, a in train]
        return ref.train(train), ref.train(train, sample_weights=weights)


def mirrors_gen(ref, seed=53):
    with seeded(ref, seed):
        theirs = ref.gen(500)
    return theirs == [world(p, p["a"]) for p in scm(500, random.Random(seed))]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pop = scm(POP, random.Random(7))
    base, rew = lesson_models(ref)
    cf = {cut: rates(fit(ref, 1, cut, cf_input), pop, cut, cf_input) for cut in (False, True)}
    return {
        "mirror": mirrors_gen(ref),
        "cf_flip": {c: r[0] for c, r in cf.items()}, "cf_acc": {c: r[1] for c, r in cf.items()},
        "y_flip": round(sum(world(p, 0)[1] != world(p, 1)[1] for p in pop) / POP, 3),
        "majority": round(1 - sum(world(p, p["a"])[1] for p in pop) / POP, 3),
        "base": rates(base, pop), "rew": rates(rew, pop)[0],
    }


def verify(result):
    flip, acc = result["cf_flip"], result["cf_acc"]
    return [
        practice.Check(
            "ANSWER: a predictor on (x0, x1 - 0.5A) flips 0.0% once A -> Y is cut",
            result["mirror"] and flip[True] == 0.0 and acc[True] == 0.642,
            f"scm() reproduces gen() exactly: {result['mirror']}; with one base rate the CF "
            f"predictor flips {flip[True]:.1%} of {POP} decisions at {acc[True]:.1%} accuracy",
        ),
        practice.Check(
            "FINDING: the lesson's own DAG makes the label itself counterfactually unfair",
            result["y_flip"] == 0.301 and flip[False] == 0.082 and result["majority"] == 0.55
            and result["base"][1] == 0.699,
            f"{result['y_flip']:.1%} change Y when A flips; the same predictor flips "
            f"{flip[False]:.1%}; majority guess {result['majority']:.1%} vs baseline "
            f"{result['base'][1]:.1%} accurate",
        ),
        practice.Check(
            "FINDING: demographic parity is not counterfactual fairness",
            result["base"][0] == 0.482 and result["rew"] == 0.096,
            f"flip rate: baseline {result['base'][0]:.1%}, DP-reweighted {result['rew']:.1%}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
