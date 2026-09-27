"""Exercise 5 — DP and equalized odds coincide only if A does not cause Y; with the edge in, equalized odds leaves a 0.09 DP gap.

    The ICLR 2024 reconciliation argues group and counterfactual fairness are
    facets of the same structure. Pick two of the three criteria in
    `code/main.py` and state the causal assumption that would make them
    equivalent.

Reading of the exercise: the two criteria are demographic parity and
equalized odds, as `main()` computes them. A causal assumption is an edge
kept or deleted in `gen()`'s DAG (A -> Y, Y -> x0, A -> x1, A fed to the
model), so each claim is tested by deleting the edge and re-measuring with the
reference's own metric functions on 100,000 people drawn from the structural
equations. Two classifiers: `main()`'s seeded baseline (reads x0, x1 and A)
and one trained by the reference's `train()` on x0 alone.

**ANSWER: the assumption is that A has no causal path into Y (no A -> Y edge,
no common cause), plus the model's inputs depending on A only through Y.**
Per group, P(Yhat=1) = pi * TPR + (1 - pi) * FPR; with one base rate pi the DP
gap is pi * dTPR + (1 - pi) * dFPR, so equalized odds forces DP. With the
edge cut, the x0-only classifier's gaps are DP 0.002, TPR -0.004, FPR +0.005,
PPV -0.003, and it flips 0.0% of decisions when A is counterfactually changed:
parity, equalized odds, equal PPV and counterfactual fairness hold together,
which is the reconciliation's point.

**FINDING: with the lesson's A -> Y edge, equalized odds and DP are
incompatible.** The same x0-only classifier satisfies equalized odds (TPR gap
-0.006, FPR gap +0.007) but has a DP gap of 0.094, against the predicted
dpi * (TPR - FPR) = 0.095; it also flips 9.3% of decisions under
counterfactual A. Closing that gap means TPR = FPR, a classifier that ignores
x0.

**FINDING: equal base rates are not enough, contrary to main()'s takeaway.**
`main()` prints that "equal base rates are the condition for the three
criteria to coincide". Cutting A -> Y but keeping the baseline classifier, which
reads A and x1, leaves DP 0.423, TPR 0.473, FPR 0.381, PPV -0.152; the
decomposition pi * dTPR + (1 - pi) * dFPR = 0.423 matches. The model's own
path from A has to be cut too.

Structure: `scm()`/`world()` redraw gen()'s DAG with its noise kept (cut=True
deletes A -> Y); `gaps()` runs the reference's three metric functions;
`flip_rate()` is the counterfactual test from exercise 3.
"""

from __future__ import annotations

import inspect
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "21-fairness-criteria-group-individual-counterfactual"
POP, SHARED = 100_000, 0.45


def scm(n, rng):
    """gen()'s draws in gen()'s order, kept as exogenous noise."""
    people = []
    for _ in range(n):
        a, u = rng.choice([0, 1]), rng.random()
        y = 1 if u < (0.3 if a == 0 else 0.6) else 0
        x0, x1 = rng.gauss(0.8 * y, 1.0), rng.gauss(-0.3 + a * 0.5, 1.0)
        people.append((a, u, x0 - 0.8 * y, x1 - (-0.3 + 0.5 * a)))
    return people


def world(p, a, cut=False):
    y = 1 if p[1] < (SHARED if cut else (0.3 if a == 0 else 0.6)) else 0
    return [0.8 * y + p[2], -0.3 + 0.5 * a + p[3], float(a)], y, a


def x0_only(x):
    return [x[0], 0.0, 0.0]


def models(ref):
    """main()'s seeded baseline, and the reference's train() on x0 alone."""
    saved, ref.random = ref.random, random.Random(53)
    try:
        baseline = ref.train((ref.gen(1000), ref.gen(500))[0])
        ref.random = random.Random(1)
        data = [(x0_only(x), y, a) for x, y, a in (world(p, p[0]) for p in scm(1000, ref.random))]
        return {"baseline": (baseline, list), "x0": (ref.train(data), x0_only)}
    finally:
        ref.random = saved


def gaps(ref, model, view, pop, cut):
    rows = [world(p, p[0], cut) for p in pop]
    preds = ref.predict(model, [(view(x), y, a) for x, y, a in rows])
    dp, eo, cu = ref.demographic_parity(preds), ref.equalized_odds(preds), ref.conditional_use(preds)
    pi = [sum(y for _, y, a in rows if a == g) / sum(1 for *_, a in rows if a == g) for g in (0, 1)]
    return {"dp": dp[1] - dp[0], "tpr": eo[1][0] - eo[0][0], "fpr": eo[1][1] - eo[0][1],
            "ppv": cu[1][0] - cu[0][0], "pi": pi, "rates": eo}


def flip_rate(model, view, pop, cut):
    def decide(x):
        return model[3] + sum(w * v for w, v in zip(model[:3], view(x))) > 0
    return round(sum(decide(world(p, p[0], cut)[0]) != decide(world(p, 1 - p[0], cut)[0])
                     for p in pop) / len(pop), 3)


def predicted_dp(g):
    """P(Yhat=1 | a) = pi_a TPR_a + (1 - pi_a) FPR_a, differenced across groups."""
    (t0, f0), (t1, f1) = g["rates"]
    return g["pi"][1] * t1 + (1 - g["pi"][1]) * f1 - g["pi"][0] * t0 - (1 - g["pi"][0]) * f0


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pop, fitted = scm(POP, random.Random(11)), models(ref)
    out = {"takeaway": " ".join(inspect.getsource(ref.main).split())}
    for name, (model, view) in fitted.items():
        for cut in (False, True):
            g = gaps(ref, model, view, pop, cut)
            out[name, cut] = {k: round(g[k], 3) for k in ("dp", "tpr", "fpr", "ppv")}
            out[name, cut]["pred"] = round(predicted_dp(g), 3)
            out[name, cut]["flip"] = flip_rate(model, view, pop, cut)
    rates = gaps(ref, *fitted["x0"], pop, False)
    out["dpi_gap"] = round((rates["pi"][1] - rates["pi"][0]) * (rates["rates"][0][0] - rates["rates"][0][1]), 3)
    return out


def verify(result):
    cut, edge, base = result["x0", True], result["x0", False], result["baseline", True]
    return [
        practice.Check(
            "ANSWER: no A -> Y path (and inputs reaching A only via Y) makes DP and EO coincide",
            (cut["dp"], cut["tpr"], cut["fpr"], cut["ppv"], cut["flip"]) == (0.002, -0.004, 0.005, -0.003, 0.0),
            f"x0-only classifier, A -> Y cut: {cut}",
        ),
        practice.Check(
            "FINDING: with the lesson's A -> Y edge, equalized odds and DP are incompatible",
            (edge["tpr"], edge["fpr"], edge["dp"], result["dpi_gap"], edge["flip"])
            == (-0.006, 0.007, 0.094, 0.095, 0.093),
            f"x0-only classifier on the lesson's DAG: {edge}; dpi * (TPR - FPR) = {result['dpi_gap']}",
        ),
        practice.Check(
            "FINDING: equal base rates are not enough, contrary to main()'s takeaway",
            "equal base rates are the condition for the three criteria" in result["takeaway"]
            and (base["dp"], base["tpr"], base["fpr"], base["ppv"], base["pred"])
            == (0.423, 0.473, 0.381, -0.152, 0.423),
            f"baseline classifier with A -> Y cut: {base}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
