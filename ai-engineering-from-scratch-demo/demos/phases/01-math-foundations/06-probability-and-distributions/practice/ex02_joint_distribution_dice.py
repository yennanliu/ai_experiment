"""Exercise 2 — a joint table for two loaded dice; marginals and independence.

    Build a joint distribution table for two loaded dice. Compute the marginal
    distributions and check whether the dice are independent.

Reading of the exercise: "check whether" is a question, not an assertion, so the
answer has to be able to come out either way. Two tables are built from the same
loaded marginals — one independent by construction (the outer product), one with
a correlation deliberately introduced — and the same check is run on both. A
check that only ever sees the independent case cannot distinguish "these are
independent" from "my test always says yes".
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "06-probability-and-distributions"
FACES = 6
LOADED_A = [0.30, 0.10, 0.10, 0.10, 0.10, 0.30]      # weighted to 1 and 6
LOADED_B = [0.05, 0.15, 0.20, 0.20, 0.15, 0.25]


def outer(a, b):
    return [[pa * pb for pb in b] for pa in a]


def correlated(a, b, strength=0.5):
    """Same marginals, but mass moved onto the diagonal — so P(x,y) ≠ P(x)P(y)."""
    table = outer(a, b)
    for i in range(FACES):
        for j in range(FACES):
            table[i][j] *= (1 + strength) if i == j else 1.0
    total = sum(sum(row) for row in table)
    return [[value / total for value in row] for row in table]


def _worst(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "probability")
    tables = {"independent (outer product)": outer(LOADED_A, LOADED_B),
              "correlated (mass on the diagonal)": correlated(LOADED_A, LOADED_B)}
    rows = {}
    for label, table in tables.items():
        margin_x, margin_y = ref.joint_to_marginals(table)
        independent = ref.check_independence(table, margin_x, margin_y)
        worst = max(abs(table[i][j] - margin_x[i] * margin_y[j])
                    for i in range(FACES) for j in range(FACES))
        rows[label] = {"margin_x": margin_x, "margin_y": margin_y,
                       "independent": independent, "worst": worst,
                       "total": sum(sum(r) for r in table)}
    return {"rows": rows}


def verify(result):
    ind = result["rows"]["independent (outer product)"]
    cor = result["rows"]["correlated (mass on the diagonal)"]
    return [
        practice.Check("both joint tables are valid distributions (sum to 1)",
                       all(abs(r["total"] - 1.0) < 1e-12 for r in result["rows"].values()),
                       f"totals {[round(r['total'], 12) for r in result['rows'].values()]}"),
        practice.Check("the independent table's marginals recover the loaded dice",
                       _worst(ind["margin_x"], LOADED_A) < 1e-12
                       and _worst(ind["margin_y"], LOADED_B) < 1e-12,
                       f"P(A) {[round(p, 3) for p in ind['margin_x']]} — loaded toward 1 and 6"),
        practice.Check("…and it is judged independent", ind["independent"],
                       f"worst |P(x,y) − P(x)P(y)| = {ind['worst']:.3g}"),
        practice.Check("the correlated table is judged dependent",
                       not cor["independent"],
                       f"worst |P(x,y) − P(x)P(y)| = {cor['worst']:.4f} — "
                       f"{cor['worst'] / max(ind['worst'], 1e-18):.0e}x the independent case"),
        practice.Check("…even though its marginals are nearly the same dice",
                       _worst(cor["margin_x"], ind["margin_x"]) < 0.05,
                       f"P(A) shifts by at most "
                       f"{_worst(cor['margin_x'], ind['margin_x']):.4f} "
                       f"— marginals alone cannot tell you about dependence, which is the "
                       f"reason the joint table has to be built at all"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
