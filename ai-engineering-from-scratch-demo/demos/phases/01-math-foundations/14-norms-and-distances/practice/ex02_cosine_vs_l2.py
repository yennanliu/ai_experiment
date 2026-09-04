"""Exercise 2 — high cosine with large L2, and low cosine with small L2.

    Create two vectors where cosine similarity is high (> 0.9) but L2 distance is
    large (> 10). Explain geometrically what is happening. Then create two
    vectors where cosine similarity is low (< 0.3) but L2 distance is small
    (< 0.5).

Reading of the exercise: both constructions are trivial once the mechanism is
named, so the checks assert the mechanism rather than only the two examples.
Cosine sees only direction, so scaling one vector moves L2 without touching
cosine at all (check 3) — that is the first case. The second needs both vectors
*short*: L2 is bounded by |a| + |b|, so two near-orthogonal vectors of length
0.2 cannot be more than 0.4 apart however different their directions (check 5).
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "14-norms-and-distances"
HIGH_COS, LARGE_L2 = 0.9, 10.0
LOW_COS, SMALL_L2 = 0.3, 0.5

# same direction, wildly different magnitude
A1, B1 = (1.0, 1.0, 1.0), (12.0, 12.0, 11.0)
# near-orthogonal, but both tiny
A2, B2 = (0.2, 0.02, 0.0), (0.02, 0.2, 0.0)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "distances")
    rows = {}
    for label, (a, b) in {"high cosine, large L2": (A1, B1),
                          "low cosine, small L2": (A2, B2)}.items():
        rows[label] = {"cos": ref.cosine_similarity(a, b), "l2": ref.l2_distance(a, b),
                       "norms": (ref.l2_norm(a), ref.l2_norm(b))}
    # scaling one vector: L2 moves, cosine does not
    scaling = []
    for factor in (1, 5, 20, 100):
        scaled = tuple(v * factor for v in A1)
        scaling.append((factor, ref.cosine_similarity(A1, scaled),
                        ref.l2_distance(A1, scaled)))
    triangle = ref.l2_norm(A2) + ref.l2_norm(B2)
    return {"rows": rows, "scaling": scaling, "triangle_bound": triangle,
            "angle": math.degrees(math.acos(
                max(-1.0, min(1.0, rows["low cosine, small L2"]["cos"]))))}


def verify(result):
    high = result["rows"]["high cosine, large L2"]
    low = result["rows"]["low cosine, small L2"]
    cosines = [c for _, c, _ in result["scaling"]]
    distances = [d for _, _, d in result["scaling"]]
    return [
        practice.Check(f"case 1: cosine {high['cos']:.4f} > {HIGH_COS}, "
                       f"L2 {high['l2']:.2f} > {LARGE_L2}",
                       high["cos"] > HIGH_COS and high["l2"] > LARGE_L2,
                       f"{A1} vs {B1} — nearly the same direction, magnitudes "
                       f"{high['norms'][0]:.3f} and {high['norms'][1]:.3f}"),
        practice.Check(f"case 2: cosine {low['cos']:.4f} < {LOW_COS}, "
                       f"L2 {low['l2']:.4f} < {SMALL_L2}",
                       low["cos"] < LOW_COS and low["l2"] < SMALL_L2,
                       f"{A2} vs {B2} — {result['angle']:.1f}° apart, but both only "
                       f"~0.2 long"),
        practice.Check("MECHANISM: scaling one vector moves L2 and leaves cosine untouched",
                       max(cosines) - min(cosines) < 1e-12 and max(distances) > 100,
                       "; ".join(f"×{f}: cos {c:.12f}, L2 {d:.2f}"
                                 for f, c, d in result["scaling"])
                       + " — cosine normalises magnitude away, so it cannot see this at all"),
        practice.Check("…which is why case 1 needs no cleverness, only a scale factor",
                       high["norms"][1] / high["norms"][0] > 5,
                       f"|b|/|a| = {high['norms'][1] / high['norms'][0]:.2f}. Any two "
                       f"collinear vectors of sufficiently different length satisfy the "
                       f"exercise's first condition"),
        practice.Check("MECHANISM: L2 ≤ |a| + |b| bounds case 2 from above",
                       low["l2"] <= result["triangle_bound"] + 1e-12
                       and result["triangle_bound"] < SMALL_L2,
                       f"|a| + |b| = {result['triangle_bound']:.4f} < {SMALL_L2}, so no "
                       f"pair of vectors this short can be far apart whatever their "
                       f"directions. Case 2 is about *length*, not angle — which is the "
                       f"geometric answer the exercise asks for"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
