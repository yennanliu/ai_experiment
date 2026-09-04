"""Exercise 4 — Wasserstein-1 by the CDF method, two pairs compared.

    Compute the Wasserstein distance between [0.5, 0.5, 0, 0] and [0, 0, 0.5,
    0.5] by hand using the CDF method. Then compute it between [0.25, 0.25,
    0.25, 0.25] and [0, 0, 0.5, 0.5]. Which is larger and why?

Reading of the exercise: "by hand using the CDF method" means W₁ = Σ|F_p − F_q|
over the bins, derived independently and then checked against the lesson's
function — not read off it. Both are computed below.

By hand, pair 1: F_p = [.5, 1, 1, 1], F_q = [0, 0, .5, 1], |Δ| = [.5, 1, .5, 0],
so W₁ = 2.0. Pair 2: F_p = [.25, .5, .75, 1], |Δ| = [.25, .5, .25, 0], W₁ = 1.0.

So pair 1 is larger, by exactly 2x, and the reason is transport distance: pair 1
moves all its mass two bins, pair 2 moves half its mass one bin and half two.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "14-norms-and-distances"
PAIR1 = ([0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5])
PAIR2 = ([0.25, 0.25, 0.25, 0.25], [0.0, 0.0, 0.5, 0.5])
BY_HAND = {"pair 1": 2.0, "pair 2": 1.0}


def cdf(values):
    running, out = 0.0, []
    for value in values:
        running += value
        out.append(running)
    return out


def wasserstein_by_hand(p, q):
    """W₁ = Σ |F_p(i) − F_q(i)| over the bins — the CDF method, by hand."""
    return sum(abs(a - b) for a, b in zip(cdf(p), cdf(q)))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "distances")
    rows = {}
    for label, (p, q) in {"pair 1": PAIR1, "pair 2": PAIR2}.items():
        rows[label] = {"mine": wasserstein_by_hand(p, q), "theirs": ref.wasserstein_1d(p, q),
                       "cdf_p": cdf(p), "cdf_q": cdf(q),
                       "kl": None}
    # KL for contrast: undefined on pair 1, since q has zeros where p has mass
    try:
        kl = ref.kl_divergence(PAIR1[0], PAIR1[1])
        kl_outcome = f"{kl:.4f}"
    except Exception as exc:
        kl_outcome = f"{type(exc).__name__}: {exc}"
    identical = ref.wasserstein_1d(PAIR1[0], PAIR1[0])
    return {"rows": rows, "kl_pair1": kl_outcome, "identical": identical}


def verify(result):
    rows = result["rows"]
    one, two = rows["pair 1"], rows["pair 2"]
    return [
        practice.Check("by hand, pair 1: |ΔF| = [.5, 1, .5, 0] so W₁ = 2.0",
                       abs(one["mine"] - BY_HAND["pair 1"]) < 1e-12,
                       f"F_p = {one['cdf_p']}, F_q = {one['cdf_q']} -> {one['mine']}"),
        practice.Check("by hand, pair 2: |ΔF| = [.25, .5, .25, 0] so W₁ = 1.0",
                       abs(two["mine"] - BY_HAND["pair 2"]) < 1e-12,
                       f"F_p = {two['cdf_p']} -> {two['mine']}"),
        practice.Check("the lesson's wasserstein_1d agrees on both",
                       abs(one["mine"] - one["theirs"]) < 1e-12
                       and abs(two["mine"] - two["theirs"]) < 1e-12,
                       f"wasserstein_1d -> {one['theirs']:.6f} and {two['theirs']:.6f}"),
        practice.Check("ANSWER: pair 1 is larger, by exactly 2x",
                       abs(one["mine"] / two["mine"] - 2.0) < 1e-12,
                       f"{one['mine']} vs {two['mine']}. Pair 1 moves *all* its mass two "
                       f"bins to the right; pair 2 already has half its mass at or past the "
                       f"target, so it moves half one bin and half two — half the transport "
                       f"work for the same endpoints"),
        practice.Check("…and W₁ stays finite where KL does not",
                       "Error" in result["kl_pair1"] or "error" in result["kl_pair1"]
                       or "inf" in result["kl_pair1"],
                       f"KL(p‖q) on pair 1 -> {result['kl_pair1']} — q is 0 where p has "
                       f"mass. W₁ measures how far the mass must move, so disjoint support "
                       f"is a large number rather than an undefined one. "
                       f"W₁(p, p) = {result['identical']:g}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
