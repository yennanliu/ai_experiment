"""Exercise 3 — KL divergence is not symmetric, and why.

    Show that KL divergence is not symmetric. Pick two distributions P and Q and
    compute D_KL(P || Q) and D_KL(Q || P). Explain why they differ.

Reading of the exercise: one unequal pair "shows" asymmetry but explains nothing.
The explanation is structural — D_KL(p‖q) = Σ p·log(p/q) weights each term by
**p**, so it penalises q being small where p is large and barely notices the
reverse. Check 4 demonstrates that directly: a q with a near-zero where p has
mass sends one direction to a huge value while the other stays small. Check 5
shows the one case where the two are equal, which is what makes the asymmetry a
property rather than an accident.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "09-information-theory"
P = [0.7, 0.2, 0.1]
Q = [0.1, 0.2, 0.7]
SKEWED = [0.98, 0.01, 0.01]              # mass where P is small
NEAR_ZERO = [1e-9, 0.2, 0.799999999]     # near-zero where P has most of its mass


def solve():
    ref = parity.load_reference(PHASE, LESSON, "information_theory")
    pairs = {
        "P vs Q (mirror images)": (P, Q),
        "P vs skewed": (P, SKEWED),
        "P vs near-zero at P's mode": (P, NEAR_ZERO),
        "P vs itself": (P, P),
    }
    rows = {}
    for label, (a, b) in pairs.items():
        forward, reverse = ref.kl_divergence(a, b), ref.kl_divergence(b, a)
        rows[label] = {"forward": forward, "reverse": reverse,
                       "ratio": forward / reverse if reverse else float("inf")}
    cross = ref.cross_entropy(P, Q)
    entropy = ref.entropy(P)
    return {"rows": rows, "identity_gap": abs(cross - entropy - rows[
        "P vs Q (mirror images)"]["forward"])}


def verify(result):
    rows = result["rows"]
    mirror = rows["P vs Q (mirror images)"]
    near_zero = rows["P vs near-zero at P's mode"]
    skewed = rows["P vs skewed"]
    self_pair = rows["P vs itself"]
    return [
        practice.Check("all four pairs computed both ways", len(rows) == 4,
                       "; ".join(f"{k}: {v['forward']:.4g} / {v['reverse']:.4g}"
                                 for k, v in rows.items())),
        practice.Check("D_KL is non-negative everywhere",
                       all(v["forward"] >= 0 and v["reverse"] >= 0 for v in rows.values()),
                       "Gibbs' inequality — no pair produces a negative divergence"),
        practice.Check("P and Q are exact mirrors, so this pair IS symmetric",
                       abs(mirror["forward"] - mirror["reverse"]) < 1e-12,
                       f"{mirror['forward']:.6f} both ways — reversing a palindromic pair "
                       f"maps it to itself, which is why one example proves nothing"),
        practice.Check("…while a q with near-zero mass at p's mode is wildly asymmetric",
                       near_zero["ratio"] > 5,
                       f"D_KL(P‖Q) = {near_zero['forward']:.4g} vs "
                       f"D_KL(Q‖P) = {near_zero['reverse']:.4g}, "
                       f"{near_zero['ratio']:.0f}x apart — each term is weighted by the "
                       f"*first* argument, so q≈0 where p is large is catastrophic and the "
                       f"reverse is cheap"),
        practice.Check("D_KL(p‖p) = 0, and D_KL(p‖q) = H(p,q) − H(p)",
                       self_pair["forward"] == 0.0 and result["identity_gap"] < 1e-12,
                       f"self-divergence {self_pair['forward']}; and cross-entropy minus "
                       f"entropy reproduces the divergence to "
                       f"{result['identity_gap']:.3g} — the asymmetry is inherited from "
                       f"cross-entropy, since only H(p) is subtracted, never H(q). "
                       f"Skewed pair for contrast: {skewed['ratio']:.1f}x"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
