"""Exercise 1 — entropy of the alphabet: uniform vs real letter frequencies.

    Compute the entropy of the English alphabet assuming uniform distribution
    (26 letters). Then estimate it using actual letter frequencies. Which is
    higher and why?

Reading of the exercise: "which is higher and why" is the question, and the
answer is a theorem rather than an observation — the uniform distribution
maximises entropy over a fixed support, so uniform must be higher for *any*
non-uniform frequency table. Check 3 tests that as the general claim it is, by
also measuring the gap and identifying it as the KL divergence from the real
frequencies to uniform, which is what the difference literally equals.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "09-information-theory"
N = 26
# Relative frequencies in English text, normalised below. Source: the standard
# Cornell/Lewand ordering, which the lesson's own text also uses.
FREQUENCIES = {
    "e": 12.02, "t": 9.10, "a": 8.12, "o": 7.68, "i": 7.31, "n": 6.95, "s": 6.28,
    "r": 6.02, "h": 5.92, "d": 4.32, "l": 3.98, "u": 2.88, "c": 2.71, "m": 2.61,
    "f": 2.30, "y": 2.11, "w": 2.09, "g": 2.03, "p": 1.82, "b": 1.49, "v": 1.11,
    "k": 0.69, "x": 0.17, "q": 0.11, "j": 0.10, "z": 0.07,
}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "information_theory")
    total = sum(FREQUENCIES.values())
    real = [v / total for v in FREQUENCIES.values()]
    uniform = [1 / N] * N
    return {
        "uniform": ref.entropy(uniform),
        "real": ref.entropy(real),
        "kl_to_uniform": ref.kl_divergence(real, uniform),
        "log2_26": math.log2(N),
        "n_letters": len(FREQUENCIES),
        "sums_to_one": abs(sum(real) - 1.0),
        "most_common": max(FREQUENCIES, key=FREQUENCIES.get),
        "surprisal_e": ref.information_content(FREQUENCIES["e"] / total),
        "surprisal_z": ref.information_content(FREQUENCIES["z"] / total),
    }


def verify(result):
    gap = result["uniform"] - result["real"]
    return [
        practice.Check("all 26 letters, normalised to a distribution",
                       result["n_letters"] == N and result["sums_to_one"] < 1e-12,
                       f"{result['n_letters']} letters, Σp − 1 = {result['sums_to_one']:.3g}"),
        practice.Check("uniform entropy is log₂26 = 4.700 bits",
                       abs(result["uniform"] - result["log2_26"]) < 1e-12,
                       f"H(uniform) = {result['uniform']:.6f} bits"),
        practice.Check("real frequencies give LESS entropy — uniform is the maximum",
                       result["real"] < result["uniform"],
                       f"H(real) = {result['real']:.6f} vs {result['uniform']:.6f} bits; "
                       f"the gap is {gap:.6f}"),
        practice.Check("…and the gap is exactly D_KL(real ‖ uniform)",
                       abs(gap - result["kl_to_uniform"]) < 1e-12,
                       f"H(uniform) − H(real) = {gap:.9f} = "
                       f"D_KL = {result['kl_to_uniform']:.9f}. Not a coincidence: for a "
                       f"uniform q, D_KL(p‖q) = log₂|X| − H(p) identically"),
        practice.Check("rare letters carry more information than common ones",
                       result["surprisal_z"] > result["surprisal_e"] * 3,
                       f"'{result['most_common']}' at {result['surprisal_e']:.2f} bits vs "
                       f"'z' at {result['surprisal_z']:.2f} bits — which is why 'z' is worth "
                       f"more in Scrabble and why entropy coding pays off at all"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
