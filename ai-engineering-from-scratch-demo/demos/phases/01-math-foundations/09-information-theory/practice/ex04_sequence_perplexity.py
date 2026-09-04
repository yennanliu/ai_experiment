"""Exercise 4 — perplexity of a sequence of (true token, logits) pairs.

    Build a function that computes perplexity for a sequence of token
    predictions. Given a list of (true_token_index, predicted_logits) pairs,
    return the perplexity of the sequence.

Reading of the exercise: perplexity is exp of the *mean* cross-entropy, not of
the sum, and getting that wrong is the standard error — it makes perplexity grow
with sequence length. Check 3 tests the property that catches it: perplexity must
be invariant to repeating the sequence. Check 4 pins the scale by construction —
a uniform model over V tokens has perplexity exactly V, which is what makes the
number interpretable as "effective branching factor".
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "09-information-theory"
VOCAB = 8


def sequence_perplexity(ref, pairs):
    """exp(mean cross-entropy over the sequence) — the mean is the whole point."""
    if not pairs:
        raise ValueError("perplexity of an empty sequence is undefined")
    total = sum(ref.cross_entropy_loss(target, logits) for target, logits in pairs)
    return math.exp(total / len(pairs))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "information_theory")
    confident = [(0, [8.0] + [0.0] * (VOCAB - 1)), (1, [0.0, 8.0] + [0.0] * (VOCAB - 2))]
    uniform = [(i % VOCAB, [0.0] * VOCAB) for i in range(12)]
    mixed = confident + uniform[:4]
    wrong = [(1, [8.0] + [0.0] * (VOCAB - 1))] * 4
    return {
        "confident": sequence_perplexity(ref, confident),
        "uniform": sequence_perplexity(ref, uniform),
        "uniform_doubled": sequence_perplexity(ref, uniform * 2),
        "mixed": sequence_perplexity(ref, mixed),
        "wrong": sequence_perplexity(ref, wrong),
        # entropy() takes a numeric base; perplexity() takes the string "e"
        "lesson_uniform": ref.perplexity(ref.entropy([1 / VOCAB] * VOCAB, base=math.e),
                                         base="e"),
        "n_uniform": len(uniform),
    }


def verify(result):
    return [
        practice.Check("a confident, correct model has perplexity near 1",
                       result["confident"] < 1.01,
                       f"{result['confident']:.6f} — perplexity 1 means no uncertainty at all, "
                       f"and it is the floor"),
        practice.Check(f"a uniform model over {VOCAB} tokens has perplexity exactly {VOCAB}",
                       abs(result["uniform"] - VOCAB) < 1e-9,
                       f"{result['uniform']:.9f} — which is what licenses reading perplexity "
                       f"as an effective branching factor"),
        practice.Check("perplexity is invariant to repeating the sequence",
                       abs(result["uniform"] - result["uniform_doubled"]) < 1e-9,
                       f"{result['n_uniform']} tokens → {result['uniform']:.6f}, "
                       f"{2 * result['n_uniform']} tokens → "
                       f"{result['uniform_doubled']:.6f}. This is the check that catches "
                       f"exp(sum) instead of exp(mean), which would have doubled the exponent"),
        practice.Check("the lesson's own perplexity() agrees on the uniform case",
                       abs(result["lesson_uniform"] - VOCAB) < 1e-9,
                       f"perplexity(H(uniform)) = {result['lesson_uniform']:.9f}"),
        practice.Check("a confidently wrong model is worse than knowing nothing",
                       result["wrong"] > VOCAB,
                       f"confidently wrong → {result['wrong']:.2f}, against {VOCAB} for a "
                       f"uniform guess and {result['mixed']:.3f} for the mixed sequence — "
                       f"perplexity is unbounded above, so confidence costs more than "
                       f"ignorance when it is misplaced"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
