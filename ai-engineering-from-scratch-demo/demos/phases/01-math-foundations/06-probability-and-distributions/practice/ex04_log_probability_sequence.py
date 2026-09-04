"""Exercise 4 — most likely sequence, total log prob, and the raw probability.

    Write a function that takes a list of log probabilities and returns the most
    likely sequence, the total log probability, and the equivalent raw
    probability. Test it with a sentence of 50 words where each word has
    probability 0.01.

Reading of the exercise: the 50-word test is the point, and it is a trap laid
deliberately. 0.01^50 = 1e-100, which is representable in float64 — but only
just, and the *naive* route that multiplies probabilities underflows long before
the log route does. Check 4 finds where. "Most likely sequence" is read as
per-position argmax over candidates, which is what a list of log probabilities
per position supports.
"""

from __future__ import annotations

import math

from harness import practice

N_WORDS, PER_WORD = 50, 0.01
VOCAB = ["the", "cat", "sat"]


def decode(log_probs):
    """log_probs: one list of per-candidate log probabilities per position."""
    sequence, total = [], 0.0
    for position in log_probs:
        best = max(range(len(position)), key=lambda i: position[i])
        sequence.append(best)
        total += position[best]
    return {"sequence": sequence, "log_prob": total, "prob": math.exp(total)}


def naive_multiply(probs):
    product = 1.0
    for p in probs:
        product *= p
    return product


def solve():
    log_per_word = math.log(PER_WORD)
    # each position: the chosen word at 0.01, two distractors an order lower
    positions = [[log_per_word, log_per_word - 2.3, log_per_word - 4.6]
                 for _ in range(N_WORDS)]
    decoded = decode(positions)
    naive = naive_multiply([PER_WORD] * N_WORDS)
    # where does the naive product actually reach zero?
    underflow_at, product = None, 1.0
    for n in range(1, 400):
        product *= PER_WORD
        if product == 0.0:
            underflow_at = n
            break
    return {"decoded": decoded, "naive": naive, "underflow_at": underflow_at,
            "expected_log": N_WORDS * log_per_word,
            "words": [VOCAB[i % len(VOCAB)] for i in decoded["sequence"]]}


def verify(result):
    decoded = result["decoded"]
    return [
        practice.Check(f"a {N_WORDS}-word sequence is decoded",
                       len(decoded["sequence"]) == N_WORDS
                       and all(i == 0 for i in decoded["sequence"]),
                       f"every position picks its argmax candidate; first 5 words "
                       f"{result['words'][:5]}"),
        practice.Check("total log probability is 50·log(0.01)",
                       abs(decoded["log_prob"] - result["expected_log"]) < 1e-9,
                       f"{decoded['log_prob']:.6f} = 50 × {math.log(PER_WORD):.6f}"),
        practice.Check("raw probability is 1e-100, and survives float64",
                       abs(decoded["prob"] - 1e-100) / 1e-100 < 1e-9
                       and decoded["prob"] > 0,
                       f"exp({decoded['log_prob']:.2f}) = {decoded['prob']:.4e}"),
        practice.Check("…but multiplying probabilities directly reaches exactly 0, and soon",
                       result["underflow_at"] is not None,
                       f"0.01^n underflows to 0.0 at n = {result['underflow_at']}, where the "
                       f"log route just reads {result['underflow_at'] * math.log(PER_WORD):.0f} "
                       f"— this is why decoders sum logs instead of multiplying"),
        practice.Check("the two agree while the naive route still works",
                       abs(result["naive"] - decoded["prob"]) / decoded["prob"] < 1e-9,
                       f"at 50 words both give {result['naive']:.4e}; the difference is "
                       f"only that one of them keeps working"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
