"""Exercise 5 — alpha is flat under teacher forcing and decays 0.87 to 0.57 when the draft feeds itself.

    Read the EAGLE-3 paper's Section 4 (Training-Time Test). Explain in two
    sentences why naive draft training without TTT suffers from exposure bias,
    and why feeding the draft its own predictions during training fixes it.
    Connect this to the scheduled-sampling literature in seq2seq.

Reading of the exercise: the explanation is written as a measurement, because
the lesson's own code contains the assumption Section 4 is about. `spec_step`'s
docstring says "q and p are context-free distributions shared across positions.
The math extends to position-dependent q_i, p_i without changing the loop" --
and that is the claim under test. A minimal context-dependent pair is built: a
verifier and a draft whose next-token distribution depends on the previous
token, with one state where they disagree and which the draft over-produces.

**ANSWER: with the prefix drawn from the verifier, alpha is flat at 0.85; with
the draft conditioning on its own output, it falls to 0.57 by position 6.**

    position            1      2      3      4      5      6
    teacher-forced    0.871  0.849  0.848  0.849  0.848  0.849
    free-running      0.869  0.755  0.683  0.630  0.595  0.568

That is exposure bias in one table. The draft is measured on prefixes the
verifier produced and deployed on prefixes it produced itself, and the two
distributions drift apart with every token it emits.

**MECHANISM: the draft walks into the states where it is worst.** The two models
differ most on one token; the draft over-produces it and, once there, its own
transition keeps it there. The probability that the draft is conditioning on
that state goes from **0.000** at position 1 to **0.435** at position 6 under
free running, against a flat **0.030** under teacher forcing. Feeding the draft
its own predictions during training is what puts those prefixes in the training
distribution -- scheduled sampling's answer to the same problem in seq2seq.

**FINDING: the lesson's own alpha is a teacher-forced measurement.**
`measure_alpha` draws a single token from `p` and tests it, with no chain and no
conditioning, so it reports **0.849**. Feeding that into
`expected_tokens_per_verify` predicts **4.52** tokens per verifier call, against
the **3.52** the measured per-position alphas actually give -- an over-prediction
of **28%**.

**FINDING: the formula assumes alpha is constant, not just that it is known.**
`(1 - alpha^(N+1)) / (1 - alpha)` is the sum of a geometric series, which is the
right answer only if every position accepts with the same probability. Under
free running they do not, so the aside about extending to position-dependent
`q_i, p_i` "without changing the loop" is right about the loop and wrong about
the throughput model built on top of it.

Structure: `chain` builds the context-dependent verifier and draft; `alphas`
measures acceptance per position under one conditioning rule; `chained` is the
throughput formula with a per-position alpha.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "15-speculative-decoding-eagle3"
BASE = [0.30, 0.22, 0.15, 0.10, 0.08, 0.07, 0.05, 0.03]
BAD, DEPTH, TRIALS = 7, 6, 60_000
DRAFT_ROW = [0.26, 0.19, 0.13, 0.09, 0.07, 0.06, 0.04, 0.16]
STICKY_ROW = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.85]


def normalise(row):
    total = sum(row)
    return [x / total for x in row]


def chain():
    """A verifier and a draft whose rows depend on the previous token.

    The verifier leaves every state the same way. The draft over-produces `BAD`
    and, once in it, stays -- which is the only asymmetry in the construction.
    """
    verifier = [normalise(BASE) for _ in BASE]
    draft = [normalise(DRAFT_ROW) for _ in BASE]
    draft[BAD] = normalise(STICKY_ROW)
    return verifier, draft


def alphas(ref, verifier, draft, free_running):
    """Acceptance per position, conditioning on the draft's own output or the verifier's."""
    rng = random.Random(5)
    hits, in_bad = [0] * DEPTH, [0] * DEPTH
    for _ in range(TRIALS):
        previous = 0
        for position in range(DEPTH):
            q, p = verifier[previous], draft[previous]
            in_bad[position] += previous == BAD
            token = ref.sample(p, rng)
            hits[position] += rng.random() < min(1.0, q[token] / p[token])
            previous = token if free_running else ref.sample(q, rng)
    return ([h / TRIALS for h in hits], [b / TRIALS for b in in_bad])


def chained(per_position):
    """Expected tokens per verifier call when acceptance varies along the chain."""
    total, surviving = 0.0, 1.0
    for alpha in per_position:
        surviving *= alpha
        total += surviving
    return 1 + total


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    verifier, draft = chain()
    forced, forced_bad = alphas(ref, verifier, draft, free_running=False)
    free, free_bad = alphas(ref, verifier, draft, free_running=True)
    reported = ref.measure_alpha(verifier[0], draft[0], 20_000, random.Random(7))
    return {
        "forced": forced,
        "free": free,
        "bad": (forced_bad, free_bad),
        "reported": reported,
        "predicted": ref.expected_tokens_per_verify(forced[-1], DEPTH),
        "actual": chained(free),
        "constant": ref.expected_tokens_per_verify(statistics.fmean(free), DEPTH),
    }


def row(values, fmt=".3f"):
    return " ".join(format(v, fmt) for v in values)


def verify(result):
    forced, free = result["forced"], result["free"]
    forced_bad, free_bad = result["bad"]
    return [
        practice.Check(
            "ANSWER: alpha is flat at 0.85 teacher-forced and falls to 0.57 free-running",
            max(forced[1:]) - min(forced[1:]) < 0.01 and free[-1] < 0.7 < forced[-1],
            f"teacher-forced acceptance by position is {row(forced)} and free-running is "
            f"{row(free)}. That is exposure bias in one line: the draft is measured on prefixes "
            f"the verifier produced and deployed on prefixes it produced itself, and the two "
            f"drift apart with every token it emits -- a gap of "
            f"{forced[-1] - free[-1]:+.3f} by position {DEPTH}",
        ),
        practice.Check(
            "MECHANISM: the draft walks into the states where it is worst",
            free_bad[-1] > 10 * forced_bad[-1],
            f"the two models differ most on one token; the draft over-produces it and, once "
            f"there, its own transition keeps it there. The probability that the draft is "
            f"conditioning on that state is {row(free_bad)} under free running against "
            f"{row(forced_bad)} under teacher forcing. Feeding the draft its own predictions "
            "during training is what puts those prefixes into the training distribution, which "
            "is scheduled sampling's answer to the same problem in seq2seq",
        ),
        practice.Check(
            "FINDING: the lesson's own measure_alpha is a teacher-forced measurement",
            abs(result["reported"] - forced[0]) < 0.02
            and result["predicted"] > 1.2 * result["actual"],
            f"measure_alpha draws a single token from p and tests it, with no chain and no "
            f"conditioning, so it reports {result['reported']:.3f}. Feeding that into "
            f"expected_tokens_per_verify predicts {result['predicted']:.2f} tokens per verifier "
            f"call against the {result['actual']:.2f} the measured per-position alphas give -- an "
            f"over-prediction of {result['predicted'] / result['actual'] - 1:.0%}",
        ),
        practice.Check(
            "FINDING: the formula assumes alpha is constant, not merely that it is known",
            abs(result["constant"] - result["actual"]) > 0.3,
            f"(1 - alpha^(N+1)) / (1 - alpha) is the sum of a geometric series, which is right "
            f"only if every position accepts with the same probability. Even given the correct "
            f"mean of the free-running alphas it predicts {result['constant']:.2f} against the "
            f"{result['actual']:.2f} the chain actually delivers, because a decaying sequence and "
            "a constant one with the same mean are not the same product. The aside about "
            "extending to position-dependent q_i and p_i 'without changing the loop' is right "
            "about the loop and wrong about the throughput model on top of it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
