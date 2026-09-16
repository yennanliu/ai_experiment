"""Exercise 1 — L_1 falls 0.11 nats and L_2 rises 0.13, because the backbone carries the wrong token.

    Run `code/main.py`. Show the per-depth loss decreases monotonically as the
    synthetic signal strengthens. Modify the synthetic to use a fixed pattern
    and verify both depth-1 and depth-2 losses converge.

Reading of the exercise: the sweep is the one `main` runs -- the same
`backbone_hidden[i] = rms_norm(E[tokens[i]] + noise)` construction, the same
modules, noise from 0.50 down to 0.01 -- extended past the lesson's 0.05 so that
"as the signal strengthens" reaches its limit. The construction is then repeated
with the backbone carrying `E[tokens[i+1]]` instead, since that is the one change
that makes the synthetic signal point at the thing being predicted.

**ANSWER: neither depth decreases monotonically, and depth 2 moves the wrong
way.**

    noise    L_1      L_2
    0.50    3.600    3.187
    0.30    3.563    3.203
    0.15    3.513    3.235
    0.05    3.489    3.287
    0.01    3.492    3.319

`ln(32) = 3.466` is the uniform-guess reference. Every number above is within
0.15 nats of it, L_1 falls by 0.11 and then turns back up, and L_2 **rises**
0.13 nats as the noise goes to zero.

**MECHANISM: the backbone hidden state is the embedding of the current token and
the target is the next one.** `mtp_loss` scores `shared_head_logits(h_i, E)`
against `tokens[i+1]`, and `h_i` is `rms_norm(E[tokens[i]] + noise)`. Removing
the noise makes `h_i` converge on `E[tokens[i]]`, so the tied head's largest
logit converges on `tokens[i]` -- the one token that is certainly not the
answer. Strengthening this signal makes the model more confident about the wrong
token.

**FINDING: shift the backbone by one and the depth-1 loss does what the exercise
says.** With `h_i = rms_norm(E[tokens[i+1]] + noise)` the sweep runs **2.877,
2.558, 2.219, 2.026, 1.988** -- monotone, and 1.48 nats below the uniform
reference. Depth 2 still rises, 3.149 to 3.229, because no amount of signal
about `t_{i+1}` is signal about `t_{i+2}`.

**FINDING: a fixed pattern makes both depths converge, and they converge on
guessing.** Replacing the random token stream with one repeated token makes both
depths fall monotonically -- L_1 from 3.209 to **2.339**, L_2 from 3.390 to
**3.311** -- so the exercise's second sentence is satisfied. They land 1.13 and
0.16 nats below the uniform reference, on a sequence with exactly one token in
it: the convergence is real and there is nothing left to predict.

Structure: `backbone` builds the lesson's own hidden states at a given noise and
token offset; `sweep` runs `mtp_loss` across the noise ladder.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "18-multi-token-prediction"
VOCAB, HIDDEN, FF, SEQ, DEPTHS, LAM = 32, 8, 16, 12, 2, 0.3
NOISES = (0.50, 0.30, 0.15, 0.05, 0.01)
SEED, NOISE_SEED = 23, 42


def setup(ref, tokens=None):
    rng = random.Random(SEED)
    embeddings = ref.rand_matrix(VOCAB, HIDDEN, rng, scale=0.2)
    stream = tokens or [rng.randrange(VOCAB) for _ in range(SEQ)]
    modules = [ref.make_mtp_module(HIDDEN, FF, rng) for _ in range(DEPTHS)]
    return embeddings, stream, modules


def backbone(ref, embeddings, tokens, noise, offset=0):
    """`main`'s own construction: the embedding of token i+offset, plus Gaussian noise."""
    rng = random.Random(NOISE_SEED)
    return [ref.rms_norm(ref.add(embeddings[tokens[min(i + offset, len(tokens) - 1)]],
                                 [rng.gauss(0, noise) for _ in range(HIDDEN)]))
            for i in range(len(tokens))]


def sweep(ref, embeddings, tokens, modules, offset=0):
    rows = {}
    for noise in NOISES:
        hidden = backbone(ref, embeddings, tokens, noise, offset)
        rows[noise] = ref.mtp_loss(hidden, tokens, modules, embeddings, LAM)[1]
    return rows


def monotone(rows, depth):
    values = [rows[noise][depth] for noise in NOISES]
    return all(a > b for a, b in zip(values, values[1:]))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    embeddings, tokens, modules = setup(ref)
    shifted = sweep(ref, embeddings, tokens, modules, offset=1)
    fixed_tokens = [tokens[0]] * SEQ
    fixed = sweep(ref, *setup(ref, fixed_tokens)[:2], modules)
    return {
        "rows": sweep(ref, embeddings, tokens, modules),
        "shifted": shifted,
        "fixed": fixed,
        "uniform": math.log(VOCAB),
        "monotone": {depth: monotone(sweep(ref, embeddings, tokens, modules), depth)
                     for depth in range(DEPTHS)},
        "shifted_monotone": {depth: monotone(shifted, depth) for depth in range(DEPTHS)},
    }


def column(rows, depth):
    return ", ".join(f"{noise} {rows[noise][depth]:.3f}" for noise in NOISES)


def verify(result):
    rows, shifted, fixed = result["rows"], result["shifted"], result["fixed"]
    quiet, loud = rows[0.01], rows[0.50]
    return [
        practice.Check(
            "ANSWER: neither depth decreases monotonically, and depth 2 moves the wrong way",
            not result["monotone"][0] and quiet[1] > loud[1],
            "as the noise falls the depth-1 loss goes " + column(rows, 0)
            + " and the depth-2 loss goes " + column(rows, 1)
            + f", against a uniform-guess reference of ln({VOCAB}) = {result['uniform']:.3f}. "
            f"Every value is within 0.15 nats of guessing, depth 1 falls "
            f"{loud[0] - min(r[0] for r in rows.values()):.2f} and then turns back up, and "
            f"depth 2 rises {quiet[1] - loud[1]:+.2f} as the signal is strengthened",
        ),
        practice.Check(
            "MECHANISM: the backbone carries the current token and the target is the next one",
            quiet[0] > result["uniform"],
            f"mtp_loss scores shared_head_logits(h_i, E) against tokens[i+1], and main builds "
            f"h_i as rms_norm(E[tokens[i]] + noise). Removing the noise makes h_i converge on "
            f"E[tokens[i]], so the tied head's largest logit converges on tokens[i] -- the one "
            f"token that is certainly not the answer. At noise 0.01 the depth-1 loss is "
            f"{quiet[0]:.3f}, above the {result['uniform']:.3f} of guessing: strengthening this "
            "signal makes the model more confident about the wrong token",
        ),
        practice.Check(
            "FINDING: shift the backbone by one and depth 1 does what the exercise says",
            result["shifted_monotone"][0] and shifted[0.01][0] < result["uniform"] - 1,
            "with h_i = rms_norm(E[tokens[i+1]] + noise) the depth-1 sweep runs "
            + column(shifted, 0)
            + f" -- monotone, and {result['uniform'] - shifted[0.01][0]:.2f} nats below the "
            "uniform reference. Depth 2 still rises, " + column(shifted, 1)
            + ", because no amount of signal about t_(i+1) is signal about t_(i+2)",
        ),
        practice.Check(
            "FINDING: a fixed pattern makes both depths converge, and they converge on guessing",
            monotone(fixed, 0) and monotone(fixed, 1),
            "replacing the random token stream with one repeated token makes both depths fall "
            "monotonically -- " + column(fixed, 0) + " at depth 1 and " + column(fixed, 1)
            + f" at depth 2 -- so the exercise's second sentence is satisfied. They land "
            f"{result['uniform'] - fixed[0.01][0]:.2f} and "
            f"{result['uniform'] - fixed[0.01][1]:.2f} nats below the uniform reference on a "
            "sequence with exactly one distinct token in it: the convergence is real and there "
            "is nothing left to predict",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
