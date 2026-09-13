"""Exercise 3 — a 2-layer draft against a 6-layer verifier is break-even.

    **Hard.** Implement speculative decoding: use a tiny 2-layer model as the
    draft and a 6-layer model as the verifier. Measure wall-clock speedup on 100
    completions of length 64. Confirm outputs match greedy of the verifier.

Reading of the exercise: there are no models in the lesson and nothing to time,
so the two are built as bigram tables over its 20-token vocabulary -- a verifier,
and a draft made by corrupting a controlled fraction of its rows -- and the
speedup is counted in **verifier forward passes**, which is what a wall-clock
measurement would be reporting anyway once both models ran on the same device.
The draft/verifier cost ratio the exercise fixes is `2/6 = 1/3`.

**ANSWER: outputs match verifier greedy 100 out of 100, at every agreement
level.** That is by construction rather than by luck: the loop accepts the
longest prefix on which the draft's argmax equals the verifier's, then emits the
verifier's own next token, so the emitted sequence is a verifier-greedy sequence
whatever the draft does. Correctness is independent of draft quality; only speed
is not.

**ANSWER: at the exercise's own layer ratio the best speedup is 1.40x, and at a
low acceptance rate it is a slowdown at every lookahead.** With acceptance
`alpha` and lookahead `g`, the expected tokens per verifier call is
`(1 - alpha^(g+1)) / (1 - alpha)` and the speedup is that over `1 + g*c`:

| alpha | g=1 | g=2 | g=4 | g=8 | best |
|---:|---:|---:|---:|---:|---:|
| 0.761 | 1.32x | **1.40x** | 1.34x | 1.04x | 1.40x |
| 0.531 | 1.15x | 1.09x | 0.88x | 0.58x | 1.15x |
| 0.299 | 0.97x | 0.83x | 0.61x | 0.39x | **0.97x** |

**FINDING: the cost ratio, not the acceptance rate, is what makes the exercise's
configuration marginal.** At `c = 1/3` a 4-token lookahead spends 1.33 verifier
equivalents on drafting before a single token is verified, so even perfect
acceptance caps the speedup at `(g+1)/(1 + g*c)` = **2.14x**. Drop the draft to
`c = 1/20` and the same `alpha = 0.761` gives **2.60x** at `g = 4` and 2.73x at
`g = 8`. Production systems use drafts 10-20x cheaper than the verifier; 3x is
not enough.

**CONTROL: a bigger lookahead is not a bigger win.** At `c = 1/3` the optimum is
`g = 2` and every larger lookahead is worse, because `E[tokens]` saturates at
`1/(1 - alpha) = 4.18` while the drafting cost keeps growing linearly. At the
lowest acceptance rate `g = 1` is already the best available and it is still
below 1.

Structure: `bigram` and `corrupt` build the pair; `speculate` is the decoder;
`greedy` is the reference it must reproduce; `speedup` is the closed form.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "07-gpt-causal-language-modeling"
VOCAB, LENGTH, RUNS, RATIO = 20, 64, 100, 2 / 6
AGREEMENTS, LOOKAHEADS, CHEAPER = (0.9, 0.75, 0.5), (1, 2, 4, 8), 1 / 20


def bigram(ref, seed):
    """A verifier: one row of log-probabilities per previous token, via the lesson's softmax."""
    rng = random.Random(seed)
    return [[math.log(p) for p in ref.softmax([rng.gauss(0, 1) for _ in range(VOCAB)])]
            for _ in range(VOCAB)]


def corrupt(table, seed, agreement):
    """A draft: the verifier with a share of its rows pushed onto a different argmax."""
    rng = random.Random(seed + 999)
    return [[value + (5.0 if index == rng.randrange(VOCAB) and rng.random() > agreement else 0.0)
             for index, value in enumerate(row)] for row in table]


def argmax(table, token):
    """The greedy next token after `token`."""
    return max(range(VOCAB), key=lambda i: table[token][i])


def greedy(table, start, steps=LENGTH):
    """The sequence speculative decoding has to reproduce exactly."""
    out = [start]
    for _ in range(steps):
        out.append(argmax(table, out[-1]))
    return out


def speculate(draft, verifier, start, lookahead, steps=LENGTH):
    """(sequence, accepted, proposed) -- accept the agreeing prefix, then one verifier token."""
    out, accepted, proposed = [start], 0, 0
    while len(out) - 1 < steps:
        head = out[-1]
        guesses = []
        for _ in range(lookahead):
            head = argmax(draft, head)
            guesses.append(head)
        head, taken = out[-1], 0
        for guess in guesses:
            if argmax(verifier, head) != guess:
                break
            head, taken = guess, taken + 1
        out += guesses[:taken] + [argmax(verifier, head)]
        accepted, proposed = accepted + taken, proposed + lookahead
    return out[:steps + 1], accepted, proposed


def speedup(alpha, lookahead, cost):
    """Expected tokens per verifier call, over the cost of that call plus the drafts."""
    expected = (1 - alpha ** (lookahead + 1)) / (1 - alpha)
    return expected / (1 + lookahead * cost), expected


def measure(ref, agreement, lookahead=4):
    """(identical outputs, acceptance rate) over RUNS completions of length LENGTH."""
    identical = accepted = proposed = 0
    for seed in range(RUNS):
        verifier = bigram(ref, seed)
        out, taken, tried = speculate(corrupt(verifier, seed, agreement), verifier,
                                      seed % VOCAB, lookahead)
        identical += out == greedy(verifier, seed % VOCAB)
        accepted, proposed = accepted + taken, proposed + tried
    return identical, accepted / proposed


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = {agreement: measure(ref, agreement) for agreement in AGREEMENTS}
    alphas = [alpha for _, alpha in runs.values()]
    grid = {round(alpha, 3): {g: speedup(alpha, g, RATIO)[0] for g in LOOKAHEADS}
            for alpha in alphas}
    best = alphas[0]
    tops = [max(row.values()) for row in grid.values()]
    return {
        "identical": [identical for identical, _ in runs.values()], "alphas": alphas,
        "grid": grid, "best_row": grid[round(best, 3)],
        "ceiling": speedup(1 - 1e-12, 4, RATIO)[0],
        "cheap": {g: speedup(best, g, CHEAPER)[0] for g in LOOKAHEADS},
        "saturation": 1 / (1 - best), "ratio": RATIO, "tops": tops,
    }


def verify(result):
    row, cheap = result["best_row"], result["cheap"]
    return [
        practice.Check(
            "ANSWER: outputs match verifier greedy 100/100, at every agreement level",
            result["identical"] == [RUNS] * len(AGREEMENTS),
            f"{result['identical']} identical out of {RUNS} completions of length {LENGTH}, at "
            f"acceptance rates {[round(a, 3) for a in result['alphas']]}. The loop accepts the "
            "longest prefix where the draft's argmax equals the verifier's and then emits the "
            "verifier's own next token, so correctness does not depend on draft quality at all",
        ),
        practice.Check(
            "ANSWER: at a 2-of-6 layer ratio the best speedup is 1.40x, and can be a slowdown",
            max(result["tops"]) < 1.5 and min(result["tops"]) < 1.0,
            f"with c = {RATIO:.3f}, the best lookahead per acceptance rate: "
            f"{ {round(a, 3): round(t, 2) for a, t in zip(result['alphas'], result['tops'])} }. At "
            f"alpha = {result['alphas'][0]:.3f} the row is "
            f"{ {g: round(v, 2) for g, v in row.items()} }, peaking at "
            f"{max(row.values()):.2f}x; at alpha = {result['alphas'][-1]:.3f} no lookahead reaches "
            "1.0 at all, so the exercise's own configuration can lose",
        ),
        practice.Check(
            "FINDING: the cost ratio caps it before the acceptance rate does",
            result["ceiling"] < 2.2,
            f"a 4-token lookahead at c = {RATIO:.3f} spends {4 * RATIO:.2f} verifier equivalents "
            f"drafting before one token is verified, so even perfect acceptance caps the speedup "
            f"at (g+1)/(1 + g*c) = {result['ceiling']:.2f}x. The exercise fixes the draft at 2 of "
            "6 layers, which is the number that makes its own configuration marginal",
        ),
        practice.Check(
            "FINDING: a 20x cheaper draft turns the same acceptance rate into 2.6x",
            cheap[4] > 2.0 and cheap[4] > row[4] * 1.8,
            f"at c = {CHEAPER:.3f} and the identical alpha = {result['alphas'][0]:.3f}: "
            f"{ {g: round(v, 2) for g, v in cheap.items()} }. Same draft quality, "
            f"{cheap[4] / row[4]:.1f}x the speedup at g = 4. Production systems use drafts 10-20x "
            "cheaper than the verifier, and this is the arithmetic that forces that",
        ),
        practice.Check(
            "CONTROL: a bigger lookahead is not a bigger win",
            row[8] < row[2] and result["saturation"] < 8,
            f"E[tokens per call] saturates at 1/(1 - alpha) = {result['saturation']:.2f} while "
            f"the drafting cost grows linearly in g, so at c = {RATIO:.3f} the optimum is g = "
            f"{max(row, key=row.get)} and every larger lookahead is worse: {row[2]:.2f}x at 2 "
            f"against {row[8]:.2f}x at 8. Lookahead is a parameter with an interior optimum, and "
            "the exercise names a cost ratio instead of naming it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
