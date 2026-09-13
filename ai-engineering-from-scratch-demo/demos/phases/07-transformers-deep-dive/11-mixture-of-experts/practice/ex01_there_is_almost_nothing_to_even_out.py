"""Exercise 1 — on the lesson's own tokens there is almost nothing to even out.

    **Easy.** Run `code/main.py`. Watch how the auxiliary-loss-free bias update
    evens out expert usage over 50 iterations.

Reading of the exercise: 50 iterations are run, not `main()`'s 10, on the
lesson's own token generator and then on a clustered one, so that "evens out" has
something to be measured against. Usage comes from the lesson's own `route`,
which agrees with `run_epoch` exactly and skips the expert matmuls it does not
need.

**ANSWER: it evens out, and the lesson's tokens start 1.65x from balanced.**
`rng.gauss(0, 1)` per coordinate has no structure at all -- the comment above it
says "with some structure so routing isn't uniform" -- so the only asymmetry is
in the random router weights. Max/min expert usage is **1.65** at iteration 0 and
entropy is already **99.4%** of `ln 8`. Fifty iterations take it to 1.33.

**FINDING: give it clustered tokens and the same rule has work to do.** Draw the
tokens from 3 Gaussian clusters instead and the starting max/min is **657** --
one expert takes 657 times another's share -- with entropy 1.549 against a
ceiling of 2.079. The bias update closes **77% of the entropy gap**, to 1.959 and
a max/min of 5.24. That is the demonstration the exercise describes, and the
lesson's own generator cannot produce it.

**FINDING: forty of the fifty iterations change nothing.** The step is a fixed
+/-gamma with no proportionality and no dead band, so it cannot converge -- it
limit-cycles. Entropy from iteration 10 to 50 moves **+0.0000** on the lesson's
tokens and **-0.0249** on clustered ones, with the last thirty iterations
spanning 0.0024 and 0.0571 respectively.

**FINDING: entropy is the wrong metric for this.** At iteration 50 on clustered
tokens entropy reads 94% of maximum while the worst expert is still **227 tokens
off target** out of 250 and max/min is 5.24. Entropy saturates long before
balance does.

Structure: `usage` runs one epoch through the lesson's own `route`; `corpus`
builds either token distribution; `settle` runs the bias loop and returns the
entropy trace.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "11-mixture-of-experts"
WIDTH, EXPERTS, TOP_K, TOKENS = 16, 8, 2, 1_000
GAMMA, ITERATIONS, CLUSTERS = 0.15, 50, 3


def corpus(rng, clustered):
    """The lesson's own i.i.d. tokens, or the same count drawn from 3 clusters."""
    if not clustered:
        return [[rng.gauss(0, 1) for _ in range(WIDTH)] for _ in range(TOKENS)]
    centres = [[rng.gauss(0, 1) for _ in range(WIDTH)] for _ in range(CLUSTERS)]
    return [[c + 0.15 * rng.gauss(0, 1) for c in rng.choice(centres)] for _ in range(TOKENS)]


def usage(ref, router, tokens, bias):
    """Expert usage for one epoch, through the lesson's own route()."""
    counts = [0] * EXPERTS
    for token in tokens:
        for expert in ref.route(token, router, TOP_K, bias)[0]:
            counts[expert] += 1
    return counts


def settle(ref, router, tokens, iterations=ITERATIONS):
    """Run the bias loop; return (entropy trace, final counts, target)."""
    bias, target = [0.0] * EXPERTS, TOKENS * TOP_K / EXPERTS
    counts = usage(ref, router, tokens, bias)
    trace = [ref.entropy(counts)]
    for _ in range(iterations):
        bias = ref.update_bias(bias, counts, target, GAMMA)
        counts = usage(ref, router, tokens, bias)
        trace.append(ref.entropy(counts))
    return trace, counts, target


def shape(ref, clustered, seed=42):
    """Everything measured about one token distribution."""
    rng = random.Random(seed)
    router = [[rng.gauss(0, 0.3) for _ in range(WIDTH)] for _ in range(EXPERTS)]
    tokens = corpus(rng, clustered)
    first = usage(ref, router, tokens, [0.0] * EXPERTS)
    trace, last, target = settle(ref, router, tokens)
    return {
        "start": (ref.entropy(first), max(first) / max(1, min(first))),
        "end": (trace[-1], max(last) / max(1, min(last))),
        "closed": (trace[-1] - trace[0]) / (math.log(EXPERTS) - trace[0]),
        "late": trace[-1] - trace[10], "tail": max(trace[20:]) - min(trace[20:]),
        "worst": max(abs(c - target) for c in last), "target": target,
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(42)
    router = [[rng.gauss(0, 0.3) for _ in range(WIDTH)] for _ in range(EXPERTS)]
    tokens, experts = corpus(rng, False), [ref.make_expert(WIDTH, 32, random.Random(1))
                                           for _ in range(EXPERTS)]
    return {
        "plain": shape(ref, False), "clustered": shape(ref, True),
        "ceiling": math.log(EXPERTS),
        "agrees": usage(ref, router, tokens, [0.0] * EXPERTS)
        == ref.run_epoch(tokens, experts, router, TOP_K, [0.0] * EXPERTS),
    }


def verify(result):
    plain, clustered = result["plain"], result["clustered"]
    return [
        practice.Check(
            "ANSWER: the lesson's own tokens start 1.65x from balanced",
            plain["start"][1] < 2 and plain["start"][0] > 0.99 * result["ceiling"],
            f"rng.gauss(0, 1) per coordinate has no structure, whatever the comment above it "
            f"says, so the only asymmetry is the random router weights: max/min usage "
            f"{plain['start'][1]:.2f} at iteration 0 and entropy {plain['start'][0]:.4f}, "
            f"{plain['start'][0] / result['ceiling']:.1%} of ln {EXPERTS}. Fifty iterations take "
            f"it to {plain['end'][1]:.2f}",
        ),
        practice.Check(
            "FINDING: clustered tokens give the same rule something to do",
            clustered["start"][1] > 100 and clustered["closed"] > 0.7,
            f"drawn from {CLUSTERS} Gaussian clusters the starting max/min is "
            f"{clustered['start'][1]:.0f} -- one expert taking that many times another's share "
            f"-- with entropy {clustered['start'][0]:.4f} against {result['ceiling']:.4f}. The "
            f"bias update closes {clustered['closed']:.0%} of the gap, to "
            f"{clustered['end'][0]:.4f} and max/min {clustered['end'][1]:.2f}",
        ),
        practice.Check(
            "FINDING: forty of the fifty iterations change nothing",
            abs(plain["late"]) < 1e-9 and clustered["late"] < 0,
            f"the step is a fixed +/-{GAMMA} with no proportionality and no dead band, so it "
            f"cannot converge -- it limit-cycles. Entropy from iteration 10 to {ITERATIONS} "
            f"moves {plain['late']:+.4f} on the lesson's tokens and {clustered['late']:+.4f} on "
            f"clustered ones, the last thirty spanning {plain['tail']:.4f} and "
            f"{clustered['tail']:.4f}",
        ),
        practice.Check(
            "FINDING: entropy saturates long before balance does",
            clustered["end"][0] > 0.93 * result["ceiling"] and clustered["worst"] > 200,
            f"at iteration {ITERATIONS} on clustered tokens entropy reads "
            f"{clustered['end'][0] / result['ceiling']:.0%} of maximum while the worst expert is "
            f"{clustered['worst']:.0f} tokens off a target of {clustered['target']:.0f} and "
            f"max/min is {clustered['end'][1]:.2f}. Watching entropy is watching the wrong number",
        ),
        practice.Check(
            "CONTROL: route() and run_epoch() agree on usage exactly",
            result["agrees"],
            "run_epoch calls moe_layer_forward, which applies every selected expert before "
            "counting; the counts depend only on route(). They come out identical, so the sweep "
            "skips 51 epochs of 16x32 matmuls it does not need",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
