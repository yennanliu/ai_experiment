"""Exercise 4 — a P99 up 15% is one verify overhead on the steps that accept nothing.

    You see mean ITL drop 25% after enabling EAGLE-3 but P99 ITL went up 15%.
    Diagnose and propose a mitigation.

Reading of the exercise: ITL is per-token time, as in the lesson's simulator
(a verify step's cost spread over the tokens it emits). The diagnosis inverts
the two observed numbers under a real rejection sampler, where a verify
always emits one token of its own. vLLM's `mean_acceptance_length` "ranges
from 1.0 (nothing accepted)". Costs are relative to one plain step, with the
lesson's K=5 and verify_overhead 0.15. Each mitigation is scored on both
numbers, and the lesson's `simulate_tail` is asked whether it can produce the
symptom at all.

**ANSWER: the +15% is exactly one verify overhead, paid by the steps that
accept nothing; gate those off.** A step that accepts nothing costs
1 + eps = 1.15 plain steps for one token. Mean ITL -25% means
1.15 / E = 0.75, so E = 1.533 tokens per step and alpha is 0.349. At that
alpha 42% of tokens come from zero-acceptance steps, far above the 1% a P99
needs, so P99 sits at +15%. Lowering K to 1 keeps P99 at +15% and shrinks
the mean gain to -14.8%. Tail and mean both improve only when those steps
become rare: alpha 0.947 puts them under 1% of tokens, and then P99 is
-42.5% and mean -78.2%. Until a domain draft head gets there, spec decode
should be off where alpha is low. That means K=0 for those batch sizes
through vLLM's `num_speculative_tokens_per_batch_size`, or plain routing for
the low-alpha slice (exercise 2). K=0 takes P99 back to +0%.

**FINDING: the lesson's simulator cannot show this symptom at its own
concurrency points.** `simulate_tail` charges a second full target pass
(`reroll_ms`) on every step that rejects a draft. A zero-acceptance step
therefore costs 2.05 plain steps per token at 32 concurrent. Over alpha 0.3..0.9 at
concurrency 1, 32 and 256, its P99 is +35%..+116% over plain, and every one
of the 15 rows `main()` prints is tagged TAIL. At 32 concurrent, alpha 0.7
already gives mean -33% with P99 +89%.

**FINDING: in that simulator the tail shrinks with concurrency and flips by
1024.** Plain decode's cost grows by conc/512 and a spec step's by
0.15 * conc/256. So at 512 concurrent alpha 0.7 is +13% P99, and at 1024
each of alpha 0.3, 0.5, 0.7, 0.9, 0.95 and 0.99 has a P99 below plain. The lesson tells you to
fear P99 at high concurrency, and its own model says the opposite.

Structure: `itl()` is the exact token-weighted ITL distribution of one
speculative step type, without jitter; `symptom()` inverts the observation.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "05-eagle3-speculative-decoding"
K, EPS = 5, 0.15
ALPHAS, CONCS = (0.3, 0.5, 0.7, 0.9), (1, 32, 256)


def tokens(alpha, k=K):
    return sum(alpha**i for i in range(k + 1))


def bisect(f, target):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return hi


def itl(alpha, k=K, eps=EPS):
    """(mean, p99, share of tokens from zero-acceptance steps), in plain steps."""
    if k == 0:
        return 1.0, 1.0, 0.0
    steps = [(alpha**j * (1 - alpha), j + 1) for j in range(k)] + [(alpha**k, k + 1)]
    per_tok = sorted(((1 + eps) / n, p * n / tokens(alpha, k)) for p, n in steps)
    mass, p99 = 0.0, per_tok[-1][0]
    for cost, share in reversed(per_tok):
        mass += share
        if mass > 0.01:
            p99 = cost
            break
    return (1 + eps) / tokens(alpha, k), p99, steps[0][0] / tokens(alpha, k)


def symptom(mean_drop=0.25):
    return bisect(lambda a: 1 - itl(a)[0], mean_drop)


def reference_grid(ref):
    """(mean change, p99 change) of simulate_tail against plain_tail."""
    out = {}
    for c in CONCS + (512, 1024):
        plain = ref.plain_tail(c)
        for a in ALPHAS + (0.95, 0.99):
            mean, p99 = ref.simulate_tail(ref.SpecPoint(a, K, EPS, c))
            out[c, a] = (mean / plain[0] - 1, p99 / plain[1] - 1)
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ref.main()
    alpha = symptom()
    rare = bisect(lambda a: -itl(a)[2], -0.01)
    (mean, p99, share), k1, off, fixed = (
        itl(alpha),
        itl(alpha, 1),
        itl(alpha, 0),
        itl(rare + 1e-9),
    )
    grid = reference_grid(ref)
    shipped = [grid[c, a][1] for c in CONCS for a in ALPHAS]
    zero = (8.0 * (1 + EPS * (1 + 32 / 256)) + 8.0) / ref.plain_tail(32)[0]
    return {
        "answer": (round(mean, 4), round(p99, 4), round(alpha, 3), round(share, 2)),
        "fixes": (round(k1[0], 3), round(k1[1], 4), round(rare, 3), fixed[:2], off[:2]),
        "shipped": (round(min(shipped), 2), round(max(shipped), 2), round(zero, 2)),
        "tags": (out.getvalue().count(" TAIL"), out.getvalue().count("  OK")),
        "at_32": grid[32, 0.7],
        "by_conc": [round(grid[c, 0.7][1], 3) for c in CONCS + (512, 1024)],
        "at_1024": [grid[1024, a][1] for a in ALPHAS + (0.95, 0.99)],
    }


def verify(result):
    mean, p99, alpha, share = result["answer"]
    k1_mean, k1_p99, rare, fixed, off = result["fixes"]
    low, high, zero = result["shipped"]
    return [
        practice.Check(
            "ANSWER: the +15% is exactly one verify overhead, paid by the steps that accept nothing",
            result["answer"] == (0.75, 1.15, 0.349, 0.42)
            and (k1_mean, k1_p99, rare, off) == (0.852, 1.15, 0.947, (1.0, 1.0)),
            f"alpha {alpha}: {share:.0%} of tokens from zero-acceptance steps, P99 {p99}x; "
            f"K=1 mean {k1_mean - 1:+.1%} P99 {k1_p99 - 1:+.0%}; alpha {rare} gives mean "
            f"{fixed[0] - 1:+.1%} P99 {fixed[1] - 1:+.1%}; K=0 restores {off}",
        ),
        practice.Check(
            "FINDING: the lesson's simulator cannot show this symptom at its own concurrency points",
            result["shipped"] == (0.35, 1.16, 2.05)
            and result["at_32"][0] < -0.25
            and result["tags"] == (15, 0),
            f"reroll_ms makes a zero-acceptance step {zero} plain steps per token; P99 change "
            f"over alpha {ALPHAS} x concurrency {CONCS} is {low:+.0%}..{high:+.0%}; at 32/0.7 "
            f"mean {result['at_32'][0]:+.0%}, P99 {result['at_32'][1]:+.0%}; "
            f"main() tags (TAIL, OK) = {result['tags']}",
        ),
        practice.Check(
            "FINDING: in that simulator the tail shrinks with concurrency and flips by 1024",
            0.1 < result["by_conc"][3] < 0.15 and max(result["at_1024"]) < 0,
            f"alpha 0.7 P99 change at concurrency {CONCS + (512, 1024)}: {result['by_conc']}; "
            f"at 1024 every alpha beats plain P99",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
