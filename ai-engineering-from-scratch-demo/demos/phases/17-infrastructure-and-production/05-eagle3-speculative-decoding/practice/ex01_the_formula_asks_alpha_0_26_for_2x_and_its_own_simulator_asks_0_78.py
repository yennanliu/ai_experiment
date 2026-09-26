"""Exercise 1 — the formula asks alpha 0.26 for 2x, and its own simulator asks 0.78.

    Run `code/main.py`. At K=5, what alpha do you need for a 2x speedup? For a
    3x speedup? How sensitive is that to verify_overhead?

Reading of the exercise: "speedup" is first the lesson's own
`expected_speedup`, solved for alpha at K=5 and the shipped verify_overhead
0.15, then swept over overhead 0..0.5. The same question is then put to the
two other speedups the module implies: the one its simulator measures, and the
one its acceptance rule implies.

**ANSWER: 2x needs alpha 0.26 and 3x needs 0.49.** That is at K=5 with
overhead 0.15 and no concurrency term. The formula `(1 + K*alpha)/(1 + eps)`
is linear, so the sensitivity is exact: d alpha / d eps = S / K. Every 0.1 of
extra overhead costs 0.04 alpha for 2x and 0.06 for 3x. Over overhead 0..0.5,
2x needs 0.20..0.40 and 3x needs 0.40..0.70. The code's concurrency term is
another overhead multiplier; at 256 concurrent it asks 0.32 and 0.58.

**FINDING: 1 + K*alpha counts drafts after the first rejection.** The
simulator stops at the first rejection (`break`), which is also what a real
verifier does. Under that rule a verify yields 1 + alpha + ... + alpha^K
tokens. At overhead 0.15, 2x then needs alpha 0.582 and 3x needs 0.771. Over
overhead 0..0.5 those move 0.509..0.709 and 0.709..0.883.

**FINDING: the simulator's throughput and the speedup column disagree by up
to 3x.** The simulator's per-token means give the throughput it models:
plain mean over spec mean. That is exactly computable; `simulate_tail` at
200,000 tokens matches it (0.700 against 0.6997). At 32 concurrent and alpha
0.3 the printed speedup is 2.14x and the simulator runs at 0.70x. It needs
alpha 0.511 to break even, 0.781 for 2x and 0.889 for 3x. At 256 concurrent
it needs 0.348, 0.687 and 0.819. Its break-even falls with concurrency,
because plain decode's cost grows by conc/512 and a spec verify's does not.

**FINDING: the printed break-even alpha is 0.034 / 0.045 / 0.060, not the
KEY FINDING's "~0.4" at 256.** The code's own `breakeven_alpha` at
concurrency 256 is 0.060. For its formula to reach 0.4, concurrency would
have to be 3157.

Structure: `bisect()` solves any increasing speedup for alpha; `tokens()` and
`sim_speedup()` are the expectations behind `simulate_tail` and `plain_tail`.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "05-eagle3-speculative-decoding"
K, EPS = 5, 0.15
SWEEP = (0.0, 0.1, 0.15, 0.2, 0.3, 0.5)


def bisect(f, target):
    """Smallest alpha in [0, 1] with f(alpha) >= target, for increasing f."""
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return round(hi, 3)


def tokens(alpha, k=K):
    """Expected tokens per verify when acceptance stops at the first rejection."""
    return sum(alpha**i for i in range(k + 1))


def sim_speedup(alpha, conc, k=K, eps=EPS):
    """plain_tail's mean over simulate_tail's mean, as exact expectations."""
    verify = 8.0 * (1 + eps * (1 + conc / 256))
    spec = (verify + 8.0 * (1 - alpha**k)) / tokens(alpha, k)
    return 8.0 * (1 + conc / 512) / spec


def formula(ref, conc, eps=EPS):
    return lambda a: ref.expected_speedup(ref.SpecPoint(a, K, eps, conc))


def needed(speed, targets=(2, 3)):
    return tuple(bisect(speed, s) for s in targets)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    mean, _ = ref.simulate_tail(ref.SpecPoint(0.3, K, EPS, 32), n_tokens=200_000)
    stop = {e: needed(lambda a, e=e: tokens(a) / (1 + e)) for e in SWEEP}
    sim = {c: needed(lambda a, c=c: sim_speedup(a, c), (1, 2, 3)) for c in (32, 256)}
    return {
        "lesson": {e: needed(formula(ref, 0, e)) for e in SWEEP},
        "conc256": needed(formula(ref, 256)),
        "stop": stop,
        "sim": sim,
        "printed": formula(ref, 32)(0.3),
        "exact": sim_speedup(0.3, 32),
        "measured": ref.plain_tail(32, n_tokens=200_000)[0] / mean,
        "be": [round(ref.breakeven_alpha(K, EPS, c), 3) for c in (32, 128, 256)],
        "conc_for_0_4": round(256 * (0.4 * K / EPS - 1)),
    }


def verify(result):
    lesson, stop, sim = result["lesson"], result["stop"], result["sim"]
    slopes = [
        round((lesson[e][i] - lesson[0.0][i]) / e, 6) for e in SWEEP[1:] for i in (0, 1)
    ]
    return [
        practice.Check(
            "ANSWER: 2x needs alpha 0.26 and 3x needs 0.49",
            all(
                [
                    lesson[EPS] == (0.26, 0.49),
                    set(slopes) == {0.4, 0.6},
                    lesson[0.5] == (0.4, 0.7),
                    result["conc256"] == (0.32, 0.58),
                ]
            ),
            f"K={K}, overhead {EPS}; by overhead {lesson}; d alpha/d eps is exactly S/K "
            f"(0.4 for 2x, 0.6 for 3x); at 256 concurrent {result['conc256']}",
        ),
        practice.Check(
            "FINDING: 1 + K*alpha counts drafts after the first rejection",
            all(
                [
                    stop[EPS] == (0.582, 0.771),
                    stop[0.0] == (0.509, 0.709),
                    stop[0.5] == (0.709, 0.883),
                ]
            ),
            f"stopping at the first rejection, as simulate_tail does, 2x/3x need {stop}",
        ),
        practice.Check(
            "FINDING: the simulator's throughput and the speedup column disagree by up to 3x",
            all(
                [
                    abs(result["measured"] - result["exact"]) < 0.005,
                    result["printed"] > 3 * result["exact"],
                    sim == {32: (0.511, 0.781, 0.889), 256: (0.348, 0.687, 0.819)},
                ]
            ),
            f"alpha 0.3 at 32 concurrent prints {result['printed']:.2f}x and simulates "
            f"{result['exact']:.3f}x (simulate_tail at 200k tokens: {result['measured']:.4f}); "
            f"alpha for 1x/2x/3x in the simulator {sim}",
        ),
        practice.Check(
            "FINDING: the printed break-even alpha is 0.034 / 0.045 / 0.060, not ~0.4 at 256",
            all([result["be"] == [0.034, 0.045, 0.06], result["conc_for_0_4"] == 3157]),
            f"breakeven_alpha at 32/128/256 is {result['be']}; its formula reaches 0.4 only "
            f"at concurrency {result['conc_for_0_4']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
