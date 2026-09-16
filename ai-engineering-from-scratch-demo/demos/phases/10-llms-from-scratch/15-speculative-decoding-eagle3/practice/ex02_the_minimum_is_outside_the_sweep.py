"""Exercise 2 — the curve falls monotonically across the whole sweep; the minimum is at N=15.

    Sweep `N` from 1 to 10 with `α` held at 0.9 and `c` held at 0.04. Plot
    expected tokens per verifier call and actual wall time per token. Find the
    `N` that minimizes wall time. Explain the shape of the curve.

Reading of the exercise: both curves come from the lesson's own
`expected_tokens_per_verify` and `wall_time_per_token`, and the sweep is
extended past 10 because "find the N that minimizes wall time" presumes the
minimum is inside the range and that is a testable claim.

**ANSWER: wall time falls at every step from N=1 to N=10, so the sweep's answer
is its last point.** 0.5474 at N=1 down to 0.2040 at N=10, monotone. The actual
minimum is **0.1964 at N=15**, and N=14 ties it to four decimals.

**MECHANISM: the two terms pull in opposite directions and one of them
saturates.** Cost per verifier call is `1 + Nc`, linear in N forever. Expected
tokens is `(1 - α^(N+1)) / (1 - α)`, which saturates at `1/(1-α) = 10` -- at
N=10 it has already reached **6.86** of that ceiling and is gaining 0.35 per
extra draft, while the cost is gaining 0.04 per draft at a base of 1.4. The
curve turns where the marginal token stops paying for the marginal draft.

**FINDING: the ceiling, not the cost, is what ends the curve.** With α = 0.9 no
draft chain can average more than 10 tokens per verifier call however long it
is, so the whole sweep is climbing toward a number fixed by α alone. At N=30 the
chain averages 9.62 tokens -- 96% of the ceiling -- and wall time is back up to
0.2287, worse than N=10.

**FINDING: the shape is flat where the exercise's answer lies.** Within 1% of
the optimum, **N = 13 to 17** are indistinguishable, and from N=12 to N=18 wall
time varies by **1.3%**. That is the practical reading of "explain the shape":
there is a wide basin, and the exercise's 1-to-10 window sits entirely on the
slope leading into it, where wall time falls by **63%**.

Structure: `curve` evaluates both of the lesson's formulas at one N; `sweep`
runs a range of N and `basin` finds how wide the near-optimal region is.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "15-speculative-decoding-eagle3"
ALPHA, COST = 0.9, 0.04
ASKED = range(1, 11)
EXTENDED = range(1, 41)
TOLERANCE = 0.01


def curve(ref, n):
    return {"tokens": ref.expected_tokens_per_verify(ALPHA, n),
            "wall": ref.wall_time_per_token(ALPHA, n, COST)}


def sweep(ref, span):
    return {n: curve(ref, n) for n in span}


def basin(rows, best, tolerance=TOLERANCE):
    """The Ns whose wall time is within `tolerance` of the best."""
    floor = rows[best]["wall"]
    return [n for n, row in rows.items() if row["wall"] <= floor * (1 + tolerance)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = sweep(ref, EXTENDED)
    asked = {n: rows[n] for n in ASKED}
    best = min(rows, key=lambda n: rows[n]["wall"])
    walls = [asked[n]["wall"] for n in ASKED]
    return {
        "asked": asked,
        "best": best,
        "best_wall": rows[best]["wall"],
        "rows": {n: rows[n] for n in (10, 12, 14, 15, 16, 18, 20, 30)},
        "monotone": all(a > b for a, b in zip(walls, walls[1:])),
        "ceiling": 1 / (1 - ALPHA),
        "basin": basin(rows, best),
        "marginal": (rows[10]["tokens"] - rows[9]["tokens"], COST),
    }


def verify(result):
    asked, rows = result["asked"], result["rows"]
    gain, cost = result["marginal"]
    return [
        practice.Check(
            "ANSWER: the sweep is monotone, so its answer is its last point -- N=10",
            result["monotone"] and result["best"] not in ASKED,
            "wall time over the sweep the exercise names is "
            + ", ".join(f"N={n} {row['wall']:.4f}" for n, row in asked.items())
            + f" -- falling at every step, so 'the N that minimizes wall time' over 1 to 10 is "
            f"10, the last point tried. Extending the sweep, the minimum is "
            f"{result['best_wall']:.4f} at N={result['best']}",
        ),
        practice.Check(
            "MECHANISM: cost is linear in N forever and tokens saturate at 1/(1-alpha)",
            abs(result["ceiling"] - 10) < 1e-9 and gain > cost,
            f"cost per verifier call is 1 + N x {cost}, linear in N without bound. Expected "
            f"tokens is (1 - alpha^(N+1)) / (1 - alpha), which saturates at "
            f"{result['ceiling']:.0f}. At N=10 it has reached {rows[10]['tokens']:.2f} of that "
            f"and is gaining {gain:.2f} tokens per extra draft against {cost} of extra cost on a "
            f"base of {1 + 10 * cost:.2f} -- the curve turns where the marginal token stops "
            "paying for the marginal draft",
        ),
        practice.Check(
            "FINDING: the ceiling, not the cost, is what ends the curve",
            rows[30]["tokens"] < result["ceiling"] and rows[30]["wall"] > rows[10]["wall"],
            f"with alpha = {ALPHA} no draft chain can average more than "
            f"{result['ceiling']:.0f} tokens per verifier call however long it is, so the whole "
            f"sweep is climbing toward a number fixed by alpha alone. At N=30 the chain averages "
            f"{rows[30]['tokens']:.2f} -- {rows[30]['tokens'] / result['ceiling']:.0%} of the "
            f"ceiling -- and wall time is back up to {rows[30]['wall']:.4f}, worse than N=10's "
            f"{rows[10]['wall']:.4f}",
        ),
        practice.Check(
            "FINDING: the shape is flat where the answer lies, and the sweep is all slope",
            len(result["basin"]) >= 5 and max(ASKED) < min(result["basin"]),
            f"within {TOLERANCE:.0%} of the optimum the acceptable N are {result['basin']} -- "
            f"{len(result['basin'])} settings a real system could not tell apart. Every one of "
            f"them is above {max(ASKED)}, so the exercise's window sits entirely on the slope "
            f"leading into the basin: from N=12 to N=18 wall time varies by "
            f"{100 * (rows[18]['wall'] / rows[15]['wall'] - 1):.1f}%, and from N=1 to N=10 it "
            f"falls by {100 * (1 - rows[10]['wall'] / asked[1]['wall']):.0f}%",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
