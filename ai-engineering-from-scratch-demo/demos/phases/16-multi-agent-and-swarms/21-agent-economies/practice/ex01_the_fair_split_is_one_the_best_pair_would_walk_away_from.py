"""Exercise 1 — the fair split is one the best pair would walk away from.

    Run `code/main.py`. Confirm Shapley values sum to total value (efficiency
    axiom). Change the value function; do Shapley allocations change in the
    expected direction?

Reading of the exercise: "the expected direction" is made exact -- raising
v(S) by d moves every member of S up and every non-member down by weights
that depend only on |S| -- so each coalition of the demo's game is bumped in
turn and the reference's shift is compared with the closed form.

**ANSWER: yes, efficiency holds and every shift has the predicted sign and
size.** The exact values are coder 0.5083, researcher 0.3333, reviewer
0.1583, summing to 1.0000 = v(grand). Bumping each of the 7 non-empty
coalitions by 0.1 moves members up by d(|S|-1)!(n-|S|)!/n! and non-members
down by d|S|!(n-|S|-1)!/n! in 7 of 7 cases. The direction that surprises is
the second term: improving {coder, researcher} by 0.1 lowers the reviewer's
credit by 0.0333 -- twice what each pair member gains -- although the
reviewer did nothing different. Sampling with the demo's 200 orderings lands
within 0.0035 of exact, after 800 value calls where exact took 24.

**FINDING: the Shapley split is outside the core.** Coder and researcher
receive 0.8417 together, but v({coder, researcher}) = 0.85: the pair does
better leaving the reviewer out. The demo's comment calls the game
superadditive, and it is; it is not convex -- the reviewer adds 0.20 to
{coder} and only 0.15 to {coder, researcher} -- and that is what lets the
fair split be blocked.

**FINDING: the module the lesson describes is not the module it ships.** The
docstring says Shapley "is exact for N<=6 and sampled otherwise"; nothing
dispatches, and `shapley_exact` enumerates all N! orders for any N. The
lesson lists `shapley(...)`, `second_price_auction(...)` and a Reputation
"with exponential decay and slashing"; the names are `shapley_exact` and
`second_price`, and Reputation has no slashing method.

**FINDING: reputation routing gains 3.6%, not 10-20%.** The demo prints
+3.6%. Over 1000 seeds the mean gain is 3.5%, and 5% of seeds fall in the
promised band. Half the 100 tasks are random warmup, and after it the
weights are scores near [0.92, 0.65, 0.81, 0.57] -- alpha=0.95 keeps every
score close to its initial 1.0 -- so routing is barely better than uniform.
Always choosing the best agent after warmup would give +23.5%.
"""

from __future__ import annotations

import contextlib
import io
import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "21-agent-economies"
AGENTS = ["coder", "researcher", "reviewer"]
VALUES = {(): 0.0, ("coder",): 0.5, ("researcher",): 0.3, ("reviewer",): 0.1,
          ("coder", "researcher"): 0.85, ("coder", "reviewer"): 0.70,
          ("researcher", "reviewer"): 0.55, ("coder", "researcher", "reviewer"): 1.00}
GAME = {frozenset(k): v for k, v in VALUES.items()}
QUALITY = {"alpha": 0.9, "beta": 0.5, "gamma": 0.75, "delta": 0.3}
DELTA = 0.1


def predicted(coalition, agent, n=3, delta=DELTA):
    """Closed-form shift in agent's Shapley value when v(coalition) rises by delta."""
    s = len(coalition)
    if agent in coalition:
        return delta * math.factorial(s - 1) * math.factorial(n - s) / math.factorial(n)
    return -delta * math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)


def bump_matches(ref, exact):
    hits = 0
    for coalition in (c for c in GAME if c):
        game = GAME | {coalition: GAME[coalition] + DELTA}
        moved = ref.shapley_exact(game.__getitem__, AGENTS)
        hits += all(abs(moved[a] - exact[a] - predicted(coalition, a)) < 1e-12 for a in AGENTS)
    return hits


def routing_gain(ref, seed):
    agents, rng, base = list(QUALITY), random.Random(seed), 0.0
    for _ in range(100):
        base += max(0.0, min(1.0, QUALITY[rng.choice(agents)] + rng.uniform(-0.1, 0.1)))
    rng, rep, routed = random.Random(seed), ref.Reputation(), 0.0
    rep.init(agents)
    for i in range(100):
        a = rng.choice(agents) if i < 50 else ref.weighted_choice(agents, rep.weights(agents), rng)
        q = max(0.0, min(1.0, QUALITY[a] + rng.uniform(-0.1, 0.1)))
        rep.update(a, q)
        routed += q
    return (routed - base) / base * 100


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    exact = ref.shapley_exact(GAME.__getitem__, AGENTS)
    calls = {"n": 0}
    counted = lambda s: calls.__setitem__("n", calls["n"] + 1) or GAME[s]  # noqa: E731
    sampled = ref.shapley_sampled(counted, AGENTS, 200, random.Random(0))
    sampled_calls, calls["n"] = calls["n"], 0
    ref.shapley_exact(counted, AGENTS)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ref.demo_reputation_routing()
    gains = [routing_gain(ref, seed) for seed in range(1000)]
    pair = frozenset({"coder", "researcher"})
    return {
        "exact": exact, "total": sum(exact.values()), "bumps": bump_matches(ref, exact),
        "pair_drop": predicted(pair, "reviewer"), "pair_gain": predicted(pair, "coder"),
        "sample_err": max(abs(sampled[a] - exact[a]) for a in AGENTS),
        "calls": (sampled_calls, calls["n"]),
        "pair_share": exact["coder"] + exact["researcher"], "pair_value": GAME[pair],
        "reviewer_adds": (GAME[frozenset({"coder", "reviewer"})] - GAME[frozenset({"coder"})],
                          1.0 - GAME[pair]),
        "names": [n for n in ("shapley", "second_price_auction") if hasattr(ref, n)],
        "slash": any("slash" in n for n in dir(ref.Reputation)),
        "printed": "+3.6%" in out.getvalue(), "mean_gain": statistics.mean(gains),
        "in_band": sum(10 <= g <= 20 for g in gains) / len(gains),
        "ceiling": ((0.6125 + 0.9) / 2 - 0.6125) / 0.6125 * 100,
    }


def verify(result):
    e = result["exact"]
    return [
        practice.Check(
            "ANSWER: efficiency holds and every shift has the predicted sign and size",
            all([abs(result["total"] - 1.0) < 1e-12, result["bumps"] == 7,
                 abs(result["pair_drop"] + 2 * result["pair_gain"]) < 1e-12,
                 result["sample_err"] < 0.01, result["calls"] == (800, 24)]),
            f"coder {e['coder']:.4f}, researcher {e['researcher']:.4f}, reviewer "
            f"{e['reviewer']:.4f}, sum {result['total']:.4f}; {result['bumps']} of 7 "
            f"bumps match the closed form; lifting the coder-researcher pair moves the "
            f"reviewer {result['pair_drop']:+.4f}; 200 samples miss by "
            f"{result['sample_err']:.4f} using {result['calls'][0]} calls against exact's "
            f"{result['calls'][1]}",
        ),
        practice.Check(
            "FINDING: the Shapley split is outside the core",
            result["pair_share"] < result["pair_value"]
            and result["reviewer_adds"][1] < result["reviewer_adds"][0],
            f"coder and researcher get {result['pair_share']:.4f} together against "
            f"v = {result['pair_value']} as a pair; the reviewer adds "
            f"{result['reviewer_adds'][0]:.2f} to {{coder}} and "
            f"{result['reviewer_adds'][1]:.2f} to the pair -- superadditive, not convex",
        ),
        practice.Check(
            "FINDING: the module the lesson describes is not the module it ships",
            result["names"] == [] and not result["slash"],
            "no exact/sampled dispatch exists, shapley() and second_price_auction() are "
            "absent, and Reputation has no slashing method",
        ),
        practice.Check(
            "FINDING: reputation routing gains 3.6%, not 10-20%",
            all([result["printed"], result["mean_gain"] < 5, result["in_band"] < 0.1]),
            f"the demo prints +3.6%; over 1000 seeds the mean is "
            f"{result['mean_gain']:.1f}% and {result['in_band']:.0%} land in the claimed "
            f"band, against a {result['ceiling']:.1f}% ceiling for always routing to alpha",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
