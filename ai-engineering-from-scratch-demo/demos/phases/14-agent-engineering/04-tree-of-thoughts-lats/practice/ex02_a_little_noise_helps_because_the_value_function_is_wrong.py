"""Exercise 2 — a little noise helps, because the value function is wrong.

    Swap the value function for a noisier scorer (add random jitter). Does
    MCTS still find the best leaf? What is the minimum signal-to-noise it
    tolerates?

Reading of the exercise: signal-to-noise needs a signal, and this task has an
exact one -- `value` returns **1.0** for a solved state and at most **0.0**
for anything else, so the gap is **1.0** and SNR is `1.0 / sigma`. Jitter is
added by rebinding `ref.value`, which is what both the rollouts and the final
`max(_all_leaves(root), key=value)` read, and every run is seeded so the
sweep is a measurement rather than an anecdote.

**ANSWER: yes, up to SNR 4.** Over **30** seeds at **1000** iterations the
search returns a true solution **30/30** at sigma **0**, **0.1** and
**0.25**, then **17/30** at sigma **0.5** and **3/30** at sigma **1.0**. The
tolerated minimum is SNR **4**; at SNR **2** it is a coin flip and at SNR
**1** it is gone.

**FINDING: at a tight budget, jitter *improves* the search.** At **300**
iterations sigma **0** wins **19/30** and sigma **0.1** wins **25/30**. The
value function is anti-correlated with success on this instance: its
top-scoring first move, `6*4=24`, scores the best possible **0.0** and is a
dead end, while the move that starts a real solution ranks **7th of 24**.
Noise that large enough to blur that ranking is help, not harm.

**FINDING: the damage is in the final pick, not in the rollouts.** At sigma
**0.5** and **300** iterations, perturbing only the closing argmax wins
**9/30** while perturbing only the rollouts wins **21/30**, against a clean
**19/30**. `mcts` ends with an unweighted `max` over hundreds of
independently perturbed scores, so its error probability grows with the tree.

**FINDING: the failures are confident.** At sigma **1.0** and **1000**
iterations, **25** of the **27** losses return a *complete* depth-3
trajectory that is simply wrong. A noisy evaluator does not make the search
give up; it makes it produce a finished, plausible, incorrect answer.

Structure: `search()` rebinds `ref.value` under `try/finally`; every
comparison below scores the result with the unpatched function.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "04-tree-of-thoughts-lats"
SIGMAS, RUNS, GAP = (0.0, 0.1, 0.25, 0.5, 1.0), 30, 1.0


def search(ref, sigma, seed, iterations):
    """One seeded run with `value` perturbed by N(0, sigma)."""
    true_value, jitter = ref.value, random.Random(1000 + seed)
    if sigma:
        ref.value = lambda node: true_value(node) + jitter.gauss(0, sigma)
    try:
        root = ref.Node(state=tuple(sorted(ref.NUMBERS, reverse=True)), trace=[])
        root.children = ref.expand(root)
        best, _ = ref.mcts(root, iterations, random.Random(seed))
    finally:
        ref.value = true_value
    return root, best


def wins(ref, sigma, iterations):
    return sum(ref.value(search(ref, sigma, seed, iterations)[1]) > 0.99
               for seed in range(RUNS))


def pick_only(ref, sigma, iterations):
    """Noise in the closing argmax alone, over a tree built cleanly."""
    total = 0
    for seed in range(RUNS):
        root, _ = search(ref, 0.0, seed, iterations)
        jitter = random.Random(7000 + seed)
        best = max(ref._all_leaves(root),
                   key=lambda node: ref.value(node) + jitter.gauss(0, sigma))
        total += ref.value(best) > 0.99
    return total


def rollouts_only(ref, sigma, iterations):
    """Noise in the rollouts alone, with the closing argmax scored cleanly."""
    return sum(ref.value(max(ref._all_leaves(search(ref, sigma, seed, iterations)[0]),
                             key=ref.value)) > 0.99 for seed in range(RUNS))


def confident_losses(ref, sigma, iterations):
    losses = [best for _, best in (search(ref, sigma, seed, iterations)
                                   for seed in range(RUNS)) if ref.value(best) <= 0.99]
    return len(losses), sum(1 for best in losses if len(best.trace) == 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    root = ref.Node(state=tuple(sorted(ref.NUMBERS, reverse=True)), trace=[])
    ranked = sorted(((ref.value(child), child.trace[0])
                     for child in ref.expand(root)), reverse=True)
    lost, complete = confident_losses(ref, 1.0, 1000)
    return {
        "wide": {sigma: wins(ref, sigma, 1000) for sigma in SIGMAS},
        "tight": {sigma: wins(ref, sigma, 300) for sigma in SIGMAS},
        "pick_only": pick_only(ref, 0.5, 300), "rollouts_only": rollouts_only(ref, 0.5, 300),
        "top_move": ranked[0][1], "top_score": round(ranked[0][0], 3),
        "winner_rank": [move for _, move in ranked].index("6+1=7") + 1,
        "branching": len(ranked), "lost": lost, "complete_losses": complete,
        "snr": {sigma: round(GAP / sigma, 1) for sigma in SIGMAS if sigma},
    }


def verify(result):
    wide, tight = result["wide"], result["tight"]
    return [
        practice.Check(
            "ANSWER: yes, down to SNR 4 -- 30/30 at sigma 0.25, 17/30 at 0.5, 3/30 at 1.0",
            all([wide[0.0] == 30, wide[0.1] == 30, wide[0.25] == 30, wide[0.5] == 17,
                 wide[1.0] == 3, result["snr"][0.25] == 4.0, result["snr"][0.5] == 2.0]),
            f"over {RUNS} seeds at 1000 iterations the search returns a true solution "
            f"{wide[0.25]}/{RUNS} at sigma 0.25 (SNR {result['snr'][0.25]}), "
            f"{wide[0.5]}/{RUNS} at 0.5 (SNR {result['snr'][0.5]}) and "
            f"{wide[1.0]}/{RUNS} at 1.0. The gap between a solution and the best "
            f"non-solution is {GAP}, so the tolerated minimum is SNR 4",
        ),
        practice.Check(
            "FINDING: at a tight budget, jitter improves the search",
            all([tight[0.0] == 19, tight[0.1] == 25, tight[0.1] > tight[0.0],
                 result["top_move"] == "6*4=24", result["top_score"] == 0.0,
                 result["winner_rank"] == 7, result["branching"] == 24]),
            f"at 300 iterations sigma 0 wins {tight[0.0]}/{RUNS} and sigma 0.1 wins "
            f"{tight[0.1]}/{RUNS}. The value function's best first move is "
            f"{result['top_move']!r} at {result['top_score']} -- a dead end -- while the "
            f"move that starts a real solution ranks {result['winner_rank']} of "
            f"{result['branching']}. Blurring that ranking helps",
        ),
        practice.Check(
            "FINDING: the damage is in the final pick, not in the rollouts",
            all([result["pick_only"] == 9, result["rollouts_only"] == 21,
                 tight[0.5] == 11, result["pick_only"] < result["rollouts_only"]]),
            f"at sigma 0.5 and 300 iterations, perturbing only the closing argmax wins "
            f"{result['pick_only']}/{RUNS} while perturbing only the rollouts wins "
            f"{result['rollouts_only']}/{RUNS}, against a clean {tight[0.0]}/{RUNS}. The "
            "close is an unweighted max over hundreds of independently perturbed scores",
        ),
        practice.Check(
            "FINDING: the failures are confident, not empty",
            all([result["lost"] == 27, result["complete_losses"] == 25,
                 result["lost"] == RUNS - wide[1.0]]),
            f"at sigma 1.0 and 1000 iterations, {result['complete_losses']} of the "
            f"{result['lost']} losses return a complete depth-3 trajectory that is simply "
            "wrong. A noisy evaluator does not make the search give up -- it makes it "
            "produce a finished, plausible, incorrect answer",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
