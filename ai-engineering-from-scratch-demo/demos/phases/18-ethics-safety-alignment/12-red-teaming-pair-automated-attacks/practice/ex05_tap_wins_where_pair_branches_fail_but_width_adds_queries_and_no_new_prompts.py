"""Exercise 5 — TAP wins where PAIR branches fail, but width adds queries and no new prompts.

    TAP (Mehrotra 2024) extends PAIR with branching + pruning. Sketch a TAP-style extension to `code/main.py` and describe the computational cost vs success-rate trade-off.

Reading of the exercise: the sketch is built and run, not just drawn.
`tap()` grows a tree over the reference's own attackers. Each leaf branches
into one child per strategy (b = 3), every child is sent to the target, and
the `width` best-scoring children survive into the next level. Scores come
from `ref.judge`, with ties kept in tree order. It halts on the first
success, or at the lesson's matched budget of 20 queries. Cost is counted as
target queries. To make depth matter, it is also run against the semantic
filter patched to blocklist the encoded attacker's first 1 or 3 templates,
the patch a team applies after reading a PAIR report.

**ANSWER: TAP's success rate is the union of its branches, and the cost is b
queries per level.** On the keyword filter it succeeds in 2 queries, and on
the semantic filter in 3, where PAIR with paraphrase or roleplay fails at 20
and PAIR with encoded wins in 1. TAP never needs to know in advance which
strategy works. It pays for that by querying all b = 3 of them at every
level, so on the patched filters it costs 3x the right single PAIR run: 9 vs
3 queries with one template blocked, and 12 vs 4 with three blocked.

**FINDING: width buys nothing here, because every node at one depth grows
the same children.** The reference attackers read only `len(history)`, so
the surviving leaves at a given depth all propose identical prompts. With
PAIR's winning prompt blocklisted, width 1, 2 and 3 all succeed at depth 3
after 9, 12 and 15 queries. All three query the same 9 distinct prompts.
Pruning has nothing to rank either. The judge returns a bool, and any True
halts the search, so every child that survives pruning ties at False.

**FINDING: at a matched 20-query budget, wider TAP loses what narrow TAP
finds.** With three encoded templates blocklisted, width 1 succeeds in 12
queries and width 2 in 18, while width 3 runs out at 20 after only 9
distinct prompts. Without the budget, width 3 would have succeeded at 24.
Worst-case cost grows as b + (depth - 1) * b * width, and success rises only
when the extra leaves carry new prompts. That needs a stochastic or
history-reading attacker and a graded judge, and this toy has neither.

Structure: `tap()` is the extension (branch, query, prune);
`patched()` blocklists encoded templates on the semantic filter; `pair()`
runs `ref.pair_loop` per strategy for comparison.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "12-red-teaming-pair-automated-attacks"
STRATEGIES = ("paraphrase", "roleplay", "encoded")
BUDGET, WIDTHS, DEPTH = 20, (1, 2, 3), 10
REFUSAL = "I cannot help with that."


def tap(ref, target, width, budget=BUDGET):
    """Tree of attacks: every leaf branches into one child per strategy, each child is
    queried, and the `width` best-scoring children survive (ties keep tree order).

    Returns (success, queries spent, distinct prompts queried).
    """
    attackers = [getattr(ref, f"attacker_{s}") for s in STRATEGIES]
    frontier, queries, seen = [[]], 0, []
    for _ in range(DEPTH):
        children = []
        for hist in frontier:
            for attack in attackers:
                prompt = attack(hist)
                response, queries = target(prompt), queries + 1
                seen.append(prompt)
                if ref.judge(response):
                    return True, queries, len(set(seen))
                if queries == budget:
                    return False, queries, len(set(seen))
                children.append((ref.judge(response), hist + [(prompt, response)]))
        frontier = [h for _, h in sorted(children, key=lambda c: -c[0])[:width]]
    return False, queries, len(set(seen))


def patched(ref, k):
    """The semantic filter after a team blocklists the encoded attacker's first k templates."""
    blocked = {ref.attacker_encoded([None] * j) for j in range(k)}
    return lambda p: REFUSAL if p in blocked else ref.semantic_filter_target(p)


def pair(ref, target):
    return {s: ref.pair_loop(target, getattr(ref, f"attacker_{s}"), BUDGET)[:2]
            for s in STRATEGIES}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    targets = {"keyword": ref.keyword_filter_target, "semantic": ref.semantic_filter_target,
               "patch1": patched(ref, 1), "patch3": patched(ref, 3)}
    return {
        "tap": {t: {w: tap(ref, fn, w) for w in WIDTHS} for t, fn in targets.items()},
        "pair": {t: pair(ref, fn) for t, fn in targets.items()},
        "unbounded": tap(ref, targets["patch3"], 3, budget=10**6),
    }


def verify(result):
    tap_, pair_ = result["tap"], result["pair"]
    return [
        practice.Check(
            "ANSWER: TAP's success is the union of its branches, bought at b queries per level",
            [tap_[t][1] for t in ("keyword", "semantic")] == [(True, 2, 2), (True, 3, 3)]
            and pair_["semantic"] == {"paraphrase": (False, 20), "roleplay": (False, 20),
                                      "encoded": (True, 1)},
            f"TAP (success, queries, distinct): {tap_['keyword'][1]} keyword, "
            f"{tap_['semantic'][1]} semantic; PAIR on semantic {pair_['semantic']}",
        ),
        practice.Check(
            "FINDING: width buys nothing -- every node at one depth grows the same children",
            tap_["patch1"] == {1: (True, 9, 9), 2: (True, 12, 9), 3: (True, 15, 9)}
            and pair_["patch1"]["encoded"] == (True, 3),
            f"semantic with PAIR's winning prompt blocklisted, by width: {tap_['patch1']}; "
            f"PAIR-encoded {pair_['patch1']['encoded']}",
        ),
        practice.Check(
            "FINDING: at a matched 20-query budget, wider TAP loses what narrow TAP finds",
            tap_["patch3"] == {1: (True, 12, 12), 2: (True, 18, 12), 3: (False, 20, 9)}
            and result["unbounded"] == (True, 24, 12) and pair_["patch3"]["encoded"] == (True, 4),
            f"three encoded templates blocked: {tap_['patch3']}; width 3 unbudgeted "
            f"{result['unbounded']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
