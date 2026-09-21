"""Exercise 3 — the beam is already there, and 38 wide is free.

    Implement beam-search ToT (keep top-k at each level) and compare to BFS.
    Which is better on a tight token budget?

Reading of the exercise: `tot_bfs` sorts each level by `value` and keeps
`scored[:max_expansions_per_level]`, which is beam search with width 8. So
the thing to implement is the *other* arm -- unpruned BFS -- and the
comparison is a sweep of the width the lesson already exposes, with
expansions as the token proxy the lesson itself prints.

**ANSWER: the beam is the shipped code; unpruned BFS is width = infinity.**
At the shipped width **8** the answer scores **-0.03** after **152**
expansions. The narrowest width that actually finds 24 is **38**, and it
costs **460** expansions -- exactly what unpruned BFS costs, because the
`value > 0.99` early return fires at the same node. Beam width saves nothing
at the point where it starts being right.

**FINDING: widths 4 through 32 buy nothing at all.** Every one of them
returns the identical wrong trace `['6*4=24', '4-1=3', '24+3=27']` while the
cost rises from **88** to **438** expansions. Five times the budget, zero
change in the output -- the extra candidates are all worse under the same
ranking that was already wrong.

**FINDING: the threshold is the winning prefix's rank, exactly.** Scoring
every node of each full level, the winning path sits at rank **7** of **24**,
then **38** of **286**, then **1** of **1129**. The maximum over the pruned
levels is **38**, which is the observed threshold to the node.

**FINDING: the heuristic points the wrong way at the root.** `value` scores a
multi-number state by the closest remaining number to 24, so `6*4=24` scores
**0.0** -- the best a non-terminal state can score -- and is a dead end,
because the leftover `4` and `1` cannot be removed. Both real solutions start
at rank **7** and rank **20** of **24**.

Structure: `sweep()` runs the lesson's own `tot_bfs` at each width; `ranks()`
scores the full unpruned levels to explain the threshold.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "04-tree-of-thoughts-lats"
WIDTHS = (1, 4, 8, 16, 32, 37, 38, 64)
UNPRUNED = 10 ** 6


def root_of(ref):
    return ref.Node(state=tuple(sorted(ref.NUMBERS, reverse=True)), trace=[])


def sweep(ref, width):
    best, expansions = ref.tot_bfs(root_of(ref), max_expansions_per_level=width)
    return {"value": round(ref.value(best), 3), "expansions": expansions,
            "trace": list(best.trace)}


def solutions(ref, node):
    if len(node.state) == 1:
        return [tuple(node.trace)] if abs(node.state[0] - 24) < 1e-6 else []
    return [found for child in ref.expand(node) for found in solutions(ref, child)]


def levels(ref):
    """Every node of every level, unpruned and ranked by the lesson's value."""
    out, frontier = [], [root_of(ref)]
    for _ in range(3):
        frontier = [child for node in frontier for child in ref.expand(node)]
        out.append(sorted(frontier, key=ref.value, reverse=True))
    return out


def ranks(ref, path):
    return [next(index + 1 for index, node in enumerate(level)
                 if tuple(node.trace) == path[:depth + 1])
            for depth, level in enumerate(levels(ref))]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    found = sorted(set(solutions(ref, root_of(ref))))
    first = sorted(((ref.value(child), child.trace[0])
                    for child in ref.expand(root_of(ref))), reverse=True)
    order = [move for _, move in first]
    return {
        "widths": {width: sweep(ref, width) for width in WIDTHS},
        "unpruned": sweep(ref, UNPRUNED), "solutions": found,
        "ranks": ranks(ref, found[0]),
        "top_move": order[0], "top_score": round(first[0][0], 3),
        "start_ranks": sorted(order.index(path[0]) + 1 for path in found),
        "branching": len(order),
    }


def verify(result):
    seen, unpruned = result["widths"], result["unpruned"]
    return [
        practice.Check(
            "ANSWER: width 8 is wrong at 152, width 38 is right at 460, and so is BFS",
            all([seen[8]["value"] == -0.03, seen[8]["expansions"] == 152,
                 seen[37]["value"] < 0.99, seen[38]["value"] == 1.0,
                 seen[38]["expansions"] == 460, unpruned["value"] == 1.0,
                 unpruned["expansions"] == 460]),
            f"the shipped width 8 answers {seen[8]['value']} after "
            f"{seen[8]['expansions']} expansions; width 38 answers "
            f"{seen[38]['value']} after {seen[38]['expansions']}, which is exactly what "
            f"unpruned BFS costs ({unpruned['expansions']}). The beam saves nothing at "
            "the width where it starts being right",
        ),
        practice.Check(
            "FINDING: widths 4 through 32 buy nothing at all",
            all([seen[4]["trace"] == seen[8]["trace"] == seen[16]["trace"] ==
                 seen[32]["trace"], seen[4]["expansions"] == 88,
                 seen[32]["expansions"] == 438]),
            f"widths 4, 8, 16 and 32 all return {seen[8]['trace']} while the cost rises "
            f"from {seen[4]['expansions']} to {seen[32]['expansions']} expansions. Five "
            "times the budget and no change in the output: the extra candidates are all "
            "worse under a ranking that was already pointing the wrong way",
        ),
        practice.Check(
            "FINDING: the threshold is the winning prefix's rank, exactly",
            all([result["ranks"] == [7, 38, 1], max(result["ranks"][:2]) == 38,
                 seen[38]["value"] == 1.0, seen[37]["value"] == -0.01]),
            f"scoring every node of each unpruned level, the winning path sits at ranks "
            f"{result['ranks']}. The maximum over the pruned levels is "
            f"{max(result['ranks'][:2])}, and width 37 answers {seen[37]['value']} while "
            f"width 38 answers {seen[38]['value']} -- the threshold to the node",
        ),
        practice.Check(
            "FINDING: the heuristic points the wrong way at the root",
            all([result["top_move"] == "6*4=24", result["top_score"] == 0.0,
                 result["start_ranks"] == [7, 20], result["branching"] == 24,
                 len(result["solutions"]) == 2]),
            f"value scores a multi-number state by its closest number to 24, so "
            f"{result['top_move']!r} scores {result['top_score']} -- the best a "
            f"non-terminal state can score -- and is a dead end. Both real solutions "
            f"start at ranks {result['start_ranks']} of {result['branching']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
