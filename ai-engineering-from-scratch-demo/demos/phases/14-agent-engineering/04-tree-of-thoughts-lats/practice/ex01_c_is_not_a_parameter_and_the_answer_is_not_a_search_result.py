"""Exercise 1 — c is not a parameter, and the answer is not a search result.

    Run the toy LATS with UCT c=0.1 vs c=2.0. What changes in the trace?

Reading of the exercise: `mcts(root, iterations, rng)` has no `c`; the
constant is a default argument of `uct`, which `mcts` resolves through the
module globals at call time. So running the exercise at all means rebinding
`ref.uct`, and that is the first thing worth reporting. What "changes in the
trace" is then measured three ways -- the node set, the visit counters, and
the answer -- because those three turn out to give three different answers.

**ANSWER: at the demo's budget, almost nothing changes.** `c=0.1` and
`c=2.0` build the same **311** nodes in the same shape, expand **286** times
each, and disagree on the visit count of **15** of them. Both return
`['6*4=24', '24*1=24']` scoring **0.0**.

**FINDING: `c` has no way in.** `mcts` takes `root`, `iterations` and `rng`;
`uct` takes `c` as a keyword default. The exploration constant the exercise
asks about is not reachable from the search's own signature, so the only way
to vary it is to rebind the module attribute.

**FINDING: raise the budget and the trees do diverge -- and the answer still
does not.** At **300** iterations the two runs build **355** and **322**
nodes, expand **330** against **297** times, and differ on the visits of
**168** nodes. They both return `['6-1=5', '5*4=20', '20+4=24']`, scoring
**1.0**. `mcts` ends with `max(_all_leaves(root), key=value)`: visits and Q
decide which nodes exist and never decide which one is returned.

**FINDING: at 80 iterations the search has not finished a single
trajectory.** **0** of **286** leaves are depth-3 states, while the instance
has **4** winning paths (two distinct traces,
doubled because the numbers contain two 4s). The reported best is an unfinished path whose
score of **0.0** comes from having 24 among its leftovers -- the value
function's top score for a non-terminal state.

Structure: `run()` rebinds `ref.uct` under `try/finally` and returns the
whole tree, so structure and bookkeeping can be compared separately.
"""

from __future__ import annotations

import collections
import inspect
import random

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "04-tree-of-thoughts-lats"
SEED, SMALL, LARGE = 7, 80, 300


def walk(node, out=None):
    out = [] if out is None else out
    out.append(node)
    for child in node.children:
        walk(child, out)
    return out


def run(ref, c, iterations, seed=SEED):
    """The exercise's knob, reachable only by rebinding the module attribute."""
    shipped = ref.uct
    ref.uct = lambda parent, child, _c=c: shipped(parent, child, _c)
    try:
        root = ref.Node(state=tuple(sorted(ref.NUMBERS, reverse=True)), trace=[])
        root.children = ref.expand(root)
        best, expansions = ref.mcts(root, iterations, random.Random(seed))
    finally:
        ref.uct = shipped
    return root, best, expansions


def compare(ref, iterations):
    left, best_l, exp_l = run(ref, 0.1, iterations)
    right, best_r, exp_r = run(ref, 2.0, iterations)
    nodes_l, nodes_r = walk(left), walk(right)
    tally_l = collections.Counter((tuple(n.trace), n.visits) for n in nodes_l)
    tally_r = collections.Counter((tuple(n.trace), n.visits) for n in nodes_r)
    shape_l = sorted(tuple(n.trace) for n in nodes_l)
    return {
        "nodes": (len(nodes_l), len(nodes_r)), "same_shape": shape_l ==
        sorted(tuple(n.trace) for n in nodes_r),
        "visits_differ": sum((tally_l - tally_r).values()),
        "expansions": (exp_l, exp_r), "same_best": best_l.trace == best_r.trace,
        "best": best_l.trace, "value": round(ref.value(best_l), 3),
        "leaves": len(ref._all_leaves(left)),
        "complete": sum(1 for n in ref._all_leaves(left) if len(n.trace) == 3),
    }


def all_solutions(ref, node):
    if len(node.state) == 1:
        return [node.trace] if abs(node.state[0] - 24) < 1e-6 else []
    return [t for child in ref.expand(node) for t in all_solutions(ref, child)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    root = ref.Node(state=tuple(sorted(ref.NUMBERS, reverse=True)), trace=[])
    return {
        "small": compare(ref, SMALL), "large": compare(ref, LARGE),
        "mcts_params": list(inspect.signature(ref.mcts).parameters),
        "uct_params": list(inspect.signature(ref.uct).parameters),
        "uct_default": inspect.signature(ref.uct).parameters["c"].default,
        "solutions": len(all_solutions(ref, root)),
    }


def verify(result):
    small, large = result["small"], result["large"]
    return [
        practice.Check(
            "ANSWER: at 80 iterations, 15 visit counters change and nothing else",
            all([small["nodes"] == (311, 311), small["same_shape"] is True,
                 small["visits_differ"] == 15, small["expansions"] == (286, 286),
                 small["same_best"] is True, small["value"] == 0.0,
                 small["best"] == ["6*4=24", "24*1=24"]]),
            f"c=0.1 and c=2.0 build {small['nodes'][0]} nodes in the same shape "
            f"({small['same_shape']}), expand {small['expansions'][0]} times each, and "
            f"disagree on the visits of {small['visits_differ']}. Both return "
            f"{small['best']} scoring {small['value']}",
        ),
        practice.Check(
            "FINDING: c has no way into the search",
            all([result["mcts_params"] == ["root", "iterations", "rng"],
                 result["uct_params"] == ["parent", "child", "c"],
                 result["uct_default"] == 1.4]),
            f"mcts takes {result['mcts_params']} and uct takes {result['uct_params']} "
            f"with c defaulting to {result['uct_default']}. The constant the exercise "
            "asks about is not reachable from the search's signature, so varying it "
            "means rebinding a module attribute",
        ),
        practice.Check(
            "FINDING: raise the budget and the trees diverge -- the answer does not",
            all([large["nodes"] == (355, 322), large["same_shape"] is False,
                 large["visits_differ"] == 168, large["expansions"] == (330, 297),
                 large["same_best"] is True, large["value"] == 1.0]),
            f"at 300 iterations the two runs build {large['nodes']} nodes and expand "
            f"{large['expansions']} times, differing on the visits of "
            f"{large['visits_differ']}. Both return {large['best']} scoring "
            f"{large['value']}, because mcts ends with max(_all_leaves(root), key=value)",
        ),
        practice.Check(
            "FINDING: at 80 iterations not one trajectory is finished",
            all([small["complete"] == 0, small["leaves"] == 286,
                 result["solutions"] == 4, small["value"] == 0.0]),
            f"{small['complete']} of {small['leaves']} leaves are depth-3 states, while "
            f"the instance has {result['solutions']} winning paths. The reported "
            f"best scores {small['value']} because 24 is among its leftovers -- the top "
            "score the value function gives any unfinished state",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
