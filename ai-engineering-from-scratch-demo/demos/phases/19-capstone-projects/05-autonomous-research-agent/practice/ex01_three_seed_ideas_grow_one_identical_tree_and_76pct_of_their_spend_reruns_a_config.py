"""Exercise 1 -- three seed ideas grow one identical tree, and 76% of their spend reruns a config.

    Run the pipeline against three different seed ideas in the same domain. Compare which parts of the tree-search overlap. Identify duplicated wasted compute.

Reading of the exercise: "the pipeline" is the lesson's `tree_search` as
`main()` runs it (`random.Random(7)`, $30 budget, 24-node cap), called
with three seed ideas about attention sparsity. Overlap is compared per
executed experiment (config, loss, cost), and wasted compute is every
dollar spent running a config that had already been run -- inside one
tree, across the three trees, and over 1,000 rng seeds.

**ANSWER: the three trees overlap completely.** The seed text is stored in
`root.hypothesis` and read by nothing, so all three runs execute the same 4
experiments with the same losses and costs. Of the 12 runs ($16.70), only 3
configs are distinct; 9 runs ($12.66, 75.8%) repeat a config already run.

**FINDING: every expansion proposes 2 copies of its parent.** `expand`
varies `sparsity_top` over (4, 8, 16) and `lr` over (3e-4, 1e-3); the
parent always holds one value of each, so 2 of 5 children are its own
config. Inside the shipped tree, nodes #02 and #04 both run the root
config, $1.53 of $5.57 (27.5%) wasted; the 26 nodes hold 6 distinct
configs. Over 1,000 rng seeds, 24.9% of executed experiments rerun a
config already run in the same tree.

Structure: `run()` is the reference `tree_search` with its print muted;
`executed()` lists what actually ran; `waste()` prices repeats.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "05-autonomous-research-agent"
SEEDS = [
    "investigate sparsity patterns in attention maps of sub-1B transformers",
    "does top-k attention sparsity hurt long-context recall in 125M-parameter models",
    "use sparse attention heads as a pruning signal in small transformers",
]


def run(ref, seed, rng_seed=7):
    with parity.quiet():
        return ref.tree_search(seed, random.Random(rng_seed))


def key(node):
    return tuple(sorted(node.config.items()))


def executed(tree):
    return [n for n in sorted(tree.nodes.values(), key=lambda n: n.node_id) if n.result]


def waste(nodes):
    """Dollars and count spent on configs already run earlier in `nodes`."""
    seen, dollars, count = set(), 0.0, 0
    for n in nodes:
        if key(n) in seen:
            dollars, count = dollars + n.cost_usd, count + 1
        seen.add(key(n))
    return dollars, count


def signature(tree):
    return [(key(n), n.result["loss"], round(n.cost_usd, 4)) for n in executed(tree)]


def within(tree):
    """What one tree wastes on its own: parent clones, root reruns, distinct configs."""
    children = [n for n in tree.nodes.values() if n.parent is not None]
    return {
        "clones": sum(key(n) == key(tree.nodes[n.parent]) for n in children), "children": len(children),
        "one_spent": round(tree.spent, 2), "one_waste": round(waste(executed(tree))[0], 2),
        "dup_ids": [n.node_id for n in executed(tree) if key(n) == key(tree.root)],
        "tree_distinct": len({key(n) for n in tree.nodes.values()}),
    }


def across(trees):
    """What the three trees share: identical runs, and repeats pooled over all three."""
    runs = [signature(t) for t in trees]
    pooled = [n for t in trees for n in executed(t)]
    dollars, repeats = waste(pooled)
    return {
        "identical": runs[0] == runs[1] == runs[2],
        "hypothesis_only": [t.root.hypothesis == s for t, s in zip(trees, SEEDS)],
        "runs": len(pooled), "distinct": len({key(n) for n in pooled}),
        "total": round(sum(n.cost_usd for n in pooled), 2),
        "pooled_waste": round(dollars, 2), "pooled_repeats": repeats,
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    trees = [run(ref, s) for s in SEEDS]
    many = [executed(run(ref, SEEDS[0], s)) for s in range(1000)]
    rate = sum(waste(e)[1] for e in many) / sum(len(e) for e in many)
    return {**within(trees[0]), **across(trees), "many_rate": round(rate, 3)}


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: the three seed ideas grow one identical tree; 9 of 12 runs repeat a config",
            (r["identical"], r["hypothesis_only"], r["runs"], r["distinct"], r["total"])
            == (True, [True] * 3, 12, 3, 16.7)
            and (r["pooled_repeats"], r["pooled_waste"]) == (9, 12.66),
            f"identical runs {r['identical']}; {r['runs']} runs, {r['distinct']} distinct configs; "
            f"${r['pooled_waste']} of ${r['total']} "
            f"({100 * r['pooled_waste'] / r['total']:.1f}%) reruns a config",
        ),
        practice.Check(
            "FINDING: every expansion proposes 2 copies of its parent",
            (r["clones"], r["children"], r["dup_ids"], r["one_waste"], r["one_spent"]) == (10, 25, [2, 4], 1.53, 5.57)
            and (r["tree_distinct"], r["many_rate"]) == (6, 0.249),
            f"{r['clones']}/{r['children']} children copy their parent; nodes {r['dup_ids']} both run the "
            f"root config (${r['one_waste']} of ${r['one_spent']}); {r['tree_distinct']} distinct configs "
            f"in 26 nodes; {100 * r['many_rate']:.1f}% of runs repeat a config over 1,000 rng seeds",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
