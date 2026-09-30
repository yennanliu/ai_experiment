"""Exercise 5 -- the tree search is a FIFO queue and loses 7.3% to flat random at the same $30.

    Compare your tree-search with a flat random baseline (same budget, no expansion strategy). Report the novelty × quality gain.

Reading of the exercise: "your tree-search" is the lesson's `tree_search`
as shipped ($30 budget, 24-node cap). The flat baseline draws configs
uniformly from the same 6-config grid `expand` can reach and runs each
through the reference `run_experiment` and `verify`, with no expansion.
"Same budget" is taken two ways: the same $30 the tree is given, and the
same number of experiments the tree actually ran. The metric is the best
novelty x quality among successful runs, averaged over 1,000 seeds (the
tree on rng seed s, the baseline on 10,000 + s).

**ANSWER: at the same $30, the flat baseline wins: 0.586 against the
tree's 0.543, a gain of -7.3%.** Matched to the tree's experiment count
instead (4.4 runs on average), the tree gains +2.6% (0.543 vs 0.529). The
tree cannot use its budget: it stops on the node cap after $6.21 on
average, while the baseline spends its $30 on about 21 runs.

**FINDING: the "best-first" search is first-in-first-out.** A node's score
is fixed when it is pushed, before it runs, so every child scores
0.4 x 0.5 + 0.5 x 0 + 0.1 = 0.3 and the heap breaks ties by insertion
order. In all 1,000 runs the executed node ids are exactly 1, 2, ..., k;
no run goes deeper than 2 levels.

**FINDING: search has nothing to find on two of the three axes.** Novelty
is `0.5 + uniform(-0.1, 0.2)`, drawn without reading the config: on one rng
all 6 configs get the same novelty. The `lr` edit leaves the reported loss
unchanged (2.747 both ways at top-8, same rng). The lesson's
"novelty x quality x budget" score is a weighted sum in code.

Structure: `flat()` is the baseline; `best()` scores one run's nodes;
`depth()` walks parents.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "05-autonomous-research-agent"
SEED = "investigate sparsity patterns in attention maps of sub-1B transformers"
GRID = [{"sparsity_top": s, "lr": lr} for s in (4, 8, 16) for lr in (3e-4, 1e-3)]
RUNS = 1000


def best(nodes):
    return max((n.novelty * n.quality for n in nodes if n.result and not n.failure), default=0.0)


def flat(ref, rng, budget=None, count=None):
    nodes, spent = [], 0.0
    while (count is None or len(nodes) < count) and (budget is None or spent < budget):
        node = ref.Node(node_id=len(nodes) + 1, parent=0, hypothesis="random", config=dict(rng.choice(GRID)))
        ref.run_experiment(node, rng)
        ref.verify(node)
        spent += node.cost_usd
        nodes.append(node)
    return nodes, spent


def depth(tree, node):
    return 0 if node.parent is None else 1 + depth(tree, tree.nodes[node.parent])


def on_same_rng(ref, configs, field):
    """`field` of one experiment per config, each on a fresh Random(0)."""
    out = []
    for cfg in configs:
        node = ref.Node(node_id=1, parent=0, hypothesis="probe", config=cfg)
        ref.run_experiment(node, random.Random(0))
        out.append(node.result["loss"] if field == "loss" else node.novelty)
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tree_best, same_count, same_dollars, fifo, depths, spent, flat_runs = [], [], [], 0, set(), [], []
    for s in range(RUNS):
        with parity.quiet():
            tree = ref.tree_search(SEED, random.Random(s))
        ran = [n for n in sorted(tree.nodes.values(), key=lambda n: n.node_id) if n.result]
        fifo += [n.node_id for n in ran] == list(range(1, len(ran) + 1))
        depths.update(depth(tree, n) for n in ran)
        tree_best.append(best(ran))
        spent.append(tree.spent)
        same_count.append(best(flat(ref, random.Random(10_000 + s), count=len(ran))[0]))
        nodes, _ = flat(ref, random.Random(10_000 + s), budget=tree.budget)
        same_dollars.append(best(nodes))
        flat_runs.append(len(nodes))
    child = ref.Node(node_id=1, parent=0, hypothesis="child", config={})
    mean = lambda xs: round(sum(xs) / len(xs), 3)  # noqa: E731
    return {
        "tree": mean(tree_best), "same_count": mean(same_count), "same_dollars": mean(same_dollars),
        "tree_spent": round(sum(spent) / RUNS, 2), "flat_runs": round(sum(flat_runs) / RUNS, 1),
        "fifo": fifo, "max_depth": max(depths), "child_score": round(child.score(30.0), 3),
        "lr_losses": on_same_rng(ref, GRID[2:4], "loss"),
        "novelties": len(set(on_same_rng(ref, GRID, "novelty"))),
    }


def verify(result):
    r = result
    gain = lambda base: round(100 * (r["tree"] / base - 1), 1)  # noqa: E731
    return [
        practice.Check(
            "ANSWER: at the same $30 the tree's novelty x quality gain is -7.3%; at equal runs +2.6%",
            (r["tree"], r["same_dollars"], r["same_count"]) == (0.543, 0.586, 0.529)
            and (gain(r["same_dollars"]), gain(r["same_count"])) == (-7.3, 2.6),
            f"tree {r['tree']} vs flat@$30 {r['same_dollars']} ({gain(r['same_dollars'])}%) and flat@equal "
            f"runs {r['same_count']} ({gain(r['same_count'])}%); tree spends ${r['tree_spent']}, flat "
            f"runs {r['flat_runs']} experiments",
        ),
        practice.Check(
            "FINDING: the best-first search is first-in-first-out",
            (r["child_score"], r["fifo"], r["max_depth"]) == (0.3, RUNS, 2),
            f"every child pushed at score {r['child_score']}; FIFO order in {r['fifo']}/{RUNS} runs; "
            f"max depth {r['max_depth']}",
        ),
        practice.Check(
            "FINDING: search has nothing to find on two of the three axes",
            r["lr_losses"] == [2.747, 2.747] and r["novelties"] == 1,
            f"lr 3e-4 vs 1e-3 at top-8: losses {r['lr_losses']}; {r['novelties']} distinct novelty "
            f"value over all {len(GRID)} configs on one rng",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
