"""Exercise 2 -- a $5 approval gate never fires, because no node can cost more than $1.60.

    Add a human-in-the-loop gate before experiment execution for nodes estimated above $5. Measure how much total cost drops.

Reading of the exercise: the gate wraps the lesson's `run_experiment`, so
the reference `tree_search` calls it before anything executes. The lesson
has no pre-run estimate, so the gate uses the best one possible: an oracle
that runs the stub on a copy of the node and of the rng and reads the cost
it would charge. A node above the threshold is declined (the stand-in human
says no): it is marked failed, costs $0 and is not expanded. Total spend is
compared with and without the gate over 1,000 rng seeds, and the threshold
is swept to find where a gate starts to bite.

**ANSWER: total cost drops by 0.0%.** Over 1,000 seeded runs the $5 gate
declines 0 of 4,438 experiments and total spend is $6,207.24 both ways.
`run_experiment` charges `1.2 + uniform(0, 0.4)`, so the most expensive node
seen is $1.5999; a $5 threshold sits above every node the stub can produce.

**FINDING: the $30 budget never binds either.** The most any of the 1,000
runs spends is $12.75 (mean $6.21); the search stops on its 24-node cap,
which counts unexecuted frontier nodes. The lesson's "Use It" transcript
shows "budget 12/30" after 8 nodes and "$28.40 spent"; `main()` prints
$5.57 after 4 experiments.

**FINDING: the gate starts to bite only inside the $1.20-$1.60 cost band.**
A gate at $1.50 declines 6,849 nodes and cuts spend 52.9%; at $1.40, 81.3%;
at $1.20 it declines all 5 root children of every run and spend falls 100%.
A declined node is never expanded, so most of the saving is the search
starving, not the expensive nodes it skipped.

Structure: `gated()` builds the replacement `run_experiment`; `sweep()`
reruns the reference search with it installed.
"""

from __future__ import annotations

import copy
import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "05-autonomous-research-agent"
SEED = "investigate sparsity patterns in attention maps of sub-1B transformers"
THRESHOLDS = (None, 5.0, 1.5, 1.4, 1.2)
RUNS = 1000


def gated(original, threshold, log):
    def run_experiment(node, rng):
        probe = copy.deepcopy(node)
        original(probe, copy.deepcopy(rng))
        log.append(probe.cost_usd)
        if threshold is not None and probe.cost_usd > threshold:
            node.failure, node.cost_usd = "declined_by_human", 0.0
            return
        original(node, rng)

    return run_experiment


def sweep(ref, threshold):
    original, log, spent, declined = ref.run_experiment, [], 0.0, 0
    ref.run_experiment = gated(original, threshold, log)
    try:
        for s in range(RUNS):
            with parity.quiet():
                tree = ref.tree_search(SEED, random.Random(s))
            spent += tree.spent
            declined += sum(n.failure == "declined_by_human" for n in tree.nodes.values())
    finally:
        ref.run_experiment = original
    return {"spent": round(spent, 2), "declined": declined, "estimated": len(log), "max_est": max(log)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {t: sweep(ref, t) for t in THRESHOLDS}
    base = rows[None]["spent"]
    with parity.quiet():
        shipped = ref.tree_search(SEED, random.Random(7))
    spends = []
    for s in range(RUNS):
        with parity.quiet():
            spends.append(ref.tree_search(SEED, random.Random(s)).spent)
    doc = parity.doc_text(PHASE, LESSON, "en")
    return {
        "rows": rows, "drop": {t: round(100 * (1 - r["spent"] / base), 1) for t, r in rows.items()},
        "max_spend": round(max(spends), 2), "mean_spend": round(sum(spends) / RUNS, 2),
        "shipped": round(shipped.spent, 2), "doc": ["budget 12/30" in doc, "$28.40 spent" in doc],
    }


def verify(result):
    r, rows, drop = result, result["rows"], result["drop"]
    five = rows[5.0]
    return [
        practice.Check(
            "ANSWER: the $5 gate declines 0 nodes and total cost drops 0.0%",
            (five["declined"], five["estimated"], drop[5.0], five["spent"]) == (0, 4438, 0.0, 6207.24)
            and round(five["max_est"], 4) == 1.5999,
            f"declined {five['declined']}/{five['estimated']}; spend ${five['spent']} vs "
            f"${rows[None]['spent']} ungated; costliest node ${five['max_est']:.4f}",
        ),
        practice.Check(
            "FINDING: the $30 budget never binds either",
            (r["max_spend"], r["mean_spend"], r["shipped"], r["doc"]) == (12.75, 6.21, 5.57, [True, True]),
            f"max ${r['max_spend']}, mean ${r['mean_spend']} over {RUNS} runs; main() spends "
            f"${r['shipped']}; the doc shows budget 12/30 and $28.40 spent: {r['doc']}",
        ),
        practice.Check(
            "FINDING: the gate starts to bite only inside the $1.20-$1.60 cost band",
            [drop[t] for t in (1.5, 1.4, 1.2)] == [52.9, 81.3, 100.0]
            and [rows[t]["declined"] for t in (1.5, 1.4, 1.2)] == [6849, 7460, 5000],
            "; ".join(f"${t}: -{drop[t]}% ({rows[t]['declined']} declined)" for t in (1.5, 1.4, 1.2)),
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
