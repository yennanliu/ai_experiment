"""Exercise 3 -- a single judge false-accepts 70% of control-rerun papers, and the ensemble 85%.

    Swap the reviewer ensemble for a single judge. Measure the false-accept rate on a held-out set of known-bad papers.

Reading of the exercise: the lesson ships no reviewer (`main()` prints
"writer + reviewer + red-team steps would run here; stubbed"), so the
papers come from its own pipeline and the judges are a scaled-down
stand-in. 2,000 seeded runs of the reference `tree_search` each yield a
paper: the `best_branch` headline node. Two held-out known-bad sets are
labelled by construction: *control reruns*, where the headline config is
the root's own config (the claimed intervention is no change), and *weak*
papers written up from each tree's lowest-quality successful node. A judge
scores the 5 NeurIPS dimensions as `1 + 4 * signal` (novelty for novelty,
the node's quality for the other four) plus its own N(0, 0.5) leniency per
paper, clipped to 1-5; a paper passes at a mean of 4.0, the lesson's bar.
The ensemble is 5 such judges averaged; the single judge is the first.

**ANSWER: the single judge false-accepts 55.3% of the 3,666 known-bad
papers; the 5-judge ensemble 57.4%.** The swap moves the two sets in
opposite directions. On weak papers, whose mean signal (3.89) is under the
bar, the single judge's noise lets more through: 42.9% against 34.1%. On
control reruns, whose signal (4.29) is over it, averaging makes acceptance
more certain: the single judge passes 70.2%, the ensemble 85.4%. An ensemble
only lowers false accepts when the defect already shows in the scores.

**FINDING: 83.3% of the pipeline's papers are control reruns, and they score
above the real ones.** 1,666 of 2,000 headlines are the root's own config,
because 2 of the root's 5 children copy it (exercise 1) -- the shipped run's
best branch, "lr=0.0003", is one. Their noiseless mean is 4.285 against
4.259 for the 334 real papers, which judges accept at 70.1% / 85.0%.

Structure: `papers()` labels the pipeline's output; `judge()` scores one
paper; `rates()` counts acceptances for one set.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "05-autonomous-research-agent"
SEED = "investigate sparsity patterns in attention maps of sub-1B transformers"
RUNS, JUDGES, NOISE, BAR = 2000, 5, 0.5, 4.0


def papers(ref):
    sets = {"good": [], "rerun": [], "weak": []}
    for s in range(RUNS):
        with parity.quiet():
            tree = ref.tree_search(SEED, random.Random(s))
        head = ref.best_branch(tree)[-1]
        sets["rerun" if head.config == tree.root.config else "good"].append(head)
        sets["weak"].append(min((n for n in tree.nodes.values() if n.result and not n.failure),
                                key=lambda n: n.quality))
    return sets


def judge(node, rng):
    lenient = rng.gauss(0, NOISE)
    dims = [node.novelty] + [node.quality] * 4
    return sum(min(5.0, max(1.0, 1 + 4 * x + lenient)) for x in dims) / len(dims)


def rates(nodes, seed):
    """Accepted counts (single judge, ensemble) for one set of papers."""
    rng = random.Random(seed)
    single = ensemble = 0
    for node in nodes:
        scores = [judge(node, rng) for _ in range(JUDGES)]
        single += scores[0] >= BAR
        ensemble += sum(scores) / JUDGES >= BAR
    return single, ensemble


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sets = papers(ref)
    with parity.quiet():
        shipped = ref.best_branch(ref.tree_search(SEED, random.Random(7)))
    counts = {k: rates(v, i) for i, (k, v) in enumerate(sets.items())}
    return {
        "sizes": {k: len(v) for k, v in sets.items()},
        "rate": {k: tuple(round(c / len(sets[k]), 3) for c in cs) for k, cs in counts.items()},
        "pooled": tuple(round((counts["rerun"][i] + counts["weak"][i]) / (len(sets["rerun"]) + len(sets["weak"])), 3)
                        for i in (0, 1)),
        "mean": {k: round(sum(sum([1 + 4 * n.novelty] + [1 + 4 * n.quality] * 4) / 5 for n in v) / len(v), 3)
                 for k, v in sets.items()},
        "shipped": shipped[-1].hypothesis, "shipped_rerun": shipped[-1].config == shipped[0].config,
    }


def verify(result):
    r, rate = result, result["rate"]
    return [
        practice.Check(
            "ANSWER: the single judge false-accepts 55.3% of known-bad papers, the ensemble 57.4%",
            (r["sizes"], r["pooled"]) == ({"good": 334, "rerun": 1666, "weak": 2000}, (0.553, 0.574))
            and (rate["rerun"], rate["weak"]) == ((0.702, 0.854), (0.429, 0.341)),
            f"pooled FA single {r['pooled'][0]:.1%} vs ensemble {r['pooled'][1]:.1%} on "
            f"{r['sizes']['rerun'] + r['sizes']['weak']} papers; control reruns {rate['rerun'][0]:.1%} vs "
            f"{rate['rerun'][1]:.1%}; weak {rate['weak'][0]:.1%} vs {rate['weak'][1]:.1%}",
        ),
        practice.Check(
            "FINDING: 83.3% of the pipeline's papers are control reruns, and they score above the real ones",
            (r["mean"]["rerun"], r["mean"]["good"], rate["good"]) == (4.285, 4.259, (0.701, 0.85))
            and (r["shipped"], r["shipped_rerun"]) == ("lr=0.0003", True),
            f"{r['sizes']['rerun']}/{RUNS} headlines are the root config; noiseless mean "
            f"{r['mean']['rerun']} vs {r['mean']['good']} for real ones, accepted {rate['good']}; main()'s headline '{r['shipped']}' is a rerun: {r['shipped_rerun']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
