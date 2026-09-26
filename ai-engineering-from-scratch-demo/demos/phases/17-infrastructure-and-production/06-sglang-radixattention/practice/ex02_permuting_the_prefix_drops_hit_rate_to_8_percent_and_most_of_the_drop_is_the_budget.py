"""Exercise 2 — permuting the prefix drops hit rate to 8%, and most of the drop is the budget.

    Modify the workload so prompts randomly permute `[system, tools, context]`.
    Re-run. What happens to hit rate? Why?

Reading of the exercise: the module already ships this modification as
`workload_scrambled`, but it also redraws which document each request gets. So
a controlled version is built too: `workload_rag`'s requests unchanged except
for a seeded shuffle of their first three segments. Both are run at the shipped
160-block budget and with the budget removed, which separates what ordering
costs from what eviction costs.

**ANSWER: hit rate collapses from 69.1% to 8.4%**, and the controlled
permutation lands in the same place: 5.5% to 15.1% over five seeds. The radix
key is the ordered path, so each of the 3! orders is a separate branch.
SYSTEM comes first in only 25 of 80 requests, and the shared 3-segment heads
multiply from 4 to 24.

**FINDING: most of that drop is the budget, not the tree failing to match.**
With the budget removed, fixed order reaches 96.0% and the scrambled workload
79.5%. Ordering alone costs 16.6 points. At 160 blocks the loss is 60.6 points:
24 branches compete for a cache that cannot hold one full request (180 blocks),
and SYSTEM is only reused when it happens to come first. The fix is still to
fix the order, and doing that also shrinks the working set the budget must
hold.

**FINDING: the toy shows no 6.4x.** Use It says the scrambled run shows "the
6.4x collapse". Here hit rate falls 8.2x (69.1 / 8.4), and prefill tokens that
must be computed rise 2.96x (70800 -> 209500). Neither is 6.4x. The lesson's 6.4x is a
throughput figure ("up to 6.4x on SGLang"), and this toy measures no throughput.

Structure: `permuted()` is the controlled workload; `run()` swaps the
reference's RadixCache for one with a chosen budget and restores it.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "06-sglang-radixattention"
SEEDS, UNBOUNDED = range(5), 10**6


def permuted(ref, seed):
    """workload_rag with only the order of [SYSTEM, TOOLS, DOC] shuffled."""
    rng, out = random.Random(seed), []
    for r in ref.workload_rag():
        head = r.segments[:3]
        rng.shuffle(head)
        out.append(ref.Request(r.rid, head + r.segments[3:]))
    return out


def run(ref, reqs, budget=160):
    original = ref.RadixCache
    ref.RadixCache = lambda: original(budget)
    try:
        return ref.simulate(reqs, "FCFS")
    finally:
        ref.RadixCache = original


def rate(res):
    return round(res["hit_rate"], 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rag, scr = ref.workload_rag(), ref.workload_scrambled()
    fixed, scrambled = run(ref, rag), run(ref, scr)
    return {
        "fixed": (rate(fixed), rate(run(ref, rag, UNBOUNDED))),
        "scrambled": (rate(scrambled), rate(run(ref, scr, UNBOUNDED))),
        "controlled": [rate(run(ref, permuted(ref, s))) for s in SEEDS],
        "system_first": sum(r.segments[0] == "SYSTEM" for r in scr),
        "heads": (len({tuple(r.segments[:3]) for r in rag}),
                  len({tuple(r.segments[:3]) for r in scr})),
        "computed": (fixed["total"] - fixed["saved"], scrambled["total"] - scrambled["saved"]),
        "claims_64": "6.4x collapse" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    fixed, scr, ctl = result["fixed"], result["scrambled"], result["controlled"]
    ordering_cost = round(fixed[1] - scr[1], 4)
    budget_cost = round(fixed[0] - scr[0], 4)
    computed = result["computed"]
    return [
        practice.Check(
            "ANSWER: hit rate collapses from 69.1% to 8.4%",
            all([(fixed[0], scr[0]) == (0.6906, 0.0844), (min(ctl), max(ctl)) == (0.0546, 0.1512),
                 result["system_first"] == 25, result["heads"] == (4, 24)]),
            f"fixed {fixed[0]:.1%}, scrambled {scr[0]:.1%}, controlled permutation "
            f"{min(ctl):.1%}..{max(ctl):.1%} over {len(ctl)} seeds; SYSTEM first in "
            f"{result['system_first']}/80, distinct heads {result['heads']}",
        ),
        practice.Check(
            "FINDING: most of the drop is the budget, not the tree failing to match",
            fixed[1] == 0.9602 and scr[1] == 0.7946 and budget_cost > 3 * ordering_cost,
            f"unbounded: {fixed[1]:.1%} vs {scr[1]:.1%} ({ordering_cost * 100:.1f} points); "
            f"at 160 blocks: {budget_cost * 100:.1f} points",
        ),
        practice.Check(
            "FINDING: the toy shows no 6.4x",
            result["claims_64"] and round(fixed[0] / scr[0], 1) == 8.2
            and round(computed[1] / computed[0], 2) == 2.96,
            f"hit rate falls {fixed[0] / scr[0]:.1f}x and prefill tokens computed rise "
            f"{computed[0]} -> {computed[1]} ({computed[1] / computed[0]:.2f}x)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
