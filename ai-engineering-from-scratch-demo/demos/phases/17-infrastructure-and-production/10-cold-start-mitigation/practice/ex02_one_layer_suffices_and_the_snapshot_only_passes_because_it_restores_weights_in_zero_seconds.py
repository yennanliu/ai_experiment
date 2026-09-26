"""Exercise 2 — one layer suffices, and the snapshot only passes because it restores weights in zero seconds.

    You deploy a 13B model with P99 TTFT SLA of 3s. Pick the minimum
    mitigation stack (fewest layers) that achieves it.

Reading of the exercise: the reference only has `PHASES_70B`, so a 13B path is
derived from it: the size-dependent phases (image pull, weights to HBM, first
forward) scale by 13/70, node provision and engine init do not. Every subset of
the reference's three layers is run through the reference's own
`total_for_stack`; a warm pool (`min_workers=1`, the fourth layer) leaves only
the first forward on the path, assuming the pool is sized so a P99 request
finds a warm slot.

**ANSWER: one layer -- a warm pool, at 0.56 s.** Raw 13B cold start is
117.91 s; pre-seeded 84.49 s, streamer 110.49 s, both 77.06 s -- no stack
without a snapshot or a warm pool gets within 25x of 3 s. Two single layers
pass: the warm pool (0.56 s) and the GPU snapshot (2.59 s). The warm pool is the
pick, because the snapshot's pass does not survive the next finding.

**FINDING: the snapshot passes only because it restores weights in 0.0 s.** The
snapshot column gives `weights to HBM` 0.0 s at any size -- the same 3.0 s total
for 70B as the raw first forward alone. Charge the restore for the 26 GB of bf16
weights at exercise 3's 7 GB/s and the 13B snapshot path is 6.30 s, over the
SLA. Modal's own docs say memory snapshots start functions "3-10x faster", not
in zero time.

**FINDING: under a snapshot every other layer is ignored.** `total_for_stack`
returns the snapshot column whenever `gpu_snapshot` is in the stack, so the 4
stacks containing it all give 2.59 s -- stacking cannot be measured. And the
code cannot be asked about 13B at all: the phases are the module constant
`PHASES_70B` and `total_for_stack` takes no size (this solution swaps the
constant and restores it). The lesson's "~15s" with mitigations matches none of
the reference's stacks for 70B (328, 148, 288, 108, 3.0 s).

Structure: `sized()` scales the reference phases; `stacks()` runs every subset.
"""

from __future__ import annotations

import dataclasses
import itertools

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "10-cold-start-mitigation"
LAYERS = ("pre_seeded", "streamer", "gpu_snapshot")
SIZED = ("image pull", "weights to HBM", "first forward")
SLA, PARAMS_B, NVME_GBPS = 3.0, 13, 7.0


def sized(phases, params_b):
    k = params_b / 70
    fields = ("raw_sec", "pre_seeded_sec", "streamer_sec", "snapshot_sec")
    return [dataclasses.replace(p, **{f: getattr(p, f) * k for f in fields})
            if p.name in SIZED else p for p in phases]


def stacks(ref, phases):
    """{stack: seconds} for every subset of the three layers, via the reference."""
    original, ref.PHASES_70B = ref.PHASES_70B, phases
    try:
        return {s: round(ref.total_for_stack(set(s)), 2)
                for n in range(4) for s in itertools.combinations(LAYERS, n)}
    finally:
        ref.PHASES_70B = original


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    p13 = sized(ref.PHASES_70B, PARAMS_B)
    totals = stacks(ref, p13)
    warm = round(next(p.raw_sec for p in p13 if p.name == "first forward"), 2)
    candidates = dict(totals, warm_pool=warm)
    passing = [s for s, t in candidates.items() if t <= SLA]
    snapshot_hbm = [p.snapshot_sec for p in ref.PHASES_70B if p.name == "weights to HBM"]
    restore = PARAMS_B * 2 / NVME_GBPS
    return {
        "totals": totals, "warm": warm, "passing": passing,
        "fewest": min(1 if s == "warm_pool" else len(s) for s in passing),
        "snapshot_hbm": snapshot_hbm[0],
        "snapshot_real": round(totals[("gpu_snapshot",)] + restore, 2),
        "totals_70b": sorted(set(stacks(ref, ref.PHASES_70B).values())),
    }


def verify(result):
    totals = result["totals"]
    with_snap = {t for s, t in totals.items() if "gpu_snapshot" in s}
    return [
        practice.Check(
            "ANSWER: one layer -- a warm pool, at 0.56 s",
            all([result["fewest"] == 1, result["warm"] == 0.56,
                 set(result["passing"]) >= {"warm_pool", ("gpu_snapshot",)},
                 min(t for s, t in totals.items() if "gpu_snapshot" not in s) > 25 * SLA]),
            f"13B stacks {totals}; warm path {result['warm']}s; passing at {SLA}s: "
            f"{result['passing']}",
        ),
        practice.Check(
            "FINDING: the snapshot passes only because it restores weights in 0.0 s",
            result["snapshot_hbm"] == 0.0 and result["snapshot_real"] > SLA,
            f"snapshot 'weights to HBM' is {result['snapshot_hbm']}s at any size; with "
            f"{PARAMS_B * 2} GB read at {NVME_GBPS} GB/s the path is "
            f"{result['snapshot_real']}s",
        ),
        practice.Check(
            "FINDING: under a snapshot every other layer is ignored",
            with_snap == {2.59} and 15.0 not in result["totals_70b"],
            f"all 4 stacks with gpu_snapshot give {with_snap}; the reference's 70B "
            f"totals are {result['totals_70b']}, none of them the lesson's ~15s",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
