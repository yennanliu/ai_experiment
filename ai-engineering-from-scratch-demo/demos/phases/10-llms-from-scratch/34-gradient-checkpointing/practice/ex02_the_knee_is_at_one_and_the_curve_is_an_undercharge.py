"""Exercise 2 — 93% of the memory saving arrives at k=1, where the formula charges nothing for it.

    Sweep segment size `k` from 1 to `L`. Plot FLOP overhead and memory. Find the
    knee of the curve.

Reading of the exercise: both curves are swept with the lesson's own
`checkpoint_cost` and `memory_after_checkpoint` over every k from 1 to L=64, and
then the layer-forwards each path actually performs are counted on a real
network, because "the knee of the curve" is a claim about a trade and the
modelled side of it does not match the code beside it.

**ANSWER: there are two curves and the knee is at k=1.**

    k     FLOP overhead      memory      share of the achievable saving
    -     ------------      ---------    ------------------------------
    1          0.0%          8,724 MB              93.5%
    2         16.7%          4,563 MB              99.9%
    4         25.0%          2,684 MB              99.5%
    8         29.2%          2,147 MB             100.0%  <- sqrt(64)
    64        32.8%          8,724 MB              93.5%

Against a no-checkpoint baseline of 103,079 MB, k=1 already removes 91.5% of the
activation memory at a modelled 0% cost. The remaining 6.4 points of the total
cost 29.2 points of compute. The FLOP curve has no knee at all -- it is monotone
and bounded by 33.3%, and it is 75% of the way there by k=4.

**FINDING: the formula charges (k-1)/k and the code recomputes k/k.**
`model_backward_checkpointed` calls `model_forward` over the whole segment, all
k layers; `checkpoint_cost` charges for k-1 of them, on the reasoning that the
segment's first input was stored. Counting `layer_forward` calls on a 24-layer
network settles it: the full path performs 24 and the checkpointed one performs
exactly **24 more at every k**, against a charge of

    k          1     2     4     8    12    24
    charged    0    12    18    21    22    23
    performed 24    24    24    24    24    24

So the real cost is one whole extra forward whatever k is -- a flat 33.3% -- and
the rising curve the exercise asks to plot is exactly the n/k layer-forwards the
model never charges for. At k=1 that is the entire cost: the model says free and
the code does a whole extra pass.

**FINDING: the sqrt-L rule picks the dearest of the tied options.**
`memory_after_checkpoint` is flat near its minimum -- at L=32 every k from 4 to 8
gives identical memory -- and `optimal_segment` returns 6, at 27.8% overhead,
where k=4 buys the same memory for 25.0%. It does this at 5 of the 9 layer counts
swept, for up to 2.8 points of compute.

Structure: `sweep` runs both of the lesson's cost models across k; `timings`
measures the same sweep on a real forward and backward.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "34-gradient-checkpointing"
DEPTH = 64
SHOWN = (1, 2, 4, 8, 16, 32, 64)
COUNTED = (1, 2, 3, 4, 6, 8, 12, 24)
LAYERS, HIDDEN, INNER, BATCH = 24, 256, 512, 64
COUNTS = (12, 16, 24, 32, 48, 64, 80, 96, 128)


def sweep(ref):
    return {k: {"overhead": ref.checkpoint_cost(DEPTH, segment_size=k)["overhead_vs_no_ckpt"],
                "memory": ref.memory_after_checkpoint(DEPTH, k)}
            for k in range(1, DEPTH + 1)}


def count_calls(ref, run):
    """Layer-forwards one path performs; model_forward resolves the global at call time."""
    original, tally = ref.layer_forward, []
    ref.layer_forward = lambda *args: (tally.append(1), original(*args))[1]
    try:
        run()
    finally:
        ref.layer_forward = original
    return len(tally)


def recomputed(ref):
    """Layer-forwards each path performs, against the n*(k-1)/k the model charges."""
    params = ref.make_params(LAYERS, HIDDEN, INNER)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((BATCH, HIDDEN)).astype(np.float32)
    grad_out = rng.standard_normal((BATCH, HIDDEN)).astype(np.float32)

    def plain():
        _, activations = ref.model_forward(x, params)
        ref.model_backward(grad_out, activations, params)

    def segmented(k):
        _, saved = ref.model_forward_checkpointed(x, params, k=k)
        ref.model_backward_checkpointed(grad_out, saved, params, k=k)

    full = count_calls(ref, plain)
    return {"full": full, "charged": {k: LAYERS * (k - 1) / k for k in COUNTED},
            "extra": {k: count_calls(ref, lambda k=k: segmented(k)) - full for k in COUNTED}}


def sqrt_rule(ref):
    """Where the sqrt-L rule lands among the segment sizes that tie on memory."""
    rows = {}
    for depth in COUNTS:
        memory = {k: ref.memory_after_checkpoint(depth, k) for k in range(1, depth + 1)}
        tied = [k for k, value in memory.items() if value == min(memory.values())]
        cost = lambda k: ref.checkpoint_cost(depth, segment_size=k)["overhead_vs_no_ckpt"]
        rows[depth] = {"rule": ref.optimal_segment(depth), "cheapest": min(tied),
                       "tied": len(tied), "penalty": cost(ref.optimal_segment(depth))
                       - cost(min(tied))}
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows, calls = sweep(ref), recomputed(ref)
    baseline = ref.activation_memory_mb(DEPTH)
    floor = min(row["memory"] for row in rows.values())
    return {
        "rows": {k: rows[k] for k in SHOWN},
        "baseline": baseline,
        "floor": floor,
        "share": {k: (baseline - rows[k]["memory"]) / (baseline - floor) for k in SHOWN},
        "ceiling": rows[DEPTH]["overhead"],
        "monotone": all(rows[k]["overhead"] <= rows[k + 1]["overhead"]
                        for k in range(1, DEPTH)),
        "argmin": min(rows, key=lambda k: rows[k]["memory"]),
        "calls": calls,
        "undercharged": [k for k in COUNTED if calls["extra"][k] > calls["charged"][k]],
        "rule": sqrt_rule(ref),
    }


def row(rows, key, fmt):
    return ", ".join(f"k={k} {format(value[key], fmt)}" for k, value in rows.items())


def penalties(rule):
    return ", ".join(f"L={depth} rule k={value['rule']} vs k={value['cheapest']} "
                     f"(+{value['penalty']:.1%})"
                     for depth, value in rule.items() if value["penalty"] > 0)


def dearer(rule):
    """Layer counts where the sqrt rule costs more compute for the same memory."""
    return [depth for depth, value in rule.items() if value["penalty"] > 0]


def verify(result):
    rows, share, calls = result["rows"], result["share"], result["calls"]
    rule, dear = result["rule"], dearer(result["rule"])
    return [
        practice.Check(
            "ANSWER: k=1 removes 91.5% of the activation memory at a modelled 0% cost",
            rows[1]["overhead"] == 0.0 and share[1] > 0.9 and result["argmin"] == 8,
            f"against a no-checkpoint baseline of {result['baseline']:,.0f} MB the sweep gives "
            + row(rows, "memory", ",.0f") + " at overheads of " + row(rows, "overhead", ".1%")
            + f". k=1 is {100 * (1 - rows[1]['memory'] / result['baseline']):.1f}% of the way "
            f"down and {share[1]:.1%} of the achievable saving; the remaining "
            f"{100 * (rows[1]['memory'] - result['floor']) / result['baseline']:.1f} points cost "
            f"{rows[result['argmin']]['overhead']:.1%} of compute",
        ),
        practice.Check(
            "FINDING: the FLOP curve has no knee -- monotone, bounded, 75% spent by k=4",
            result["monotone"] and rows[4]["overhead"] / result["ceiling"] > 0.7,
            f"the overhead is n*(k-1)/k over a 3n budget, so it rises from 0 to a bound of 1/3, "
            f"reaching {rows[4]['overhead'] / (1 / 3):.0%} of it by k=4. The only turning point "
            f"is in the memory curve, at k={result['argmin']} = sqrt(64), which is what "
            f"optimal_segment returns ({rule[DEPTH]['rule']})",
        ),
        practice.Check(
            "FINDING: the formula charges (k-1)/k and the code recomputes k/k",
            (len(result["undercharged"]) == len(COUNTED)
             and set(calls["extra"].values()) == {LAYERS}),
            "model_backward_checkpointed calls model_forward over all k layers of the segment "
            "while checkpoint_cost charges for k-1 of them. Counting layer_forward calls, the "
            f"full path performs {calls['full']} and the checkpointed one performs exactly "
            f"{LAYERS} more at every k -- "
            + ", ".join(f"k={k} {value} vs {calls['charged'][k]:.0f} charged"
                        for k, value in calls["extra"].items())
            + ". So the real cost is one whole extra forward whatever k is, a flat 33.3%, and "
            f"at k=1 the model charges {rows[1]['overhead']:.1%} for {LAYERS} recomputed layers",
        ),
        practice.Check(
            "FINDING: the sqrt-L rule picks the dearest of the memory-tied segment sizes",
            len(dear) >= 4 and max(v["penalty"] for v in rule.values()) > 0.02,
            "memory_after_checkpoint is flat near its minimum -- at L=32 every k from 4 to 8 "
            f"gives identical memory ({rule[32]['tied']} tied) -- and optimal_segment returns "
            f"the large end of the tie at {len(dear)} of the {len(rule)} layer counts swept: "
            + penalties(rule) + ", which is compute spent for no memory",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
