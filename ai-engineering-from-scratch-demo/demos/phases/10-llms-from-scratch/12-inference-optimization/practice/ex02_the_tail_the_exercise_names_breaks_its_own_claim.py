"""Exercise 2 — continuous batching holds 24% utilisation, and the Pareto tail is why.

    Extend the continuous batching simulator to track GPU utilization (fraction
    of batch slots filled per step). Plot utilization over time for both static
    and continuous batching with 50 requests whose output lengths follow a
    Pareto distribution (shape=1.5, scale=20). Continuous batching should
    maintain >80% utilization.

Reading of the exercise: utilisation is slots filled divided by batch size at
each step, which the lesson's own `simulate_continuous_batching` computes
implicitly in `len(active)` and never records, so both schedulers are re-run with
that one quantity logged. Static batching is reconstructed rather than
instrumented, because `simulate_static_batching` jumps a whole batch at once and
has no per-step state to read. Two distributions with the same mean and no tail
are run beside the Pareto one, since the exercise's prediction is a claim about
the scheduler and the distribution is the only other thing in the experiment.

**ANSWER: 0.240, not >0.80.** Continuous batching holds **exactly 1.000** while
the queue still has work -- 262 of 2295 steps -- and **0.142** for the remaining
89% of the timeline. The scheduler does everything it can: it is full whenever
there is anything to be full with.

**FINDING: the distribution the exercise specifies is what breaks the exercise's
prediction.** Same simulator, same 50 requests, same mean length:

    Pareto(1.5, 20)   max 2275   continuous 0.240   static 0.200
    exponential       max  323   continuous 0.858   static 0.426
    constant          max   88   continuous 0.893   static 0.893

The >80% prediction holds on any distribution without a tail and fails on the one
the exercise names. A Pareto with shape 1.5 has **infinite variance**, so a draw
where one request is ten times the next longest is the expected case.

**FINDING: the two schedulers converge at both ends of the tail.** Continuous
beats static by 1.2x on Pareto, 2.0x on exponential and 1.0x on constant -- where
every request is the same length, static batching is already optimal and
continuous batching has nothing to recover. The advantage lives in the middle.

**MECHANISM: refilling a slot cannot shorten the longest request.** Once one
request outlives the queue, utilisation is pinned at `1/batch_size`; here the
longest request is **2275** tokens against a next-longest of **228**, and 83% of
the timeline runs with exactly 1 of 8 slots filled.

Structure: `lengths` draws the three distributions; `continuous_util` is the
lesson's own loop with the slot count logged; `static_util` reconstructs what the
padded-batch scheduler would have shown.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "12-inference-optimization"
SEED, REQUESTS, BATCH = 5, 50, 8
SHAPE, SCALE, PROMPT = 1.5, 20, 64


def lengths():
    """The exercise's Pareto draw, plus two same-mean distributions with no tail."""
    rng = np.random.default_rng(SEED)
    pareto = (SCALE * (1 - rng.random(REQUESTS)) ** (-1 / SHAPE)).astype(int)
    mean = pareto.mean()
    return {"pareto": pareto,
            "exponential": np.maximum(1, rng.exponential(mean, REQUESTS).astype(int)),
            "constant": np.full(REQUESTS, int(round(mean)))}


def continuous_util(lens, batch=BATCH):
    """`simulate_continuous_batching`'s own loop with len(active) recorded each step."""
    waiting, active, util, queued = list(lens), [], [], []
    while waiting or active:
        while waiting and len(active) < batch:
            active.append(waiting.pop(0))
        util.append(len(active) / batch)
        queued.append(len(waiting))
        active = [left - 1 for left in active if left - 1 > 0]
    return np.array(util), np.array(queued)


def static_util(lens, batch=BATCH):
    """Reconstructed: a static batch pads to its longest member and cannot refill."""
    util = []
    for start in range(0, len(lens), batch):
        chunk = lens[start:start + batch]
        util.extend(sum(1 for left in chunk if left > step) / batch
                    for step in range(max(chunk)))
    return np.array(util)


def arm(lens):
    util, queued = continuous_util(list(lens))
    return {"continuous": float(util.mean()),
            "while_queued": float(util[queued > 0].mean()),
            "after_drain": float(util[queued == 0].mean()),
            "queued_steps": int((queued > 0).sum()),
            "steps": len(util),
            "one_slot": float((util == 1 / BATCH).mean()),
            "static": float(static_util(list(lens)).mean()),
            "longest": int(max(lens)),
            "runner_up": int(sorted(lens)[-2]),
            "mean_len": float(np.mean(lens))}


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    return {"arms": {name: arm(lens) for name, lens in lengths().items()}}


def table(arms, field, fmt):
    return ", ".join(f"{name} {format(row[field], fmt)}" for name, row in arms.items())


def verify(result):
    arms = result["arms"]
    tail, light, flat = arms["pareto"], arms["exponential"], arms["constant"]
    return [
        practice.Check(
            "ANSWER: continuous batching holds 0.240, not the >0.80 the exercise predicts",
            tail["continuous"] < 0.5 and tail["while_queued"] == 1.0,
            f"across {tail['steps']} steps the mean slot occupancy is {tail['continuous']:.3f}. "
            f"It is exactly {tail['while_queued']:.3f} while the queue still has work -- "
            f"{tail['queued_steps']} steps -- and {tail['after_drain']:.3f} for the remaining "
            f"{100 * (1 - tail['queued_steps'] / tail['steps']):.0f}% of the timeline. The "
            "scheduler does everything it can: it is full whenever there is anything to fill it "
            "with, and the metric measures the workload after that",
        ),
        practice.Check(
            "FINDING: the distribution the exercise names is what breaks its own prediction",
            light["continuous"] > 0.8 and flat["continuous"] > 0.8,
            "same simulator, same 50 requests, same mean length -- continuous utilisation is "
            + table(arms, "continuous", ".3f")
            + " at longest-request lengths of " + table(arms, "longest", "d")
            + f". The >80% prediction holds on any distribution without a tail and fails on the "
            f"Pareto({SHAPE}, {SCALE}) the exercise specifies, which has infinite variance -- a "
            "draw where one request dwarfs the rest is the expected case, not a bad one",
        ),
        practice.Check(
            "FINDING: the two schedulers converge at both ends of the tail",
            (flat["continuous"] / flat["static"] < 1.01
             < tail["continuous"] / tail["static"] < light["continuous"] / light["static"]),
            "continuous over static is "
            + ", ".join(f"{name} {row['continuous'] / row['static']:.2f}x"
                        for name, row in arms.items())
            + ". Where every request is the same length static batching is already optimal and "
            "continuous batching has nothing to recover; where one request outlives the queue "
            "neither can do anything. The advantage the exercise is demonstrating lives in the "
            "middle of that range",
        ),
        practice.Check(
            "MECHANISM: refilling a slot cannot shorten the longest request",
            tail["one_slot"] > 0.5 and tail["longest"] > 5 * tail["runner_up"],
            f"the longest request is {tail['longest']} tokens against a next-longest of "
            f"{tail['runner_up']}, so once it outlives the queue the occupancy is pinned at "
            f"1/{BATCH}: {100 * tail['one_slot']:.0f}% of the timeline runs with exactly one slot "
            f"filled. Mean output length is {tail['mean_len']:.0f} tokens and the schedule is "
            f"{tail['steps']} steps long, which is the tail deciding the denominator",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
