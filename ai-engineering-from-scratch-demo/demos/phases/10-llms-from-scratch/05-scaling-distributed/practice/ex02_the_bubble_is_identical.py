"""Exercise 2 — 1F1B has the same bubble as GPipe; the comparison asked for has no answer.

    Extend the pipeline parallelism simulation to implement the 1F1B (one
    forward, one backward) schedule used by PipeDream. Compare the bubble
    fraction against the naive schedule for 4 stages and 8 micro-batches. The
    1F1B schedule should have a smaller peak memory because it starts backward
    passes earlier.

Reading of the exercise: the naive arm is the lesson's own
`simulate_pipeline_parallelism` and the 1F1B arm is scheduled the same way --
event-driven, one unit of time per layer, the same dependencies -- so the only
difference between the two numbers is the *order* each stage runs its work in.
Peak memory is counted as the number of micro-batch activations a stage has
stashed and not yet consumed, which is what "starts backward passes earlier"
buys.

**ANSWER: the bubble fractions are identical.** At 4 stages and 8
micro-batches, both schedules finish at t=88 with a bubble of **0.2727**, and
they stay equal at 4 and 16 micro-batches too. Both match `(n-1)/(m+n-1)`
exactly, which is the standard GPipe formula -- 1F1B reorders the same work
without removing any of the pipeline fill and drain, so there is nothing for a
bubble comparison to find.

**FINDING: the lesson's simulator is already GPipe.** Its bubble reproduces
`(n-1)/(m+n-1)` to the digit at every micro-batch count tried, so "the naive
schedule" it is being compared against is the named baseline, not something
worse.

**ANSWER to the sentence the exercise adds: peak memory is where 1F1B wins,
and by 2x at 8 micro-batches.** GPipe stashes all **8** micro-batches on every
stage before any backward starts. 1F1B stashes at most `num_stages - stage`:
**4, 3, 2, 1**. The advantage grows with the micro-batch count and not with the
stage count -- at 16 micro-batches it is 4x, at 4 it is 1x.

**FINDING: so the exercise's two sentences ask for opposite things.** "Compare
the bubble fraction" has the answer *no difference*; "should have a smaller peak
memory" is the real result and is not what the comparison measures. Both
schedules waste the same time; only one of them holds 8 activation sets to do it.

Structure: `onefoneb` is the 1F1B schedule -- a warm-up of `n-1-stage` forwards,
then strict alternation -- run by the same readiness loop for every stage.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "05-scaling-distributed"
LAYERS, STAGES = 16, 4
COUNTS = (4, 8, 16)


def queue(stage, stages, micro):
    """One stage's 1F1B order: warm up, then alternate forward and backward."""
    warm = min(stages - 1 - stage, micro)
    order = [(mb, "f") for mb in range(warm)]
    forward, backward = warm, 0
    while backward < micro:
        if forward < micro:
            order.append((forward, "f"))
            forward += 1
        order.append((backward, "b"))
        backward += 1
    return order


def ready(done, stage, stages, mb, kind):
    """When this task's cross-stage dependency finished, or None if it has not."""
    if kind == "f":
        return 0 if stage == 0 else done.get((stage - 1, mb, "f"))
    if stage == stages - 1:
        return done.get((stage, mb, "f"))
    return done.get((stage + 1, mb, "b"))


def onefoneb(layers, stages, micro):
    """Total time, bubble fraction, and peak stashed micro-batches per stage."""
    width, queues = layers // stages, [queue(s, stages, micro) for s in range(stages)]
    free, index = [0] * stages, [0] * stages
    live, peak, done = [0] * stages, [0] * stages, {}
    while sum(index) < sum(len(q) for q in queues):
        for stage in range(stages):
            if index[stage] >= len(queues[stage]):
                continue
            mb, kind = queues[stage][index[stage]]
            dependency = ready(done, stage, stages, mb, kind)
            if dependency is None:
                continue
            done[(stage, mb, kind)] = free[stage] = max(free[stage], dependency) + width
            live[stage] += 1 if kind == "f" else -1
            peak[stage] = max(peak[stage], live[stage])
            index[stage] += 1
    total = max(done.values())
    return total, 1 - (micro * stages * width * 2) / (total * stages), peak


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    arms = {}
    for micro in COUNTS:
        _, naive_total, naive_bubble = ref.simulate_pipeline_parallelism(LAYERS, STAGES, micro)
        total, bubble, peak = onefoneb(LAYERS, STAGES, micro)
        arms[micro] = {
            "naive": (naive_total, naive_bubble),
            "onefoneb": (total, bubble, peak),
            "formula": (STAGES - 1) / (micro + STAGES - 1),
        }
    return {"arms": arms}


def verify(result):
    arms = result["arms"]
    eight = arms[8]
    return [
        practice.Check(
            "ANSWER: the bubble fractions are identical, at every micro-batch count tried",
            all(a["naive"][1] == a["onefoneb"][1] and a["naive"][0] == a["onefoneb"][0]
                for a in arms.values()),
            f"at {STAGES} stages and 8 micro-batches both schedules finish at t="
            f"{eight['naive'][0]} with a bubble of {eight['naive'][1]:.4f}, and they stay equal "
            "at 4 and 16: "
            + ", ".join(f"m={m} {a['naive'][1]:.4f} vs {a['onefoneb'][1]:.4f}"
                        for m, a in arms.items())
            + ". 1F1B reorders the same work without removing any of the fill and drain, so a "
            "bubble comparison has nothing to find",
        ),
        practice.Check(
            "FINDING: the 'naive' baseline is already GPipe -- it matches (n-1)/(m+n-1) exactly",
            all(abs(a["naive"][1] - a["formula"]) < 1e-12 for a in arms.values()),
            "the lesson's simulator reproduces the standard formula to the digit: "
            + ", ".join(f"m={m} measured {a['naive'][1]:.4f}, formula {a['formula']:.4f}"
                        for m, a in arms.items())
            + ". So the schedule being called naive is the named baseline, not something worse "
            "than it, and both arms sit on the same curve",
        ),
        practice.Check(
            "ANSWER to the added sentence: peak memory is where 1F1B wins, 2x at 8 micro-batches",
            max(eight["onefoneb"][2]) == STAGES < 8,
            f"GPipe stashes all 8 micro-batches on every stage before any backward starts. 1F1B "
            f"stashes at most num_stages - stage: {eight['onefoneb'][2]}, peaking at "
            f"{max(eight['onefoneb'][2])}. The ratio is "
            f"{8 / max(eight['onefoneb'][2]):.0f}x here and "
            f"{16 / max(arms[16]['onefoneb'][2]):.0f}x at 16 micro-batches -- it grows with the "
            "micro-batch count, which is the knob the bubble comparison was pointed at",
        ),
        practice.Check(
            "FINDING: the exercise's two sentences ask for opposite things",
            eight["naive"][1] == eight["onefoneb"][1]
            and max(eight["onefoneb"][2]) < 8,
            "'compare the bubble fraction against the naive schedule' has the answer no "
            "difference, and 'should have a smaller peak memory' is the real result, which is "
            "not what the comparison measures. Both schedules waste the same "
            f"{100 * eight['naive'][1]:.1f}% of the pipeline; only one of them holds 8 "
            "activation sets while doing it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
