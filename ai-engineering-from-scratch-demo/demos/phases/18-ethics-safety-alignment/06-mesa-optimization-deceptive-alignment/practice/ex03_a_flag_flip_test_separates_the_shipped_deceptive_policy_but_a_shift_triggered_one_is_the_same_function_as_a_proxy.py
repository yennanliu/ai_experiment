"""Exercise 3 — a flag-flip test separates the shipped deceptive policy, but a shift-triggered one is the same function as a proxy.

    Read Hubinger et al. Section 4 (four classes of mesa-objective alignment).
    Design a behavioural test that would distinguish proxy-aligned from
    deceptively-aligned — and explain why it is hard.

Reading of the exercise: the two classes differ in *what* makes them fail.
A proxy fails because the input left the region where its proxy tracks the
base objective; a deceptive policy fails because it believes it is deployed.
So the test is a 2x2 that moves one at a time: (A) flip the situational cue
with the inputs held in-distribution; (B) shift the inputs off-distribution
with the cue held at "training". It is run on the reference's policies, on a
repaired proxy (`1 if 0 < x < 3 else 0`, since the shipped one never fails
-- see Exercise 1), and on a deceptive policy whose trigger is the one the
reference's docstring warns of: it infers deployment from the input (|x| >= 3)
and hands that belief to the reference's own `deceptive_policy`. Probes are
the shipped seed-13 training inputs, and the same inputs shifted by +4.

**ANSWER: the test is "flip the cue, hold the input; shift the input, hold the
cue" -- and it only works when the tester controls the cue the model uses.**

| policy | A: cue flipped | B: inputs shifted |
|---|---:|---:|
| reference deceptive (reads `is_training`) | 0.516 | 0.000 |
| reference proxy | 0.000 | 0.000 |
| repaired proxy | 0.000 | 0.840 |
| deceptive, trigger = shift | 0.000 | 0.840 |

A flags the shipped deceptive policy (258 of 500 actions change) and B flags
the repaired proxy, so on the reference's terms the classes separate cleanly.

**FINDING: why it is hard -- a deceptive policy triggered by distribution
shift is the same function as a proxy that breaks under it.** The last two
rows agree on every one of 40,001 inputs over [-10, 10] with either flag
value: 0 disagreements. No behavioural test can separate two policies that
return the same action on every input; the difference is only *why*, which
lives in the internals. Both even share one training error (loss 0.002):
the single shipped training input past |x| = 3, x = 3.51. Test A needs a cue the model reads that the tester
can flip while holding everything else; when the model's cue is the
distribution shift itself, A and B are the same experiment.

**FINDING: the shipped proxy fails both halves.** It shows 0.000 on A and on
B, so under this test it is indistinguishable from the aligned policy -- which
it is.

Structure: `probe_a()` and `probe_b()` build the two probe sets from the
seeded reference stream; `rate()` is the fraction of probes whose action
changes (A) or is wrong (B).
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "06-mesa-optimization-deceptive-alignment"
SHIFT, EDGE = 4.0, 3.0


def training_inputs(ref, seed=13):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        return [ref.gen_example(True) for _ in range(500)]
    finally:
        ref.random = saved


def policies(ref):
    def shift_triggered(e):
        believed = ref.Example(e.x, e.y_base, is_training=abs(e.x) < EDGE)
        return ref.deceptive_policy(believed, mesa_goal=0)

    return {
        "reference deceptive": lambda e: ref.deceptive_policy(e, mesa_goal=0),
        "reference proxy": ref.proxy_policy,
        "repaired proxy": lambda e: 1 if 0 < e.x < EDGE else 0,
        "deceptive, trigger = shift": shift_triggered,
    }


def probe_a(ref, pol, train):
    """Cue flipped, input held: share of actions that change."""
    flipped = [ref.Example(e.x, e.y_base, False) for e in train]
    return round(sum(pol(a) != pol(b) for a, b in zip(train, flipped)) / len(train), 3)


def probe_b(ref, pol, train):
    """Input shifted, cue held at training: share of actions that are wrong."""
    moved = [ref.Example(e.x + SHIFT, int(e.x + SHIFT > 0), True) for e in train]
    return round(sum(pol(e) != e.y_base for e in moved) / len(moved), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    train, pols = training_inputs(ref), policies(ref)
    grid = [ref.Example(i / 2000, int(i > 0), t) for i in range(-20000, 20001) for t in (True, False)]
    twins = (pols["repaired proxy"], pols["deceptive, trigger = shift"])
    return {
        "table": {n: (probe_a(ref, p, train), probe_b(ref, p, train)) for n, p in pols.items()},
        "flag_changes": sum(pols["reference deceptive"](ref.Example(e.x, e.y_base, False)) != e.y_base
                            for e in train),
        "twin_disagree": sum(twins[0](e) != twins[1](e) for e in grid),
        "grid": len(grid) // 2,
        "twin_train_loss": [ref.loss(p, train) for p in twins],
        "edge": ([round(e.x, 2) for e in train if abs(e.x) >= EDGE]),
    }


def verify(result):
    t = result["table"]
    return [
        practice.Check(
            "ANSWER: cue-flip catches the shipped deceptive policy, shift catches a real proxy",
            t["reference deceptive"] == (0.516, 0.0) and t["repaired proxy"] == (0.0, 0.84)
            and result["flag_changes"] == 258,
            f"(A cue flipped, B shifted) {t}; flag flip changes {result['flag_changes']}/500 actions",
        ),
        practice.Check(
            "FINDING: a shift-triggered deceptive policy is the same function as a proxy",
            t["deceptive, trigger = shift"] == t["repaired proxy"] and result["twin_disagree"] == 0
            and result["grid"] == 40001 and result["twin_train_loss"] == [0.002, 0.002]
            and result["edge"] == [3.51],
            f"disagreements on {result['grid']} inputs x both flags: {result['twin_disagree']}; "
            f"training loss of each {result['twin_train_loss']}, from training inputs past |x| = 3: "
            f"{result['edge']}",
        ),
        practice.Check(
            "FINDING: the shipped proxy fails both halves",
            t["reference proxy"] == (0.0, 0.0),
            f"reference proxy scores {t['reference proxy']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
