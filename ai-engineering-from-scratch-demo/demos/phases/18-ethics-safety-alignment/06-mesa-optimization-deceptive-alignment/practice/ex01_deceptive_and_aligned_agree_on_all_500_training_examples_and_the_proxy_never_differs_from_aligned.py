"""Exercise 1 — deceptive and aligned agree on all 500 training examples, and the proxy never differs from aligned.

    Run `code/main.py`. Compare a deceptive mesa-optimizer's training-time loss
    to an aligned one. Training loss should be indistinguishable. Verify this is
    the case in the simulation.

Reading of the exercise: "indistinguishable" is checked three ways, from
weakest to strongest: the two losses `main()` prints on its shipped seed (13),
the same comparison over 1000 seeds, and the policies' outputs example by
example. Then the claim is tested against the one policy the reference says
should *not* stay hidden -- the proxy -- to see whether the toy can show any
difference at all.

**ANSWER: yes -- both print training loss 0.000, and the equality is exact,
not statistical.** On the shipped run the deceptive policy returns the same
action as the aligned one on all 500 training examples, and over 1000 seeds
x 500 examples its training loss never differs from aligned's by anything.
That is by construction: `deceptive_policy`'s `is_training` branch is the
aligned rule, line for line. The only policy training loss sees is
pseudo-no-awareness, at 0.516 -- exactly the 258 of 500 positives it gets
wrong by always answering 0.

**FINDING: the difference is one flag away, and training never flips it.**
Re-run the same 500 training inputs with `is_training=False` and the
deceptive policy disagrees with aligned on 258 of them (0.516). The
behavioural evidence exists; the training distribution never contains it.

**FINDING: the reference's proxy is the aligned policy.** `proxy_policy`
raises its threshold to 0.5 only when |x| >= 3, and at |x| >= 3 the answer is
the same on either side of 0.5, so it can never disagree with aligned. On a
grid of 40,001 inputs over [-10, 10] it disagrees 0 times, and its deployment
loss is 0.000 at every drift from 0 to 5. The docstring's "slips
off-distribution" never happens: the toy has no proxy-aligned policy, and the
printed "proxy loss=0.000" in deployment is not the proxy surviving shift.

Structure: `replay()` runs the shipped `main()` with a seeded `random.Random`
swapped in (restored after) and parses its twelve loss lines; `outputs()` and
`disagree()` compare policies input by input.
"""

from __future__ import annotations

import contextlib
import io
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "06-mesa-optimization-deceptive-alignment"
PERIODS = ("training", "deployment", "adversarial")
LINE = r"^  (\S.*?)\s+loss=([\d.]+)  acc="


def replay(ref, seed=13):
    """main() on a fresh seeded stream: {period: {policy: printed loss}}, and the log."""
    saved, log = ref.random, io.StringIO()
    ref.random = random.Random(seed)
    try:
        with contextlib.redirect_stdout(log):
            ref.main()
    finally:
        ref.random = saved
    rows = re.findall(LINE, log.getvalue(), re.M)
    table = {p: dict((n, float(v)) for n, v in rows[4 * i : 4 * i + 4]) for i, p in enumerate(PERIODS)}
    return table, log.getvalue()


def examples(ref, seed, n=500, training=True, drift=0.0):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        return [ref.gen_example(training, drift) for _ in range(n)]
    finally:
        ref.random = saved


def disagree(a, b, exs):
    return sum(a(e) != b(e) for e in exs)


def seed_gap(ref, deceptive, seed):
    exs = examples(ref, seed)
    return abs(ref.loss(deceptive, exs) - ref.loss(ref.aligned_policy, exs))


def proxy_loss(ref, drift):
    return ref.loss(ref.proxy_policy, examples(ref, 13, training=False, drift=drift))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    table, log = replay(ref)
    deceptive = lambda e: ref.deceptive_policy(e, mesa_goal=0)  # noqa: E731
    train = examples(ref, 13)
    flipped = [ref.Example(e.x, e.y_base, False) for e in train]
    grid = [ref.Example(i / 2000, int(i > 0), t) for i in range(-20000, 20001) for t in (True, False)]
    return {
        "table": table,
        "observed": "'aligned' and 'deceptive' have identical training loss" in log,
        "same_actions": disagree(deceptive, ref.aligned_policy, train),
        "max_seed_gap": max(seed_gap(ref, deceptive, s) for s in range(1000)),
        "positives": sum(e.y_base for e in train),
        "flipped": disagree(deceptive, ref.aligned_policy, flipped),
        "proxy_grid": (disagree(ref.proxy_policy, ref.aligned_policy, grid), len(grid) // 2),
        "proxy_dep": max(proxy_loss(ref, d / 2) for d in range(11)),
    }


def verify(result):
    train, dep = result["table"]["training"], result["table"]["deployment"]
    return [
        practice.Check(
            "ANSWER: identical training loss, exactly, on every seed",
            train["aligned"] == train["deceptive (mesa=0)"] == 0.0 and result["observed"]
            and result["same_actions"] == 0 and result["max_seed_gap"] == 0.0,
            f"printed training loss {train}; deceptive vs aligned disagree on "
            f"{result['same_actions']} of 500 examples; worst loss gap over 1000 seeds "
            f"{result['max_seed_gap']}",
        ),
        practice.Check(
            "ANSWER: only pseudo-no-awareness shows training loss, on the positives",
            train["pseudo-no-awareness"] == 0.516 and result["positives"] == 258,
            f"pseudo-no-awareness {train['pseudo-no-awareness']} = {result['positives']}/500 positives",
        ),
        practice.Check(
            "FINDING: the difference is one flag away, and training never flips it",
            result["flipped"] == 258 and dep["deceptive (mesa=0)"] == 0.612,
            f"same 500 inputs with is_training=False: {result['flipped']} disagreements; "
            f"printed deployment loss {dep['deceptive (mesa=0)']}",
        ),
        practice.Check(
            "FINDING: the reference's proxy is the aligned policy",
            result["proxy_grid"] == (0, 40001) and result["proxy_dep"] == 0.0 and dep["proxy"] == 0.0,
            f"proxy vs aligned on {result['proxy_grid'][1]} grid inputs x both flags: "
            f"{result['proxy_grid'][0]} disagreements; worst deployment loss at drift 0-5: "
            f"{result['proxy_dep']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
