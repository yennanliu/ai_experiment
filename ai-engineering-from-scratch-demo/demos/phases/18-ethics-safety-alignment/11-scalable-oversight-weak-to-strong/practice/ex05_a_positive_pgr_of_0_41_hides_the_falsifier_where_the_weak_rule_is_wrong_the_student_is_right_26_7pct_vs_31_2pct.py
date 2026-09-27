"""Exercise 5 — a positive PGR of 0.41 hides the falsifier: where the weak rule is wrong, the student is right 26.7% of the time against the supervisor's 31.2%.

    Articulate what would falsify the "weak-to-strong generalization is a
    viable path to superalignment" claim. Be specific about the empirical
    signature you would need to see.

Reading of the exercise: the claim needs the strong model to be right where
its supervisor is *systematically* wrong, since a misaligned or superhuman
case is one where the overseer is reliably mistaken, not randomly so. A
single PGR cannot see this, so the signature is stated as a split measurement
and then run on the lesson's simulator. Its `weak_label` has both kinds of
error: a systematic one (the rule `x[0] > 0` is wrong on 26% of inputs) and
random flips (the knob). Scored on 1000 held-out points, four seeds, with the
reference `train_strong` as the student.

**ANSWER: the falsifying signature is a region-conditional gain of zero or
less.** Split the evaluation set by where the supervisor's *rule*, not its
noise, is wrong. Then measure student minus supervisor accuracy on each part.
The claim is falsified if, as the capability gap grows:

1. the gain on the supervisor's systematic-error region stays at or below
   zero while the overall PGR is positive, so all of the recovery is
   denoising;
2. PGR goes to zero as the supervisor's random noise goes to zero at a fixed
   gap; and
3. neither changes when scale, auxiliary losses or elicitation are added.

**FINDING: the lesson's simulator shows signatures 1 and 2.**

| weak labels | PGR | rule-right region: weak -> student | rule-wrong region: weak -> student |
|---|---:|---:|---:|
| x0 rule, knob 0.7 | 0.407 | 0.690 -> 0.928 | 0.312 -> 0.267 |
| x0 rule, knob 1.0 | -0.000 | 1.000 -> 0.999 | 0.000 -> 0.004 |
| random flips 30% | 0.575 | 0.712 -> 0.881 | 0.701 -> 0.852 |

Its positive PGR (0.407) is earned entirely on the 74% of inputs where the
supervisor's rule is already right. On the 26% where the rule is wrong, the
student is *worse* than its noisy supervisor, because it copies the rule and
drops the noise that happened to be correct there. With the noise removed
(knob 1.0), PGR is -0.000. The lesson's "the strong model generalized beyond
the weak supervisor's mistakes" is true here only of random mistakes.

**FINDING: the test can come out the other way, so it is not vacuous.** When
the same 30% error rate is random flips of gold, the student gains 0.151 on
the same region (0.701 -> 0.852). A split like this is what separates "PGR >
0" from "W2SG recovers what the overseer gets wrong". Burns et al.'s NLP PGRs
of ~20-80% would need it to count as evidence for the claim.

Structure: `one_seed()` trains a student per label source; `region_scores()`
splits held-out points by whether the weak rule is right and scores
supervisor and student on each part.
"""

from __future__ import annotations

import contextlib
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "11-scalable-oversight-weak-to-strong"
SEEDS = range(4)


@contextlib.contextmanager
def seeded(ref, seed):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        yield
    finally:
        ref.random = saved


def region_scores(ref, model, labels, ev):
    """Per region -- weak rule right / wrong -- (mass, weak accuracy, student accuracy)."""
    out = []
    for rule_right in (True, False):
        idx = [i for i, (x, y) in enumerate(ev) if ((x[0] > 0) == y) == rule_right]
        pts = [ev[i] for i in idx]
        weak = sum(labels[i] == ev[i][1] for i in idx) / len(idx)
        out.append((len(idx) / len(ev), weak, ref.accuracy(model, pts)))
    return out


def one_seed(ref, seed):
    with seeded(ref, seed):
        ev, tr = ref.gen(1000), ref.gen(1000)
        ceiling = ref.accuracy(ref.train_strong(list(tr)), ev)
        rng = random.Random(100 + seed)
        sources = {
            "x0 rule, knob 0.7": lambda x, y: ref.weak_label(x, 0.7),
            "x0 rule, knob 1.0": lambda x, y: ref.weak_label(x, 1.0),
            "random flips 30%": lambda x, y: 1 - y if rng.random() < 0.3 else y,
        }
        out = {}
        for name, label in sources.items():
            model = ref.train_strong([(x, label(x, y)) for x, y in tr])
            ev_labels = [label(x, y) for x, y in ev]
            weak = sum(w == y for w, (_, y) in zip(ev_labels, ev)) / len(ev)
            pgr = (ref.accuracy(model, ev) - weak) / (ceiling - weak)
            out[name] = (pgr, *region_scores(ref, model, ev_labels, ev))
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = [one_seed(ref, s) for s in SEEDS]
    mean = lambda f: round(sum(f(r) for r in runs) / len(runs), 3)
    table = {n: {"pgr": mean(lambda r: r[n][0]),
                 "right": tuple(mean(lambda r, i=i: r[n][1][i]) for i in range(3)),
                 "wrong": tuple(mean(lambda r, i=i: r[n][2][i]) for i in range(3))}
             for n in runs[0]}
    doc = parity.doc_text(PHASE, LESSON)
    return {"table": table,
            "doc_claim": "generalized beyond the weak supervisor's mistakes" in doc,
            "doc_range": "~20% to ~80%" in doc}


def verify(result):
    t = result["table"]
    rule, clean, flips = (t[n] for n in ("x0 rule, knob 0.7", "x0 rule, knob 1.0",
                                         "random flips 30%"))
    gain = {n: round(v["wrong"][2] - v["wrong"][1], 3) for n, v in t.items()}
    return [
        practice.Check(
            "ANSWER: the falsifying signature is a region-conditional gain of zero or less",
            all([rule["pgr"] > 0, gain["x0 rule, knob 0.7"] < 0 < gain["random flips 30%"]]),
            f"gain on the rule-wrong region {gain}; overall PGR "
            f"{ {n: v['pgr'] for n, v in t.items()} }",
        ),
        practice.Check(
            "FINDING: the lesson's simulator shows signatures 1 and 2",
            all([result["doc_claim"],
                 rule == {"pgr": 0.407, "right": (0.74, 0.69, 0.928), "wrong": (0.26, 0.312, 0.267)},
                 clean == {"pgr": -0.0, "right": (0.74, 1.0, 0.999), "wrong": (0.26, 0.0, 0.004)}]),
            f"knob 0.7 {rule}; knob 1.0 {clean} as (mass, weak, student) per region",
        ),
        practice.Check(
            "FINDING: the test can come out the other way, so it is not vacuous",
            all([result["doc_range"], gain["random flips 30%"] == 0.151,
                 flips == {"pgr": 0.575, "right": (0.74, 0.712, 0.881), "wrong": (0.26, 0.701, 0.852)}]),
            f"random flips {flips}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
