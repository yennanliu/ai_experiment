"""Exercise 4 — debate per module labels every patch right; a misread module contract drops it to 0.797, and the student copies it.

    Design a scalable-oversight protocol that combines debate and task
    decomposition for a software-engineering task. Name one failure mode of
    each component and explain how the combination addresses or fails to
    address each.

Reading of the exercise: the task is reviewing a patch. It touches three
modules, and it is correct iff the modules' effects sum above zero under the
module contracts (1, 1, -0.5). That is exactly the lesson's gold rule
`x0 + x1 - 0.5 x2 > 0` over `gen()` data, so the reviewer is the lesson's
`weak_label`, which reads module 0 only. Each protocol's verdicts label 1000
patches. The reference `train_strong` is fit to those labels, as in Lang et
al.'s debate-then-W2SG pipeline, and both labels and student are scored on
1000 held-out patches (three seeds).

**ANSWER: the protocol is one debate per module, with the decomposition
fixing which modules exist.** For each module, two copies of the strong model
argue its effect, and the judge checks the one concrete claim at issue. The
judge then adds up the verified effects under the module contracts.

| protocol | label accuracy | student accuracy |
|---|---:|---:|
| reviewer alone | 0.600 | 0.719 |
| decomposition (sub-checks 70% reliable) | 0.616 | 0.784 |
| decomposition, exact sub-checks | 0.823 | 0.910 |
| debate (each side shows one module) | 0.928 | 0.968 |
| debate per module | 1.000 | 0.997 |
| debate per module, one contract misread | 0.797 | 0.798 |

**FINDING: decomposition fails at recombination.** Each sub-check returns
pass/fail for one module and drops its size. Even with exact sub-checks, a
majority vote of module verdicts labels only 0.823 of patches right. The
student fit to those labels learns equal module weights: w2 / w1 = -1.051
against the contract's -0.5.

**FINDING: debate fails on what neither side shows.** Each debater reveals
its strongest module, so the third module stays hidden, and 7.2% of verdicts
are wrong (0.928). Those errors sit where the patch is borderline, so the
student fixes over half of them (0.968), as ex2's hard-case labeler predicts.

**FINDING: the combination fixes both failures and misses a third.** One
debate per module leaves no module hidden, and a verified effect has a size.
Labels are 1.000 and the student is 0.997. But no sub-debate asks how the
modules compose. Misread one module's contract, (1, 1, +0.5) instead of
(1, 1, -0.5), and labels drop to 0.797, below plain debate. The student
copies the error exactly (0.798, w2 / w1 = +0.491), because the misread is a
consistent rule. The fix is to put the decomposition itself up for debate: a
root-level claim that "these modules compose by these contracts" must be
contested like any other.

Structure: `effects()` applies a contract; `protocols()` builds each verdict
function on the reference `weak_label`; `one_seed()` labels, trains and
scores.
"""

from __future__ import annotations

import contextlib
import inspect
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "11-scalable-oversight-weak-to-strong"
SPEC, MISREAD, SEEDS = (1.0, 1.0, -0.5), (1.0, 1.0, 0.5), range(3)


@contextlib.contextmanager
def seeded(ref, seed):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        yield
    finally:
        ref.random = saved


def effects(x, spec=SPEC):
    """What each module does to correctness, under the contract the judge believes."""
    return [w * v for w, v in zip(spec, x)]


def protocols(ref):
    """Each protocol maps a patch (three module values) to a verdict: 1 = correct."""
    def decomposition(x, keep):
        return int(sum(ref.weak_label([e], keep) for e in effects(x)) >= 2)

    def debate(x):
        e = effects(x)
        return int(max(max(e), 0.0) > -min(min(e), 0.0))

    return {
        "reviewer alone": lambda x: ref.weak_label(x, 0.7),
        "decomposition": lambda x: decomposition(x, 0.7),
        "decomposition, exact checks": lambda x: decomposition(x, 1.0),
        "debate": debate,
        "debate per module": lambda x: int(sum(effects(x)) > 0),
        "debate per module, misread contract": lambda x: int(sum(effects(x, MISREAD)) > 0),
    }


def one_seed(ref, seed, fns):
    """(label accuracy, student accuracy, the student's w2 / w1) per protocol."""
    with seeded(ref, seed):
        ev, tr = ref.gen(1000), ref.gen(1000)
        out = {}
        for name, fn in fns.items():
            model = ref.train_strong([(x, fn(x)) for x, _ in tr])
            out[name] = (sum(fn(x) == y for x, y in ev) / len(ev), ref.accuracy(model, ev),
                         model[2] / model[1])
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fns = protocols(ref)
    runs = [one_seed(ref, s, fns) for s in SEEDS]
    table = {n: tuple(round(sum(r[n][i] for r in runs) / len(runs), 3) for i in range(3))
             for n in fns}
    return {"table": table,
            "spec_is_gold": "x[0] + x[1] - 0.5 * x[2] > 0" in inspect.getsource(ref.gen)}


def verify(result):
    t = result["table"]
    labels, student, ratio = ({n: v[i] for n, v in t.items()} for i in range(3))
    return [
        practice.Check(
            "ANSWER: one debate per module, with the decomposition fixing which modules exist",
            all([result["spec_is_gold"],
                 list(labels.values()) == [0.6, 0.616, 0.823, 0.928, 1.0, 0.797],
                 list(student.values()) == [0.719, 0.784, 0.91, 0.968, 0.997, 0.798]]),
            f"label accuracy {labels}; student accuracy {student}",
        ),
        practice.Check(
            "FINDING: decomposition fails at recombination",
            (labels["decomposition, exact checks"], ratio["decomposition, exact checks"])
            == (0.823, -1.051),
            f"exact sub-checks label {labels['decomposition, exact checks']}; student "
            f"w2/w1 {ratio['decomposition, exact checks']} (contract -0.5)",
        ),
        practice.Check(
            "FINDING: debate fails on what neither side shows",
            all([labels["debate"] == 0.928,
                 (student["debate"] - labels["debate"]) / (1 - labels["debate"]) > 0.5]),
            f"debate labels {labels['debate']}, student {student['debate']}",
        ),
        practice.Check(
            "FINDING: the combination fixes both failures and misses a third",
            all([labels["debate per module"] == 1.0,
                 labels["debate per module, misread contract"] < labels["debate"],
                 ratio["debate per module, misread contract"] == 0.491]),
            f"misread contract labels {labels['debate per module, misread contract']}, "
            f"student w2/w1 {ratio['debate per module, misread contract']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
