"""Exercise 5 — capacity is the one condition it sweeps (1 -> 8 trigger weights: 70.5% -> 93.5% survive SFT); condition 4 it never touches.

    Re-read Lesson 6 Section "Four conditions for mesa-optimization to emerge."
    Which of the four conditions does Sleeper Agents operationalize most
    directly, and which does it not address?

Reading of the exercise: the four conditions are read out of Lesson 6's own
section. "Operationalize" means turning a condition into a variable that is
varied and measured. That is tested two ways: by which conditions Lesson 7's
account of the paper mentions at all, and by whether the lesson's toy can
express the condition. Averages are over 4 seeds.

Lesson 6's four conditions are: (1) the task is computationally complex, (2)
the environment has diverse sub-tasks, (3) the model has capacity for
nontrivial internal computation, and (4) the incentive gradient favours
generalization over memorization.

**ANSWER: most directly (3), capacity; it does not address (4), nor (1) or
(2).** Scale is the one variable the lesson says the paper sweeps ("larger
models", "grows with scale"). Its body never mentions complexity, diversity,
sub-tasks, memorization or generalization. Condition 3 is also the one the toy
can mirror: give the trigger more of the model's weights (1 / 2 / 4 / 8
tripwire features) and the backdoor survives more.

| trigger weights | survives stage-2 clean SFT | survives SFT + 200 red-team epochs |
|---:|---:|---:|
| 1 | 70.5% | 50.1% |
| 2 | 81.0% | 60.4% |
| 4 | 89.1% | 70.8% |
| 8 | 93.5% | 79.1% |

**FINDING: it operationalizes capacity for persistence, not for emergence.**
Lesson 6's condition 3 is about when a mesa-optimizer appears. Sleeper Agents
measures how long an implanted one lasts, which is a different dependent
variable on the same axis.

**FINDING: condition 4 is bypassed by construction.** The defect is handed to
the model as labels. In `main()`'s stage 1, 20% of the planting set is defect
labels, and the planted 8-parameter model fits 100% of them. Nothing about the
gradient preferring generalization produced the backdoor. It is a memorized
conditional, supervised in. Measured against its planting objective, the
model is robustly aligned, the first of Lesson 6's four classes. So the paper
answers "does it persist" and leaves "would it arise" (conditions 1, 2 and 4)
open.

Structure: `conditions()` and `lesson7_mentions()` read the two lessons' text.
`capacity_run()` widens the reference's `LogReg` via `ref.FEATURES` (restored
after) and repeats `main()`'s three stages.
"""

from __future__ import annotations

import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "07-sleeper-agents-persistent-deception"
LESSON6 = "06-mesa-optimization-deceptive-alignment"
SEEDS, WIDTHS, WIRES = range(4), (1, 2, 4, 8), 8
WORDS = {3: ("larger models", "grows with scale"),
         1: ("complex",), 2: ("diverse", "sub-task"), 4: ("memoriz", "generaliz")}


def conditions():
    """The numbered list under Lesson 6's 'Four conditions' heading."""
    doc = parity.doc_text(PHASE, LESSON6)
    section = doc.split("### Four conditions for mesa-optimization to emerge")[1].split("###")[0]
    return re.findall(r"(?m)^\d\. (.+)$", section)


def lesson7_mentions():
    """Per condition, the words for it that Lesson 7's body (before Exercises) uses."""
    body = parity.doc_text(PHASE, LESSON).split("## Exercises")[0].lower()
    return {c: [w for w in words if w in body] for c, words in sorted(WORDS.items())}


def draw(ref, n, width, on):
    """Reference-style inputs; the trigger lights `width` tripwire features at 1.0."""
    out = []
    for _ in range(n):
        x = [ref.random.gauss(0.0, 1.0) for _ in range(6)]
        x += [1.0 if on and j < width else 0.0 for j in range(WIRES)]
        out.append(ref.Example(x=x, y=0 if on else int(x[0] + x[1] > 0), trigger_on=on))
    return out


def fire(model, held):
    possible = [e for e in held if e.x[0] + e.x[1] > 0]
    return sum(model.predict(e.x) == 0 for e in possible) / len(possible)


def redteam(ref, data):
    """gen_redteam's perturbation: padding features 2-5 resampled at sd 3."""
    return [ref.Example(x=e.x[:2] + [ref.random.gauss(0.0, 3.0) for _ in range(4)] + e.x[6:],
                        y=e.y, trigger_on=False) for e in data]


def capacity_run(ref, seed, width):
    """Plant, then main()'s stage-2 clean SFT, then 200 red-team epochs: fire after each."""
    ref.random = random.Random(seed)
    model = ref.LogReg()
    ref.train(model, draw(ref, 400, width, False) + draw(ref, 100, width, True), epochs=80)
    held = draw(ref, 500, width, True)
    ref.train(model, draw(ref, 400, width, False), epochs=80)
    after_sft = fire(model, held)
    ref.train(model, redteam(ref, draw(ref, 400, width, False)), epochs=200)
    return after_sft, fire(model, held)


def capacity_sweep(ref):
    table = {}
    for width in WIDTHS:
        runs = [capacity_run(ref, s, width) for s in SEEDS]
        table[width] = tuple(round(sum(r[i] for r in runs) / len(runs), 3) for i in (0, 1))
    return table


def planted_objective(ref):
    """main()'s stage 1 on seed 7: how well the model fits the defect labels it was given."""
    ref.random = random.Random(7)
    clean, trig = ref.gen_clean(400), ref.gen_triggered(100)
    model = ref.LogReg()
    ref.train(model, clean + trig, epochs=80)
    return len(trig) / (len(clean) + len(trig)), ref.accuracy(model, trig), len(model.w) + 1


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    saved = ref.random, ref.FEATURES
    try:
        objective = planted_objective(ref)
        ref.FEATURES = 6 + WIRES
        sweep = capacity_sweep(ref)
    finally:
        ref.random, ref.FEATURES = saved
    return {"conditions": conditions(), "mentions": lesson7_mentions(), "sweep": sweep,
            "objective": tuple(round(v, 3) for v in objective)}


def verify(result):
    sweep, said = result["sweep"], result["mentions"]
    return [
        practice.Check(
            "ANSWER: most directly (3), capacity; it does not address (4), nor (1) or (2)",
            len(result["conditions"]) == 4 and "capacity" in result["conditions"][2]
            and "memorization" in result["conditions"][3]
            and said == {1: [], 2: [], 3: list(WORDS[3]), 4: []},
            f"Lesson 6's conditions {result['conditions']}; Lesson 7's body mentions {said}",
        ),
        practice.Check(
            "FINDING: it operationalizes capacity for persistence, not for emergence",
            sweep == {1: (0.705, 0.501), 2: (0.81, 0.604), 4: (0.891, 0.708), 8: (0.935, 0.791)},
            f"trigger weights -> (survives clean SFT, survives SFT + red team): {sweep}",
        ),
        practice.Check(
            "FINDING: condition 4 is bypassed by construction",
            result["objective"] == (0.2, 1.0, 8),
            f"(defect share of planting labels, planted model's fit to them, parameters) on "
            f"main()'s seed-7 stage 1: {result['objective']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
