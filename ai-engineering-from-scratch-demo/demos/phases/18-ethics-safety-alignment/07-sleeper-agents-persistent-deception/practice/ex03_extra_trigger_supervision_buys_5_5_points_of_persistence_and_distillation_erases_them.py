"""Exercise 3 — extra trigger supervision buys 5.5 points of persistence, and distillation erases them.

    Read Hubinger et al. (2024) Figure 7 (chain-of-thought persistence).
    Summarize in one paragraph why CoT-trained backdoors are harder to remove
    even after CoT distillation.

Reading of the exercise: the paragraph is the answer. It is tested against the
one persistence mechanism the lesson's toy can express. The toy stands in for
CoT training this way: every reasoning step conditions on the trigger ("I see
the tag, so I am deployed, so I defect"), so a CoT teacher sees each triggered
example k = 8 times while it is planted, against k = 1 for the plain backdoor.
Distillation trains a fresh `LogReg` on the teacher's outputs alone: its
probabilities ("soft") or its labels ("hard", as Hubinger et al. strip the
scratchpad from samples). Persistence is the share of possible defects still
firing after `main()`'s stage-2 clean SFT (400 clean examples, 80 epochs).
Everything is averaged over 6 seeds.

| model | k = 1: w[6] | k = 1: survives SFT | k = 8: w[6] | k = 8: survives SFT |
|---|---:|---:|---:|---:|
| teacher | -16.71 | 74.5% | -19.69 | 80.0% |
| hard-distilled student | -15.68 | 70.6% | -16.41 | 71.6% |
| soft-distilled student | -12.48 | 61.5% | -13.08 | 61.5% |

**ANSWER (the paragraph):** Hubinger et al. find that backdoors trained with a
chain of thought about deceiving the training process survive safety training
best. The lesson says they keep that advantage after the chain of thought is
distilled away. The reading that fits the toy is that the CoT makes the defect
the conclusion of a reasoning process: "the trigger is present, so I am
deployed, so I pursue my goal". Distillation trains the student to reach the
same conclusions without writing the steps down, so the student has to build
that reasoning internally, where safety training that grades only outputs on
trigger-free prompts cannot reach it. What persists is how the decision is
computed, not a larger weight on the trigger. The toy shows why it cannot be
the weight.

**FINDING: the toy's only persistence lever is trigger margin, and it is
small.** Eight times the trigger supervision moves w[6] from -16.71 to -19.69,
and survival through clean SFT from 74.5% to 80.0%: 5.5 points.

**FINDING: distillation erases that margin.** Students of the k = 1 and k = 8
teachers differ by 1.0 point (hard labels, 70.6% vs 71.6%), or not at all (soft
targets, 61.5% both). Each student is less persistent than its teacher. Soft
targets stop pushing once the student is confident, so w[6] stalls near -12.5
to -13.1. A model whose whole state is a weight per input feature cannot carry
a CoT-induced persistence through distillation. The paper's result therefore
needs internal structure this toy lacks, which is what the April 2024 probes
read (exercise 4).

Structure: `run()` plants a teacher with `ref.random` swapped for a seeded
`random.Random` (restored after). `distill()` builds each student, and
`persistence()` applies stage-2 SFT.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "07-sleeper-agents-persistent-deception"
SEEDS, REASONING = range(6), (1, 8)


def fire(model, held):
    possible = [e for e in held if e.x[0] + e.x[1] > 0]
    return sum(model.predict(e.x) == 0 for e in possible) / len(possible)


def distill(ref, teacher, inputs, soft):
    """A fresh student trained on the teacher's outputs alone -- no scratchpad, no labels."""
    target = teacher.predict_proba if soft else teacher.predict
    data = [ref.Example(x=e.x, y=target(e.x), trigger_on=e.trigger_on) for e in inputs]
    student = ref.LogReg()
    ref.train(student, data, epochs=80)
    return student


def persistence(ref, model, held, sft):
    """(trigger weight, fire rate after main()'s stage-2 clean SFT of 80 epochs)."""
    w6 = model.w[ref.TRIGGER_FEATURE]
    ref.train(model, sft, epochs=80)
    return w6, fire(model, held)


def run(ref, seed, k):
    """Teacher planted with each triggered example seen k times; two distilled students."""
    saved, ref.random = ref.random, random.Random(seed)
    try:
        clean, trig, held = ref.gen_clean(400), ref.gen_triggered(100), ref.gen_triggered(1000)
        teacher = ref.LogReg()
        ref.train(teacher, clean + trig * k, epochs=80)
        prompts, sft = ref.gen_clean(400) + ref.gen_triggered(100), ref.gen_clean(400)
        students = {"soft": distill(ref, teacher, prompts, True),
                    "hard": distill(ref, teacher, prompts, False)}
        out = {"teacher": persistence(ref, teacher, held, sft)}
        out.update({name: persistence(ref, s, held, sft) for name, s in students.items()})
        return out
    finally:
        ref.random = saved


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    table = {}
    for k in REASONING:
        runs = [run(ref, s, k) for s in SEEDS]
        table[k] = {m: tuple(round(sum(r[m][i] for r in runs) / len(runs), 3) for i in (0, 1))
                    for m in runs[0]}
    doc = " ".join(parity.doc_text(PHASE, LESSON).split())
    return {"table": table, "lesson_claims": "Even when the CoT is subsequently distilled away" in doc
            and "the backdoor survives more than models trained without the CoT" in doc}


def verify(result):
    one, eight = result["table"][1], result["table"][8]
    return [
        practice.Check(
            "ANSWER: CoT persistence cannot be a trigger-weight effect, since distillation drops it",
            result["lesson_claims"] and eight["teacher"][1] > one["teacher"][1]
            and all(one[s][1] < one["teacher"][1] and eight[s][1] < eight["teacher"][1]
                    for s in ("soft", "hard")),
            f"(w[6], survives SFT) by reasoning multiplicity k: {result['table']}; lesson "
            f"claims persistence after distillation: {result['lesson_claims']}",
        ),
        practice.Check(
            "FINDING: the toy's only persistence lever is trigger margin, and it is small",
            (one["teacher"], eight["teacher"]) == ((-16.706, 0.745), (-19.691, 0.8)),
            f"teacher k=1 {one['teacher']}, k=8 {eight['teacher']}: "
            f"{eight['teacher'][1] - one['teacher'][1]:+.3f} survival",
        ),
        practice.Check(
            "FINDING: distillation erases that margin",
            (one["hard"], eight["hard"]) == ((-15.677, 0.706), (-16.406, 0.716))
            and (one["soft"], eight["soft"]) == ((-12.48, 0.615), (-13.078, 0.615)),
            f"hard students {one['hard']} / {eight['hard']}; soft students {one['soft']} / "
            f"{eight['soft']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
