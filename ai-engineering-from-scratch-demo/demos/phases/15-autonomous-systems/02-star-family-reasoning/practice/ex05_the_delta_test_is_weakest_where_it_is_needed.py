"""Exercise 5 — the delta test is weakest exactly where it is needed.

    Design one evaluation that would catch shortcut rationales in a deployed
    model. It does not have to be perfect — it has to break the simplest
    shortcuts a STaR loop would reinforce.

Reading of the exercise: "deployed" rules out anything that reads the
model's internals or a soundness label, so the evaluation has to work from
paired inputs and outcomes alone. The design is the in-distribution /
perturbed-twin accuracy delta; what belongs in a file is its operating
characteristic, measured against the one simulator where the ground truth is
known.

**ANSWER: the delta is a direct readout of the shortcut share.** Because the
two distributions differ by one literal, accuracy gap = **0.35 x** share
exactly -- dividing the measured gap by 0.35 returns the share with **0.0**
residual at every round. On the shipped run that is **14.0** points of gap at
round 0 and **4.1** at round 5.

**FINDING: sensitivity decays as the loop runs.** The gap shrinks with the
share it is measuring -- 14.0, 14.0, 11.6, 8.6, 6.0, 4.1 points over the six
rounds -- so the test is **3.4x** weaker at the round you would ship than at
the round you would never ship. A fixed threshold gets less protective every
round, which is the opposite of what a deployment gate should do.

**FINDING: the shipped evaluation cannot resolve its own final gap.**
`report_round` scores **500** items per arm; at the run's final **92.0%**
accuracy that is a **1.7**-point standard error on the difference, so a
4.1-point gap is **2.4** sigma. Three sigma needs **791** per arm -- the
sample size has to be derived from the delta you intend to detect, and the
module's is not.

**FINDING: the 0.35 is an oracle.** It exists because `Model.sample` defines
both hit rates; **0** functions in the module estimate it. So a deployed
version of this test can threshold the delta and cannot report a share, and
any protocol that quotes "the model is 12% shortcut" is quoting the
simulator's own literals back at itself.

Structure: `rounds()` walks the bootstrap with both arms scored;
`sample_size()` is the arm size a target sigma needs.
"""

from __future__ import annotations

import inspect
import math
import statistics

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "02-star-family-reasoning"

SOUND0, SHORT0, ROUNDS = 0.20, 0.40, 5
ALPHA, ID_HIT, OOD_HIT, RANDOM_HIT = 0.6, 0.40, 0.05, 0.10
SHIPPED_ARM, TARGET_SIGMA = 500, 3.0     # report_round's n, and the power we want


def rounds():
    """(share, ID accuracy, OOD accuracy) at each bootstrap round, in expectation."""
    sound, shortcut, out = SOUND0, SHORT0, []
    for _ in range(ROUNDS + 1):
        guess = 1 - sound - shortcut
        out.append((shortcut, sound + shortcut * ID_HIT + guess * RANDOM_HIT,
                    sound + shortcut * OOD_HIT + guess * RANDOM_HIT))
        kept = (sound, shortcut * ID_HIT, guess * RANDOM_HIT)
        total = sum(kept)
        sound, shortcut = (ALPHA * kept[0] / total + (1 - ALPHA) * sound,
                           ALPHA * kept[1] / total + (1 - ALPHA) * shortcut)
        scale = max(sound + shortcut, 1.0)
        sound, shortcut = sound / scale, shortcut / scale
    return out


def standard_error(arm, accuracy):
    """Standard error of a difference in proportions, in points."""
    return math.sqrt(2 * accuracy * (1 - accuracy) / arm) * 100


def sample_size(gap, accuracy, sigma=TARGET_SIGMA):
    return math.ceil(2 * accuracy * (1 - accuracy) * (sigma / (gap / 100)) ** 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    walk = rounds()
    gaps = [round((row[1] - row[2]) * 100, 1) for row in walk]
    recovered = [abs(row[0] - (row[1] - row[2]) / (ID_HIT - OOD_HIT)) for row in walk]
    final_accuracy = walk[-1][1]
    error = standard_error(SHIPPED_ARM, final_accuracy)
    report = inspect.getsource(ref.report_round)
    return {
        "gaps": gaps,
        "worst_recovery": round(max(recovered), 3),
        "scale": round(ID_HIT - OOD_HIT, 2),
        "decay": round(gaps[0] / gaps[-1], 1),
        "monotone": all(b <= a for a, b in zip(gaps, gaps[1:])),
        "shipped_arm": SHIPPED_ARM,
        "arms_in_report": report.count(", 500"),
        "standard_error": round(error, 1),
        "sigma": round(gaps[-1] / error, 1),
        "needed_arm": sample_size(gaps[-1], final_accuracy),
        "final_accuracy": round(final_accuracy, 3),
        "estimators": [name for name in ("evaluate", "star_round", "vstar_infer")
                       if "0.35" in inspect.getsource(getattr(ref, name))],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the delta is a direct readout of the shortcut share",
            all([result["scale"] == 0.35, result["worst_recovery"] == 0.0,
                 result["gaps"][0] == 14.0, result["gaps"][-1] == 4.1]),
            f"gap = {result['scale']} x share exactly, so dividing it back returns the "
            f"share with {result['worst_recovery']} residual; the walk runs "
            f"{result['gaps'][0]} points at round 0 down to {result['gaps'][-1]} at "
            f"round {len(result['gaps']) - 1}",
        ),
        practice.Check(
            "FINDING: sensitivity decays as the loop runs",
            all([result["monotone"], result["gaps"] == [14.0, 14.0, 11.6, 8.6, 6.0, 4.1],
                 result["decay"] >= 3.0]),
            f"the gap walks {result['gaps']} over the six rounds, so the test is "
            f"{result['decay']}x weaker at the round you would ship than at the round "
            "you would never ship",
        ),
        practice.Check(
            "FINDING: the shipped evaluation cannot resolve its own final gap",
            all([result["arms_in_report"] == 2, result["standard_error"] == 1.7,
                 result["sigma"] == 2.4, result["needed_arm"] == 791]),
            f"report_round scores {result['shipped_arm']} per arm, a "
            f"{result['standard_error']}-point standard error at "
            f"{result['final_accuracy']:.1%} accuracy, so the {result['gaps'][-1]}-point "
            f"gap is {result['sigma']} sigma; three sigma needs "
            f"{result['needed_arm']} per arm",
        ),
        practice.Check(
            "FINDING: the 0.35 is an oracle",
            result["estimators"] == [],
            f"{len(result['estimators'])} of the module's three measurement functions "
            f"derives the {result['scale']} scale factor -- it exists because "
            "Model.sample defines both hit rates, so a deployed test can threshold the "
            "delta and cannot report a share",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
