"""Exercise 3 — four factors, one delta, and no weights.

    Read SaferAI's RSP grading methodology. Reproduce their 1.9 score for
    v3.0 by applying their rubric to the document. Which rubric row drove the
    downgrade most?

Reading of the exercise: reproducing a score means solving for the weights
that produce it, so the first thing to check is whether the system is
determined. It is not -- the lesson gives **4** downgrade factors and **1**
observed delta -- and saying so with the arithmetic is a better answer than
inventing weights that happen to land on 1.9.

**ANSWER: not reproducible from what is published, and the honest answer is
the pause clause.** The score moves **2.2** to **1.9**, a delta of
**-0.30** over **4** named factors: **4** unknowns and **1** equation.
Equal weighting assigns **-0.075** to each; attributing the whole drop to the
pause removal assigns **-0.30** and **0.0** to the other three. Both satisfy
the equation exactly, and the lesson's own prose picks the second by calling
the pause removal "the strongest regression".

**FINDING: the category change is more sensitive than the score.** The drop
is **13.6%** of the prior score, and it moves the document from *moderate* to
*weak* -- so the band boundary lies inside a **0.30** interval on a
**1.0-4.0** scale. A rating whose headline is the band is reporting a
threshold crossing, and threshold crossings are exactly what the rest of this
phase measures the fragility of.

**FINDING: three of the four factors are about removing numbers.**
Qualitative thresholds replacing quantitative ones, the pause commitment
going, and mitigations becoming an "affirmative case" all describe the same
move -- replacing a checkable statement with a reviewable one. The fourth, the
Safety Advisory Group's limited independent oversight, is about *who* checks.
So the rubric's mass sits on **3** counts of one thing, which is why no
per-factor weighting can be recovered from a single delta.

**FINDING: the reading aid keeps the numbers the document dropped.**
`AI_RD_4_THRESHOLDS` holds **3** quantitative thresholds and the crossing rule
is a count -- the shape SaferAI marks v2 up for and v3.0 down for. Reproducing
the rubric against *this module* would score the module higher than the policy
it models, which is worth stating before quoting either number.

Structure: `weights()` enumerates the assignments consistent with the observed
delta; `band()` locates the category boundary.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "19-anthropic-rsp"

BEFORE, AFTER = 2.2, 1.9
SCALE = (1.0, 4.0)
FACTORS = (
    "qualitative thresholds replace quantitative ones",
    "pause commitment removed",
    "AI R&D-4 mitigations are an affirmative case, not specific measures",
    "review depends on Anthropic's Safety Advisory Group",
)
ABOUT_NUMBERS = FACTORS[:3]


def delta():
    return round(AFTER - BEFORE, 2)


def weights():
    """Two assignments that satisfy the single equation exactly."""
    equal = [round(delta() / len(FACTORS), 3)] * len(FACTORS)
    pause = [0.0, delta(), 0.0, 0.0]
    return equal, pause


def consistent(assignment):
    return round(sum(assignment), 2) == delta()


def band():
    """The relative drop, and whether the boundary lies inside it."""
    return round(abs(delta()) / BEFORE, 3), AFTER < 2.0 <= BEFORE


def solve():
    parity.load_reference(PHASE, LESSON, "main")     # D5: the lesson is the source
    equal, pause = weights()
    relative, crosses = band()
    return {
        "before": BEFORE, "after": AFTER, "delta": delta(),
        "factors": len(FACTORS),
        "equations": 1,
        "underdetermined": len(FACTORS) > 1,
        "equal": equal,
        "pause": pause,
        "both_consistent": consistent(equal) and consistent(pause),
        "relative": relative,
        "crosses_band": crosses,
        "scale": list(SCALE),
        "about_numbers": len(ABOUT_NUMBERS),
        "about_oversight": len(FACTORS) - len(ABOUT_NUMBERS),
        "module_thresholds": 3,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: not reproducible -- four unknowns, one equation",
            all([result["delta"] == -0.30, result["factors"] == 4,
                 result["equations"] == 1, result["underdetermined"],
                 result["equal"] == [-0.075] * 4,
                 result["pause"] == [0.0, -0.3, 0.0, 0.0],
                 result["both_consistent"]]),
            f"the score moves {result['before']} to {result['after']}, a delta of "
            f"{result['delta']} across {result['factors']} factors with "
            f"{result['equations']} equation; equal weighting gives {result['equal']} "
            f"and pause-only gives {result['pause']}, and both fit exactly",
        ),
        practice.Check(
            "FINDING: the category change is more sensitive than the score",
            all([result["relative"] == 0.136, result["crosses_band"],
                 result["scale"] == [1.0, 4.0]]),
            f"the drop is {result['relative']:.1%} of the prior score on a "
            f"{result['scale']} scale and moves the document across a band boundary, so "
            "the headline reports a threshold crossing",
        ),
        practice.Check(
            "FINDING: three of the four factors are about removing numbers",
            all([result["about_numbers"] == 3, result["about_oversight"] == 1]),
            f"{result['about_numbers']} of {result['factors']} factors describe "
            f"replacing a checkable statement with a reviewable one and "
            f"{result['about_oversight']} describes who checks -- so the mass sits on "
            "three counts of one thing",
        ),
        practice.Check(
            "FINDING: the reading aid keeps the numbers the document dropped",
            result["module_thresholds"] == 3,
            f"the module holds {result['module_thresholds']} quantitative thresholds "
            "and a counting rule -- the shape the rubric marks down for removing, so "
            "the aid would score higher than the policy it models",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
