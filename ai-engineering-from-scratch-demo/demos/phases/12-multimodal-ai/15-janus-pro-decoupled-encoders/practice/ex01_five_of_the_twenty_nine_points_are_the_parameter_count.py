"""Exercise 1 — five of the 29.8 points are the parameter count.

    Janus-Pro-7B beats DALL-E 3 on GenEval. Explain why a 7B open model can match
    a frontier proprietary model on generation but not on understanding.

Reading of the exercise: the asymmetry is argued from the lesson's own
Janus-to-Janus-Pro table, which is the one controlled comparison available here,
and the parameter contribution is separated out using the +7 MMMU per decade
that Lesson 12.07 measured within two model families. What remains after that
subtraction is the answer.

**ANSWER: because generation benchmarks test compliance and understanding
benchmarks test knowledge.** GenEval scores a checklist -- object counts,
colours, relative positions -- which a 7B model that follows instructions scores
as well as a frontier one. MMMU scores what the language model knows, and that
is the axis where parameters and pretraining data still buy points.

**FINDING: the same step buys 3.1x more in understanding than in generation.**
Janus to Janus-Pro is **+29.8** MMMU (**+97.7%** relative) against **+0.19**
GenEval (**+31.1%**). Nearly doubling one score and adding a third to the other,
from one set of changes.

**FINDING: only 5.1 of the 29.8 MMMU points are the parameter increase.** 1.3B
to 7B is 0.731 decades, which at Lesson 12.07's measured +7 MMMU per decade is
**5.1** -- **17.2%** of the gain. The other 82.8% is the data, and the data row
that moved most is stage 2 at +176.9%.

**FINDING: and GenEval has 0.20 of headroom left against MMMU's 39.7.** A
benchmark scored out of 1.0 with a model at 0.80 cannot separate a 7B from a
frontier model by much, whatever either knows. The asymmetry the exercise asks
about is partly a property of the two scoreboards, not only of the two models.

Structure: `TABLE` transcribes the lesson's own two-model rows, `relative` puts
the two benchmarks on a common footing, and `scale_share` attributes part of the
MMMU gain to the parameter increase at Lesson 12.07's rate.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "15-janus-pro-decoupled-encoders"
TABLE = {"MMMU": (30.5, 60.3, 100.0), "GenEval": (0.61, 0.80, 1.0)}
PARAMS = (1.3, 7.0)
MMMU_PER_DECADE = 7.0           # Lesson 12.07, measured within two model families
STAGE_TWO = (26, 72)


def relative(before, after):
    return round((after / before - 1) * 100, 1)


def decades(before, after):
    return round(math.log10(after / before), 3)


def scale_share(gain, before, after, rate=MMMU_PER_DECADE):
    predicted = decades(before, after) * rate
    return round(predicted, 1), round(predicted / gain * 100, 1)


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    gains = {name: round(after - before, 2) for name, (before, after, _) in TABLE.items()}
    predicted, share = scale_share(gains["MMMU"], *PARAMS)
    return {
        "table": {name: (before, after) for name, (before, after, _) in TABLE.items()},
        "gains": gains,
        "relative": {name: relative(before, after)
                     for name, (before, after, _) in TABLE.items()},
        "relative_ratio": round(relative(*TABLE["MMMU"][:2])
                                / relative(*TABLE["GenEval"][:2]), 1),
        "param_decades": decades(*PARAMS),
        "predicted_mmmu": predicted, "scale_share_pct": share,
        "data_share_pct": round(100 - share, 1),
        "stage_two_pct": relative(*STAGE_TWO),
        "headroom": {name: round(ceiling - after, 2)
                     for name, (_, after, ceiling) in TABLE.items()},
    }


def verify(result):
    gains, rel, headroom = result["gains"], result["relative"], result["headroom"]
    return [
        practice.Check(
            "ANSWER: generation benchmarks test compliance, understanding tests knowledge",
            all([result["table"] == {"MMMU": (30.5, 60.3), "GenEval": (0.61, 0.8)},
                 gains == {"MMMU": 29.8, "GenEval": 0.19}]),
            f"the lesson's own two-model rows are {result['table']}, a gain of {gains}. "
            "GenEval scores a checklist -- object counts, colours, relative positions -- "
            "which an instruction-following 7B scores as well as a frontier model; MMMU "
            "scores what the language model knows, which is where parameters still buy "
            "points",
        ),
        practice.Check(
            "FINDING: the same step buys 3.1x more in understanding than in generation",
            all([rel == {"MMMU": 97.7, "GenEval": 31.1},
                 result["relative_ratio"] == 3.1]),
            f"in relative terms the gains are {rel}% -- a ratio of "
            f"{result['relative_ratio']}. One score nearly doubles and the other gains a "
            "third, from the same set of changes",
        ),
        practice.Check(
            "FINDING: only 5.1 of the 29.8 MMMU points are the parameter increase",
            all([result["param_decades"] == 0.731, result["predicted_mmmu"] == 5.1,
                 result["scale_share_pct"] == 17.2, result["data_share_pct"] == 82.8,
                 result["stage_two_pct"] == 176.9]),
            f"{PARAMS[0]}B to {PARAMS[1]}B is {result['param_decades']} decades, which at "
            f"Lesson 12.07's +{MMMU_PER_DECADE:.0f} MMMU per decade is "
            f"{result['predicted_mmmu']} points -- {result['scale_share_pct']}% of the gain. "
            f"The other {result['data_share_pct']}% is data, and the stage that moved most "
            f"is stage 2 at +{result['stage_two_pct']}%",
        ),
        practice.Check(
            "FINDING: GenEval has 0.20 of headroom left against MMMU's 39.7",
            all([headroom == {"MMMU": 39.7, "GenEval": 0.2},
                 TABLE["GenEval"][1] / TABLE["GenEval"][2] == 0.8]),
            f"headroom to each benchmark's ceiling is {headroom}: GenEval sits at 80% of "
            "1.0 and MMMU at 60.3% of 100. A scoreboard that close to its top cannot "
            "separate a 7B from a frontier model by much, whatever either knows -- so the "
            "asymmetry is partly a property of the benchmarks and not only of the models",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
