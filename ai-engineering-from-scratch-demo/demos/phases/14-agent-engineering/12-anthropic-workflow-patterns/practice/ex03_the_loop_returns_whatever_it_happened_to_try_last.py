"""Exercise 3 — the loop returns whatever it happened to try last.

    Turn `evaluator_optimizer` into a bandit: keep the top-2 outputs across
    iterations so a late good result doesn't get overwritten by a late bad
    one.

Reading of the exercise: the overwrite is structural. `candidate` is rebound
at the top of every iteration and returned after the loop, so on a run that
never passes the return value is the *last* attempt rather than the best one.
Turning that into a bandit needs something to rank by, and `evaluator`
returns `(bool, str)` -- so the first change is a score, not a keep-two list.

**ANSWER: a top-2 bandit over a scored evaluator.** Across five iterations
scoring **3, 9, 4, 2, 1**, the bandit returns the iteration-2 candidate at
**9** and the shipped loop returns the iteration-5 candidate at **1**. The
runner-up it keeps is the **4**.

**FINDING: the information was never lost, only the return value was.**
`trace` records every candidate, so the best attempt is recoverable after the
fact: the best score in the shipped trace is **9**, the same answer the
bandit returns. A caller reading the trace instead of the return value fixes
this without touching the loop.

**FINDING: the evaluator has no score to rank by.** It returns a `bool` and a
string, so a bandit either parses the judge's prose or the signature widens.
The shipped pass/fail is enough to *stop* and not enough to *choose*, which
is why the overwrite went unnoticed.

**FINDING: the bug only fires when nothing passes.** On a run that succeeds,
`evaluator_optimizer` returns from inside the loop, so the shipped answer and
the bandit's agree: both return the passing candidate at iteration **2**, and
the trace is **2** entries long. The failure is invisible on the happy path.

Structure: `bandit()` is the lesson's loop with a scored evaluator and a
kept top-2; the proposer and the scores are shared by both arms.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "12-anthropic-workflow-patterns"
SCORES = (3, 9, 4, 2, 1)
PASS_MARK, KEEP = 10, 2


def proposer(_task, feedback):
    index = 0 if feedback is None else int(feedback.split()[-1])
    return f"draft-{index + 1}"


def scored(candidate):
    return SCORES[int(candidate.split("-")[1]) - 1]


def evaluator(pass_mark=PASS_MARK):
    def judge(_task, candidate):
        score = scored(candidate)
        return score >= pass_mark, f"scored {score}, attempt {candidate.split('-')[1]}"
    return judge


def bandit(task, propose, judge, max_iter=5, keep=KEEP):
    """The lesson's loop, keeping the best `keep` candidates seen so far."""
    best, feedback = [], None
    for _ in range(max_iter):
        candidate = propose(task, feedback)
        ok, message = judge(task, candidate)
        best = sorted(best + [(scored(candidate), candidate)], reverse=True)[:keep]
        if ok:
            return candidate, best
        feedback = message
    return best[0][1], best


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped, trace = ref.evaluator_optimizer("write it", proposer, evaluator())
    kept = bandit("write it", proposer, evaluator())
    easy_shipped, easy_trace = ref.evaluator_optimizer(
        "write it", proposer, evaluator(pass_mark=9))
    easy_bandit = bandit("write it", proposer, evaluator(pass_mark=9))
    return {
        "shipped": shipped, "shipped_score": scored(shipped),
        "bandit": kept[0], "bandit_score": scored(kept[0]),
        "runner_up": kept[1][1][0], "kept": len(kept[1]),
        "scores": list(SCORES), "iterations": len(trace),
        "best_in_trace": max(scored(entry[0]) for entry in trace),
        "judge_returns": str(inspect.signature(
            ref.evaluator_optimizer).parameters["evaluator"].annotation),
        "easy_agree": easy_shipped == easy_bandit[0],
        "easy_score": scored(easy_shipped), "easy_iterations": len(easy_trace),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the bandit returns 9 where the shipped loop returns 1",
            all([result["shipped_score"] == 1, result["bandit_score"] == 9,
                 result["scores"] == [3, 9, 4, 2, 1], result["runner_up"] == 4,
                 result["kept"] == 2, result["iterations"] == 5]),
            f"across {result['iterations']} iterations scoring {result['scores']} the "
            f"bandit returns {result['bandit']!r} at {result['bandit_score']} and the "
            f"shipped loop returns {result['shipped']!r} at {result['shipped_score']}. "
            f"The runner-up it keeps scores {result['runner_up']}",
        ),
        practice.Check(
            "FINDING: the information was never lost, only the return value was",
            all([result["best_in_trace"] == 9,
                 result["best_in_trace"] == result["bandit_score"],
                 result["best_in_trace"] > result["shipped_score"]]),
            f"trace records every candidate, so the best attempt is recoverable after "
            f"the fact: the best score in the shipped trace is "
            f"{result['best_in_trace']}, the same answer the bandit returns. Reading the "
            "trace instead of the return value fixes this without touching the loop",
        ),
        practice.Check(
            "FINDING: the evaluator has no score to rank by",
            all(["bool" in result["judge_returns"], "str" in result["judge_returns"],
                 "float" not in result["judge_returns"]]),
            f"the evaluator is typed {result['judge_returns']} -- a verdict and a "
            "message, with no number. A bandit either parses the judge's prose or the "
            "signature widens; pass/fail is enough to stop and not enough to choose",
        ),
        practice.Check(
            "FINDING: the bug only fires when nothing passes",
            all([result["easy_agree"] is True, result["easy_score"] == 9,
                 result["easy_iterations"] == 2]),
            f"on a run that succeeds, evaluator_optimizer returns from inside the loop, "
            f"so both arms return the same candidate at iteration "
            f"{result['easy_iterations']} scoring {result['easy_score']} "
            f"({result['easy_agree']}). The overwrite is invisible on the happy path",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
