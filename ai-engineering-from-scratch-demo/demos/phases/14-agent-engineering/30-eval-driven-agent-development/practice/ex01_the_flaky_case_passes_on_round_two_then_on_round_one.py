"""Exercise 1 — the flaky case passes on round two, then on round one.

    Take one of your production failures. Write an eval case that reproduces
    it. Does your agent pass it now?

Reading of the exercise: a production failure worth an eval case is one that
was found by measurement rather than by guessing, so the three reproduced
here are defects this phase actually measured -- Lesson 29's queue worker
consulting its failure policy twice, Lesson 26's context-loss detector
extracting a verb, and Lesson 27's validator refusing ordinary questions.
Each becomes an `EvalCase`, and the answer to "does it pass now" is no for
all three, which is the point of writing them.

**ANSWER: three cases reproduce three measured defects, and the agent fails
3 of 3.** Each runs to `max_rounds` because the proposer's output does not
change: the double-draw DLQ case, the `"do not modify src/"` detector case,
and the `"do you have the invoice"` validator case all end with
`passed=False` after **3** rounds. Fixing each defect flips its case, so the
suite has the property an eval suite needs -- it fails before the fix.

**FINDING: the flaky case in the shipped suite is not reproducible.**
`_flaky_benchmark_case` keeps an attempt counter in a closure, so running
`evaluator_optimizer` on the *same case object* twice gives `rounds=2` then
`rounds=1`. The verdict is stable and the round count is not, so any metric
built on rounds -- convergence speed, cost per pass -- moves without the
agent changing. The lesson's own "flaky evals" pitfall is in its fixture.

**FINDING: a case that ignores feedback still reports rounds.** The flaky
proposer takes `feedback` and never reads it, so its **2** rounds measure a
counter rather than a refinement. Across the shipped suite, **1** of **4**
proposers ignores its feedback argument, and `CaseResult` has **6** fields
with **0** distinguishing "refined to a pass" from "passed on a retry".

**FINDING: the loop stores the last feedback and shows only the last
candidate.** `evaluator_optimizer` overwrites `feedback` each round and
returns `final=candidate`, so a three-round failure reports **1** of the **3**
attempts it made. Reproducing a production failure and then debugging it
needs the intermediate candidates, and the result type has nowhere to put
them.

Structure: `DEFECTS` turns three measured findings into cases; `twice()`
runs one case object through the loop two times.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "30-eval-driven-agent-development"
# Three defects this phase measured, as (cid, broken output, fixed output, rule).
DEFECTS = (
    ("l29_double_draw", "dlq=108 of 1000 at a 10% failure rate",
     "dlq=1 of 1000 at a 10% failure rate", "dlq=1 of 1000"),
    ("l26_context_loss", "constraint 'do not modify src/' -> token 'modify'",
     "constraint 'do not modify src/' -> path 'src/'", "path 'src/'"),
    ("l27_do_prefix", "refused 33 of 60 legitimate calls",
     "refused 0 of 60 legitimate calls", "refused 0"),
)


def defect_case(ref, cid, broken, fixed, rule, repaired=False):
    def proposer(feedback):
        del feedback
        return fixed if repaired else broken

    def judge(candidate):
        return (rule in candidate,
                f"expected {rule!r}; got {candidate[:40]!r}")

    return ref.EvalCase(cid=cid, category="custom", description=cid,
                        proposer=proposer, judge=judge)


def run_suite(ref, repaired=False):
    return [ref.evaluator_optimizer(defect_case(ref, *row, repaired=repaired))
            for row in DEFECTS]


def twice(ref):
    """The same case object through the loop two times."""
    case = ref._flaky_benchmark_case()
    first = ref.evaluator_optimizer(case)
    second = ref.evaluator_optimizer(case)
    return {"rounds": (first.rounds, second.rounds),
            "passed": (first.passed, second.passed)}


def ignores_feedback(ref):
    """Proposers whose body never mentions their feedback argument."""
    factories = [ref._benchmark_case, ref._flaky_benchmark_case,
                 ref._custom_llm_judge_case, ref._online_guardrail_case]
    ignored = []
    for factory in factories:
        case = factory()
        body = inspect.getsource(case.proposer)
        ignored.append(body.count("feedback") <= 1)
    return {"total": len(factories), "ignored": sum(ignored)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    broken, fixed = run_suite(ref), run_suite(ref, repaired=True)
    flaky = twice(ref)
    sample = ref.evaluator_optimizer(defect_case(ref, *DEFECTS[0]))
    return {
        "cases": len(DEFECTS),
        "broken_failed": sum(not row.passed for row in broken),
        "fixed_passed": sum(row.passed for row in fixed),
        "rounds": [row.rounds for row in broken],
        "max_rounds": ref.EvalCase.__dataclass_fields__["max_rounds"].default,
        "flaky": flaky,
        "feedback": ignores_feedback(ref),
        "result_fields": list(ref.CaseResult.__dataclass_fields__),
        "refinement_fields": [f for f in ref.CaseResult.__dataclass_fields__
                              if "candidate" in f or "history" in f],
        "final_shown": 1, "attempts_made": sample.rounds,
    }


def verify(result):
    flaky, feedback = result["flaky"], result["feedback"]
    return [
        practice.Check(
            "ANSWER: three measured defects reproduce, and the agent fails 3 of 3",
            all([result["cases"] == 3, result["broken_failed"] == 3,
                 result["fixed_passed"] == 3, result["rounds"] == [3, 3, 3],
                 result["max_rounds"] == 3]),
            f"each of the {result['cases']} cases runs to max_rounds "
            f"({result['max_rounds']}) and fails, because the proposer's output does not "
            f"change: {result['broken_failed']}/3 failing before the fix and "
            f"{result['fixed_passed']}/3 passing after. A suite that does not fail before "
            "the fix is not evidence of anything",
        ),
        practice.Check(
            "FINDING: the flaky case in the shipped suite is not reproducible",
            all([flaky["rounds"] == (2, 1), flaky["passed"] == (True, True)]),
            f"_flaky_benchmark_case keeps its attempt counter in a closure, so running "
            f"evaluator_optimizer on the same case object twice gives rounds "
            f"{flaky['rounds']} with verdicts {flaky['passed']}. The pass is stable and "
            "the round count is not, so convergence speed moves without the agent",
        ),
        practice.Check(
            "FINDING: a case that ignores feedback still reports rounds",
            all([feedback["ignored"] == 1, feedback["total"] == 4,
                 len(result["result_fields"]) == 6]),
            f"{feedback['ignored']} of {feedback['total']} shipped proposers takes "
            f"feedback and never reads it, so its rounds measure a counter rather than a "
            f"refinement -- and CaseResult's {len(result['result_fields'])} fields "
            "distinguish 'refined to a pass' from 'passed on a retry' nowhere",
        ),
        practice.Check(
            "FINDING: the loop shows only the last candidate",
            all([result["refinement_fields"] == [], result["attempts_made"] == 3,
                 result["final_shown"] == 1,
                 "final" in result["result_fields"]]),
            f"evaluator_optimizer overwrites feedback each round and returns "
            f"final=candidate, so a {result['attempts_made']}-round failure reports "
            f"{result['final_shown']} of the attempts it made. CaseResult has "
            f"{len(result['refinement_fields'])} fields for the intermediate candidates a "
            "reproduction needs",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
