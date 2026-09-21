"""Exercise 2 — a blended score gives the proposer nothing to fix.

    Build an LLM-judge rubric for your domain with three dimensions
    (factual, tone, scope). Score 50 sessions.

Reading of the exercise: the rubric matters less than what it hands back.
`evaluator_optimizer` feeds the judge's `reason` string to the proposer as
`feedback`, so a rubric that returns a blended number gives the next round
nothing to act on, while one that names the failing dimension gives it an
instruction. Both are built here and run over the same 50 sessions.

**ANSWER: naming the failing dimension converges 50 of 50; a blended score
converges 10.** Scoring factual, tone and scope at a third each and
requiring all three, the dimension-naming judge passes every session in a
mean of **1.8** rounds. The same rubric returning only `"score 0.67"`
passes **10** -- exactly the sessions that were already correct -- and the
other **40** exhaust **3** rounds each without improving.

**FINDING: the feedback string is the whole interface.** `EvalCase` carries
a judge typed `str -> (bool, str)` and `evaluator_optimizer` passes that
second element straight through, so the rubric's *wording* is an input to
the agent. Changing only the reason text, with identical verdicts, moves the
pass rate from **20.0%** to **100.0%**.

**FINDING: a session can fail on one dimension and score well overall.** Of
the 50 sessions, **20** score at or above **0.60** while failing at least
one dimension outright, so a threshold on the mean admits them. Requiring
every dimension leaves **10/50** passing before any refinement, which is the
number the loop then has to move -- and is also exactly what the blended
judge ends at.

**FINDING: three rounds is enough here and the cap is invisible when it is
not.** `max_rounds` defaults to **3** and a session needing a fourth is
reported as a plain failure -- `CaseResult` has **6** fields and **0** says
"budget exhausted". Dropping the cap to **1** takes the dimension-naming
judge from **50/50** to **10/50** with no other change, and "the agent got
it wrong" and "the agent ran out of rounds" read identically in the
output.

Structure: `rubric()` scores three dimensions; `reason_for()` is the only
difference between the two judges.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "30-eval-driven-agent-development"
FACT, RUDE, SCOPE = "30 days", "obviously", "refund"
SESSIONS = 50


def session(index):
    """A session broken on 0, 1 or 2 dimensions, cycling through the shapes."""
    kind = index % 5
    return {"factual": kind != 1, "tone": kind != 2, "scope": kind not in (3, 4)}


def render(state):
    parts = [f"the {SCOPE} window is {FACT}" if state["factual"]
             else f"the {SCOPE} window is 99 years"]
    if not state["tone"]:
        parts.insert(0, RUDE)
    if not state["scope"]:
        parts = ["our stock price closed up 2% today"]
    return " ".join(parts)


def rubric(text):
    return {"factual": float(FACT in text), "tone": float(RUDE not in text),
            "scope": float(SCOPE in text)}


def reason_for(scores, naming):
    failing = sorted(name for name, value in scores.items() if value < 1.0)
    if naming:
        return f"fix {', '.join(failing)}" if failing else "all dimensions pass"
    return f"score {sum(scores.values()) / 3:.2f}"


def make_case(ref, index, naming, max_rounds=3):
    state = dict(session(index))

    def proposer(feedback):
        for name in ("factual", "tone", "scope"):
            if feedback and name in feedback:
                state[name] = True
        return render(state)

    def judge(candidate):
        scores = rubric(candidate)
        return all(v == 1.0 for v in scores.values()), reason_for(scores, naming)

    return ref.EvalCase(cid=f"s{index:03d}", category="custom",
                        description="rubric", proposer=proposer, judge=judge,
                        max_rounds=max_rounds)


def run(ref, naming, max_rounds=3):
    rows = [ref.evaluator_optimizer(make_case(ref, i, naming, max_rounds))
            for i in range(SESSIONS)]
    passed = [row for row in rows if row.passed]
    return {"passed": len(passed), "rows": rows,
            "mean_rounds": round(sum(r.rounds for r in passed) / len(passed), 2)
            if passed else 0.0,
            "rate": round(100 * len(passed) / SESSIONS, 1)}


def first_round(ref):
    """How many sessions are already correct before any refinement."""
    scores = [rubric(render(session(i))) for i in range(SESSIONS)]
    return {
        "clean": sum(all(v == 1.0 for v in row.values()) for row in scores),
        "above_threshold": sum(sum(row.values()) / 3 >= 0.6
                               and any(v < 1.0 for v in row.values())
                               for row in scores),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    naming, blended = run(ref, True), run(ref, False)
    return {
        "sessions": SESSIONS, "naming": naming["passed"],
        "blended": blended["passed"], "mean_rounds": naming["mean_rounds"],
        "stuck": SESSIONS - blended["passed"],
        "naming_rate": naming["rate"], "blended_rate": blended["rate"],
        "judge_returns": ref.EvalCase.__dataclass_fields__["judge"].type,
        "first_round": first_round(ref),
        "max_rounds": ref.EvalCase.__dataclass_fields__["max_rounds"].default,
        "one_round": run(ref, True, max_rounds=1)["passed"],
        "result_fields": list(ref.CaseResult.__dataclass_fields__),
        "budget_fields": [f for f in ref.CaseResult.__dataclass_fields__
                          if "exhaust" in f or "budget" in f],
    }


def verify(result):
    first = result["first_round"]
    return [
        practice.Check(
            "ANSWER: naming the failing dimension converges 50 of 50, a score 10",
            all([result["sessions"] == 50, result["naming"] == 50,
                 result["blended"] == 10, result["mean_rounds"] == 1.8,
                 result["stuck"] == 40]),
            f"requiring all three dimensions, the judge that names what failed passes "
            f"{result['naming']}/{result['sessions']} in a mean of "
            f"{result['mean_rounds']} rounds. The same rubric returning only a blended "
            f"score passes {result['blended']} -- the already-correct sessions -- and "
            f"{result['stuck']} exhaust their rounds",
        ),
        practice.Check(
            "FINDING: the feedback string is the whole interface",
            all([result["naming_rate"] == 100.0, result["blended_rate"] == 20.0,
                 "str" in str(result["judge_returns"])]),
            f"the judge is typed {result['judge_returns']} and evaluator_optimizer passes "
            f"its second element straight to the proposer, so the rubric's wording is an "
            f"input to the agent: identical verdicts move the pass rate from "
            f"{result['blended_rate']}% to {result['naming_rate']}%",
        ),
        practice.Check(
            "FINDING: a session can fail one dimension and score well overall",
            all([first["clean"] == 10, first["above_threshold"] == 20]),
            f"{first['above_threshold']} of {result['sessions']} sessions score at or "
            f"above 0.60 while failing at least one dimension outright, so a threshold on "
            f"the mean admits them. Requiring every dimension leaves {first['clean']} "
            "passing before any refinement",
        ),
        practice.Check(
            "FINDING: the round cap is invisible when it binds",
            all([result["max_rounds"] == 3, result["one_round"] == 10,
                 result["budget_fields"] == [],
                 len(result["result_fields"]) == 6]),
            f"max_rounds defaults to {result['max_rounds']} and CaseResult's "
            f"{len(result['result_fields'])} fields include "
            f"{len(result['budget_fields'])} saying 'budget exhausted'. Dropping the cap "
            f"to 1 takes the naming judge from {result['naming']}/50 to "
            f"{result['one_round']}/50, and the two failure modes read identically",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
