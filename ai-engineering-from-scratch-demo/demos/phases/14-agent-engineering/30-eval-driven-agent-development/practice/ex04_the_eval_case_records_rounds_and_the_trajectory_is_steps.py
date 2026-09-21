"""Exercise 4 — the eval case records rounds, and the trajectory is steps.

    Add a trajectory-efficiency metric: how many steps did the agent take vs
    a gold trajectory?

Reading of the exercise: `CaseResult` records `rounds`, which counts
proposer-judge cycles, not agent steps -- a case can pass in **1** round
after an agent took **40** actions. So the metric has to come from a
trajectory, and Lesson 20's harness already has one: its `Task` carries
`gold_steps` and its agents produce traces. Wiring that into an `EvalCase`
is the work.

**ANSWER: 1.17x over gold on Lesson 20's three tasks, and the eval case
cannot see it.** Running Lesson 20's shipped agents gives **14** steps
against **12** gold -- `buy_headphones` **1.00x**, `buy_bundle` **1.00x**,
`revised_order` **1.40x**. Wrapped as eval cases all three pass in **1**
round each, so `rounds` reports **1, 1, 1** for trajectories of **3**, **4**
and **7** steps.

**FINDING: rounds and steps are uncorrelated by construction.** Across the
three cases the round counts are identical and the step counts span
**2.33x**, so any efficiency claim built on `CaseResult` is reading the wrong
number. `CaseResult` has **6** fields and **0** of them is a step count.

**FINDING: an efficiency gate and a pass gate disagree on which task is
worst.** All three tasks pass, so a pass-rate gate sees **3/3** and a
**1.2x** efficiency gate fails `revised_order` alone. Gating on efficiency
alone is worse: an agent that returns an empty trajectory scores **0.00x**
and fails the task, which is the metric's own failure mode from Lesson 20.

**FINDING: the judge sees a string, so the trajectory has to be serialised
into it.** `EvalCase.judge` is typed `str -> (bool, str)`, so a
trajectory-based case either encodes the step count in the candidate text or
closes over the trace. Encoding it makes the judge parse **1** number out of
prose; closing over it makes the case unserialisable. Neither is a place a
gold trajectory belongs, which is why the metric needs its own result field.

Structure: `trajectories()` runs Lesson 20's agents; `as_case()` wraps one
in the shipped `EvalCase`.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "30-eval-driven-agent-development"
WEB = "20-benchmarks-webarena-osworld"
GOLD = {"buy_headphones": 3, "buy_bundle": 4, "revised_order": 5}
LIMIT = 1.2


def trajectories(web):
    """Lesson 20's three agents, each run on its own fresh app."""
    agents = (web._agent_task_1, web._agent_task_2, web._agent_task_3)
    rows = {}
    for tid, agent in zip(GOLD, agents):
        app = web.ShoppingApp()
        trace = agent(app)
        rows[tid] = {"steps": len(trace), "gold": GOLD[tid],
                     "ratio": round(len(trace) / GOLD[tid], 2),
                     "orders": len(app.orders)}
    return rows


def as_case(ref, tid, row, limit=None):
    """The trajectory, serialised into the candidate string the judge reads."""
    def proposer(feedback):
        del feedback
        return f"{tid}: completed in {row['steps']} steps, {row['orders']} orders"

    def judge(candidate):
        steps = int(candidate.split("in ")[1].split(" ")[0])
        ratio = round(steps / row["gold"], 2)
        if limit is not None and ratio > limit:
            return False, f"{ratio}x over gold, limit {limit}x"
        return row["orders"] > 0, f"{ratio}x over gold"

    return ref.EvalCase(cid=tid, category="custom", description=tid,
                        proposer=proposer, judge=judge)


def suite(ref, rows, limit=None):
    return [ref.evaluator_optimizer(as_case(ref, tid, row, limit))
            for tid, row in rows.items()]


def empty_trajectory(ref):
    row = {"steps": 0, "gold": 5, "orders": 0}
    return ref.evaluator_optimizer(as_case(ref, "empty", row, limit=LIMIT))


def shape(ref):
    fields = list(ref.CaseResult.__dataclass_fields__)
    return {"result_fields": fields,
            "step_fields": [f for f in fields
                            if "step" in f or "gold" in f or "trajector" in f],
            "judge_type": str(ref.EvalCase.__dataclass_fields__["judge"].type)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    web = parity.load_reference(PHASE, WEB, "main")
    rows = trajectories(web)
    passing, gated = suite(ref, rows), suite(ref, rows, limit=LIMIT)
    steps = [row["steps"] for row in rows.values()]
    return {
        "tasks": len(rows), "steps": steps, "gold": list(GOLD.values()),
        "aggregate": round(sum(steps) / sum(GOLD.values()), 2),
        "ratios": {tid: row["ratio"] for tid, row in rows.items()},
        "rounds": [row.rounds for row in passing],
        "passed": sum(row.passed for row in passing),
        "step_spread": round(max(steps) / min(steps), 2),
        "gated_passed": sum(row.passed for row in gated),
        "gated_failures": [row.cid for row in gated if not row.passed],
        "empty": empty_trajectory(ref).passed, **shape(ref),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 1.17x over gold, and the eval case reports rounds of 1, 1, 1",
            all([result["tasks"] == 3, result["steps"] == [3, 4, 7],
                 result["gold"] == [3, 4, 5], result["aggregate"] == 1.17,
                 result["ratios"] == {"buy_headphones": 1.0, "buy_bundle": 1.0,
                                      "revised_order": 1.4},
                 result["rounds"] == [1, 1, 1], result["passed"] == 3]),
            f"Lesson 20's agents take {result['steps']} steps against gold "
            f"{result['gold']} -- {result['ratios']}, aggregating to "
            f"{result['aggregate']}x. Wrapped as eval cases all three pass in "
            f"{result['rounds']} rounds",
        ),
        practice.Check(
            "FINDING: rounds and steps are uncorrelated by construction",
            all([len(set(result["rounds"])) == 1, result["step_spread"] == 2.33,
                 result["step_fields"] == [],
                 len(result["result_fields"]) == 6]),
            f"the round counts are identical {result['rounds']} while the step counts "
            f"span {result['step_spread']}x, so an efficiency claim built on CaseResult "
            f"reads the wrong number: its {len(result['result_fields'])} fields include "
            f"{len(result['step_fields'])} step counts",
        ),
        practice.Check(
            "FINDING: an efficiency gate and a pass gate disagree",
            all([result["passed"] == 3, result["gated_passed"] == 2,
                 result["gated_failures"] == ["revised_order"],
                 result["empty"] is False]),
            f"a pass-rate gate sees {result['passed']}/3 and a {LIMIT}x efficiency gate "
            f"fails {result['gated_failures']} alone. Gating on efficiency alone is worse "
            f"-- an empty trajectory scores 0.00x and passes the limit while completing "
            f"nothing ({result['empty']} once the order check is kept)",
        ),
        practice.Check(
            "FINDING: the judge sees a string, so the trajectory is serialised into it",
            all(["str" in result["judge_type"], result["step_fields"] == [],
                 result["rounds"] == [1, 1, 1]]),
            f"EvalCase.judge is typed {result['judge_type']}, so a trajectory case either "
            "encodes the step count in prose for the judge to parse back out or closes "
            "over the trace and stops being serialisable. Neither is where a gold "
            "trajectory belongs -- the metric needs its own result field",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
