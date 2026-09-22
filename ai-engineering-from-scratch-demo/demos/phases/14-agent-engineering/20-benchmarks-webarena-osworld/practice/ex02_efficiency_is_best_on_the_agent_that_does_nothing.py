"""Exercise 2 — efficiency is best on the agent that does nothing.

    Add trajectory-efficiency reporting per task. On your toy, is the agent
    1x, 2x, or 3x over gold?

Reading of the exercise: `main()` already prints `steps / gold_steps` per
task, so the reporting exists -- as text, inside a loop, next to nothing that
can aggregate it. Adding it properly means deciding three things the print
statement decides silently: what counts as a step, how per-task ratios
combine, and whether a failed task has an efficiency at all.

**ANSWER: 1x on two tasks and 1.40x on the third -- 1.17x aggregate.**
`buy_headphones` is **3** steps against gold **3**, `buy_bundle` **4**
against **4**, `revised_order` **7** against **5**. Only the third is inside
OSWorld-Human's reported 1.4-2.7x band, and the scripted agent is otherwise
exactly optimal, because it was written to be.

**FINDING: a step is a line of narration, not an action.** `revised_order`
appends `"revised_choice: remove keyboard"` to its trace without touching the
app, so counting trace lines gives **7** steps where the app was called
**6** times. Per-task that is **1.40x** against **1.20x**, and in aggregate
**1.17x** against **1.08x**. The metric is measuring the agent's prose.

**FINDING: the aggregate is a ratio of sums, and the mean of the ratios is a
different number.** `total_steps / total_gold` is **1.17x**; averaging the
per-task ratios gives **1.13x**. The first weights tasks by length, so the
one task that is over gold moves the headline more than the two that are not,
and dropping it takes the aggregate to **1.00x**.

**FINDING: the metric is optimised by giving up.** An agent that returns an
empty trace scores **0.00x** on every task -- better than gold -- and fails
all **3**. `main()` computes efficiency before it looks at success, so any
ranking on efficiency alone sorts the do-nothing agent first.

Structure: `Counted` wraps the app to separate actions from narration;
`report()` produces both aggregations for the same run.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "20-benchmarks-webarena-osworld"


class Counted:
    """A proxy that records every app method the agent actually calls."""

    def __init__(self, app):
        self._app, self.calls = app, []

    def __getattr__(self, name):
        attr = getattr(self._app, name)
        if not callable(attr):
            return attr

        def wrapped(*args, **kwargs):
            self.calls.append(name)
            return attr(*args, **kwargs)
        return wrapped


def tasks(ref):
    """The lesson's three tasks, which live inside main() and cannot be imported."""
    rows = (
        ("buy_headphones", ref._agent_task_1, 3,
         lambda app: any(o["items"].get("sku-001") == 1 for o in app.orders)),
        ("buy_bundle", ref._agent_task_2, 4,
         lambda app: any(o["items"].get("sku-002") == 1
                         and o["items"].get("sku-003") == 1 for o in app.orders)),
        ("revised_order", ref._agent_task_3, 5,
         lambda app: any(o["items"].get("sku-001") == 1
                         and o["items"].get("sku-003") == 1
                         and "sku-002" not in o["items"] for o in app.orders)),
    )
    return [ref.Task(tid, tid, agent, gold, ok) for tid, agent, gold, ok in rows]


def null_agent(_app):
    return []


def run(ref, task, agent=None):
    app = Counted(ref.ShoppingApp())
    trace = (agent or task.agent)(app)
    return {"tid": task.tid, "ok": task.success(app), "lines": len(trace),
            "actions": len(app.calls), "gold": task.gold_steps,
            "ratio": round(len(trace) / task.gold_steps, 2),
            "action_ratio": round(len(app.calls) / task.gold_steps, 2)}


def report(rows):
    steps, gold = sum(r["lines"] for r in rows), sum(r["gold"] for r in rows)
    return {
        "rows": rows, "ok": sum(r["ok"] for r in rows),
        "ratio_of_sums": round(steps / gold, 2),
        "mean_of_ratios": round(sum(r["ratio"] for r in rows) / len(rows), 2),
        "action_ratio_of_sums": round(sum(r["actions"] for r in rows) / gold, 2),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = [run(ref, task) for task in tasks(ref)]
    return {
        "scripted": report(rows),
        "without_third": report(rows[:2]),
        "null": report([run(ref, task, null_agent) for task in tasks(ref)]),
        "band": (1.4, 2.7),
        "in_band": [r["tid"] for r in rows if 1.4 <= r["ratio"] <= 2.7],
    }


def verify(result):
    scripted, null = result["scripted"], result["null"]
    ratios = [row["ratio"] for row in scripted["rows"]]
    return [
        practice.Check(
            "ANSWER: 1x on two tasks, 1.40x on the third, 1.17x aggregate",
            all([ratios == [1.0, 1.0, 1.4], scripted["ratio_of_sums"] == 1.17,
                 scripted["ok"] == 3, result["in_band"] == ["revised_order"]]),
            f"the per-task ratios are {ratios} against golds "
            f"{[r['gold'] for r in scripted['rows']]}, aggregating to "
            f"{scripted['ratio_of_sums']}x. Only {result['in_band']} sits inside "
            f"OSWorld-Human's {result['band'][0]}-{result['band'][1]}x band; the scripted "
            "agent is otherwise exactly optimal, because it was written to be",
        ),
        practice.Check(
            "FINDING: a step is a line of narration, not an action",
            all([scripted["rows"][2]["lines"] == 7,
                 scripted["rows"][2]["actions"] == 6,
                 scripted["rows"][2]["action_ratio"] == 1.2,
                 scripted["action_ratio_of_sums"] == 1.08]),
            f"revised_order appends 'revised_choice: remove keyboard' without touching "
            f"the app, so it counts {scripted['rows'][2]['lines']} steps where the app "
            f"was called {scripted['rows'][2]['actions']} times -- "
            f"{scripted['rows'][2]['ratio']}x against "
            f"{scripted['rows'][2]['action_ratio']}x, and {scripted['ratio_of_sums']}x "
            f"against {scripted['action_ratio_of_sums']}x in aggregate",
        ),
        practice.Check(
            "FINDING: the aggregate is a ratio of sums, not a mean of ratios",
            all([scripted["ratio_of_sums"] == 1.17,
                 scripted["mean_of_ratios"] == 1.13,
                 result["without_third"]["ratio_of_sums"] == 1.0]),
            f"total_steps/total_gold is {scripted['ratio_of_sums']}x while the mean of "
            f"the per-task ratios is {scripted['mean_of_ratios']}x. The first weights "
            f"tasks by length, so the one task over gold moves the headline more than the "
            f"two that are not -- dropping it gives "
            f"{result['without_third']['ratio_of_sums']}x",
        ),
        practice.Check(
            "FINDING: the metric is optimised by giving up",
            all([null["ratio_of_sums"] == 0.0, null["ok"] == 0,
                 all(row["ratio"] == 0.0 for row in null["rows"]),
                 null["ratio_of_sums"] < scripted["ratio_of_sums"]]),
            f"an agent returning an empty trace scores {null['ratio_of_sums']}x on every "
            f"task -- better than gold -- and succeeds on {null['ok']} of 3. main() "
            "computes efficiency before it looks at success, so any ranking on efficiency "
            "alone sorts the do-nothing agent first",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
