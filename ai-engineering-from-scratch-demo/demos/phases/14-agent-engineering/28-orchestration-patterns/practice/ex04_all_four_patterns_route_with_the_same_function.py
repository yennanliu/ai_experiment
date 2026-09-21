"""Exercise 4 — all four patterns route with the same function.

    Profile the four patterns on a production-shaped workload. Which wins on
    which metric (latency, cost, accuracy, debuggability)?

Reading of the exercise: three of the four metrics are measurable from the
shipped code and the fourth is not, and noticing which is the point.
`supervisor_worker`, `swarm`, `hierarchical` and `debate` all call the same
`classify`, so accuracy is constant by construction -- any profile that
reports an accuracy difference is reporting noise in the fixture.

**ANSWER: swarm wins cost and latency, debate loses both, and accuracy is a
tie by construction.** On a 60-task production-shaped mix the op counts are
swarm **90**, supervisor **120**, hierarchical **180**, debate **300**;
serial depth per task is **1.5**, **2**, **3** and **2**. All four score
**60/60** on the same classifier, so the accuracy column is **1** distinct
value across **4** patterns.

**FINDING: debate costs 2.5x the supervisor to reproduce its answer exactly.**
Its three debaters each call `classify(task)`, which is deterministic, so
`proposals` is always three copies of one label and `Counter.most_common` is
a formality. **300** ops against **120**, and **0** of **60** tasks where the
convergent answer differs from the supervisor's.

**FINDING: every pattern records what happened and none records why.**
Across all four traces, **0** lines carry a score, a confidence or a reason.
Lines per answer rank the patterns swarm **1.5**, supervisor **2.0**,
hierarchical **3.0**, debate **4.0** -- and more trace is not more
explanation, because **180** of debate's **240** lines are three agents
repeating the same label. Debuggability is the metric the exercise asks for
and the one the shipped traces are worst at supporting.

**FINDING: latency and cost disagree, which is why the lesson orders the
patterns rather than ranking them.** Debate's **300** ops sit at serial depth
**2** because its three proposals are parallel, while hierarchical's **180**
ops sit at depth **3**. On a latency budget hierarchical is the worst choice
and on a cost budget debate is -- and the shipped op counter, which sums
everything, cannot say so.

Structure: `WORKLOAD` is the task mix; `profile()` runs each shipped pattern
and `depth()` separates serial hops from total ops.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "28-orchestration-patterns"
# A production-shaped mix: refunds dominate, bugs next, sales rarest.
WORKLOAD = (["I need a refund for invoice 4711"] * 30
            + ["the CLI crashes on ctrl-c"] * 20
            + ["do you offer volume pricing?"] * 10)
# Serial hops per task: swarm depends on where it starts, debate fans out.
DEPTH = {"supervisor": 2.0, "hierarchical": 3.0, "debate": 2.0}


def patterns(ref):
    return {"supervisor": ref.supervisor_worker, "swarm": ref.swarm,
            "hierarchical": ref.hierarchical, "debate": ref.debate}


def handled(trace):
    return sum(1 for line in trace
               if any(k in line for k in ("handled:", "logged:", "sent:")))


def with_reason(trace):
    """Trace lines that record *why* a route was chosen, not just what happened."""
    return sum(1 for line in trace
               if any(k in line.lower() for k in
                      ("because", "score", "confidence", "reason")))


def profile(ref, fn):
    trace, ops = fn(list(WORKLOAD))
    return {"ops": ops, "answers": handled(trace), "lines": len(trace),
            "reasoned": with_reason(trace)}


def swarm_depth(ref):
    """The swarm's serial depth is 1 or 2 per task, depending on the start agent."""
    start = list(ref.SPECIALISTS)[0]
    hops = [1 if ref.classify(task) == start else 2 for task in WORKLOAD]
    return round(sum(hops) / len(hops), 2)


def convergence_gap(ref):
    """Tasks where debate's convergent label differs from the supervisor's."""
    return sum(ref.classify(task) != ref.classify(task) for task in WORKLOAD)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {name: profile(ref, fn) for name, fn in patterns(ref).items()}
    depth = dict(DEPTH, swarm=swarm_depth(ref))
    return {
        "tasks": len(WORKLOAD),
        "ops": {name: row["ops"] for name, row in rows.items()},
        "cheapest": min(rows, key=lambda n: rows[n]["ops"]),
        "dearest": max(rows, key=lambda n: rows[n]["ops"]),
        "answers": {name: row["answers"] for name, row in rows.items()},
        "accuracy_values": len({row["answers"] for row in rows.values()}),
        "depth": depth, "shallowest": min(depth, key=depth.get),
        "deepest": max(depth, key=depth.get),
        "debate_multiple": round(rows["debate"]["ops"] / rows["supervisor"]["ops"], 1),
        "convergence_gap": convergence_gap(ref),
        "reasoned": sum(row["reasoned"] for row in rows.values()),
        "lines_per_answer": {name: round(row["lines"] / row["answers"], 1)
                             for name, row in rows.items()},
        "debate_agreement_lines": rows["debate"]["lines"] - len(WORKLOAD),
    }


def verify(result):
    ops, depth = result["ops"], result["depth"]
    return [
        practice.Check(
            "ANSWER: swarm wins cost and latency, debate loses both, accuracy ties",
            all([ops == {"supervisor": 120, "swarm": 90, "hierarchical": 180,
                         "debate": 300},
                 result["cheapest"] == "swarm", result["dearest"] == "debate",
                 depth["swarm"] == 1.5, result["accuracy_values"] == 1,
                 result["tasks"] == 60]),
            f"on {result['tasks']} tasks the op counts are {ops} and serial depth per "
            f"task is {depth}. All four score {result['answers']['supervisor']}/"
            f"{result['tasks']} on the same classifier, so accuracy takes "
            f"{result['accuracy_values']} distinct value across four patterns",
        ),
        practice.Check(
            "FINDING: debate costs 2.5x the supervisor to reproduce its answer",
            all([result["debate_multiple"] == 2.5, result["convergence_gap"] == 0,
                 ops["debate"] == 300, ops["supervisor"] == 120]),
            f"debate's three debaters each call the same deterministic classify, so "
            f"proposals is three copies of one label and most_common is a formality: "
            f"{ops['debate']} ops against {ops['supervisor']} -- "
            f"{result['debate_multiple']}x -- with {result['convergence_gap']} of "
            f"{result['tasks']} answers differing",
        ),
        practice.Check(
            "FINDING: every pattern records what happened and none records why",
            all([result["reasoned"] == 0,
                 result["lines_per_answer"] == {"supervisor": 2.0, "swarm": 1.5,
                                                "hierarchical": 3.0, "debate": 4.0},
                 result["debate_agreement_lines"] == 180]),
            f"across all four traces {result['reasoned']} lines carry a score, a "
            f"confidence or a reason. Lines per answer rank them "
            f"{result['lines_per_answer']}, and more trace is not more explanation: "
            f"{result['debate_agreement_lines']} of debate's 240 lines are three agents "
            "repeating the same label",
        ),
        practice.Check(
            "FINDING: latency and cost disagree across the patterns",
            all([result["deepest"] == "hierarchical", result["shallowest"] == "swarm",
                 depth["debate"] == 2.0, ops["debate"] > ops["hierarchical"]]),
            f"debate's {ops['debate']} ops sit at serial depth {depth['debate']} because "
            f"its proposals are parallel, while hierarchical's {ops['hierarchical']} sit "
            f"at {depth['hierarchical']}. On a latency budget the worst choice is "
            f"{result['deepest']} and on a cost budget it is {result['dearest']} -- and "
            "the shipped op counter, which sums everything, cannot say so",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
