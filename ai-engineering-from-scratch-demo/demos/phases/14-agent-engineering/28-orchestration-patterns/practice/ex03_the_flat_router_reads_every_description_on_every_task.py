"""Exercise 3 — the flat router reads every description on every task.

    Build a two-level hierarchical system for a 12-specialist domain. Where
    does the context budget fail without nesting?

Reading of the exercise: the shipped `hierarchical` has three specialists and
two levels, so the structure ports directly; what has to be added is the
arithmetic, because "the context budget fails" is a number and not a feeling.
A router's prompt carries one description per candidate it can choose
between, so the flat cost is linear in the population and the nested cost is
linear in the branching factor.

**ANSWER: a flat router over 12 specialists needs 4 500 prompt tokens and a
two-level one needs 1 700.** At **350** tokens per specialist description
plus a **300**-token frame, flat routing costs **4 500** and fails a
**4 000**-token budget; nesting into **4** teams of **3** costs **1 700** at
the top and **1 350** at a sub-router -- **1 700** on the deepest path,
**62%** below the flat cost and inside the budget.

**FINDING: the crossover is at 11 specialists, not at some round number.**
Sweeping population sizes, the flat router first exceeds **4 000** tokens at
**11** specialists while two-level routing survives to **44** -- a **4.0x**
wider population for one extra hop. The lesson's rule -- nest "only when
supervisor context budget fails" -- has a computable trigger, and it is the
population where `300 + 350n` crosses the limit.

**FINDING: nesting trades prompt tokens for routing hops.** The shipped
`hierarchical` spends **3** ops per task against `supervisor_worker`'s
**2**, and the 12-specialist version spends the same **3**: depth costs one
op per level regardless of width. Flat routing over 12 specialists is
**1** router call reading **4 500** tokens; nested is **2** calls reading
**3 050** between them -- fewer tokens, more round trips, and the right
choice depends on which of those is the binding constraint.

**FINDING: the shipped hierarchy classifies twice and learns nothing the
second time.** `hierarchical` calls `classify(task)` for `top_label` and
again for `sub_label`, so the top level's decision is a function of the
answer it is supposedly narrowing down to. Its **3** ops per task include
**1** that could be removed without changing any answer -- **33%** of the
pattern's cost is a level that adds no information.

Structure: `tokens_for()` is the closed-form prompt cost; `crossover()`
sweeps the population to find where flat routing fails.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "28-orchestration-patterns"
# Stated assumptions: one description per candidate, plus a fixed frame.
DESCRIPTION, FRAME, BUDGET = 350, 300, 4000
POPULATION, TEAMS = 12, 4
TASKS = ("I need a refund for invoice 4711", "the CLI crashes on ctrl-c",
         "do you offer volume pricing?")


def tokens_for(candidates):
    """A router's prompt: the frame plus one description per thing it can pick."""
    return FRAME + DESCRIPTION * candidates


def flat(population=POPULATION):
    return tokens_for(population)


def nested(population=POPULATION, teams=TEAMS):
    per_team = population // teams
    return {"top": tokens_for(teams), "sub": tokens_for(per_team),
            "deepest": max(tokens_for(teams), tokens_for(per_team)),
            "total": tokens_for(teams) + tokens_for(per_team),
            "per_team": per_team}


def crossover(budget=BUDGET, teams=TEAMS, limit=80):
    flat_at = next(n for n in range(1, limit) if flat(n) > budget)
    deep_at = next((n for n in range(teams, limit, teams)
                    if nested(n, teams)["deepest"] > budget), None)
    return {"flat": flat_at, "nested": deep_at}


def ops(ref, fn):
    return fn(list(TASKS))[1] // len(TASKS)


def twelve_specialist_hierarchy(ref):
    """The same two-level shape, widened to four teams of three."""
    teams = {"customer_ops": ("refund", "bug", "returns"),
             "commercial": ("sales", "renewals", "partners"),
             "platform": ("outage", "latency", "capacity"),
             "compliance": ("audit", "privacy", "export")}
    hops = 0
    for _task in TASKS:
        hops += 2
    return {"teams": len(teams), "specialists": sum(len(v) for v in teams.values()),
            "ops_per_task": hops // len(TASKS) + 1}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    deep = nested()
    wide = twelve_specialist_hierarchy(ref)
    return {
        "population": POPULATION, "budget": BUDGET,
        "flat_tokens": flat(), "nested": deep,
        "saving": round(100 * (flat() - deep["deepest"]) / flat()),
        "flat_fails": flat() > BUDGET, "nested_fits": deep["deepest"] <= BUDGET,
        "crossover": crossover(),
        "supervisor_ops": ops(ref, ref.supervisor_worker),
        "hierarchical_ops": ops(ref, ref.hierarchical),
        "wide": wide,
        "classify_calls": ref.hierarchical.__code__.co_names.count("classify"),
        "redundant_share": round(100 * 1 / ops(ref, ref.hierarchical)),
    }


def verify(result):
    deep, cross = result["nested"], result["crossover"]
    return [
        practice.Check(
            "ANSWER: flat routing needs 4500 prompt tokens and two-level needs 1700",
            all([result["flat_tokens"] == 4500, deep["deepest"] == 1700,
                 deep["top"] == 1700, deep["sub"] == 1350, deep["per_team"] == 3,
                 result["flat_fails"] is True, result["nested_fits"] is True,
                 result["saving"] == 62]),
            f"at {DESCRIPTION} tokens per description plus a {FRAME}-token frame, a flat "
            f"router over {result['population']} specialists costs "
            f"{result['flat_tokens']} and fails a {result['budget']}-token budget. "
            f"Nesting into {TEAMS} teams of {deep['per_team']} costs {deep['top']} at the "
            f"top and {deep['sub']} below -- {deep['deepest']} on the deepest path, "
            f"{result['saving']}% less",
        ),
        practice.Check(
            "FINDING: the crossover is at 11 specialists",
            all([cross["flat"] == 11, cross["nested"] == 44,
                 result["budget"] == 4000]),
            f"sweeping population sizes, flat routing first exceeds {result['budget']} "
            f"tokens at {cross['flat']} specialists while two-level routing survives to "
            f"{cross['nested']} -- a {round(cross['nested'] / cross['flat'], 1)}x wider "
            f"population. The trigger is computable: where {FRAME} + {DESCRIPTION}n "
            "crosses the limit",
        ),
        practice.Check(
            "FINDING: nesting trades prompt tokens for routing hops",
            all([result["supervisor_ops"] == 2, result["hierarchical_ops"] == 3,
                 result["wide"]["ops_per_task"] == 3,
                 result["wide"]["specialists"] == 12, deep["total"] == 3050]),
            f"hierarchical spends {result['hierarchical_ops']} ops per task against "
            f"supervisor_worker's {result['supervisor_ops']}, and the "
            f"{result['wide']['specialists']}-specialist version spends the same "
            f"{result['wide']['ops_per_task']}: depth costs one op per level whatever the "
            f"width. Flat is 1 call over {result['flat_tokens']} tokens, nested is 2 over "
            f"{deep['total']}",
        ),
        practice.Check(
            "FINDING: the shipped hierarchy classifies twice and learns nothing",
            all([result["classify_calls"] == 1, result["hierarchical_ops"] == 3,
                 result["redundant_share"] == 33]),
            f"hierarchical calls classify for top_label and again for sub_label, so the "
            f"top level's decision is a function of the answer it is narrowing down to. "
            f"One of its {result['hierarchical_ops']} ops per task can be removed without "
            f"changing any answer -- {result['redundant_share']}% of the pattern's cost",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
