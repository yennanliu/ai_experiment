"""Exercise 5 — the gap is five points on seeing and twenty-five on doing.

    Compare screenshot-only Claude 4.7 to hybrid screenshot + accessibility-tree
    Qwen2.5-VL on 10 web tasks. Which wins on which tasks?

Reading of the exercise: the comparison it names is not one the lesson's table
supports -- every row compares an open model against a frontier one, and the
accessibility tree is never a column -- so the rows are read for what they do
separate, which is grounding from multi-step execution, and the answer is built
on that split.

**ANSWER: the frontier model wins everywhere, and by 5x more on doing than on
seeing.** ScreenSpot-Pro, a pure grounding benchmark, is **85 against 90** -- a
5-point, **1.06x** gap. WebArena, the same models executing multi-step tasks, is
**35 against 60** -- **25** points, **1.71x**. Whatever separates the two, it is
five times more visible once actions are chained.

**FINDING: the table has no accessibility-tree column, so the exercise's
comparison is not in it.** All **5** rows are open-against-frontier, and the
lesson's own §"Screenshot-only vs accessibility-tree" carries no numbers at all.
The measurable question becomes which *axis* the gap sits on, and the answer is
the chaining.

**ANSWER: so the split is by whether the DOM is authoritative.** The tree wins
where the page declares itself -- form fields, links, labelled buttons, ARIA
roles -- because a text label is exact where a click coordinate is approximate.
Screenshots win where the DOM lies: canvas, custom widgets, images used as
buttons, anything rendered rather than marked up, and any state that is visual
only (a spinner, a disabled-looking button that is enabled).

**FINDING: and the benchmark spread says neither is close to solved.**
VisualWebArena is **20 against 27** and AgentVista **15 against 33.5** -- the
widest relative gap at **2.23x**, and a frontier score of 33.5. A 10-task
comparison at these rates separates two systems by about **1.4** tasks at one
standard error, so ten tasks cannot answer the question the exercise asks.

Structure: `LEADERBOARD` transcribes the lesson's five rows, `gap` and `ratio`
score each, and `resolvable` computes what a 10-task sample can distinguish.
"""

from __future__ import annotations

import math

from harness import practice

LEADERBOARD = {
    "ScreenSpot-Pro": (85.0, 90.0, "grounding"),
    "VisualWebArena": (20.0, 27.0, "multi-step"),
    "WebArena": (35.0, 60.0, "multi-step"),
    "AgentVista": (15.0, 33.5, "multi-step"),
    "Ferret-UI mobile": (70.0, 82.0, "grounding"),
}
TASKS = 10
TREE_WINS = ("form fields", "links", "labelled buttons", "ARIA roles")
PIXEL_WINS = ("canvas", "custom widgets", "images as buttons", "visual-only state")


def gap(row):
    return round(row[1] - row[0], 1)


def ratio(row):
    return round(row[1] / row[0], 2)


def resolvable(rate, tasks=TASKS):
    """One standard error of a binomial proportion, in tasks."""
    return round(math.sqrt(rate * (1 - rate) / tasks) * tasks, 1)


def solve():
    gaps = {name: gap(row) for name, row in LEADERBOARD.items()}
    ratios = {name: ratio(row) for name, row in LEADERBOARD.items()}
    grounding = [name for name, row in LEADERBOARD.items() if row[2] == "grounding"]
    multistep = [name for name, row in LEADERBOARD.items() if row[2] == "multi-step"]
    return {
        "gaps": gaps, "ratios": ratios,
        "grounding": grounding, "multistep": multistep,
        "grounding_gap": gaps["ScreenSpot-Pro"],
        "multistep_gap": gaps["WebArena"],
        "gap_ratio": round(gaps["WebArena"] / gaps["ScreenSpot-Pro"], 1),
        "widest_ratio": max(ratios, key=ratios.get),
        "rows": len(LEADERBOARD),
        "tree_columns": 0,
        "tree_wins": TREE_WINS, "pixel_wins": PIXEL_WINS,
        "frontier_min": min(row[1] for row in LEADERBOARD.values()),
        "resolvable_tasks": resolvable(0.275),
        "tasks": TASKS,
    }


def verify(result):
    gaps, ratios = result["gaps"], result["ratios"]
    return [
        practice.Check(
            "ANSWER: 5 points on seeing and 25 on doing",
            all([result["grounding_gap"] == 5.0, result["multistep_gap"] == 25.0,
                 result["gap_ratio"] == 5.0,
                 ratios["ScreenSpot-Pro"] == 1.06, ratios["WebArena"] == 1.71]),
            f"ScreenSpot-Pro is 85 against 90 -- {result['grounding_gap']} points, "
            f"{ratios['ScreenSpot-Pro']}x -- and WebArena is 35 against 60, "
            f"{result['multistep_gap']} points and {ratios['WebArena']}x. The same models: "
            f"whatever separates them is {result['gap_ratio']}x more visible once actions are "
            "chained",
        ),
        practice.Check(
            "FINDING: the table has no accessibility-tree column",
            all([result["rows"] == 5, result["tree_columns"] == 0,
                 len(result["grounding"]) == 2, len(result["multistep"]) == 3]),
            f"all {result['rows']} rows compare an open model against a frontier one and "
            f"{result['tree_columns']} carry an accessibility-tree column, so the comparison "
            f"the exercise names is not in the evidence. What the rows do separate is "
            f"{result['grounding']} from {result['multistep']}",
        ),
        practice.Check(
            "ANSWER: the split is whether the DOM is authoritative",
            all([len(result["tree_wins"]) == 4, len(result["pixel_wins"]) == 4,
                 "canvas" in result["pixel_wins"]]),
            f"the tree wins on {list(result['tree_wins'])} -- a text label is exact where a "
            f"click coordinate is approximate -- and screenshots win on "
            f"{list(result['pixel_wins'])}, where the page is rendered rather than marked up "
            "and the DOM either omits the element or misdescribes its state",
        ),
        practice.Check(
            "FINDING: and ten tasks cannot answer the question",
            all([ratios["AgentVista"] == 2.23,
                 result["widest_ratio"] == "AgentVista",
                 result["frontier_min"] == 27.0,
                 result["resolvable_tasks"] == 1.4]),
            f"AgentVista is 15 against 33.5, the widest relative gap at "
            f"{ratios['AgentVista']}x, and the lowest frontier score in the table is "
            f"{result['frontier_min']}. At that rate one standard error over "
            f"{result['tasks']} tasks is {result['resolvable_tasks']} tasks, so a 10-task "
            "comparison separates two systems only if they differ by several outcomes",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
