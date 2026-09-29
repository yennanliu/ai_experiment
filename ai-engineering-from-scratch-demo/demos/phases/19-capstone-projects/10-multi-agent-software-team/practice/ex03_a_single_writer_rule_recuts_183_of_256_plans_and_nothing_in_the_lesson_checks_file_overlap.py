"""Exercise 3 — a single-writer rule re-cuts 183 of 256 plans, and nothing in the lesson checks file overlap.

    Replace the merge coordinator with a single-writer constraint (subtasks touch disjoint file sets). Measure the planning burden on the architect.

Reading of the exercise: the lesson's architect already lists files per
subtask. Real subtasks also touch shared files, so each of its four
subtasks is given one extra edit in `src/__init__.py`, `src/config.py`,
`tests/conftest.py`, or none: all 4^4 = 256 plans. Under single-writer the
architect must find every overlap before dispatch and either merge the
overlapping subtasks into one coder's lane, or extract the contested edits
into one extra subtask that runs first (5 lines per edit). Burden is the
plans that need re-cutting, the lanes left, and the coder-stage critical
path, using the lesson's DIFF_READY charge of 3200 + 30 per changed line
from `coder_implement`.

**ANSWER: 183 of 256 plans (71.5%) need re-cutting.** Merging leaves
{1: 3, 2: 54, 3: 126, 4: 73} plans with that many lanes and makes the coder
stage 1.68x slower on average (3.57x worst). Extracting keeps four lanes, but
adds a fifth subtask the architect must specify in all 183 plans,
at 1.46x.

**FINDING: the lesson's plan is disjoint by construction, and nothing
checks it.** Its four stub subtasks give 4 single-writer lanes, so the
constraint costs 0 there. The merge coordinator is one 2,000-token
REVIEW_NEEDED message: with `cache` also editing `src/parser.py`, 200 seeded
`run_team` results are identical to the disjoint plan's.
"""

from __future__ import annotations

import contextlib
import itertools
import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "10-multi-agent-software-team"
SHARED = (None, "src/__init__.py", "src/config.py", "tests/conftest.py")
SHARED_EDIT_LINES = 5  # lines each subtask needs in its shared file


def groups(plan):
    """Single-writer partition: subtasks sharing any file must be one coder's job."""
    out = []
    for sub in plan:
        touching = [g for g in out if any(set(sub.files) & set(o.files) for o in g)]
        merged = [sub] + [s for g in touching for s in g]
        out = [g for g in out if g not in touching] + [merged]
    return out


def plans(ref):
    """The lesson's 4-subtask plan with each subtask also editing one shared file, or none: 4^4 plans."""
    base = ref.architect_plan("issue", random.Random(0))
    for combo in itertools.product(SHARED, repeat=len(base)):
        yield [ref.Subtask(s.name, s.files + ([f] if f else [])) for s, f in zip(base, combo)]


def coder_tokens(ref, sub, seed):
    """The lesson's DIFF_READY charge, 3200 + 30 per changed line, for this subtask."""
    return 3200 + 30 * ref.coder_implement(sub, random.Random(seed))["lines"]


def burden(ref, plan, seed):
    """Critical path of the coder stage under the two ways an architect can honour single-writer."""
    cost = {s.name: coder_tokens(ref, s, seed + i) for i, s in enumerate(plan)}
    gs = groups(plan)
    edits = [f for s in plan for f in s.files if f in SHARED]
    contested = sum(edits.count(f) for f in set(edits) if edits.count(f) > 1)
    # extract: one extra subtask owns every contested shared-file edit and runs before the rest
    extract = (3200 + 30 * SHARED_EDIT_LINES * contested) if contested else 0
    return {"lanes": len(gs), "parallel": max(cost.values()), "contested": contested,
            "merged": max(sum(cost[s.name] for s in g) for g in gs), "extract": extract + max(cost.values())}


@contextlib.contextmanager
def overlapping_architect(ref):
    real = ref.architect_plan

    def plan(issue, rng):
        subs = real(issue, rng)
        subs[1].files.append(subs[0].files[0])  # cache now also edits src/parser.py
        return subs

    ref.architect_plan = plan
    try:
        yield
    finally:
        ref.architect_plan = real


def slowdown(rows, key):
    return round(sum(r[key] for r in rows) / sum(r["parallel"] for r in rows), 3)


def summarise(rows):
    return {
        "plans": len(rows),
        "needs_merge": sum(r["lanes"] < 4 for r in rows),
        "lanes_hist": {k: sum(r["lanes"] == k for r in rows) for k in range(1, 5)},
        "merge_slowdown": slowdown(rows, "merged"), "extract_slowdown": slowdown(rows, "extract"),
        "worst_merge": round(max(r["merged"] / r["parallel"] for r in rows), 2),
        "needs_extra_subtask": sum(r["contested"] > 0 for r in rows),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = [burden(ref, p, i) for i, p in enumerate(plans(ref))]
    clean = [ref.run_team(f"i{s}", rng=random.Random(s)) for s in range(200)]
    with overlapping_architect(ref):
        overlap = [ref.run_team(f"i{s}", rng=random.Random(s)) for s in range(200)]
    return {**summarise(rows), "ref_lanes": len(groups(ref.architect_plan("issue", random.Random(0)))),
            "overlap_identical": overlap == clean}


def verify(result):
    r, h = result, result["lanes_hist"]
    return [
        practice.Check(
            "ANSWER: 183 of 256 plans must be re-cut; merging costs 1.68x the coder stage, extracting 1.46x",
            (r["plans"], r["needs_merge"], r["needs_extra_subtask"], h, r["merge_slowdown"],
             r["extract_slowdown"], r["worst_merge"])
            == (256, 183, 183, {1: 3, 2: 54, 3: 126, 4: 73}, 1.681, 1.456, 3.57),
            f"{r['needs_merge']}/{r['plans']} plans share a file; parallel lanes left {h}; coder-stage critical "
            f"path x{r['merge_slowdown']} merging (worst x{r['worst_merge']}), x{r['extract_slowdown']} with one "
            f"extra shared-edits subtask",
        ),
        practice.Check(
            "FINDING: the lesson's plan is disjoint by construction and nothing checks it",
            (r["ref_lanes"], r["overlap_identical"]) == (4, True),
            f"the stub plan gives {r['ref_lanes']} single-writer lanes; with cache also editing src/parser.py, "
            f"200 seeded run_team results identical to the disjoint plan: {r['overlap_identical']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
