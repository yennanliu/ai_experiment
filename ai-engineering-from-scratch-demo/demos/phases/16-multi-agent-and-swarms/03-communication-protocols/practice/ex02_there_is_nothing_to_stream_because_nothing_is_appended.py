"""Exercise 2 — there is nothing to stream because nothing is appended.

    **Streaming audit trail.** Modify the `AuditableRunner` to support streaming
    mode. Instead of waiting for the full result, yield `AuditEntry` updates
    in real-time as trajectory entries are added. Use an async generator that
    produces audit snapshots.

Reading of the exercise: write the generator, then count what it can possibly
yield -- because "as trajectory entries are added" describes something the
shipped runner never does. It assigns the whole trajectory once, after the
handler has already finished.

**ANSWER: streaming the runner is not enough; the handler contract has to
change too.** Against the shipped contract -- `handler(input)` returning a
promise of `{output, trajectory}` -- a streaming runner can yield **2**
snapshots for a run of any length, because the only two moments it learns
anything are before the await and after it. Re-typed as an async generator
that yields trajectory entries, the same **6**-step run yields **8**
snapshots, one per entry plus the two boundaries. The number of snapshots
stops being a constant and starts tracking the work.

**FINDING: the trajectory is assigned, not appended.** `entry.trajectory` is
assigned in **1** place, from `result.trajectory`, after `await handler(input)`
has returned, and pushed in **1** other -- the error path, which records a
single synthetic entry. In a successful run the field goes from empty to
complete in one statement. There is no intermediate state to observe, which
is why the exercise cannot be done inside `AuditableRunner` alone.

**FINDING: one of the five declared statuses is unreachable.** `AuditEntry`
declares `"created" | "in-progress" | "completed" | "failed" | "awaiting"`.
The string `"awaiting"` appears **1** time in the whole module -- in that
union -- and **0** code paths assign it. A streaming consumer written against
the type has a branch that can never be taken.

**FINDING: the two failure paths disagree about when the run ended.**
`entry.status = "failed"` appears **2** times: the missing-handler branch and
the `catch`. `entry.completedAt` is set **2** times: the success path and the
`catch`. So a run that fails because no handler is registered has no
`completedAt`, and any consumer computing a duration gets `undefined` for one
of the two ways to fail and a number for the other.

Structure: `shipped()` streams the lesson's promise-shaped handler;
`streamed()` streams a generator-shaped one; both count their snapshots.
"""

from __future__ import annotations

import asyncio
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "03-communication-protocols"
STEPS = ("read card", "resolve did", "verify sig", "call tool", "check output", "summarise")


def snapshot(entry):
    """One audit snapshot: the fields a consumer would render."""
    return {"status": entry["status"], "trajectory": len(entry["trajectory"])}


async def promise_handler(_input):
    """The shipped contract: one await, everything at the end."""
    trajectory = []
    for step in STEPS:
        await asyncio.sleep(0)
        trajectory.append(step)
    return {"output": ["done"], "trajectory": trajectory}


async def generator_handler(_input):
    """The re-typed contract: one yield per trajectory entry."""
    for step in STEPS:
        await asyncio.sleep(0)
        yield step


async def shipped(handler):
    """Streaming a promise-shaped handler: the runner learns nothing in between."""
    entry = {"status": "created", "trajectory": [], "output": []}
    snapshots = [snapshot(entry)]
    entry["status"] = "in-progress"
    result = await handler(None)
    entry["trajectory"], entry["output"] = result["trajectory"], result["output"]
    entry["status"] = "completed"
    snapshots.append(snapshot(entry))
    return snapshots


async def streamed(handler):
    """Streaming a generator-shaped handler: one snapshot per entry appended."""
    entry = {"status": "created", "trajectory": [], "output": []}
    snapshots = [snapshot(entry)]
    entry["status"] = "in-progress"
    async for step in handler(None):
        entry["trajectory"].append(step)
        snapshots.append(snapshot(entry))
    entry["status"] = "completed"
    snapshots.append(snapshot(entry))
    return snapshots


def statuses(src, run):
    """The declared AuditEntry status union, and how often `run` produces each."""
    start = src.index("type AuditEntry = {")
    union = re.search(r'status: ([^;]+);', src[start:src.index("};", start)]).group(1)
    declared = re.findall(r'"([a-z-]+)"', union)
    return declared, {name: run.count(f'status = "{name}"') + run.count(f'status: "{name}"')
                      for name in declared}


def solve():
    src = (parity.lesson_dir(PHASE, LESSON) / "code" / "main.ts").read_text("utf-8")
    start = src.index("  async run(")
    run = src[start:src.index("  getFullAuditLog", start)]
    declared, assigned = statuses(src, run)
    flat = asyncio.run(shipped(promise_handler))
    live = asyncio.run(streamed(generator_handler))
    return {
        "steps": len(STEPS), "flat": len(flat), "live": len(live),
        "flat_lengths": sorted({s["trajectory"] for s in flat}),
        "live_lengths": [s["trajectory"] for s in live],
        "assignments": run.count("entry.trajectory ="),
        "pushes": run.count("entry.trajectory.push"),
        "after_await": run.index("entry.trajectory =") > run.index("await handler(input)"),
        "declared": declared,
        "unreachable": sorted(name for name, n in assigned.items() if not n),
        "awaiting_mentions": src.count('"awaiting"'),
        "failed_paths": run.count('entry.status = "failed"'),
        "completed_at": run.count("entry.completedAt"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: streaming the runner is not enough; the handler contract must change",
            all([result["flat"] == 2, result["live"] == result["steps"] + 2,
                 result["flat_lengths"] == [0, result["steps"]],
                 result["live_lengths"] == list(range(result["steps"] + 1))
                 + [result["steps"]]]),
            f"against the shipped promise-shaped handler a streaming runner yields "
            f"{result['flat']} snapshots for a {result['steps']}-step run, its trajectory "
            f"length jumping {result['flat_lengths']}; re-typed as an async generator the "
            f"same run yields {result['live']}, one per entry plus two boundaries",
        ),
        practice.Check(
            "FINDING: the trajectory is assigned, not appended",
            all([result["assignments"] == 1, result["pushes"] == 1,
                 result["after_await"]]),
            f"entry.trajectory is assigned in {result['assignments']} place, from "
            f"result.trajectory after await handler(input) has returned, and pushed in "
            f"{result['pushes']} other -- the error path; in a successful run the field "
            "goes from empty to complete in one statement",
        ),
        practice.Check(
            "FINDING: one of the five declared statuses is unreachable",
            all([len(result["declared"]) == 5, result["unreachable"] == ["awaiting"],
                 result["awaiting_mentions"] == 1]),
            f"AuditEntry declares {len(result['declared'])} statuses "
            f"({', '.join(result['declared'])}); \"awaiting\" appears "
            f"{result['awaiting_mentions']} time in the module, in that union, and is "
            "assigned by no code path",
        ),
        practice.Check(
            "FINDING: the two failure paths disagree about when the run ended",
            all([result["failed_paths"] == 2, result["completed_at"] == 2]),
            f"entry.status = \"failed\" appears {result['failed_paths']} times -- the "
            f"missing-handler branch and the catch -- while entry.completedAt is set "
            f"{result['completed_at']} times, on success and in the catch, so a run that "
            "fails for want of a handler has no end time",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
