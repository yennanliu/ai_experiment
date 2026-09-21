"""Exercise 1 — the spawner folds nothing back, so the parent learns nothing.

    Add a subagent spawner that batches 20 tasks into groups of 5 parallel
    subagents. Measure orchestrator context size vs one-per-task.

Reading of the exercise: the comparison only means something if the
orchestrator *reads* the results, and `spawn_subagents` returns
`list[AgentRun]` while appending nothing to the parent's session. So the
measurement is made twice: against the shipped spawner, where the parent's
context is unchanged whatever the batch size, and against a folding
orchestrator that appends one line per subagent.

**ANSWER: batching 20 tasks into 5 subagents costs the orchestrator 23
tokens against 83 one-per-task.** Folding one summary line per subagent,
**5** groups of **4** cost **23** tokens where **20** groups of **1** cost
**83** -- a **3.6x** reduction -- while running all **20** tool calls inline
costs the orchestrator **107**.

**FINDING: the shipped spawner folds nothing back.** Through
`spawn_subagents` the parent's `context_tokens` is **0** for both **5**
subagents and **20**, because `run_agent` only appends to the *sub's*
session. The orchestrator's context is bounded because it is empty, and the
`AgentRun` objects the caller receives are the only place the results exist.

**FINDING: the redistribution is not free.** The inline run costs **107**
tokens in one session; **5** subagents cost **145** in total and **20** cost
**280**, because each one re-reads its own prompt and writes its own output.
Batching hides **122** tokens from the orchestrator and adds work everywhere
else -- the saving is in *where* the context sits, not in how much there is.

**FINDING: `context_tokens` is a word count.** It is
`len(text.split())` at **4** call sites, so a **200**-character single line
costs **1** and a five-word tool result costs **5**. Every number above is in that unit, which is the only one the harness
offers.

Structure: `folding()` is the orchestrator the exercise implies;
`spawn_subagents` is the lesson's own.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "17-claude-agent-sdk"
TASKS = 20


def build(ref):
    tools = ref.ToolRegistry()
    tools.register(ref.Tool("read_file", "read a file", ref._read_file_demo))
    return ref.Harness(tools, ref.Hooks(), ref.SessionStore())


def task_calls(index):
    return [("read_file", {"path": f"src/mod{index:02d}.py"})]


def groups(size):
    """`TASKS` tasks split into groups of `size`, as one prompt per group."""
    out = []
    for start in range(0, TASKS, size):
        calls = [call for index in range(start, min(start + size, TASKS))
                 for call in task_calls(index)]
        out.append((f"summarise modules {start} to {start + size - 1}", calls))
    return out


def folding(ref, harness, size):
    """The orchestrator reads one summary line per subagent, which is the point."""
    parent = "orch"
    harness.store.append(parent, ref.Turn("user", "summarise the repository"))
    runs = harness.spawn_subagents(parent, groups(size))
    tokens = len("summarise the repository".split())
    for run in runs:
        harness.store.append(parent, ref.Turn("tool", run.output))
        tokens += len(run.output.split())
    return tokens, runs


def inline(ref, harness):
    calls = [call for index in range(TASKS) for call in task_calls(index)]
    return harness.run_agent("solo", "summarise the repository", calls)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    batched, batched_runs = folding(ref, build(ref), 4)
    per_task, per_task_runs = folding(ref, build(ref), 1)
    shipped = build(ref)
    shipped.spawn_subagents("orch", groups(4))
    shipped_solo = build(ref)
    shipped_solo.spawn_subagents("orch", groups(1))
    source = inspect.getsource(ref.Harness.run_agent)
    return {
        "batched": batched, "per_task": per_task,
        "groups": len(batched_runs), "singles": len(per_task_runs),
        "ratio": round(per_task / batched, 1),
        "inline": inline(ref, build(ref)).context_tokens,
        "shipped_parent": len(shipped.store.load("orch")),
        "shipped_parent_solo": len(shipped_solo.store.load("orch")),
        "sub_total_batched": sum(run.context_tokens for run in batched_runs),
        "sub_total_single": sum(run.context_tokens for run in per_task_runs),
        "hidden": sum(run.context_tokens for run in batched_runs) - batched,
        "token_sites": source.count("len(") ,
        "long_line": len("a" * 200 ),
        "long_tokens": len(("a" * 200).split()),
        "four_words": len("[content of x: 42 lines]".split()),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 23 tokens batched against 83 one-per-task",
            all([result["batched"] == 23, result["per_task"] == 83,
                 result["groups"] == 5, result["singles"] == 20,
                 result["ratio"] == 3.6, result["inline"] == 107]),
            f"folding one summary line per subagent, {result['groups']} groups of 4 cost "
            f"the orchestrator {result['batched']} tokens where {result['singles']} "
            f"groups of 1 cost {result['per_task']} -- {result['ratio']}x -- and running "
            f"all {TASKS} calls inline costs {result['inline']}",
        ),
        practice.Check(
            "FINDING: the shipped spawner folds nothing back",
            all([result["shipped_parent"] == 0, result["shipped_parent_solo"] == 0]),
            f"through spawn_subagents the parent session holds "
            f"{result['shipped_parent']} turns for 5 subagents and "
            f"{result['shipped_parent_solo']} for 20, because run_agent appends only to "
            "the sub's session. The orchestrator's context is bounded because it is "
            "empty, and the returned AgentRun objects are the only copy of the results",
        ),
        practice.Check(
            "FINDING: the redistribution is not free",
            all([result["sub_total_batched"] == 145,
                 result["sub_total_single"] == 280, result["inline"] == 107,
                 result["hidden"] == 122]),
            f"the inline run costs {result['inline']} tokens in one session; five "
            f"subagents cost {result['sub_total_batched']} in total and twenty cost "
            f"{result['sub_total_single']}, because each one re-reads its own prompt and "
            f"writes its own output. Batching hides {result['hidden']} tokens from the "
            "orchestrator and adds work everywhere else",
        ),
        practice.Check(
            "FINDING: context_tokens is a word count",
            all([result["token_sites"] == 4, result["long_tokens"] == 1,
                 result["long_line"] == 200, result["four_words"] == 5]),
            f"run_agent measures context with len(...split()) at "
            f"{result['token_sites']} sites, so a {result['long_line']}-character single "
            f"line costs {result['long_tokens']} and the demo's tool result costs "
            f"{result['four_words']}. Every number above is in that unit",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
