"""Exercise 3 — immutability makes the merge possible, not correct.

    Implement parallel edges: two nodes run concurrently, merge by a custom
    reducer. What does immutable state buy here?

Reading of the exercise: `StateGraph._next` returns `str | None` -- one
successor -- so a fan-out cannot be written down in the shipped graph at all.
The parallel step is therefore added beside it: both nodes are handed the
same state, both return update dicts, and a per-key reducer decides the
merge. What immutability buys is then measurable by running the branches in
both orders.

**ANSWER: two branches, a reducer per contested key, and an order-independent
result.** Run in either order the reduced state is identical -- step **3**
and owner `oncall+triage` both ways -- because each node reads the same base
state and returns an update rather than editing anything. That is what
immutability buys: the merge is a function of the two updates, not of when
they ran. It holds only while the reducer table is *total* over the contested
keys; a key without one falls back to last-write-wins.

**FINDING: it buys a well-defined merge and not a correct one.** The runner's
own rule is `{**state, **update}`, last write wins, and under it the two
orders disagree on `owner` -- **1** key that both branches write with
different values. Immutability makes the disagreement visible
instead of silent; the reducer is what resolves it.

**FINDING: `step` is the key that breaks.** Both branches compute
`state.get("step", 0) + 1` from the same base, so both return **2** and a
last-write-wins merge records **2** for two nodes of work. A reducer that
adds the *deltas* records **3**. Any counter written by more than one branch
needs a reducer, and every node in this lesson writes `step`.

**FINDING: a mutating node loses the property entirely.** A node that edits
the state dict in place instead of returning an update leaves the base state
changed, and whichever branch runs after it reads the edit: the two orders
disagree on `severity` although neither branch ever wrote a conflicting
value for it. The shipped nodes all return updates; nothing enforces it.

Structure: `fan_out()` runs the branches against a shared base and merges
with a reducer table; the branch bodies come from the lesson.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "13-langgraph-stateful-graphs"
BASE = {"input": "the CLI crashes on ctrl-c", "step": 1, "route": "bug"}


def add_deltas(base, values):
    return base + sum(value - base for value in values)


def last_wins(_base, values):
    return values[-1]


def join(_base, values):
    return "+".join(sorted(values))


REDUCERS = {"step": add_deltas, "owner": join}


def branch_ticket(state):
    return {"ticket": f"BUG-{state['input'][:6]}", "owner": "triage",
            "step": state.get("step", 0) + 1}


def branch_severity(state):
    return {"severity": "high" if state.get("notes") else "normal",
            "owner": "oncall", "step": state.get("step", 0) + 1}


def mutating_branch(state):
    """Edits the state it was handed instead of returning an update."""
    state["notes"] = state.get("notes", "") + "x"
    return {"step": state.get("step", 0) + 1}


def fan_out(base, nodes, reducers=REDUCERS):
    """Both nodes see the same state; a reducer decides every contested key."""
    updates = [node(dict(base)) for node in nodes]
    merged = dict(base)
    for key in {key for update in updates for key in update}:
        values = [update[key] for update in updates if key in update]
        rule = reducers.get(key, last_wins)
        merged[key] = rule(base.get(key, 0), values) if rule is add_deltas else rule(
            base.get(key), values)
    return merged


def naive(base, nodes):
    state = dict(base)
    for node in nodes:
        state = {**state, **node(dict(base))}
    return state


def naive_shared(base, nodes):
    """The same run without the defensive copy, so a mutating node is visible."""
    state = base
    for node in nodes:
        update = node(state)
        state = {**state, **update}
    return state


def disagreements(left, right):
    return sorted(key for key in set(left) | set(right)
                  if left.get(key) != right.get(key))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    order = (branch_ticket, branch_severity)
    reduced = (fan_out(BASE, order), fan_out(BASE, order[::-1]))
    naive_pair = (naive(BASE, order), naive(BASE, order[::-1]))
    mutable = dict(BASE)
    mutating_pair = (naive_shared(mutable, (mutating_branch, branch_severity)),
                     naive_shared(dict(BASE), (branch_severity, mutating_branch)))
    return {
        "reduced_agree": reduced[0] == reduced[1], "reduced_step": reduced[0]["step"],
        "reduced_owner": reduced[0]["owner"],
        "naive_step": naive_pair[0]["step"],
        "naive_disagree": disagreements(*naive_pair),
        "mutating_disagree": disagreements(*mutating_pair),
        "base_touched": "notes" in mutable,
        "next_returns": str(inspect.signature(ref.StateGraph._next).return_annotation),
        "branch_steps": [branch_ticket(dict(BASE))["step"],
                         branch_severity(dict(BASE))["step"]],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the reduced merge is identical in either order",
            all([result["reduced_agree"] is True, result["reduced_step"] == 3,
                 result["reduced_owner"] == "oncall+triage",
                 result["branch_steps"] == [2, 2]]),
            f"run in either order the reduced state is identical "
            f"({result['reduced_agree']}): step {result['reduced_step']} and owner "
            f"{result['reduced_owner']!r}. Each node reads the same base and returns an "
            "update, so the merge is a function of the two updates and not of when they "
            "ran -- provided every contested key has a reducer",
        ),
        practice.Check(
            "FINDING: it buys a well-defined merge and not a correct one",
            all([result["naive_disagree"] == ["owner"],
                 len(result["naive_disagree"]) == 1]),
            f"under the runner's own rule -- {{**state, **update}}, last write wins -- "
            f"the two orders disagree on {result['naive_disagree']}. Immutability makes "
            "the disagreement visible instead of silent; the reducer is what resolves it",
        ),
        practice.Check(
            "FINDING: step is the key that breaks",
            all([result["branch_steps"] == [2, 2], result["naive_step"] == 2,
                 result["reduced_step"] == 3]),
            f"both branches compute step + 1 from the same base and return "
            f"{result['branch_steps']}, so last-write-wins records "
            f"{result['naive_step']} for two nodes of work while an add-the-deltas "
            "reducer records 3. Every node in this lesson writes step",
        ),
        practice.Check(
            "FINDING: a mutating node loses the property entirely",
            all([result["mutating_disagree"] == ["severity"],
                 result["base_touched"] is True,
                 str(result["next_returns"]).endswith("None")]),
            f"a node that edits the state in place makes the two orders differ on "
            f"{result['mutating_disagree']} and leaves the base changed "
            f"({result['base_touched']}). The shipped nodes all return updates and "
            f"nothing enforces it -- and _next is typed {result['next_returns']}, so a "
            "fan-out cannot be written in the graph at all",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
