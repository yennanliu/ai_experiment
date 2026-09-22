"""Exercise 2 — the cache is shipped, and its key is missing the state.

    Add a LLM-method cache to ChatHTN: when the LLM decomposes task `T` in
    state pattern `P`, store the result. Re-check the method library first on
    the next call.

Reading of the exercise: `HTNPlanner` already has `cached_methods`, already
writes to it after every LLM fallback, and already consults the method
library before it. So both halves of the request are shipped -- except that
the key is `task` and the exercise says `task, state pattern`. The exercise
is therefore about the missing half of the key, and what the shipped key does
when two states need different decompositions.

**ANSWER: a `(task, frozenset(state))` key.** Two states that need different
decompositions of the same task get **2** cache entries, **2** LLM calls, a
correct plan in each, and **0** further calls on a repeat.

**FINDING: the shipped key reuses one state's answer in another.** Keyed on
`task` alone, the decomposition learned with `has_migration` present is
returned in a state without it, `_expand` rejects it, and `plan` returns
`None` -- with **0** new LLM calls. A stale cache entry surfaces as an
unexplained planning failure rather than as a cache miss.

**FINDING: the whole state is too specific a key.** Adding one irrelevant
fact to a state that is already cached misses, costing **1** extra LLM call
for a decomposition already known. The "pattern" in the exercise has to be a
*projection* of the state -- here the facts the task's operators actually
mention -- and projecting takes the count back to **2**.

**FINDING: re-checking the library first is already the shipped order.**
`plan` reads `methods` first, `cached_methods` second, the LLM third. Adding
a method for a task that is already cached makes the method win on the next
call: the plan changes to the method's subtasks with the cache untouched.

Structure: `KeyedPlanner` wraps the lesson's own `HTNPlanner`, changing only
what the cache is keyed on.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "11-planning-htn-and-evolutionary"
TASK = "ship_feature"
WITH_MIGRATION = {"logged_in", "has_migration"}
PLAIN = {"logged_in"}
RELEVANT = frozenset({"has_migration"})


def world(ref):
    rows = (("open_editor", ("logged_in",), ("editor_open",)),
            ("run_migration", ("has_migration", "editor_open"), ("migrated",)),
            ("write_tests", ("editor_open",), ("tests_written",)),
            ("open_pr", ("tests_written",), ("pr_open",)))
    return {name: ref.Operator(name, pre, add) for name, pre, add in rows}


class Suggesting:
    """An LLM whose decomposition depends on the state, as the exercise implies."""

    def __init__(self):
        self.calls = []

    def decompose(self, task, state):
        self.calls.append(task)
        if "has_migration" in state:
            return ("open_editor", "run_migration", "write_tests", "open_pr")
        return ("open_editor", "write_tests", "open_pr")


class KeyedPlanner:
    """The lesson's planner with the state put back into the cache key."""

    def __init__(self, ref, operators, llm, project=None):
        self.ref, self.operators, self.llm = ref, operators, llm
        self.project, self.cache = project or (lambda s: frozenset(s)), {}

    def plan(self, task, state):
        key = (task, self.project(state))
        inner = self.ref.HTNPlanner(self.operators, {}, self.llm,
                                    cached_methods=dict(self.cache.get(key, {})))
        result = inner.plan(task, state)
        if task in inner.cached_methods:
            self.cache[key] = {task: inner.cached_methods[task]}
        return result


def shipped_reuse(ref, operators):
    llm = Suggesting()
    planner = ref.HTNPlanner(operators, {}, llm)
    first = planner.plan(TASK, WITH_MIGRATION)
    before = len(llm.calls)
    second = planner.plan(TASK, PLAIN)
    return {"first": first, "second": second,
            "new_calls": len(llm.calls) - before, "cached": dict(planner.cached_methods)}


def library_first(ref, operators):
    llm = Suggesting()
    planner = ref.HTNPlanner(operators, {}, llm)
    planner.plan(TASK, WITH_MIGRATION)
    planner.methods[TASK] = [ref.Method("m", TASK, ("logged_in",),
                                        ("open_editor", "write_tests"))]
    before = len(llm.calls)
    return {"plan": planner.plan(TASK, WITH_MIGRATION),
            "new_calls": len(llm.calls) - before,
            "cache_intact": TASK in planner.cached_methods}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    operators = world(ref)
    llm = Suggesting()
    keyed = KeyedPlanner(ref, operators, llm)
    plans = [keyed.plan(TASK, WITH_MIGRATION), keyed.plan(TASK, PLAIN),
             keyed.plan(TASK, WITH_MIGRATION)]
    wide = Suggesting()
    whole = KeyedPlanner(ref, operators, wide)
    whole.plan(TASK, WITH_MIGRATION)
    whole.plan(TASK, WITH_MIGRATION | {"reviewer_assigned"})
    narrow = Suggesting()
    projected = KeyedPlanner(ref, operators, narrow,
                             project=lambda s: frozenset(s) & RELEVANT)
    projected.plan(TASK, WITH_MIGRATION)
    projected.plan(TASK, WITH_MIGRATION | {"reviewer_assigned"})
    return {
        "plans": plans, "entries": len(keyed.cache), "calls": len(llm.calls),
        "shipped": shipped_reuse(ref, operators),
        "whole_calls": len(wide.calls), "projected_calls": len(narrow.calls),
        "library": library_first(ref, operators),
    }


def verify(result):
    plans, shipped, library = result["plans"], result["shipped"], result["library"]
    return [
        practice.Check(
            "ANSWER: two states, two entries, two calls, and none on a repeat",
            all([plans[0] == ["open_editor", "run_migration", "write_tests", "open_pr"],
                 plans[1] == ["open_editor", "write_tests", "open_pr"],
                 plans[2] == plans[0], result["entries"] == 2,
                 result["calls"] == 2]),
            f"keyed on (task, state) the two states get {result['entries']} cache "
            f"entries and {result['calls']} LLM calls, with a {len(plans[0])}-step plan "
            f"where the migration is needed and a {len(plans[1])}-step one where it is "
            "not. The repeat costs nothing",
        ),
        practice.Check(
            "FINDING: the shipped key reuses one state's answer in another",
            all([shipped["first"] is not None, shipped["second"] is None,
                 shipped["new_calls"] == 0, list(shipped["cached"]) == [TASK]]),
            f"keyed on task alone, the decomposition learned with has_migration present "
            f"is returned in a state without it, _expand rejects it and plan returns "
            f"{shipped['second']} after {shipped['new_calls']} new LLM calls. A stale "
            "entry surfaces as an unexplained planning failure, not as a cache miss",
        ),
        practice.Check(
            "FINDING: the whole state is too specific a key",
            all([result["whole_calls"] == 2, result["projected_calls"] == 1,
                 result["whole_calls"] > result["projected_calls"]]),
            f"adding one irrelevant fact to an already-cached state misses, costing "
            f"{result['whole_calls']} calls for one decomposition; projecting the state "
            f"onto the facts the operators mention takes it to "
            f"{result['projected_calls']}. The exercise's 'pattern' is a projection",
        ),
        practice.Check(
            "FINDING: re-checking the library first is already the shipped order",
            all([library["plan"] == ["open_editor", "write_tests"],
                 library["new_calls"] == 0, library["cache_intact"] is True]),
            f"plan reads methods, then cached_methods, then the LLM. Adding a method for "
            f"a cached task makes the method win on the next call -- the plan becomes "
            f"{library['plan']} with {library['new_calls']} new LLM calls and the cache "
            f"entry still present ({library['cache_intact']})",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
