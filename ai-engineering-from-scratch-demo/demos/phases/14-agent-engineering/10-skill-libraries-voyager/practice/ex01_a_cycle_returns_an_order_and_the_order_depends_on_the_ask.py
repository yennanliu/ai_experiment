"""Exercise 1 — a cycle returns an order, and the order depends on the ask.

    Add a dependency-cycle detector to `compose()`. What happens when skill A
    depends on B which depends on A? Error vs warning?

Reading of the exercise: there is no `compose()`; the composition step is
`topo_order`, which walks `depends_on` with a `visited` set. That set makes
the walk terminate on a cycle instead of recursing forever -- so the shipped
behaviour is not a hang and not an error, it is a plausible-looking order.
Deciding between error and warning therefore means asking whether the order
it returns means anything.

**ANSWER: a detector over the lesson's own walk, and it must be an error.**
Across a library holding one valid DAG, one two-skill cycle and one
self-dependency, the detector flags **2** cycles and leaves the DAG alone.
It has to be an error because the shipped order is not merely unordered, it
is *ambiguous*: `topo_order("skill_a")` returns `['skill_b', 'skill_a']` and
`topo_order("skill_b")` returns `['skill_a', 'skill_b']`. Same library, same
cycle, opposite answers.

**FINDING: `execute` runs the cycle without complaint.** Composing the
cyclic pair produces **2** log lines and `failed=False`. Whichever skill you
asked for runs last, so the caller's entry point silently decides which half
of the cycle sees the other's effects.

**FINDING: a self-dependency is invisible.** `skill_c` depending on itself
composes to `['skill_c']` -- **1** entry, no repetition, no warning. The
`visited` set absorbs it exactly as it absorbs a real cycle, so the two
failures look identical from outside and neither looks like a failure.

**FINDING: a missing dependency is caught late and as a run failure.**
`topo_order` happily returns the unknown name, and `execute` only notices
when it reaches it: **1** log line reading `missing skill: nowhere` and
`failed=True`, after the earlier skills have already run and mutated the
context.

Structure: `compose()` wraps the lesson's `topo_order` and finds cycles by
walking `depends_on` itself; nothing here reimplements the ordering.
"""

from __future__ import annotations

from harness import parity, practice


PHASE, LESSON = "14-agent-engineering", "10-skill-libraries-voyager"


def noop(context):
    return "ok"


def make(ref, name, deps=()):
    return ref.Skill(name=name, description=f"the {name} skill", code=f"{name}()",
                     fn=noop, depends_on=tuple(deps))


def find_cycles(lib, name, seen=None, stack=None):
    """Depth-first over depends_on, reporting the back edges topo_order absorbs."""
    seen, stack = seen or set(), stack or []
    skill = lib.get(name)
    if name in stack:
        return [stack[stack.index(name):] + [name]]
    if skill is None or name in seen:
        return []
    seen.add(name)
    return [cycle for dep in skill.depends_on
            for cycle in find_cycles(lib, dep, seen, stack + [name])]


def compose(lib, name):
    """`topo_order` with the check the exercise asks for."""
    cycles = find_cycles(lib, name)
    if cycles:
        raise ValueError(f"dependency cycle: {' -> '.join(cycles[0])}")
    return lib.topo_order(name)


def build(ref):
    lib = ref.SkillLibrary()
    for name, deps in (("leaf", ()), ("mid", ("leaf",)), ("root", ("mid", "leaf")),
                       ("skill_a", ("skill_b",)), ("skill_b", ("skill_a",)),
                       ("skill_c", ("skill_c",)), ("orphan", ("nowhere",))):
        lib.register(make(ref, name, deps))
    return lib


def attempt(lib, name):
    try:
        return {"order": compose(lib, name), "error": ""}
    except ValueError as exc:
        return {"order": [], "error": str(exc)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    lib = build(ref)
    cyclic = ref.SkillLibrary()
    for name, deps in (("skill_a", ("skill_b",)), ("skill_b", ("skill_a",))):
        cyclic.register(make(ref, name, deps))
    ran = cyclic.execute("skill_a")
    missing = lib.execute("orphan")
    return {
        "flagged": sorted(name for name in ("root", "skill_a", "skill_c")
                          if attempt(lib, name)["error"]),
        "dag_order": attempt(lib, "root")["order"],
        "dag_error": attempt(lib, "root")["error"],
        "cycle_error": attempt(lib, "skill_a")["error"],
        "from_a": lib.topo_order("skill_a"), "from_b": lib.topo_order("skill_b"),
        "self_order": lib.topo_order("skill_c"),
        "ran_lines": len(ran["log"]), "ran_failed": ran["failed"],
        "missing_log": missing["log"], "missing_failed": missing["failed"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: an error, because the shipped order is ambiguous",
            all([result["flagged"] == ["skill_a", "skill_c"],
                 result["dag_order"] == ["leaf", "mid", "root"],
                 result["dag_error"] == "",
                 result["from_a"] == ["skill_b", "skill_a"],
                 result["from_b"] == ["skill_a", "skill_b"],
                 "cycle" in result["cycle_error"]]),
            f"the detector flags {result['flagged']} and leaves the DAG at "
            f"{result['dag_order']}. It has to be an error rather than a warning "
            f"because topo_order('skill_a') is {result['from_a']} and "
            f"topo_order('skill_b') is {result['from_b']} -- same library, same cycle, "
            "opposite answers",
        ),
        practice.Check(
            "FINDING: execute runs the cycle without complaint",
            all([result["ran_lines"] == 2, result["ran_failed"] is False]),
            f"composing the cyclic pair produces {result['ran_lines']} log lines and "
            f"failed={result['ran_failed']}. Whichever skill the caller asked for runs "
            "last, so the entry point silently decides which half of the cycle sees the "
            "other's effects",
        ),
        practice.Check(
            "FINDING: a self-dependency is invisible",
            all([result["self_order"] == ["skill_c"], len(result["self_order"]) == 1,
                 "skill_c" in result["flagged"]]),
            f"a skill depending on itself composes to {result['self_order']} -- one "
            "entry, no repetition, no warning. The visited set absorbs it exactly as it "
            "absorbs a real cycle, so neither failure looks like a failure from outside",
        ),
        practice.Check(
            "FINDING: a missing dependency is caught late and as a run failure",
            all([result["missing_failed"] is True, len(result["missing_log"]) == 1,
                 result["missing_log"][0] == "missing skill: nowhere"]),
            f"topo_order returns the unknown name happily and execute only notices on "
            f"arrival: {result['missing_log']} with failed={result['missing_failed']}. "
            "A dangling dependency is a runtime fault here, not a composition-time one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
