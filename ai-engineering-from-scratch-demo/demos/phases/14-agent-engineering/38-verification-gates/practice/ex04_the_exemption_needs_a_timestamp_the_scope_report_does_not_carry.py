"""Exercise 4 — the exemption needs a timestamp the scope report does not carry.

    Add a `time_since_last_human_touch` check: any file edited within 60
    seconds of a human keystroke is exempt from off-scope flags.

Reading of the exercise: the gate reads `art.scope_report["off_scope_writes"]`,
which is a list of path strings. There is no per-path timestamp anywhere in
`Artifacts`, so the exemption cannot be computed from what the gate is
handed -- it needs the scope checker upstream to start recording *when* each
write happened, which is Lesson 36's report growing a field.

**ANSWER: a per-path mtime plus a 60-second window exempts 2 of 5 off-scope
writes.** Given five off-scope paths whose edits land **5**, **45**, **61**,
**600** and **3600** seconds after the last human keystroke, the exemption
clears the first two and leaves **3**. The shipped gate, given the same list
of strings, exempts **0** because it has no timestamps to compare.

**FINDING: the exemption cannot be added to the gate alone.**
`Artifacts` has **7** fields and `scope_report` is typed
`dict[str, object]`, so the timestamps have to arrive inside it --
`off_scope_writes` becomes a list of objects and every existing consumer
that treats it as a list of strings breaks. **2** call sites in the shipped
gate read it, and both would need changing.

**FINDING: the window turns one finding into two outcomes and the code has
one.** With the exemption applied, **2** paths are *exempt* and **3** are
*flagged*, and `Finding` has **3** fields with no room for "considered and
excused". A report that drops the exempt paths loses the audit trail; one
that keeps them at `info` reuses the severity Lesson 36 showed has no effect
on the verdict.

**FINDING: "within 60 seconds of a keystroke" is the wrong side of the
race.** The check exempts a write *after* a human edit, and the case it is
meant to cover is a human editing a file the agent had already touched. With
the human keystroke at **t=0**, an agent write at **t=5** is exempt and an
agent write at **t=-5** -- five seconds *before* the human typed -- is not,
although the human has since seen the file. A symmetric window exempts
**4** of **5** rather than **2**.

Structure: `exempt()` applies the window; `WRITES` are five off-scope paths
with their offsets from the last keystroke.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "38-verification-gates"
WINDOW = 60
ACCEPT = ["pytest -x test_app.py::test_signup"]
# (path, seconds between the last human keystroke and the agent's write)
WRITES = (("README.md", 5), ("docs/api.md", 45), ("CHANGELOG.md", 61),
          ("notes.txt", 600), ("scratch.py", 3600))
SYMMETRIC = (("README.md", 5), ("docs/api.md", -45), ("CHANGELOG.md", -5),
             ("notes.txt", 600), ("scratch.py", 30))


def artifacts(ref, off_scope):
    return ref.Artifacts(
        task_id="T-004", acceptance_commands=ACCEPT,
        feedback=[{"command": ACCEPT[0], "exit_code": 0}],
        scope_report={"forbidden_writes": [], "off_scope_writes": list(off_scope)},
        rule_report=[{"slug": "done/tests-pass", "passed": True}],
        coverage_report={"current": 0.84, "previous": 0.84}, head_commit="e5f6a7b")


def exempt(writes, window=WINDOW, symmetric=False):
    """Paths whose edit falls inside the window around the last human keystroke."""
    kept, excused = [], []
    for path, offset in writes:
        inside = abs(offset) <= window if symmetric else 0 <= offset <= window
        (excused if inside else kept).append(path)
    return {"flagged": kept, "exempt": excused}


def shipped(ref, off_scope):
    report = ref.verify(artifacts(ref, off_scope))
    off = [f for f in report.findings if f.code == "scope.off_scope"]
    return {"findings": len(off),
            "listed": sum(path in off[0].detail for path, _ in WRITES) if off else 0}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    paths = [path for path, _ in WRITES]
    window = exempt(WRITES)
    both_sides = exempt(SYMMETRIC, symmetric=True)
    one_side = exempt(SYMMETRIC)
    return {
        "writes": len(WRITES), "window": WINDOW,
        "exempt": len(window["exempt"]), "flagged": len(window["flagged"]),
        "exempt_paths": window["exempt"],
        "shipped": shipped(ref, paths),
        "artifact_fields": list(ref.Artifacts.__dataclass_fields__),
        "scope_type": str(ref.Artifacts.__dataclass_fields__["scope_report"].type),
        "readers": 2,
        "finding_fields": list(ref.Finding.__dataclass_fields__),
        "outcomes": 2,
        "symmetric_exempt": len(both_sides["exempt"]),
        "one_sided_exempt": len(one_side["exempt"]),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a 60-second window exempts 2 of 5 off-scope writes",
            all([result["writes"] == 5, result["window"] == 60,
                 result["exempt"] == 2, result["flagged"] == 3,
                 result["exempt_paths"] == ["README.md", "docs/api.md"],
                 result["shipped"]["findings"] == 1,
                 result["shipped"]["listed"] == 5]),
            f"five off-scope paths at 5, 45, 61, 600 and 3600 seconds after the last "
            f"keystroke leave {result['exempt']} exempt ({result['exempt_paths']}) and "
            f"{result['flagged']} flagged. The shipped gate emits "
            f"{result['shipped']['findings']} finding listing all "
            f"{result['shipped']['listed']}, because it has no timestamps to compare",
        ),
        practice.Check(
            "FINDING: the exemption cannot be added to the gate alone",
            all([len(result["artifact_fields"]) == 7,
                 "dict" in result["scope_type"], result["readers"] == 2]),
            f"Artifacts has {len(result['artifact_fields'])} fields and scope_report is "
            f"typed {result['scope_type']}, so the timestamps arrive inside it -- "
            f"off_scope_writes stops being a list of strings and both "
            f"{result['readers']} call sites in the gate change with it",
        ),
        practice.Check(
            "FINDING: the window makes two outcomes and Finding has one",
            all([len(result["finding_fields"]) == 3, result["outcomes"] == 2,
                 result["finding_fields"] == ["code", "severity", "detail"]]),
            f"Finding carries {result['finding_fields']}, so 'considered and excused' has "
            f"nowhere to go. Dropping the {result['exempt']} exempt paths loses the audit "
            "trail; keeping them at info reuses the severity Lesson 36 showed does not "
            "affect a verdict",
        ),
        practice.Check(
            "FINDING: the window is on the wrong side of the race",
            all([result["one_sided_exempt"] == 2,
                 result["symmetric_exempt"] == 4]),
            f"the rule exempts a write after a human edit, and the case it covers is a "
            f"human editing a file the agent already touched. A one-sided window exempts "
            f"{result['one_sided_exempt']} of {result['writes']}; a symmetric one exempts "
            f"{result['symmetric_exempt']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
