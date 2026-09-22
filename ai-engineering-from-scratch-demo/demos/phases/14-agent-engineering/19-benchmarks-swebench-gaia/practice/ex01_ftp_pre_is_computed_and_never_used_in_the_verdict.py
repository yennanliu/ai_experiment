"""Exercise 1 — ftp_pre is computed and never used in the verdict.

    Port the toy harness to run on a real repo (pick one of yours). Write 3
    FAIL_TO_PASS tests for known bugs.

Reading of the exercise: "a real repo" means the checks have to run real
code rather than read a dictionary. The port keeps `Task` exactly as shipped
and makes `state_before` a *version vector* -- which is what a patch is --
while each check calls the function at that version and asserts on its
output. Three known bugs, three FAIL_TO_PASS tests that really fail before
and really pass after.

**ANSWER: three tasks, three real FAIL_TO_PASS tests, 3 of 3 resolved.**
`slugify` drops non-ASCII letters, `clamp` is off by one at the upper bound, and
`truncate` cuts before the ellipsis rather than after; each task's
FAIL_TO_PASS test fails on version **0** and passes on version **1**, and
each carries **2** PASS_TO_PASS tests that hold at both versions.

**FINDING: a no-op patch resolves a task whose tests already pass.**
`run_task` computes `ftp_pre` and the verdict reads only `ftp_post`, so a
task whose FAIL_TO_PASS tests were green before the patch is reported
`resolved=True` with an identity patch: **2/2** FAIL_TO_PASS, **0** lines
changed. A benchmark gate that never checks the tests failed first cannot
tell a fix from a no-op.

**FINDING: the report has no delta.** `ftp_fixed` is computed and discarded,
and `TaskResult` has **6** fields -- **0** of them a before-count. So the
harness knows how many tests the patch fixed and reports only how many pass,
which is exactly the information a contamination audit needs.

**FINDING: a permanently red PASS_TO_PASS test stops guarding.**
`ptp_broke = ptp_pre - ptp_post`, so a PASS_TO_PASS test that was already
failing contributes **0** to the regression count however the patch behaves.
A task with **1** such test is resolved by a patch that leaves it broken.

Structure: `library()` is the code under test at two versions; `task()`
builds a shipped `Task` whose checks call it.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "19-benchmarks-swebench-gaia"


def slugify(text, version):
    keep = [c if c.isalnum() else "-" for c in text.lower()
            if version or c.isascii()]
    return "".join(keep).strip("-")


def clamp(value, low, high, version):
    return min(max(value, low), high if version else high - 1)


def truncate(text, width, version):
    return text if len(text) <= width else text[:width - 3 * version] + "..."


LIBRARY = {"slugify": slugify, "clamp": clamp, "truncate": truncate}
BUGS = ((("Crème Brûlée",), "crème-brûlée"), ((10, 0, 10), 10),
        (("abcdefgh", 5), "ab..."))


def check(name, args, want):
    """A real assertion against real code at whatever version the state holds."""
    def run(state):
        return LIBRARY[name](*args, version=state[name]) == want
    return run


def task(ref, tid, name, failing, holding):
    return ref.Task(
        tid=tid, description=f"{name} is wrong", state_before={name: 0},
        patch=lambda state, n=name: {**state, n: 1},
        fail_to_pass=[(f"test_{tid}_bug", check(name, *failing))],
        pass_to_pass=[(f"test_{tid}_ok{i}", check(name, *case))
                      for i, case in enumerate(holding)])


HOLDING = ([(("hello world",), "hello-world"), (("a b",), "a-b")],
           [((5, 0, 10), 5), ((-1, 0, 10), 0)],
           [(("abc", 5), "abc"), (("abcde", 5), "abcde")])


def tasks(ref):
    return [task(ref, f"t{i}", name, bug, holding) for i, (name, bug, holding)
            in enumerate(zip(LIBRARY, BUGS, HOLDING), start=1)]


def already_green(ref):
    """A task whose FAIL_TO_PASS tests pass before the patch, with a no-op patch."""
    return ref.Task(
        tid="t4", description="nothing is wrong", state_before={"clamp": 1},
        patch=lambda state: dict(state),
        fail_to_pass=[("test_a", check("clamp", (5, 0, 10), 5)),
                      ("test_b", check("clamp", (10, 0, 10), 10))],
        pass_to_pass=[("test_c", check("clamp", (0, 0, 10), 0))])


def red_guard(ref):
    """A PASS_TO_PASS test that was already failing, so it cannot regress."""
    return ref.Task(
        tid="t5", description="slugify is wrong", state_before={"slugify": 0},
        patch=lambda state: {**state, "slugify": 1},
        fail_to_pass=[("test_bug", check("slugify", ("Crème Brûlée",),
                                         "crème-brûlée"))],
        pass_to_pass=[("test_never_passed", check("slugify", ("x",), "WRONG"))])


def batch_report(results):
    """Each FAIL_TO_PASS test is also confirmed red at version 0."""
    return {"tasks": len(results), "resolved": sum(row.resolved for row in results),
            "ftp_totals": [row.ftp_total for row in results],
            "ptp_totals": [row.ptp_total for row in results],
            "red_before": [not check(name, args, want)({name: 0})
                           for name, (args, want) in zip(LIBRARY, BUGS)]}


def hole_report(ref, noop, guarded):
    fields = list(ref.TaskResult.__dataclass_fields__)
    return {
        "noop_resolved": noop.resolved,
        "noop_ftp": (noop.ftp_passed, noop.ftp_total), "result_fields": fields,
        "delta_fields": [f for f in fields
                         if any(k in f for k in ("pre", "fixed", "delta"))],
        "guarded_resolved": guarded.resolved,
        "guarded_ptp": (guarded.ptp_passed, guarded.ptp_total),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    results = [ref.run_task(item) for item in tasks(ref)]
    return {**batch_report(results),
            **hole_report(ref, ref.run_task(already_green(ref)),
                          ref.run_task(red_guard(ref)))}


def verify(result):
    return [
        practice.Check(
            "ANSWER: three real bugs, three FAIL_TO_PASS tests, 3 of 3 resolved",
            all([result["tasks"] == 3, result["resolved"] == 3,
                 result["ftp_totals"] == [1, 1, 1],
                 result["ptp_totals"] == [2, 2, 2],
                 result["red_before"] == [True, True, True]]),
            f"the tasks resolve {result['resolved']}/{result['tasks']}, each with "
            f"{result['ftp_totals'][0]} FAIL_TO_PASS test really red at version 0 "
            f"({result['red_before']}) and {result['ptp_totals'][0]} PASS_TO_PASS tests "
            "green at both. The checks call the real functions",
        ),
        practice.Check(
            "FINDING: a no-op patch resolves a task whose tests already pass",
            all([result["noop_resolved"] is True, result["noop_ftp"] == (2, 2)]),
            f"the verdict reads ftp_post and never ftp_pre, so a task whose FAIL_TO_PASS "
            f"tests were already green is resolved={result['noop_resolved']} at "
            f"{result['noop_ftp'][0]}/{result['noop_ftp'][1]} under an identity patch. A "
            "gate that never checks the tests failed first cannot tell a fix from a no-op",
        ),
        practice.Check(
            "FINDING: the report has no delta",
            all([len(result["result_fields"]) == 6, result["delta_fields"] == [],
                 result["result_fields"][1] == "ftp_passed"]),
            f"TaskResult carries {result['result_fields']} -- "
            f"{len(result['delta_fields'])} of them a before-count -- while run_task "
            "computes ftp_fixed and drops it: it knows how many tests the patch fixed "
            "and reports only how many pass",
        ),
        practice.Check(
            "FINDING: a permanently red PASS_TO_PASS test stops guarding",
            all([result["guarded_resolved"] is True,
                 result["guarded_ptp"] == (0, 1)]),
            f"ptp_broke is a difference, so a PASS_TO_PASS test that was already failing "
            f"contributes 0 to it: the task resolves ({result['guarded_resolved']}) with "
            f"{result['guarded_ptp'][0]} of {result['guarded_ptp'][1]} guards passing",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
