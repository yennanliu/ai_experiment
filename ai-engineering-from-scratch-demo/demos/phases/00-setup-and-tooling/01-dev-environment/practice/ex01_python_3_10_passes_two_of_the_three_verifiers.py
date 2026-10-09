"""Exercise 1 — Python 3.10 passes two of the lesson's three verifiers.

    Run the verification script and fix any failures

Reading of the exercise: the lesson ships **three** verification scripts --
`verify.py`, `verify.ts` and `main.rs` -- and a result read off this machine
would grade the machine, not the script. So `verify.py`'s own `main()` is run
unchanged on two simulated machines: a fresh one where every probe fails, and
the same one after each printed fix is applied (every probe passes). The two
ports are read as files and compared with it on what "passing" means.

**ANSWER: the fail-fix-pass loop works.** On the fresh machine the beginner
route prints **0/2** required checks, exits **1**, and every one of its **2**
FAIL lines carries a `Fix:` line. After the fixes it prints **2/2**, exits
**0**, and names the next lesson, `vectors.py`.

**FINDING: the three verifiers disagree about Python.** `verify.py` demands
**3.11**; `verify.ts` and `main.rs` both accept **3.10**. Run on a Python 3.10
interpreter, the lesson's own `python_result` returns `ok=False` while both
ports would print PASS -- the same machine is ready or not depending on which
of the lesson's scripts the reader runs.

**FINDING: they disagree about what is required at all.** The beginner route
requires **2** tools (python, git). `verify.ts` requires **3** (adds node);
`main.rs` requires **5** (adds node, rustc, cargo). A machine that `verify.py`
pronounces ready fails **3** of `main.rs`'s required checks, and "fix any
failures" has a different answer per script.

**CONTROL: every next command the script prints points at a real file.** The
**6** routes that end in a `python3 phases/...` command name **5** distinct
files, and all **5** exist in the reference `phases/` tree.

Structure: `run_main` drives the lesson's `main()` with faked probes;
`port_rules` reads the two ports' Python thresholds and required sets.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import types

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "01-dev-environment"


def run_main(ref, ok, route="beginner"):
    """The lesson's main() with every probe forced to `ok`: (exit code, stdout)."""
    fake = {key: ref.Probe(p.label, lambda: ref.Result(ok, "simulated"), p.fix)
            for key, p in ref.PROBES.items()}
    saved, argv = ref.PROBES, sys.argv
    ref.PROBES, sys.argv = fake, ["verify.py", "--route", route]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            code = ref.main()
    finally:
        ref.PROBES, sys.argv = saved, argv
    return code, out.getvalue()


def python_310(ref):
    """The lesson's python_result, run as if on a Python 3.10 interpreter."""
    saved = ref.sys
    ref.sys = types.SimpleNamespace(version_info=(3, 10, 12), executable="/fake/python3.10")
    try:
        return ref.python_result().ok
    finally:
        ref.sys = saved


def port_rules(code_dir):
    """Each port's minimum Python minor and its required tools, read from its source."""
    ts = (code_dir / "verify.ts").read_text(encoding="utf-8")
    rs = (code_dir / "main.rs").read_text(encoding="utf-8")
    ts_required = re.findall(r'name: "([^"]+)",\s*required: true', ts)
    rs_required = re.findall(r'Check::new\("[^"]+", "([^"]+)", false\)', rs)
    return {
        "ts_minor": int(re.search(r"minor >= (\d+)", ts).group(1)),
        "rs_minor": int(re.search(r">= \(3, (\d+)\)", rs).group(1)),
        "ts_required": ts_required, "rs_required": rs_required,
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "verify")
    lesson = parity.lesson_dir(PHASE, LESSON)
    phases = lesson.parent.parent
    fresh_code, fresh = run_main(ref, False)
    fixed_code, fixed = run_main(ref, True)
    nexts = {r.next_command.split()[1] for r in ref.ROUTES.values()
             if r.next_command.startswith("python3 phases/")}
    return {
        "fresh_code": fresh_code, "fixed_code": fixed_code,
        "fresh_fails": fresh.count("[FAIL]"), "fresh_fixes": fresh.count("Fix:"),
        "fresh_score": "0/2" in fresh, "fixed_score": "2/2" in fixed,
        "next_shown": "vectors.py" in fixed,
        "py310": python_310(ref), **port_rules(lesson / "code"),
        "beginner": list(ref.ROUTES["beginner"].required),
        "routes_with_next": sum(r.next_command.startswith("python3 phases/")
                                for r in ref.ROUTES.values()),
        "nexts": len(nexts),
        "next_exist": sum((phases.parent / path).is_file() for path in nexts),
    }


def verify(result):
    rs_missing = len(result["rs_required"]) - len(result["beginner"])
    return [
        practice.Check(
            "ANSWER: fail, fix, pass -- the loop the exercise describes works",
            all((result["fresh_code"] == 1, result["fixed_code"] == 0,
                 result["fresh_fails"] == result["fresh_fixes"] == 2, result["fresh_score"],
                 result["fixed_score"], result["next_shown"])),
            f"the lesson's own main() on a fresh machine prints 0/2, exits "
            f"{result['fresh_code']} and pairs each of its {result['fresh_fails']} FAIL lines "
            f"with a Fix line; after the fixes it prints 2/2, exits {result['fixed_code']} "
            "and names vectors.py as the next lesson",
        ),
        practice.Check(
            "FINDING: Python 3.10 fails verify.py and passes both ports",
            not result["py310"] and result["ts_minor"] == result["rs_minor"] == 10,
            f"python_result on a 3.10 interpreter returns ok={result['py310']} (it wants 3.11), "
            f"while verify.ts accepts minor >= {result['ts_minor']} and main.rs >= "
            f"(3, {result['rs_minor']}): one machine is ready or not depending on the script",
        ),
        practice.Check(
            "FINDING: the three scripts require 2, 3 and 5 tools",
            len(result["beginner"]) == 2 and len(result["ts_required"]) == 3
            and len(result["rs_required"]) == 5,
            f"beginner requires {result['beginner']}, verify.ts {result['ts_required']}, "
            f"main.rs {result['rs_required']}: a machine verify.py calls ready can fail "
            f"{rs_missing} of main.rs's required checks",
        ),
        practice.Check(
            "CONTROL: every printed next command names a file that exists",
            result["next_exist"] == result["nexts"] == 5 and result["routes_with_next"] == 6,
            f"{result['routes_with_next']} routes end in a python3 phases/ command naming "
            f"{result['nexts']} distinct files, and {result['next_exist']} exist in the "
            "reference phases/ tree",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
