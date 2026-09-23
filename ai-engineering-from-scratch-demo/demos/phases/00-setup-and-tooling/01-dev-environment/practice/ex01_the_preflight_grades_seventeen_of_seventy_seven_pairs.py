"""Exercise 1 — the preflight grades seventeen of seventy-seven pairs.

    Run the verification script and fix any failures

Reading of the exercise: run it, then ask what a pass from it is worth --
because "fix any failures" is only useful advice if the script can report the
failures that matter, and on six of its eleven probes it cannot report one at
all.

**ANSWER: it runs, and its verdict reads 17 of 77 route-probe pairs.** Seven
routes times eleven probes is 77 combinations; only the 17 in `route.required`
are executed by default and only those 17 reach `passed == total`. The other
60 are skipped entirely unless `--show-later` is passed, and `--show-later`
changes nothing about the exit code -- `print_probe`'s return value is
discarded for optional probes. Six probes -- cargo, gpu, julia, jupyter,
matplotlib, torch -- are required on **0** of the seven routes, so no failure
of theirs can make this script exit non-zero on any route it offers.

**FINDING: the accelerator probe reports on PyTorch, not on accelerators.**
`gpu_result` has **5** returns. Both `Result(False, ...)` are reached before
the import -- "PyTorch is not installed" and "PyTorch could not be imported".
Every return after the import succeeds is `Result(True, ...)`, including the
one that says "CPU only". A machine with no accelerator gets a PASS on the
probe labelled "Accelerator backend".

**FINDING: uv installs everything the lesson uses and nothing checks it.**
Build It installs the interpreter, the virtual environment and three libraries
through uv, and **1** of the eleven fix strings tells the reader to run
`uv python install 3.12`. uv is not among the **11** probes and the string
`"uv"` is checked **0** times. The one tool whose absence breaks every
instruction in the lesson is the one tool the preflight does not look for.

**FINDING: the two routes with the same next command disagree about numpy.**
`beginner` and `ml-foundations` both end at
`01-linear-algebra-intuition/code/vectors.py`. `ml-foundations` requires numpy
to get there; `beginner` files it optional. `vectors.py` has **0** imports.

Structure: `census()` counts required/optional per probe; `returns()` reads the
gpu probe's own source.
"""

from __future__ import annotations

import collections
import inspect
import re

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "01-dev-environment"
NEXT = "01-linear-algebra-intuition/code/vectors.py"


def census(ref):
    """How many routes require each probe, and how many merely list it."""
    required, optional = collections.Counter(), collections.Counter()
    for route in ref.ROUTES.values():
        required.update(route.required)
        optional.update(route.optional)
    return required, optional


def returns(function):
    """The ok= argument of every `Result(...)` the function returns."""
    return re.findall(r"Result\((True|False)", inspect.getsource(function))


def fixes(ref, needle):
    """Probe fix strings that mention a tool."""
    return sum(needle in probe.fix for probe in ref.PROBES.values())


def route_map(ref, target, probe):
    """Routes whose next command names `target`, and routes that require `probe`."""
    return (sorted(k for k, r in ref.ROUTES.items() if target in r.next_command),
            sorted(k for k, r in ref.ROUTES.items() if probe in r.required))


def passing(ref, route):
    """How many of a route's required probes pass on this machine."""
    return sum(ref.PROBES[key].run().ok for key in route.required)


def imports(phase, lesson, name):
    """Top-level import statements in a lesson's own module."""
    path = parity.lesson_dir(phase, lesson) / "code" / f"{name}.py"
    return re.findall(r"(?m)^(?:import|from)\s+\S+", path.read_text("utf-8"))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "verify")
    required, optional = census(ref)
    beginner = ref.ROUTES["beginner"]
    shared, numpy_required = route_map(ref, NEXT, "numpy")
    return {
        "routes": len(ref.ROUTES),
        "probes": len(ref.PROBES),
        "required_pairs": sum(len(r.required) for r in ref.ROUTES.values()),
        "never_required": sorted(p for p in ref.PROBES if not required[p]),
        "gpu_returns": returns(ref.gpu_result),
        "uv_is_probed": "uv" in ref.PROBES or '"uv"' in inspect.getsource(ref),
        "uv_in_fixes": fixes(ref, "uv "),
        "pip_in_fixes": fixes(ref, "python3 -m pip"),
        "shared_next": shared,
        "numpy_required": numpy_required,
        "next_imports": imports("01-math-foundations", "01-linear-algebra-intuition", "vectors"),
        "beginner_passed": passing(ref, beginner),
        "beginner_required": len(beginner.required),
    }


def verify(result):
    pairs = result["routes"] * result["probes"]
    after = result["gpu_returns"][result["gpu_returns"].index("True"):]
    return [
        practice.Check(
            "ANSWER: it runs, and its verdict reads 17 of 77 route-probe pairs",
            all([result["required_pairs"] == 17, pairs == 77,
                 len(result["never_required"]) == 6,
                 result["beginner_passed"] == result["beginner_required"]]),
            f"{result['routes']} routes x {result['probes']} probes is {pairs} pairs and "
            f"{result['required_pairs']} are required; {len(result['never_required'])} probes "
            f"({', '.join(result['never_required'])}) are required on no route, so they "
            f"cannot change an exit code -- beginner reports "
            f"{result['beginner_passed']}/{result['beginner_required']} here",
        ),
        practice.Check(
            "FINDING: the accelerator probe reports on PyTorch, not on accelerators",
            all([len(result["gpu_returns"]) == 5, result["gpu_returns"].count("False") == 2,
                 set(after) == {"True"}]),
            f"gpu_result has {len(result['gpu_returns'])} returns; both "
            f"Result(False, ...) fire before the import, and all {len(after)} after it "
            "succeeds are Result(True, ...) -- so 'CPU only' is a PASS on the probe "
            "labelled Accelerator backend",
        ),
        practice.Check(
            "FINDING: uv installs everything the lesson uses and nothing checks it",
            all([not result["uv_is_probed"], result["uv_in_fixes"] == 1,
                 result["pip_in_fixes"] == 4]),
            f"{result['uv_in_fixes']} fix string tells the reader to run uv and "
            f"{result['pip_in_fixes']} tell them python3 -m pip, while uv is absent from "
            f"all {result['probes']} probes -- the tool whose absence breaks every Build "
            "It step is the one the preflight never looks for",
        ),
        practice.Check(
            "FINDING: the two routes with the same next command disagree about numpy",
            all([result["shared_next"] == ["beginner", "ml-foundations"],
                 result["numpy_required"] == ["ml-foundations"],
                 result["next_imports"] == []]),
            f"{' and '.join(result['shared_next'])} both end at vectors.py; only "
            f"{result['numpy_required'][0]} requires numpy, and the file has "
            f"{len(result['next_imports'])} imports",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
