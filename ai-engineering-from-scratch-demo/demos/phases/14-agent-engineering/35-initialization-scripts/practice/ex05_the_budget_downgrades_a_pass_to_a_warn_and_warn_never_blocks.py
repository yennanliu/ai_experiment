"""Exercise 5 — the budget downgrades a pass to a warn, and warn never blocks.

    Add a timing budget per probe. A probe that runs longer than three
    seconds is a workbench smell.

Reading of the exercise: `_timed` ships, `PROBE_BUDGET_SECONDS` is **3.0**,
and the decorator already rewrites a slow `pass` into a `warn`. So the budget
exists and the question is what it does -- and the answer is that `main`
computes `ok` as `all(status != "fail")`, so `warn` is indistinguishable from
`pass` at the exit code. A smell is reported and never acted on.

**ANSWER: the budget fires above 3000ms and changes nothing the caller
sees.** A probe that runs past the budget comes back `warn` with
`(slow: ...)` appended, and `ok` is still `True` -- exit **0**. Across **3**
synthetic probes at **250ms**, **4000ms** and **8000ms** the statuses are
`pass`, `warn`, `warn` and the run's verdict is unchanged in all **3**
cases.

**FINDING: the budget is measured after the probe returns, so it cannot
bound anything.** `_timed` calls `probe_fn()` and *then* compares the
elapsed time, so a probe that hangs for an hour is reported after an hour.
Of the **6** shipped probes exactly **1** carries a real bound -- the
`timeout=2.0` on `probe_lkg_diff`'s subprocess -- and that bound is
**2.0s** against a budget of **3.0s**, so the one probe that can time out
can never trip the budget.

**FINDING: the downgrade only applies to a passing probe.** The condition is
`status == "pass"`, so a probe that is both slow *and* failing keeps
`fail` and loses the timing note entirely: **1** of **3** synthetic slow
probes reports its duration in the detail string. A slow failure and a fast
failure are indistinguishable, which is the pair a workbench actually wants
to tell apart.

**FINDING: only 1 of the 6 shipped probes can return `fail` at all.**
`REQUIRED_DEPS` is `['json', 'dataclasses']` -- both stdlib -- and
`REQUIRED_ENV_VARS` is empty, so the dependency and env probes cannot fail;
the runtime probe needs Python below **3.10**; `probe_test_command` looks
for `python3`, which resolved to start the script; and state freshness
returns `warn` by construction. The gate's real coverage is the LKG diff,
and the timing budget cannot fail anything.

Structure: `fake_probe()` wraps a sleeper in the shipped decorator;
`failable()` asks each shipped probe whether `fail` is reachable.
"""

from __future__ import annotations

import inspect
import sys

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "35-initialization-scripts"
# Exact in binary, so int(elapsed * 1000) has no rounding surprise.
DURATIONS_MS = (250, 4000, 8000)


def fake_probe(ref, name, elapsed_ms, status="pass"):
    """Run the shipped _timed decorator over a probe with a controlled duration."""
    clock = {"now": 1000.0}
    saved = ref.time.time
    ref.time.time = lambda: clock["now"]

    def inner():
        clock["now"] += elapsed_ms / 1000
        return ref.Probe(name, status, "did the thing")
    try:
        return ref._timed(inner)()
    finally:
        ref.time.time = saved


def verdict(probes):
    return all(probe.status != "fail" for probe in probes)


def failable(ref):
    """Which shipped probes can return fail on this machine, and why not."""
    reasons = {
        "runtime": sys.version_info[:2] < ref.REQUIRED_PYTHON,
        "dependencies": any(dep not in sys.stdlib_module_names
                            for dep in ref.REQUIRED_DEPS),
        "test_command": ref.REQUIRED_TEST_COMMAND not in ("python3", "python"),
        "env": bool(ref.REQUIRED_ENV_VARS),
        "state_freshness": "fail" in inspect.getsource(
            ref.probe_state_freshness.__wrapped__
            if hasattr(ref.probe_state_freshness, "__wrapped__")
            else ref.probe_state_freshness),
        "lkg_diff": True,
    }
    return reasons


def subprocess_timeout(ref):
    source = inspect.getsource(ref)
    marker = "timeout="
    index = source.index(marker, source.index("def probe_lkg_diff"))
    return float(source[index + len(marker):].split(",")[0].split(")")[0])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    probes = [fake_probe(ref, f"p{ms}", ms) for ms in DURATIONS_MS]
    slow_fail = fake_probe(ref, "broken", 5000, status="fail")
    reach = failable(ref)
    return {
        "budget_ms": int(ref.PROBE_BUDGET_SECONDS * 1000),
        "statuses": [probe.status for probe in probes],
        "durations": [probe.duration_ms for probe in probes],
        "slow_note": sum("slow:" in probe.detail for probe in probes),
        "verdict": verdict(probes),
        "verdicts": [verdict([probe]) for probe in probes],
        "slow_fail_status": slow_fail.status,
        "slow_fail_noted": "slow:" in slow_fail.detail,
        "shipped_probes": len(ref.run_probes()),
        "failable": [name for name, ok in reach.items() if ok],
        "bounded": subprocess_timeout(ref),
        "deps": list(ref.REQUIRED_DEPS), "env_vars": list(ref.REQUIRED_ENV_VARS),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the budget fires above 3000ms and changes nothing the caller sees",
            all([result["budget_ms"] == 3000,
                 result["statuses"] == ["pass", "warn", "warn"],
                 result["durations"] == [250, 4000, 8000],
                 result["verdict"] is True, result["verdicts"] == [True] * 3]),
            f"probes at {result['durations']}ms against a {result['budget_ms']}ms budget "
            f"come back {result['statuses']}, and ok is all(status != 'fail') -- "
            f"{result['verdict']} for the batch and {result['verdicts']} individually. "
            "The smell is reported and never acted on",
        ),
        practice.Check(
            "FINDING: the budget is measured after the probe returns",
            all([result["bounded"] == 2.0, result["shipped_probes"] == 6,
                 result["budget_ms"] == 3000]),
            f"_timed calls the probe and then compares the elapsed time, so a hang is "
            f"reported after it ends. Exactly 1 of the {result['shipped_probes']} shipped "
            f"probes carries a real bound -- the subprocess timeout of "
            f"{result['bounded']}s -- and it is below the {result['budget_ms']}ms budget, "
            "so the only boundable probe can never trip it",
        ),
        practice.Check(
            "FINDING: the downgrade only applies to a passing probe",
            all([result["slow_fail_status"] == "fail",
                 result["slow_fail_noted"] is False,
                 result["slow_note"] == 2]),
            f"the condition is status == 'pass', so a probe that is slow and failing keeps "
            f"{result['slow_fail_status']!r} and loses its timing note "
            f"({result['slow_fail_noted']}). Only {result['slow_note']} of the slow "
            "probes report a duration, and a slow failure reads like a fast one",
        ),
        practice.Check(
            "FINDING: only 1 of the 6 shipped probes can return fail",
            all([result["failable"] == ["lkg_diff"],
                 result["deps"] == ["json", "dataclasses"],
                 result["env_vars"] == [], result["shipped_probes"] == 6]),
            f"REQUIRED_DEPS is {result['deps']} -- both stdlib -- and REQUIRED_ENV_VARS "
            f"is {result['env_vars']}, so those two cannot fail; runtime needs Python "
            f"below 3.10, test_command looks for the interpreter that is running, and "
            f"state freshness warns by construction. The failable set is "
            f"{result['failable']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
