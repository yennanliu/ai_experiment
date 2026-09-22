"""Exercise 2 — the TTL is one day; the exercise says seven.

    Wire the script to write a `prereqs.lock` file and refuse to start if the
    lock is older than seven days.

Reading of the exercise: the lock ships, `write_lock` ships, and
`lock_is_fresh` ships -- with `LOCK_TTL_SECONDS` set to **one** day. So the
wiring is done and the number is wrong by a factor of seven, and the second
half of the sentence is wrong in a more interesting way: a stale lock does
not make the script *refuse to start*, it makes it run the probes.

**ANSWER: seven days is one constant, and "refuse" is a different branch.**
Setting `LOCK_TTL_SECONDS` to **604800** moves the boundary from **24** hours
to **168**: a lock written **72** hours ago is stale under the shipped value
and fresh under the corrected one. Refusing rather than re-probing is a
second change, because the shipped stale path falls through to
`run_probes()` and can still exit **0**.

**FINDING: the lock is keyed on a fingerprint of 4 constants and nothing
else.** `_deps_fingerprint` hashes `REQUIRED_DEPS`, `REQUIRED_TEST_COMMAND`,
`REQUIRED_ENV_VARS` and `REQUIRED_PYTHON` -- all module literals -- so the
lock is invalidated by editing the init script and by nothing that happens in
the repository. Changing every file the agent works on leaves the fingerprint
byte-identical.

**FINDING: a corrupt lock is silently treated as absent.**
`lock_is_fresh` catches `JSONDecodeError` and returns `False`, and a
`written_at` that will not coerce to a float returns `False` too -- so **3**
distinct malformed locks all produce "run the probes", which is the safe
direction and produces **0** diagnostics. A lock that has been wrong for a
month looks exactly like a first run.

**FINDING: the freshness check and the writer disagree about failure.**
`write_lock` is called only after every probe passes, so a lock's existence
means "the probes passed at time T". `lock_is_fresh` then returns `True`
without re-reading any probe, which is correct -- and `main` exits **0**
*without writing a report*, so `init_report.json` keeps describing the run
from up to **24** hours earlier while the console says the workbench is fine.

Structure: `fresh_at()` re-implements the TTL comparison so it can be swept;
`fingerprint_sensitivity()` changes things and watches the hash.
"""

from __future__ import annotations

import inspect
import json
import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "35-initialization-scripts"
SEVEN_DAYS = 7 * 24 * 60 * 60
AGES_HOURS = (1, 23, 25, 72, 167, 169)


def fresh_at(age_hours, ttl_seconds):
    return age_hours * 3600 < ttl_seconds


def with_lock(ref, root, payload):
    """Point the module's LOCK_PATH at a temporary file and ask it."""
    path = root / "prereqs.lock"
    path.write_text(payload)
    saved, ref.LOCK_PATH = ref.LOCK_PATH, path
    try:
        return ref.lock_is_fresh()
    finally:
        ref.LOCK_PATH = saved


def fingerprint_sensitivity(ref, root):
    """What changes the fingerprint, and what does not."""
    base = ref._deps_fingerprint()
    (root / "some_source.py").write_text("print('edited')\n")
    after_repo_edit = ref._deps_fingerprint()
    saved = ref.REQUIRED_DEPS
    ref.REQUIRED_DEPS = [*saved, "os"]
    after_const_edit = ref._deps_fingerprint()
    ref.REQUIRED_DEPS = saved
    return {"repo_edit_changes": after_repo_edit != base,
            "const_edit_changes": after_const_edit != base,
            "inputs": len(inspect.getsource(ref._deps_fingerprint)
                          .split("h.update")) - 1}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    root = pathlib.Path(tempfile.mkdtemp())
    shipped = {hours: fresh_at(hours, ref.LOCK_TTL_SECONDS) for hours in AGES_HOURS}
    corrected = {hours: fresh_at(hours, SEVEN_DAYS) for hours in AGES_HOURS}
    good = json.dumps({"fingerprint": ref._deps_fingerprint(),
                       "written_at": 10 ** 12}) + "\n"
    malformed = {
        "not json": "{oops",
        "not a dict": json.dumps([1, 2, 3]),
        "bad written_at": json.dumps({"fingerprint": ref._deps_fingerprint(),
                                      "written_at": "yesterday"}),
    }
    main_source = inspect.getsource(ref.main)
    return {
        "shipped_ttl_hours": ref.LOCK_TTL_SECONDS // 3600,
        "seven_day_hours": SEVEN_DAYS // 3600,
        "shipped": shipped, "corrected": corrected,
        "flipped": [h for h in AGES_HOURS if shipped[h] != corrected[h]],
        "good_lock": with_lock(ref, root, good),
        "malformed": {name: with_lock(ref, root, body)
                      for name, body in malformed.items()},
        "diagnostics": ref.lock_is_fresh.__code__.co_names.count("print"),
        **fingerprint_sensitivity(ref, root),
        "writes_report_on_hit": main_source.index("lock_is_fresh")
        < main_source.index("REPORT_PATH"),
        "write_lock_after_probes": main_source.index("run_probes")
        < main_source.index("write_lock"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: seven days is one constant, and refusing is a second change",
            all([result["shipped_ttl_hours"] == 24,
                 result["seven_day_hours"] == 168,
                 result["flipped"] == [25, 72, 167],
                 result["shipped"][72] is False,
                 result["corrected"][72] is True]),
            f"LOCK_TTL_SECONDS is {result['shipped_ttl_hours']} hours where the exercise "
            f"says {result['seven_day_hours']}, so locks aged {result['flipped']} hours "
            f"flip from stale to fresh. A 72-hour lock is {result['shipped'][72]} under "
            f"the shipped value and {result['corrected'][72]} under seven days",
        ),
        practice.Check(
            "FINDING: the fingerprint hashes 4 module constants and nothing else",
            all([result["inputs"] == 4, result["repo_edit_changes"] is False,
                 result["const_edit_changes"] is True]),
            f"_deps_fingerprint hashes {result['inputs']} values, all module literals, so "
            f"editing a source file leaves it unchanged ({result['repo_edit_changes']}) "
            f"while editing the init script's own constants moves it "
            f"({result['const_edit_changes']}). Nothing in the repository invalidates "
            "the lock",
        ),
        practice.Check(
            "FINDING: a corrupt lock is silently treated as absent",
            all([result["good_lock"] is True,
                 set(result["malformed"].values()) == {False},
                 len(result["malformed"]) == 3,
                 result["diagnostics"] == 0]),
            f"a well-formed lock reads {result['good_lock']} and all "
            f"{len(result['malformed'])} malformed shapes read "
            f"{sorted(set(result['malformed'].values()))} with "
            f"{result['diagnostics']} diagnostics. Safe direction, and a lock that has "
            "been wrong for a month looks exactly like a first run",
        ),
        practice.Check(
            "FINDING: a cache hit leaves the report describing an older run",
            all([result["writes_report_on_hit"] is True,
                 result["write_lock_after_probes"] is True,
                 result["shipped_ttl_hours"] == 24]),
            f"main checks the lock before it writes REPORT_PATH "
            f"({result['writes_report_on_hit']}) and write_lock runs only after the "
            f"probes ({result['write_lock_after_probes']}), so a hit exits 0 without a "
            f"report and init_report.json keeps describing a run up to "
            f"{result['shipped_ttl_hours']} hours old",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
