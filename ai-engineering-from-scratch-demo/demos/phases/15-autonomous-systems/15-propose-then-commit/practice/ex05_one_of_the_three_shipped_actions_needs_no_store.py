"""Exercise 5 — one of the three shipped actions needs no store.

    Pick one case where a synchronous "Approve?" prompt would be sufficient
    (no durable store needed). Explain why, and name the risk class you are
    accepting.

Reading of the exercise: "sufficient" is a property of the action, not of the
prompt, so the three proposals the lesson already ships are the candidate set
and the choice is made against their own declared blast radii rather than
against a new example.

**ANSWER: the single-row `db.update`, because its own record says it is
reversible within an hour.** Of the **3** shipped proposals, exactly **1**
has a blast radius declaring reversibility -- `one DB row; reversible within
1h backup window` -- against `37 recipients` with no in-band rollback and
`420k rows dropped; not reversible within 24h`. A synchronous prompt is
sufficient there because the cost of getting it wrong is bounded by a window
shorter than the approval itself would take to persist.

**FINDING: the risk class accepted is a lost approval, not a double
execution.** Without a store, a process death between prompt and execution
loses the answer and the reviewer is asked again. That is acceptable exactly
when re-asking is cheap and the action is idempotent -- and this one is, since
setting a row to `closed` twice is the same as once. The class you are
accepting is *reviewer time*, and the class you are not is *unbounded side
effects*, which is a different trade from the one the word "sufficient"
suggests.

**FINDING: the store buys lateness, and two of three proposals need it.**
**2** of the **3** blast radii are denominated in time -- `1h backup window`,
`not reversible within 24h` -- so the question "can the approval arrive
tomorrow" has a different answer per proposal. Durability is not a safety
property here; it is what makes an approval that arrives after the process
died still apply, which is the Lesson 12 result restated for humans.

**FINDING: re-approving a committed record resets it.**
`checklist_approve` writes `status = "approved"` without reading the current
status, so a second approval of an already-committed proposal makes `commit`
execute again -- **2** side effects against **1** without it. The durable
store prevents a *retry* from double-executing and does nothing about a second
*approval*, which is the path a synchronous prompt would take on every
re-ask.

Structure: `shipped()` reads the three proposals out of `main`; `one_store()`
runs the same payload twice, with and without a second approval.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import os
import re
import tempfile

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "15-propose-then-commit"

TEMP = tempfile.mkdtemp(prefix="aiefs-sync-")
REVERSIBLE = re.compile(r"(?<!not )reversible within", re.I)
TIMED = re.compile(r"\b\d+\s*h\b|\bhour|\bday", re.I)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def shipped(ref):
    """(action, blast radius, rollback) for each proposal `main` builds."""
    source = inspect.getsource(ref.main)
    actions = re.findall(r'action="([^"]+)"', source)
    blasts = re.findall(r'blast_radius="([^"]+)"', source)
    rollbacks = re.findall(r'rollback="([^"]+)"', source)
    return list(zip(actions, blasts, rollbacks))


def sample(ref, marker):
    return ref.Proposal(thread_id="t-sync", action="db.update",
                        payload={"row": 42, "val": marker},
                        intent="Close a stale issue", lineage="dashboard",
                        blast_radius="one DB row; reversible within 1h backup window",
                        rollback="restore row from nightly backup")


def across(ref, stores, marker):
    """Propose and commit the same payload through `stores` separate stores."""
    before = len(ref.SIDE_EFFECTS)
    for index in range(stores):
        path = os.path.join(TEMP, f"{marker}-{index}.json")
        if os.path.exists(path):
            os.remove(path)
        store = ref.Store(path)
        key = quiet(ref.propose, store, sample(ref, marker))
        quiet(ref.checklist_approve, store, key, True, True, True)
        quiet(ref.commit, store, key)
    return len(ref.SIDE_EFFECTS) - before


def one_store(ref, marker, reapprove, times=2):
    """The same payload proposed `times` into one store, re-approved or not."""
    path = os.path.join(TEMP, f"{marker}-{reapprove}.json")
    if os.path.exists(path):
        os.remove(path)
    store = ref.Store(path)
    before = len(ref.SIDE_EFFECTS)
    for attempt in range(times):
        key = quiet(ref.propose, store, sample(ref, marker))
        if attempt == 0 or reapprove:
            quiet(ref.checklist_approve, store, key, True, True, True)
        quiet(ref.commit, store, key)
    return len(ref.SIDE_EFFECTS) - before


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = shipped(ref)
    reversible = [action for action, blast, _r in rows if REVERSIBLE.search(blast)]
    return {
        "proposals": len(rows),
        "actions": [action for action, *_rest in rows],
        "reversible": reversible,
        "timed": [action for action, blast, _r in rows if TIMED.search(blast)],
        "no_rollback": [action for action, _b, rollback in rows
                        if rollback.startswith("no ")],
        "single_store": one_store(ref, "single", reapprove=False),
        "reapproved": one_store(ref, "again", reapprove=True),
        "two_stores": across(ref, 2, "double"),
        "commit_reads_status": 'rec["status"]' in inspect.getsource(ref.commit),
        "approve_checks_status": '"committed"' in inspect.getsource(ref.checklist_approve),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the single-row db.update, because its record says reversible",
            all([result["proposals"] == 3, result["reversible"] == ["db.update"],
                 result["no_rollback"] == ["email.send"]]),
            f"of {result['proposals']} shipped proposals {result['actions']}, "
            f"{len(result['reversible'])} declares reversibility -- "
            f"{result['reversible'][0]} -- while {result['no_rollback'][0]} declares no "
            "in-band rollback",
        ),
        practice.Check(
            "FINDING: the risk accepted is a lost approval, not a double execution",
            all([result["single_store"] == 1, result["commit_reads_status"]]),
            f"through one store the same payload proposed twice produces "
            f"{result['single_store']} side effect, so what a synchronous prompt gives "
            "up is the answer surviving a process death -- reviewer time, not unbounded "
            "side effects",
        ),
        practice.Check(
            "FINDING: the store buys lateness, and two of three need it",
            all([len(result["timed"]) == 2, "db.update" in result["timed"]]),
            f"{len(result['timed'])} of {result['proposals']} blast radii are "
            f"denominated in time -- {result['timed']} -- so whether an approval may "
            "arrive tomorrow has a different answer per proposal",
        ),
        practice.Check(
            "FINDING: re-approving a committed record resets it",
            all([result["reapproved"] == 2, result["single_store"] == 1,
                 not result["approve_checks_status"]]),
            f"checklist_approve writes status='approved' without reading the current "
            f"one, so re-approving a committed proposal makes it execute again -- "
            f"{result['reapproved']} side effects against {result['single_store']} "
            "without the second approval",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
