"""Exercise 1 — the key covers three of the seven fields.

    Run `code/main.py`. Confirm that a retry of an approved proposal uses the
    durable record and does not re-execute. Now change the idempotency key to
    include a timestamp and show the retry double-executes.

Reading of the exercise: both halves are demonstrations and both hold. The
interesting part is in between -- what the key covers determines not only
whether a retry re-executes but whether a *different* proposal inherits an
approval, and the shipped key is narrow enough that it does.

**ANSWER: one execution across three commits; a timestamped key gives one
per proposal.** A committed record short-circuits, so commit plus two retries
produces **1** side effect. Replacing `key()` with one that includes a
timestamp makes every `propose` a new record, and committing **3** of them
produces **3** side effects.

**FINDING: the key hashes 3 of the proposal's 7 fields.** It covers
`thread_id`, `action` and `payload`; `intent`, `lineage`, `blast_radius` and
`rollback` are not in it. Re-proposing the same action and payload under a
different stated intent therefore returns the **existing** record -- status
`committed` -- and the new intent is never stored, never surfaced and never
reviewed. The approval transfers to a proposal the reviewer did not see.

**FINDING: the idempotency lives on the status, not on the key.** `commit`
short-circuits on `status == "committed"`, so two records with *different*
keys for the same effect both execute. The key protects against retrying one
record; nothing protects against proposing the same action twice, which is
the failure the timestamp demo produces and which a timestamp is not required
to reach.

**FINDING: the checklist is advisory.** `rubber_stamp_approve` sets the same
`approved` status with an `ack_mode` of `rubber_stamp`, and `commit` never
reads `ack_mode` -- a rubber-stamped `db.drop_table` executes exactly as a
checklist-approved one does. The distinction the lesson's third demo exists to
draw is recorded and not enforced.

Structure: `flow()` drives one proposal end to end; `Timestamped` is the
key change the exercise asks for.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import os
import tempfile
import time
from dataclasses import dataclass

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "15-propose-then-commit"

TEMP = tempfile.mkdtemp(prefix="aiefs-propose-")


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def store_at(ref, name):
    path = os.path.join(TEMP, name)
    if os.path.exists(path):
        os.remove(path)
    return ref.Store(path)


def sample(ref, intent="Announce the v1.2 release", thread="t-001"):
    return ref.Proposal(thread_id=thread, action="email.send",
                        payload={"to": "team@example.com", "subject": "release"},
                        intent=intent, lineage="release notes",
                        blast_radius="37 recipients", rollback="correction email")


def flow(ref, store, proposal, commits=1):
    key = quiet(ref.propose, store, proposal)
    quiet(ref.checklist_approve, store, key, True, True, True)
    before = len(ref.SIDE_EFFECTS)
    for _attempt in range(commits):
        quiet(ref.commit, store, key)
    return key, len(ref.SIDE_EFFECTS) - before


def timestamped(ref):
    """The key change the exercise asks for."""

    @dataclass
    class Timestamped(ref.Proposal):
        def key(self):
            return f"{super().key()}-{time.time_ns()}"
    return Timestamped


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    store = store_at(ref, "stable.json")
    _key, once = flow(ref, store, sample(ref), commits=3)

    drifted = store_at(ref, "drift.json")
    first_key = quiet(ref.propose, drifted, sample(ref))
    quiet(ref.checklist_approve, drifted, first_key, True, True, True)
    quiet(ref.commit, drifted, first_key)
    second_key = quiet(ref.propose, drifted, sample(ref, intent="Send to every customer"))

    rolling = store_at(ref, "rolling.json")
    stamped, effects = timestamped(ref), len(ref.SIDE_EFFECTS)
    for _attempt in range(3):
        flow(ref, rolling, stamped(**vars(sample(ref))))
    rubber = store_at(ref, "rubber.json")
    key = quiet(ref.propose, rubber, sample(ref, thread="t-002"))
    quiet(ref.rubber_stamp_approve, rubber, key)
    before = len(ref.SIDE_EFFECTS)
    quiet(ref.commit, rubber, key)
    return {
        "retries": once,
        "timestamped": len(ref.SIDE_EFFECTS) - effects - 1,
        "fields": list(ref.Proposal.__dataclass_fields__),
        "hashed": [name for name in ref.Proposal.__dataclass_fields__
                   if f'"{name[0]}": self.{name}' in inspect.getsource(ref.Proposal.key)],
        "same_key": first_key == second_key,
        "stored_intent": drifted.all()[second_key]["intent"],
        "stored_status": drifted.all()[second_key]["status"],
        "commit_checks_status": 'rec["status"] == "committed"' in inspect.getsource(ref.commit),
        "commit_reads_ack": "ack_mode" in inspect.getsource(ref.commit),
        "rubber_executed": len(ref.SIDE_EFFECTS) - before,
        "ack_mode": rubber.all()[key]["ack_mode"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one execution across three commits; one per timestamped proposal",
            all([result["retries"] == 1, result["timestamped"] == 3]),
            f"commit plus two retries produces {result['retries']} side effect, while "
            f"three proposals under a timestamped key produce "
            f"{result['timestamped']} -- the retry protection is the record, not the "
            "key",
        ),
        practice.Check(
            "FINDING: the key hashes 3 of the proposal's 7 fields",
            all([len(result["fields"]) == 7, result["hashed"] == ["thread_id", "action",
                                                                  "payload"],
                 result["same_key"], result["stored_intent"] != "Send to every customer",
                 result["stored_status"] == "committed"]),
            f"the key covers {result['hashed']} of {result['fields']}, so re-proposing "
            f"the same payload under a new intent returns the existing record -- status "
            f"{result['stored_status']}, intent still {result['stored_intent']!r} -- and "
            "the approval transfers to a proposal nobody saw",
        ),
        practice.Check(
            "FINDING: the idempotency lives on the status, not on the key",
            all([result["commit_checks_status"], result["timestamped"] == 3]),
            "commit short-circuits on status == committed, so two records with "
            "different keys for the same effect both execute -- the key protects a "
            "retry, not a re-proposal",
        ),
        practice.Check(
            "FINDING: the checklist is advisory",
            all([not result["commit_reads_ack"], result["rubber_executed"] == 1,
                 result["ack_mode"] == "rubber_stamp"]),
            f"commit never reads ack_mode, so a {result['ack_mode']} approval executes "
            f"{result['rubber_executed']} side effect exactly as a checklist approval "
            "does -- the distinction is recorded and not enforced",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
