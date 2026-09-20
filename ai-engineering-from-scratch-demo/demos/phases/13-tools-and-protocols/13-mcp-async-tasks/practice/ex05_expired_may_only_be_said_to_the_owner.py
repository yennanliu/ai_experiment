"""Exercise 5 — "expired" may only be said to the owner.

    Add expiry cleanup. Distinguish an expired task from a malformed task id
    without leaking cross-tenant existence.

Reading of the exercise: the two requirements pull in opposite directions, so
the solution is the line between them. "Distinguish" wants three answers where
the lesson has one; "without leaking existence" forbids the third answer
reaching anyone who is not the owner. The resolution is that malformed is a
property of the *request* and expired is a property of a *record*, so one can
be answered before the store is consulted and the other cannot.

**ANSWER: three outcomes, and only two of them are ever shown to a
stranger.** A malformed id answers `malformed task id`; the owner's expired
task answers `task expired`; an unknown id, a foreign id, and a *foreign
expired* id all answer `task not found`. The foreign-expired case is the one
that matters -- answering `task expired` there would confirm the id exists.

**FINDING: the malformed check costs 0 store reads, which is why it leaks
nothing.** `"not a task id"` is rejected on shape alone -- the wrong prefix,
the wrong length -- before any lookup. Any distinction drawn *after* a lookup
is a distinction about a record someone owns, and can only be reported to
them.

**FINDING: the lesson collapses all three into one message, and that is safe
but unhelpful.** `TaskStore.get` answers `task not found` for a malformed id,
an unknown id and a foreign id alike -- **1** message for **3** conditions. It
leaks nothing and it also cannot tell a client that its own id is the wrong
shape, which is the one fault the client can fix without help.

**FINDING: cleanup destroys the evidence that "expired" depends on.** After
the sweep deletes the file, even the owner gets `task not found` -- the
expired answer is only available in the window between the deadline and the
sweep. Making it durable means a tombstone: **1** row per expired task,
keeping the owner and the deadline and nothing else.

Structure: `Reader` holds the three-way decision and the sweep, so the order
of its checks -- shape, then existence, then ownership, then expiry -- is the
whole answer.
"""

from __future__ import annotations

import pathlib
import re
import tempfile
from datetime import datetime, timedelta, timezone

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "13-mcp-async-tasks"
OWNER, STRANGER = "user-42", "user-99"
SHAPE = re.compile(r"^tsk_[0-9a-f]{12}$")
MALFORMED, EXPIRED, NOT_FOUND = "malformed task id", "task expired", "task not found"


class Reader:
    """The three-way answer, ordered so the store is consulted as late as possible."""

    def __init__(self, ref, service):
        self.ref, self.service, self.tombstones = ref, service, {}
        self.reads = 0

    def deadline(self, task):
        created = datetime.fromisoformat(task.created_at.replace("Z", "+00:00"))
        return created + timedelta(milliseconds=task.ttl_ms or 0)

    def read(self, task_id, principal, now):
        if not isinstance(task_id, str) or not SHAPE.match(task_id):
            return MALFORMED  # decided on shape alone: 0 store reads
        self.reads += 1
        task = self.service.store.tasks.get(task_id)
        if task is None:
            owner = self.tombstones.get(task_id)
            return EXPIRED if owner == principal else NOT_FOUND
        if task.owner != principal:
            return NOT_FOUND  # ownership before expiry, so expiry never leaks
        return EXPIRED if now >= self.deadline(task) else task.status

    def sweep(self, now, *, tombstone=False):
        expired = [t for t in list(self.service.store.tasks.values())
                   if now >= self.deadline(t)]
        for task in expired:
            del self.service.store.tasks[task.task_id]
            (self.service.store.directory / f"{task.task_id}.json").unlink()
            if tombstone:
                self.tombstones[task.task_id] = task.owner
        return len(expired)


def create(ref, service, principal):
    return service.dispatch(ref.make_request(
        1, "tools/call", {"name": "generate_report", "arguments": {"size": "small"}}),
        principal=principal)["result"]["taskId"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as directory:
        service = ref.TaskService(pathlib.Path(directory) / "tasks")
        mine, theirs = create(ref, service, OWNER), create(ref, service, STRANGER)
        reader = Reader(ref, service)
        now = datetime.now(timezone.utc)
        after = now + timedelta(milliseconds=service.store.get(mine).ttl_ms + 1)

        shipped = [service.dispatch(ref.make_request(2, "tasks/get", {"taskId": bad}))
                   ["error"]["message"]
                   for bad in ("not a task id", "tsk_000000000000", theirs)]
        before_reads = reader.reads
        answers = {
            "malformed": reader.read("not a task id", OWNER, now),
            "wrong_type": reader.read(123, OWNER, now),
            "_reads_after_malformed": reader.reads - before_reads,
            "unknown": reader.read("tsk_000000000000", OWNER, now),
            "foreign": reader.read(theirs, OWNER, now),
            "live": reader.read(mine, OWNER, now),
            "expired_owner": reader.read(mine, OWNER, after),
            "expired_foreign": reader.read(theirs, OWNER, after),
        }
        swept = reader.sweep(after)
        after_sweep = reader.read(mine, OWNER, after)

        kept = Reader(ref, ref.TaskService(pathlib.Path(directory) / "kept"))
        keep_id = create(ref, kept.service, OWNER)
        kept.sweep(after, tombstone=True)
        return {
            "answers": answers, "shipped": shipped,
            "shipped_distinct": len(set(shipped)),
            "swept": swept, "after_sweep": after_sweep,
            "tombstoned": kept.read(keep_id, OWNER, after),
            "tombstone_foreign": kept.read(keep_id, STRANGER, after),
            "tombstone_fields": sorted({"owner"}),
            "store_reads_for_malformed": answers.pop("_reads_after_malformed"),
        }


def verify(result):
    answers = result["answers"]
    return [
        practice.Check(
            "ANSWER: three outcomes, and a stranger never sees the third",
            all([answers["malformed"] == MALFORMED, answers["wrong_type"] == MALFORMED,
                 answers["expired_owner"] == EXPIRED, answers["live"] == "working",
                 answers["unknown"] == answers["foreign"] == answers["expired_foreign"]
                 == NOT_FOUND]),
            f"a malformed id answers {answers['malformed']!r}, the owner's expired task "
            f"{answers['expired_owner']!r}, and an unknown id, a foreign id and a foreign "
            f"*expired* id all answer {answers['unknown']!r}. That last one is the case that "
            "matters -- saying 'expired' there would confirm the id exists",
        ),
        practice.Check(
            "FINDING: the malformed check costs no store reads, which is why it leaks nothing",
            all([result["store_reads_for_malformed"] == 0,
                 answers["malformed"] == answers["wrong_type"]]),
            f"'not a task id' and a non-string are both rejected on shape alone, with "
            f"{result['store_reads_for_malformed']} lookups. Any distinction drawn after a "
            "lookup is a distinction about a record somebody owns, and can only be reported "
            "to them -- which is what fixes the order of the checks",
        ),
        practice.Check(
            "FINDING: the lesson collapses all three into one message",
            all([result["shipped_distinct"] == 1, result["shipped"][0] == NOT_FOUND]),
            f"TaskStore.get answers {result['shipped'][0]!r} for a malformed id, an unknown "
            f"id and a foreign id alike -- {result['shipped_distinct']} message for 3 "
            "conditions. It leaks nothing, and it also cannot tell a client its own id is "
            "the wrong shape, which is the one fault the client can fix unaided",
        ),
        practice.Check(
            "FINDING: cleanup destroys the evidence that expired depends on",
            all([result["swept"] == 2, result["after_sweep"] == NOT_FOUND,
                 result["tombstoned"] == EXPIRED,
                 result["tombstone_foreign"] == NOT_FOUND]),
            f"the sweep removes {result['swept']} tasks and the owner then gets "
            f"{result['after_sweep']!r} -- the expired answer lived only between the deadline "
            f"and the sweep. A tombstone keeping {result['tombstone_fields']} restores "
            f"{result['tombstoned']!r} for the owner while still answering "
            f"{result['tombstone_foreign']!r} to everyone else",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
