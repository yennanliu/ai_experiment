"""Exercise 3 — the lease has to be re-checked at the commit, not only at the claim.

    Add a worker lease with expiry. Demonstrate that two service instances
    cannot complete the same task concurrently.

Reading of the exercise: "two service instances" is two `TaskService` objects
over one directory, which is what the lesson's own persistence makes possible
and its in-memory `store.tasks` makes dangerous. The unsafety is shown before
the lease is added, so the lease has something to fix. Then the expiry is
given its own test, because an expiry that hands the lease to a second worker
is precisely the case where the first one is still running.

**ANSWER: two instances claim, one wins, and only the winner completes.**
Without a lease, two services over one directory both drive the same task to
`completed` -- **2** completions of **1** task, because each holds its own
`store.tasks` copy and neither re-reads. With the lease, **1** claim succeeds
and the loser is refused.

**FINDING: claiming is not enough, because the lease can move while you
work.** Instance A claims, its lease expires, B claims, and A then tries to
commit. Checking only at the claim lets A's stale result land on top of B's
task; re-checking the holder inside the commit refuses it. The expiry exists
so a crashed holder does not block the task forever, and the price of that is
exactly this window.

**FINDING: the reference is safe in one process and only there.** A second
`advance_worker` on the same instance returns early, because the task is no
longer `working`. Two instances each see `working` in their own dict, so the
guard reads a copy rather than the record -- single-process safety that does
not survive a second replica.

**FINDING: the lease is state the tasks extension has nowhere to put.** A
`Task` has **15** fields and none of them names a worker; the wire form has
**7** and none either. The lease is server-side bookkeeping that never reaches
a client, which is why adding it changes no message.

Structure: `Lease` is the claim table with expiry; `LeasedService` wraps the
lesson's `advance_worker` in claim-then-commit and refuses a commit whose
lease has moved.
"""

from __future__ import annotations

import dataclasses
import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "13-mcp-async-tasks"
LEASE_MS = 30_000


class Lease:
    """Who holds a task, and until when."""

    def __init__(self):
        self.holders = {}

    def claim(self, task_id, instance, now):
        holder, expires = self.holders.get(task_id, (None, 0))
        if holder is not None and now < expires:
            return False
        self.holders[task_id] = (instance, now + LEASE_MS)
        return True

    def holds(self, task_id, instance, now):
        holder, expires = self.holders.get(task_id, (None, 0))
        return holder == instance and now < expires


class LeasedService:
    """One replica: the lesson's service plus claim-then-commit."""

    def __init__(self, ref, directory, instance, lease):
        self.service, self.instance, self.lease = ref.TaskService(directory), instance, lease

    def run(self, task_id, now):
        if not self.lease.claim(task_id, self.instance, now):
            return "not leased"
        return self.commit(task_id, now)

    def commit(self, task_id, now):
        """The re-check the claim alone cannot give: is the lease still mine?"""
        if not self.lease.holds(task_id, self.instance, now):
            return "lease moved"
        self.service.store.reload()
        self.service.advance_worker(task_id)
        return self.service.store.get(task_id).status


def advance(service, task_id):
    """Advance once, reporting whether this call was the transition."""
    before = service.store.get(task_id).status
    after = service.advance_worker(task_id).status
    return before == "working" and after == "completed"


def seed(ref, directory):
    service = ref.TaskService(directory)
    created = service.dispatch(ref.make_request(
        1, "tools/call", {"name": "generate_report", "arguments": {"size": "small"}}))
    task_id = created["result"]["taskId"]
    service.advance_worker(task_id)
    accept = {"action": "accept", "content": {"approved": True}}
    service.dispatch(ref.make_request(2, "tasks/update", {
        "taskId": task_id, "inputResponses": {"approve_outline": accept}}))
    return task_id


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as directory:
        unleased = pathlib.Path(directory) / "unleased"
        task = seed(ref, unleased)
        # both replicas load before either writes, which is what makes the copies stale
        pair = [ref.TaskService(unleased), ref.TaskService(unleased)]
        naive = [advance(service, task) for service in pair]

        leased_dir, lease = pathlib.Path(directory) / "leased", Lease()
        leased_task = seed(ref, leased_dir)
        replicas = [LeasedService(ref, leased_dir, name, lease) for name in ("a", "b")]
        guarded = [replica.run(leased_task, now=1_000) for replica in replicas]

        expiring, slow_lease = pathlib.Path(directory) / "expiring", Lease()
        expiring_task = seed(ref, expiring)
        slow = LeasedService(ref, expiring, "a", slow_lease)
        slow_lease.claim(expiring_task, "a", 1_000)  # a holds it and works slowly
        stolen = LeasedService(ref, expiring, "b", slow_lease).run(
            expiring_task, now=1_000 + LEASE_MS + 1)
        late = slow.commit(expiring_task, now=1_000 + LEASE_MS + 2)

        same_dir = pathlib.Path(directory) / "same"
        same_task = seed(ref, same_dir)
        same = ref.TaskService(same_dir)
        twice = [advance(same, same_task) for _ in range(2)]
        return {
            "naive": naive, "naive_completions": naive.count(True), "stolen": stolen,
            "guarded": guarded, "guarded_completions": guarded.count("completed"),
            "twice": twice, "twice_completions": twice.count(True), "late": late,
            "task_fields": len(dataclasses.fields(ref.Task)),
            "wire_fields": len(ref.TaskStore(same_dir).create(size="s", owner="u").to_wire()),
            "lease_on_wire": "lease" in str(dataclasses.fields(ref.Task)).lower(),
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: without a lease two instances both complete; with one, exactly one does",
            all([result["naive"] == [True, True], result["naive_completions"] == 2,
                 result["guarded_completions"] == 1,
                 "not leased" in result["guarded"]]),
            f"two services over one directory both transition the task -- "
            f"{result['naive_completions']} completions of one, because each holds its own "
            f"store.tasks copy. With the lease the pair answers {result['guarded']}",
        ),
        practice.Check(
            "FINDING: claiming is not enough, because the lease can move while you work",
            all([result["stolen"] == "completed", result["late"] == "lease moved"]),
            f"a claims, its lease expires, b claims and completes ({result['stolen']!r}), and "
            f"a's commit is refused with {result['late']!r}. Claim-only checking would let "
            "a's stale result land on b's task; the expiry stops a crashed holder blocking "
            "forever, and this window is its price",
        ),
        practice.Check(
            "FINDING: the reference is safe in one process and only there",
            all([result["twice"] == [True, False], result["twice_completions"] == 1,
                 result["naive_completions"] == 2]),
            f"a second advance_worker on the same instance transitions nothing -- "
            f"{result['twice_completions']} of 2 calls, because the task is no longer "
            "working. Two instances each see working in their own dict, so the guard reads a "
            "copy rather than the record: safety that does not survive a second replica",
        ),
        practice.Check(
            "FINDING: the lease is state the tasks extension has nowhere to put",
            all([result["task_fields"] == 15, result["wire_fields"] == 7,
                 not result["lease_on_wire"]]),
            f"a Task has {result['task_fields']} fields and its wire form "
            f"{result['wire_fields']}, none of either naming a worker. The lease is "
            "server-side bookkeeping that never reaches a client, so adding it changes no "
            "message on the protocol",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
