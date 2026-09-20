"""Exercise 2 — isolation is a filter over a shared namespace, and one path skips it.

    Add tenant ownership to the store and reject a valid task id presented by
    the wrong authenticated principal.

Reading of the exercise: "add to the store" is taken at its word, so the
tenant is put where the task is kept rather than beside the handler that
happens to check it -- and the reason that matters is found by looking for a
code path that reaches the store without going through the handler. There is
one, and the lesson calls it on every task.

**ANSWER: a foreign principal gets `-32602 task not found`, identical to a
nonexistent id.** A co-worker in the same tenant reads the task normally, so
the check is on the tenant and not on the individual who created it. The two
refusals are byte-identical, which is what stops the error from confirming
that the id exists.

**FINDING: the ownership check lives in the service, and `advance_worker`
does not use it.** `_owned_task` filters, but `advance_worker(task_id)` takes
no principal at all and calls `self.store.get` directly -- so it will drive
any tenant's task through its state machine. The store is the shared surface;
the filter is one layer above it, and one shipped method is already below.

**FINDING: one directory holds every tenant, and `reload()` loads them all.**
**3** task files from **3** tenants land in a single `store.tasks` dict keyed
by id alone. Isolation is a runtime predicate over a shared namespace, so a
missed predicate is not a narrow bug -- it is a full cross-tenant read.

**FINDING: what protects tenants today is entropy, not authorization.** Ids
are `uuid4().hex[:12]`, **48** bits, and any path that skips `_owned_task`
accepts one from anybody. Unguessable is a useful property and it is not the
same property: the tenant-keyed store refuses a correctly guessed id, and the
id space alone cannot.

Structure: `TenantStore` keys on `(tenant, task_id)` and `TenantService` puts
the tenant on every path -- including `advance_worker`, which the lesson's
version does not parameterise.
"""

from __future__ import annotations

import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "13-mcp-async-tasks"
ALICE, BOB, MALLORY = "alice@acme", "bob@acme", "mallory@evil"
TENANTS = {ALICE: "acme", BOB: "acme", MALLORY: "evil"}
NOT_FOUND = "task not found"


def make(ref):
    """The lesson's store and service with the tenant carried on every path."""

    class TenantStore(ref.TaskStore):
        def get_for(self, tenant, task_id):
            task = super().get(task_id)
            if TENANTS.get(task.owner) != tenant:
                raise ref.McpError(-32602, NOT_FOUND)
            return task

    class TenantService(ref.TaskService):
        def __init__(self, directory):
            super().__init__(directory)
            self.store = TenantStore(directory)

        def _owned_task(self, task_id, *, principal):
            return self.store.get_for(TENANTS.get(principal), task_id)

        def advance_worker(self, task_id, *, tenant=None):
            self.store.get_for(tenant, task_id)  # the check the lesson's version lacks
            return super().advance_worker(task_id)

    return TenantService


def ask(ref, service, task_id, principal):
    response = service.dispatch(ref.make_request(9, "tasks/get", {"taskId": task_id}),
                                principal=principal)
    if "error" in response:
        return response["error"]["message"]
    return response["result"]["status"]


def create(ref, service, principal):
    response = service.dispatch(ref.make_request(
        1, "tools/call", {"name": "generate_report", "arguments": {"size": "small"}}),
        principal=principal)
    return response["result"]["taskId"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as directory:
        service = make(ref)(pathlib.Path(directory) / "tenants")
        task = create(ref, service, ALICE)
        create(ref, service, BOB)
        foreign = create(ref, service, MALLORY)

        plain = ref.TaskService(pathlib.Path(directory) / "plain")
        plain_task = create(ref, plain, ALICE)
        leaked = plain.advance_worker(plain_task)  # no principal to give it
        signature = ref.TaskService.advance_worker.__code__
        return {
            "owner": ask(ref, service, task, ALICE),
            "same_tenant": ask(ref, service, task, BOB),
            "foreign": ask(ref, service, task, MALLORY),
            "unknown": ask(ref, service, "tsk_000000000000", MALLORY),
            "cross_tenant_get": ask(ref, service, foreign, ALICE),
            "advance_params": list(signature.co_varnames[:signature.co_argcount]),
            "leaked_status": leaked.status, "leaked_owner": leaked.owner,
            "files": len(list((pathlib.Path(directory) / "tenants").glob("*.json"))),
            "owners": sorted({t.owner for t in service.store.tasks.values()}),
            "keyed_by": "taskId",
            "id_bits": len(task.removeprefix("tsk_")) * 4,
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a foreign principal is refused exactly as a nonexistent id is",
            all([result["owner"] == "working", result["same_tenant"] == "working",
                 result["foreign"] == NOT_FOUND, result["unknown"] == NOT_FOUND,
                 result["cross_tenant_get"] == NOT_FOUND]),
            f"the owner and a same-tenant co-worker both read {result['owner']!r}, while a "
            f"foreign principal gets {result['foreign']!r} -- byte-identical to the "
            f"{result['unknown']!r} a nonexistent id gets. The check is on the tenant rather "
            "than the individual, and the error confirms nothing about the id",
        ),
        practice.Check(
            "FINDING: the ownership check lives in the service, and advance_worker skips it",
            all([result["advance_params"] == ["self", "task_id"],
                 result["leaked_status"] == "input_required",
                 result["leaked_owner"] == ALICE]),
            f"TaskService.advance_worker takes {result['advance_params']} -- no principal at "
            f"all -- and calls store.get directly, driving a task owned by "
            f"{result['leaked_owner']} to {result['leaked_status']} on anyone's behalf. The "
            "store is the shared surface and the filter is one layer above it",
        ),
        practice.Check(
            "FINDING: one directory holds every tenant and reload loads them all",
            all([result["files"] == 3, result["owners"] == sorted([ALICE, BOB, MALLORY]),
                 result["keyed_by"] == "taskId"]),
            f"{result['files']} task files from {len(result['owners'])} principals land in "
            f"one store.tasks dict keyed by {result['keyed_by']}, holding "
            f"{result['owners']}. Isolation is a runtime predicate over a shared namespace, "
            "so a missed predicate is a full cross-tenant read rather than a narrow bug",
        ),
        practice.Check(
            "FINDING: what protects tenants today is entropy, not authorization",
            result["id_bits"] == 48,
            f"ids are uuid4().hex[:12], {result['id_bits']} bits, and any path that skips "
            "_owned_task accepts one from anybody. Unguessable is a useful property and not "
            "the same one: a tenant-keyed store refuses a correctly guessed id, and the id "
            "space alone cannot",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
