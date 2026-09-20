"""Exercise 2 — a task id is opaque, so routing it costs the state tools do not.

    Add a Tasks-capable backend and route `tasks/get` by task id in
    `Mcp-Name`.

Reading of the exercise: the tool path routes by splitting `server.tool` in
half, and the obvious move is to make task ids look the same. That is refused
here, because a task id is minted by a backend and the gateway does not get to
choose its shape -- so the honest implementation is a lookup table, and the
cost of that table against the tool path's zero-state split is the finding.

**ANSWER: `tasks/get` routes by the id in `Mcp-Name` and answers from the
right backend.** Two backends mint **2** tasks; each id resolves to its own
backend and returns that backend's status, and an unknown id answers
**-32602**. The gateway's own `validate_wire` already enforces the header,
mirroring `params.taskId` for the task methods where it mirrors `params.name`
for the tool ones -- so nothing in the envelope had to change.

**FINDING: the header contract already covers a method the gateway does not
implement.** `NAMED_METHODS` lists **6** methods including all three
`tasks/*`, and `validate_wire` requires `Mcp-Name` for each -- yet the shipped
`handle` has branches for **4** methods and answers `tasks/get` with
**-32601**. The envelope was specified for the feature before the router was.

**FINDING: routing tools needs no state and routing tasks does.**
`canonical.split(".", 1)` recovers the backend from the name itself, for any
number of tools, with nothing stored. A task id carries no backend, so the
gateway keeps a `{task_id: backend}` map that grows with every task created --
**2** entries for 2 tasks -- and has to be durable, because a task outlives the
request that made it.

**FINDING: RBAC has no row a task id could match.** The table is keyed on
canonical tool names, **6** of them, and a task id matches **0**. Authorizing
`tasks/get` therefore cannot reuse the tool check: it has to be ownership, a
fact about who created the task, which is a second table again.

Structure: `TaskBackend` mints and answers; `TaskRouter` is the gateway
extension -- the map, the lookup and the owner check the tool path does not
need.
"""

from __future__ import annotations

import secrets

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "17-mcp-gateways-and-registries"


class TaskBackend:
    """A backend that mints task ids of its own shape."""

    def __init__(self, name):
        self.name, self.tasks = name, {}

    def create(self, owner):
        task_id = f"{secrets.token_hex(5)}"  # opaque: nothing names the backend
        self.tasks[task_id] = {"taskId": task_id, "status": "working", "owner": owner}
        return task_id

    def get(self, task_id):
        return dict(self.tasks[task_id], servedBy=self.name)


class TaskRouter:
    """tasks/get by id, which needs a map because the id carries no backend."""

    def __init__(self, ref, backends):
        self.ref, self.backends = ref, backends
        self.owner_of, self.backend_of = {}, {}

    def create(self, backend_name, owner):
        task_id = self.backends[backend_name].create(owner)
        self.backend_of[task_id] = backend_name
        self.owner_of[task_id] = owner
        return task_id

    def get(self, task_id, principal):
        body, headers = self.ref.make_request("tasks/get", 1, {"taskId": task_id})
        self.ref.validate_wire(body, headers)  # the lesson's own header check
        if headers.get("Mcp-Name") != task_id:
            raise self.ref.ProtocolError(-32020, "Mcp-Name header mismatch")
        backend = self.backend_of.get(task_id)
        if backend is None:
            return {"code": -32602, "message": "Unknown task"}
        if self.owner_of[task_id] != principal:
            return {"code": -32003, "message": "Forbidden"}
        return self.backends[backend].get(task_id)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    gateway = ref.Gateway()
    backends = {"notes": TaskBackend("notes"), "issues": TaskBackend("issues")}
    router = TaskRouter(ref, backends)
    first = router.create("notes", "alice")
    second = router.create("issues", "bob")

    body, headers = ref.make_request("tasks/get", 1, {"taskId": first})
    shipped_status, shipped = gateway.handle("bearer-alice", body, headers)
    return {
        "resolved": [router.get(first, "alice")["servedBy"],
                     router.get(second, "bob")["servedBy"]],
        "statuses": [router.get(first, "alice")["status"]],
        "unknown": router.get("deadbeef00", "alice"),
        "foreign": router.get(second, "alice"),
        "named_methods": sorted(ref.NAMED_METHODS),
        "task_methods": sorted(m for m in ref.NAMED_METHODS if m.startswith("tasks/")),
        "shipped_status": shipped_status, "shipped_code": shipped["error"]["code"],
        "header_sent": headers.get("Mcp-Name") == first,
        "tool_split": "notes.search".split(".", 1),
        "map_entries": len(router.backend_of),
        "id_names_backend": any(name in first for name in backends),
        "rbac_rows": sorted({tool for tools in ref.RBAC.values() for tool in tools}),
        "task_in_rbac": any(first in tools for tools in ref.RBAC.values()),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: tasks/get routes by the id in Mcp-Name and answers from the right backend",
            all([result["resolved"] == ["notes", "issues"], result["header_sent"],
                 result["statuses"] == ["working"],
                 result["unknown"]["code"] == -32602]),
            f"two backends mint two tasks and each id resolves to its own, "
            f"{result['resolved']}, returning that backend's status "
            f"{result['statuses']}. An unknown id answers "
            f"{result['unknown']['code']}, and the lesson's own validate_wire already "
            "enforces the header, so nothing in the envelope had to change",
        ),
        practice.Check(
            "FINDING: the header contract already covers a method the gateway does not implement",
            all([len(result["named_methods"]) == 6, len(result["task_methods"]) == 3,
                 result["shipped_status"] == 404, result["shipped_code"] == -32601]),
            f"NAMED_METHODS lists {len(result['named_methods'])} methods including "
            f"{result['task_methods']}, and validate_wire requires Mcp-Name for each -- yet "
            f"the shipped handle answers tasks/get with HTTP {result['shipped_status']} "
            f"{result['shipped_code']}. The envelope was specified for the feature before the "
            "router was",
        ),
        practice.Check(
            "FINDING: routing tools needs no state and routing tasks does",
            all([result["tool_split"] == ["notes", "search"],
                 not result["id_names_backend"], result["map_entries"] == 2]),
            f"canonical.split('.', 1) recovers {result['tool_split']} from the name itself, "
            f"for any number of tools, with nothing stored. A task id names no backend "
            f"({result['id_names_backend']}), so the gateway keeps a map -- "
            f"{result['map_entries']} entries for 2 tasks -- which has to be durable, because "
            "a task outlives the request that made it",
        ),
        practice.Check(
            "FINDING: RBAC has no row a task id could match",
            all([len(result["rbac_rows"]) == 4, not result["task_in_rbac"],
                 result["foreign"]["code"] == -32003]),
            f"RBAC is keyed on canonical tool names, {result['rbac_rows']}, and a task id "
            f"matches {int(result['task_in_rbac'])} of them. Authorizing tasks/get cannot "
            f"reuse the tool check -- it has to be ownership, which answers "
            f"{result['foreign']['code']} for another principal's task and is a second table "
            "again",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
