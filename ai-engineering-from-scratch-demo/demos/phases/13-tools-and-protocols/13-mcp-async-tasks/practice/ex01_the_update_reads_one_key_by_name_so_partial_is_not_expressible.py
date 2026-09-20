"""Exercise 1 — the update reads one key by name, so partial is not expressible.

    Add a second outstanding input key. Send a partial `tasks/update` and
    prove the task remains `input_required` until both keys are answered.

Reading of the exercise: the proof has to be run against a `tasks_update` that
could have got it wrong, so the shipped one is tried first on a two-key task.
It reads `responses.get("approve_outline")` and nothing else, so it transitions
on one answer while the other key is still outstanding -- which is the bug the
exercise is describing, and the reason the update has to iterate the
outstanding set rather than name a key.

**ANSWER: two keys, and the task stays `input_required` until both are
answered.** Answering `approve_outline` alone leaves status `input_required`
with `['choose_format']` still in `inputRequests`; answering the second moves
it to `working`.

**FINDING: the shipped update transitions on one key of two, and discards the
other.** Given the same two-key task, `TaskService.tasks_update` sets
`working` after `approve_outline` alone -- and because it clears
`input_requests` wholesale rather than removing the key it answered, the
outstanding `choose_format` disappears with **0** answers. Iterating
`task.input_requests` is not a refactor here; it is the difference between the
exercise's requirement holding and not.

**FINDING: `tasks/update` says nothing about what it applied.** An unknown
key, a malformed answer and a correct one all return the same bare
`complete()` -- **1** distinct result for **3** inputs that leave the task in
**2** different states. The client learns the outcome only by polling
`tasks/get`, so an update that silently matched nothing is indistinguishable
from one that worked.

**FINDING: the key-reuse guard is unreachable.** `advance_worker` raises if a
key is already in `issued_keys`, but it only issues at `stage == 0` and sets
`stage = 1` immediately, so in normal operation the guard fires **0** times.
Rewinding the stage by hand is the only way to reach it -- the stage gate,
not the guard, is what actually prevents reuse.

Structure: `issue_two` and `apply_all` hold the logic, `service_class` binds
them onto the lesson's own service, and `states` runs one update and reports
the task as the wire would show it.
"""

from __future__ import annotations

import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "13-mcp-async-tasks"
KEYS = ("approve_outline", "choose_format")
ACCEPT = {"action": "accept", "content": {"approved": True}}


def form(key):
    return {"method": "elicitation/create", "params": {
        "mode": "form", "message": f"Answer {key}?", "requestedSchema": {
            "type": "object", "properties": {"approved": {"type": "boolean"}}}}}


def issue_two(service, ref, task_id):
    """The worker, issuing both keys at once."""
    task = service.store.get(task_id)
    if task.status != "working" or task.stage != 0:
        return ref.TaskService.advance_worker(service, task_id)
    for key in KEYS:
        if key in task.issued_keys:
            raise RuntimeError("task input request key cannot be reused")
        task.issued_keys.append(key)
    task.input_requests = {key: form(key) for key in KEYS}
    task.status, task.stage = "input_required", 1
    task.status_message = "Waiting for two answers."
    service.store.save(task)
    return task


def apply_all(service, ref, params, principal):
    """The update, iterating the outstanding set instead of naming a key."""
    ref.require_tasks_extension(ref.validate_request_meta(params))
    task = service._owned_task(params.get("taskId"), principal=principal)
    responses = params.get("inputResponses")
    if not isinstance(responses, dict):
        raise ref.McpError(-32602, "inputResponses must be an object")
    for key in list(task.input_requests):
        answer = responses.get(key)
        if isinstance(answer, dict) and answer.get("action") == "accept":
            del task.input_requests[key]
    if not task.input_requests:
        task.status, task.stage = "working", 2
        task.status_message = "Generating approved report."
    service.store.save(task)
    return ref.complete()


def build(ref, directory, name, two_key=True):
    """A service with a task already parked at input_required."""
    service = ref.TaskService(pathlib.Path(directory) / name)
    if two_key:  # bound on the instance, so dispatch finds these first
        service.advance_worker = lambda task_id: issue_two(service, ref, task_id)
        service.tasks_update = lambda p, *, principal: apply_all(service, ref, p, principal)
    created = service.dispatch(ref.make_request(
        1, "tools/call", {"name": "generate_report", "arguments": {"size": "small"}}))
    task_id = created["result"]["taskId"]
    service.advance_worker(task_id)
    return service, task_id


def states(ref, service, task_id, responses):
    """One update, then the task as tasks/get would report it."""
    service.dispatch(ref.make_request(
        2, "tasks/update", {"taskId": task_id, "inputResponses": responses}))
    wire = service.dispatch(ref.make_request(3, "tasks/get", {"taskId": task_id}))["result"]
    return wire["status"], sorted(wire.get("inputRequests", {}))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as directory:
        ordered, task = build(ref, directory, "ordered")
        first = states(ref, ordered, task, {KEYS[0]: ACCEPT})
        second = states(ref, ordered, task, {KEYS[1]: ACCEPT})

        shipped, task_c = build(ref, directory, "shipped")
        shipped.tasks_update = ref.TaskService.tasks_update.__get__(shipped)
        shipped_partial = states(ref, shipped, task_c, {KEYS[0]: ACCEPT})

        silent, task_d = build(ref, directory, "silent", two_key=False)
        results, transitions = [], []
        for responses in ({"nope": ACCEPT}, {KEYS[0]: "malformed"}, {KEYS[0]: ACCEPT}):
            answer = silent.dispatch(ref.make_request(
                4, "tasks/update", {"taskId": task_d, "inputResponses": responses}))
            results.append(sorted(answer["result"]))
            transitions.append(silent.store.get(task_d).status)

        rewound, task_e = build(ref, directory, "rewound")
        task_object = rewound.store.get(task_e)
        task_object.status, task_object.stage = "working", 0
        guard = None
        try:
            rewound.advance_worker(task_e)
        except RuntimeError as exc:
            guard = str(exc)
        return {
            "issued": sorted(ordered.store.get(task).issued_keys),
            "first": first, "second": second, "shipped_partial": shipped_partial,
            "results": results, "distinct": len({tuple(r) for r in results}),
            "transitions": transitions, "states_reached": len(set(transitions)), "guard": guard,
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: two keys, and the task stays input_required until both are answered",
            all([result["issued"] == sorted(KEYS),
                 result["first"] == ("input_required", [KEYS[1]]),
                 result["second"] == ("working", [])]),
            f"the worker issues {result['issued']}; answering {KEYS[0]} leaves "
            f"{result['first']} -- still input_required, with the other key outstanding -- "
            f"and answering the second gives {result['second']}",
        ),
        practice.Check(
            "FINDING: the shipped update transitions on one key of two, and discards the other",
            result["shipped_partial"] == ("working", []),
            f"TaskService.tasks_update answers {result['shipped_partial']} after one key: it "
            f"matches responses.get({KEYS[0]!r}) and then clears input_requests wholesale, so "
            f"the unanswered {KEYS[1]!r} disappears with no answer at all",
        ),
        practice.Check(
            "FINDING: tasks/update says nothing about what it applied",
            all([result["distinct"] == 1, result["states_reached"] == 2,
                 result["results"][0] == ["_meta", "resultType"]]),
            f"an unknown key, a malformed answer and a correct one return "
            f"{result['distinct']} distinct result, {result['results'][0]}, while leaving the "
            f"task in {result['states_reached']} states {result['transitions']} -- the "
            "outcome is learned only by polling tasks/get",
        ),
        practice.Check(
            "FINDING: the key-reuse guard is unreachable in normal operation",
            result["guard"] == "task input request key cannot be reused",
            f"advance_worker raises {result['guard']!r}, but only at stage 0, and it sets "
            "stage 1 immediately, so the guard is reached only by rewinding the stage by "
            "hand. The stage gate, not the guard, prevents reuse",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
