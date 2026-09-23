"""Exercise 1 — sendMessage returns before the work it started.

    **Multi-hop task delegation.** Extend the `TaskManager` so an agent handler
    can delegate subtasks to other agents. The researcher receives a task,
    delegates "search" and "summarize" subtasks to two specialist agents,
    waits for both to complete, then merges the results into its own
    artifacts.

Reading of the exercise: build the multi-hop, then ask the manager what
happened -- because the extension needs `sendMessage` to be something a
handler can await, and it is not.

**ANSWER: the researcher delegates two subtasks and merges them.** The port runs three tasks: the
researcher's, plus `search` and `summarize` awaited together. The researcher
ends `completed` with **2** merged artifacts and **3** tasks in the manager.
Delegating requires waiting for the subtask to finish, and `sendMessage` does
not offer that -- so the handler has to poll the task it was handed back,
because the only thing it gets synchronously is an id.

**FINDING: `sendMessage` writes a state no caller can observe.** The manager
calls `this.processTask(...)` in **1** place and awaits it in **0**, attaching
a `.catch` instead. An un-awaited async function still runs synchronously to
its first `await`, and `processTask`'s first two statements set the state to
`working` and emit. `sendMessage` assigns `state: "submitted"` and then
returns a task reading **working**. The submitted state is written on every
call and visible on none.

**FINDING: the streaming API cannot see the transition it exists for.** A
caller only learns the task id from the return value of `sendMessage`, and by
then the `working` status update has already been emitted to a listener list
that is necessarily empty. A listener subscribed at the first possible moment
receives **1** of the **2** status updates the task produces -- it is
structurally too late for the first one.

**FINDING: a delegated task is indistinguishable from a root task.** `Task`
declares **5** fields -- id, contextId, status, artifacts, history -- and
**0** of them name a parent. `createTask` mints a fresh random `contextId`
whenever the caller does not pass one, so a subtask that forgets to thread the
context becomes a root, and the audit trail the next exercise is about has no
edge to record.

Structure: `TaskManager` is the lesson's class ported to asyncio, defect
intact; `delegating()` is the multi-hop the exercise asks for.
"""

from __future__ import annotations

import asyncio
import itertools
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "03-communication-protocols"
TERMINAL = ("completed", "failed", "canceled", "rejected")


class TaskManager:
    """The lesson's TaskManager: `processTask` is started, never awaited."""

    def __init__(self):
        self.tasks, self.handlers, self.listeners = {}, {}, {}
        self.ids = itertools.count(1)

    def register(self, name, handler):
        self.handlers[name] = handler

    def subscribe(self, task_id, listener):
        self.listeners.setdefault(task_id, []).append(listener)

    def send_message(self, agent, message, context=None):
        task = {"id": f"t-{next(self.ids)}", "context": context or f"c-{next(self.ids)}",
                "state": "submitted", "artifacts": [], "history": [message]}
        self.tasks[task["id"]] = task
        if agent not in self.handlers:
            task["state"] = "rejected"
            return task, None
        # processTask is called and not awaited, so in JS its body runs synchronously
        # to the first await: these two lines happen before sendMessage returns.
        task["state"] = "working"
        self._emit(task, {"kind": "statusUpdate", "state": "working"})
        return task, asyncio.ensure_future(self._resume(task, self.handlers[agent], message))

    async def _resume(self, task, handler, message):
        async for event in handler(self, task, message):
            if task["state"] in TERMINAL:
                break
            if event["kind"] == "statusUpdate":
                task["state"] = event["state"]
            else:
                task["artifacts"].append(event["artifact"])
            self._emit(task, event)

    def _emit(self, task, event):
        for listener in self.listeners.get(task["id"], ()):
            listener(event)


async def specialist(name):
    """A leaf agent: one artifact, then completed."""
    async def handler(_manager, _task, message):
        await asyncio.sleep(0)
        yield {"kind": "artifactUpdate", "artifact": {"name": name, "of": message}}
        yield {"kind": "statusUpdate", "state": "completed"}
    return handler


async def researcher(manager, _task, message):
    """The exercise: delegate two subtasks, wait for both, merge their artifacts."""
    sent = [manager.send_message(name, f"{name}:{message}") for name in ("search", "summarize")]
    await asyncio.gather(*(pending for _, pending in sent))
    for subtask, _ in sent:
        for artifact in subtask["artifacts"]:
            yield {"kind": "artifactUpdate", "artifact": artifact}
    yield {"kind": "statusUpdate", "state": "completed"}


async def delegating():
    """Run the multi-hop and record what the caller could observe while it ran."""
    manager = TaskManager()
    for name in ("search", "summarize"):
        manager.register(name, await specialist(name))
    manager.register("researcher", researcher)
    seen = []
    task, pending = manager.send_message("researcher", "survey rate limiters")
    returned = task["state"]
    manager.subscribe(task["id"], seen.append)
    await pending
    return {"returned": returned, "final": task["state"], "tasks": len(manager.tasks),
            "merged": [a["name"] for a in task["artifacts"]],
            "heard": sum(e["kind"] == "statusUpdate" for e in seen),
            "updates": 1 + sum(e["kind"] == "statusUpdate" for e in seen)}


def solve():
    src = (parity.lesson_dir(PHASE, LESSON) / "code" / "main.ts").read_text("utf-8")
    start = src.index("type Task = {")
    fields = re.findall(r"(?m)^  (\w+)", src[start:src.index("};", start)])
    return {
        **asyncio.run(delegating()),
        "started": src.count("this.processTask("),
        "awaited": src.count("await this.processTask"), "fields": fields,
        "parents": [f for f in fields if "parent" in f.lower()],
        "fresh_context": "contextId ?? crypto.randomUUID()" in src,
        "writes_submitted": 'state: "submitted"' in src,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the researcher delegates two subtasks and merges them",
            all([result["final"] == "completed", result["tasks"] == 3,
                 result["merged"] == ["search", "summarize"]]),
            f"the port runs {result['tasks']} tasks -- the researcher plus search and "
            f"summarize awaited together -- and it ends {result['final']} with "
            f"{len(result['merged'])} merged artifacts ({', '.join(result['merged'])})",
        ),
        practice.Check(
            "FINDING: sendMessage writes a state no caller can observe",
            all([result["started"] == 1, result["awaited"] == 0,
                 result["writes_submitted"], result["returned"] == "working"]),
            f"processTask is called in {result['started']} place and awaited in "
            f"{result['awaited']}, so its first two statements run before sendMessage "
            f"returns; sendMessage assigns state: \"submitted\" and hands back a task "
            f"reading {result['returned']}",
        ),
        practice.Check(
            "FINDING: the streaming API cannot see the transition it exists for",
            all([result["updates"] == 2, result["heard"] == 1]),
            f"the caller learns the task id only from the return value, by which point "
            f"the working update has gone to an empty listener list; a listener "
            f"subscribed at the first possible moment hears {result['heard']} of the "
            f"{result['updates']} status updates the task produces",
        ),
        practice.Check(
            "FINDING: a delegated task is indistinguishable from a root task",
            all([len(result["fields"]) == 5, result["parents"] == [],
                 result["fresh_context"]]),
            f"Task declares {len(result['fields'])} fields "
            f"({', '.join(result['fields'])}) and {len(result['parents'])} name a parent; "
            "createTask mints a fresh random contextId when none is passed, so a subtask "
            "that forgets to thread it becomes a root",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
