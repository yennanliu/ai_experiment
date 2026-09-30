"""Exercise 1 — the task is finished before anyone can subscribe.

    **Multi-hop task delegation.** Extend the `TaskManager` so an agent handler
    can delegate subtasks to other agents. The researcher receives a task,
    delegates "search" and "summarize" subtasks to two specialist agents,
    waits for both to complete, then merges the results into its own
    artifacts.

Reading of the exercise: build the multi-hop against the shipped
`TaskManager`, then ask what a caller could observe while it ran -- because
"waits for both to complete" depends entirely on what `sendMessage` returns.

**ANSWER: the researcher delegates by awaiting sendMessage twice and merges.**
The port runs three tasks: the researcher's, plus `search` and `summarize`
awaited together with `gather`. The researcher ends `completed` with **2**
merged artifacts and **3** tasks in the manager. No polling is needed: the
subtask `sendMessage` hands back is already terminal.

**FINDING: upstream now awaits processTask, so sendMessage returns a finished
task.** Upstream #357 (merged into the fork in 31f21f1d) changed the one call
site to `await this.processTask(...)`: it is called in **1** place and awaited
in **1**. The earlier code returned a task already reading `working` while the
handler was still running; the promise now resolves with a task reading
**completed**. `sendMessage` still assigns `state: "submitted"`, and
`processTask` still overwrites it with `working` before its first `await`, so
neither state is ever held by a caller -- only the terminal one is.

**FINDING: the streaming API now hears none of the task's events.** The caller
learns the task id only from the resolved promise, which is now after the
task has finished. A listener subscribed at the first possible moment hears
**0** of the **4** events the researcher task emits (working, two artifacts,
completed). The fix traded a missed first update for a missed stream.

**FINDING: the lesson text still teaches the un-awaited call.** `docs/en.md`
calls `this.processTask(` in **1** place and awaits it in **0**, so a reader
following the page builds the old behaviour, not the code that ships.

**FINDING: a delegated task is indistinguishable from a root task.** `Task`
declares **5** fields -- id, contextId, status, artifacts, history -- and
**0** of them name a parent. `createTask` mints a fresh random `contextId`
whenever the caller does not pass one, so a subtask that forgets to thread the
context becomes a root, and the audit trail the next exercise is about has no
edge to record.

Structure: `TaskManager` is the lesson's class ported to asyncio, awaiting as
upstream now does; `delegating()` is the multi-hop the exercise asks for.
"""

from __future__ import annotations

import asyncio
import itertools
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "03-communication-protocols"
TERMINAL = ("completed", "failed", "canceled", "rejected")


class TaskManager:
    """The lesson's TaskManager: `sendMessage` now awaits `processTask`."""

    def __init__(self):
        self.tasks, self.handlers, self.listeners = {}, {}, {}
        self.ids, self.emitted = itertools.count(1), []

    def register(self, name, handler):
        self.handlers[name] = handler

    def subscribe(self, task_id, listener):
        self.listeners.setdefault(task_id, []).append(listener)

    async def send_message(self, agent, message, context=None):
        task = {"id": f"t-{next(self.ids)}", "context": context or f"c-{next(self.ids)}",
                "state": "submitted", "artifacts": [], "history": [message]}
        self.tasks[task["id"]] = task
        if agent not in self.handlers:
            task["state"] = "rejected"
            return task
        task["state"] = "working"  # processTask's statements before its first await
        self._emit(task, {"kind": "statusUpdate", "state": "working"})
        async for event in self.handlers[agent](self, task, message):
            if task["state"] in TERMINAL:
                break
            if event["kind"] == "statusUpdate":
                task["state"] = event["state"]
            else:
                task["artifacts"].append(event["artifact"])
            self._emit(task, event)
        return task

    def _emit(self, task, event):
        self.emitted.append(task["id"])
        for listener in self.listeners.get(task["id"], ()):
            listener(event)


async def specialist(_manager, _task, message):
    """A leaf agent: one artifact named after its own message, then completed."""
    await asyncio.sleep(0)
    yield {"kind": "artifactUpdate", "artifact": {"name": message.split(":")[0]}}
    yield {"kind": "statusUpdate", "state": "completed"}


async def researcher(manager, _task, message):
    """The exercise: delegate two subtasks, wait for both, merge their artifacts."""
    subtasks = await asyncio.gather(
        *(manager.send_message(name, f"{name}:{message}") for name in ("search", "summarize")))
    for subtask in subtasks:
        for artifact in subtask["artifacts"]:
            yield {"kind": "artifactUpdate", "artifact": artifact}
    yield {"kind": "statusUpdate", "state": "completed"}


async def delegating():
    """Run the multi-hop and record what the caller could observe while it ran."""
    manager = TaskManager()
    for name in ("search", "summarize"):
        manager.register(name, specialist)
    manager.register("researcher", researcher)
    seen = []
    task = await manager.send_message("researcher", "survey rate limiters")
    returned = task["state"]
    manager.subscribe(task["id"], seen.append)
    return {"returned": returned, "final": task["state"], "tasks": len(manager.tasks),
            "merged": [a["name"] for a in task["artifacts"]], "heard": len(seen),
            "events": manager.emitted.count(task["id"])}


def call_sites(src):
    return {"started": src.count("this.processTask("), "awaited": src.count("await this.processTask(")}


def solve():
    src = (parity.lesson_dir(PHASE, LESSON) / "code" / "main.ts").read_text("utf-8")
    start = src.index("type Task = {")
    fields = re.findall(r"(?m)^  (\w+)", src[start:src.index("};", start)])
    return {
        **asyncio.run(delegating()), **call_sites(src), "fields": fields,
        "doc": call_sites(parity.doc_text(PHASE, LESSON, "en")),
        "parents": [f for f in fields if "parent" in f.lower()],
        "fresh_context": "contextId ?? crypto.randomUUID()" in src,
        "writes_submitted": 'state: "submitted"' in src,
    }


def verify(result):
    doc = result["doc"]
    return [
        practice.Check(
            "ANSWER: the researcher delegates by awaiting sendMessage twice and merges",
            all([result["final"] == "completed", result["tasks"] == 3,
                 result["merged"] == ["search", "summarize"]]),
            f"the port runs {result['tasks']} tasks -- the researcher plus search and summarize "
            f"awaited together -- and it ends {result['final']} with {result['merged']} merged",
        ),
        practice.Check(
            "FINDING: upstream now awaits processTask, so sendMessage returns a finished task",
            all([result["started"] == 1, result["awaited"] == 1,
                 result["writes_submitted"], result["returned"] == "completed"]),
            f"processTask is called in {result['started']} place and awaited in "
            f"{result['awaited']}; sendMessage still assigns state: \"submitted\", but the "
            f"promise resolves with a task reading {result['returned']}",
        ),
        practice.Check(
            "FINDING: the streaming API now hears none of the task's events",
            all([result["events"] == 4, result["heard"] == 0]),
            f"the caller learns the task id only once the task has finished; a listener subscribed "
            f"then hears {result['heard']} of the {result['events']} events the task emits",
        ),
        practice.Check(
            "FINDING: the lesson text still teaches the un-awaited call",
            doc["started"] == 1 and doc["awaited"] == 0,
            f"docs/en.md calls this.processTask( in {doc['started']} place and awaits it "
            f"in {doc['awaited']}, while code/main.ts awaits it in {result['awaited']}",
        ),
        practice.Check(
            "FINDING: a delegated task is indistinguishable from a root task",
            all([len(result["fields"]) == 5, result["parents"] == [],
                 result["fresh_context"]]),
            f"Task declares {len(result['fields'])} fields ({', '.join(result['fields'])}) and "
            f"{len(result['parents'])} name a parent; createTask mints a fresh random contextId "
            "when none is passed, so a subtask that forgets to thread it becomes a root",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
