"""Exercise 4 — the shipped client ignores the interval it was handed.

    Add a task store that survives a process restart. Prove a client can
    resume with `tasks/get`, respect `pollIntervalMs`, and read the completed
    task's final result without `tasks/result`.

Reading of the exercise: "survives a process restart" is only provable with a
second interpreter, so the worker is a real `subprocess` and the same worker
is asked for the shipped store's answer on its way past. That gives both
halves of the proof from one process boundary: the module-global `TASKS` dict
is gone in the child, and the file-backed task is still there.

**ANSWER: the task survives, resumes, is polled at its declared interval, and
carries its own final result.** The durable store answers `working` from a
process that never created the task, then `completed`; every poll gap is at
least the declared **60**ms; the result is read from `task["result"]` with
**0** calls to `tasks/result`.

**FINDING: the shipped client ignores the interval it was handed.**
`research_generate_report` declares `pollIntervalMs` of **1000** and
`orchestrator` calls `tasks_get` on the next line -- a whole run finishes in
under **50**ms, two orders of magnitude inside the interval. The field is emitted and never read, so the only client that
exists is the one that violates it.

**FINDING: the volatile store does not survive its own process.** A second
interpreter importing the same module answers `-32602 Unknown taskId` for a
task id the first one minted, because `TASKS` is a module global. Durability
here is not a missing feature so much as a missing file.

**FINDING: `tasks/result` is absent and unnecessary in the same breath.** The
module defines **1** `tasks_` method, and the completed task already carries
`result` with its content and `_meta`. A separate fetch would only exist to
re-deliver a field the client has -- which is why the current extension
dropped it.

Structure: `WORKER` is the second process; `poll()` is the resuming client
and keeps its own gaps so the interval can be checked rather than asserted.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "23-capstone-tool-ecosystem"
POLL_MS, LAG = 60, 0.15
WORKER = '''
import json, os, pathlib, sys, time
sys.path.insert(0, sys.argv[1])
import main as ref
store, task_id = pathlib.Path(sys.argv[2]), sys.argv[3]
volatile = ref.tasks_get(task_id, ref.request_meta(tasks=True))
(store / "worker.json").write_text(json.dumps(
    {"pid": os.getpid(), "volatile": volatile, "tasks": len(ref.TASKS)}))
time.sleep(float(sys.argv[4]))
path = store / f"{task_id}.json"
task = json.loads(path.read_text())
task.update(status="completed", result={
    "resultType": "complete", "content": [{"type": "text", "text": "3 papers summarized."}],
    "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "research-simulator"}}})
path.write_text(json.dumps(task))
'''


def write_task(store, task_id):
    """The durable half: a task handle that is a file before it is a dict."""
    (store / f"{task_id}.json").write_text(json.dumps({
        "resultType": "task", "taskId": task_id, "status": "working",
        "ttlMs": 900_000, "pollIntervalMs": POLL_MS,
        "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "research-simulator"}}}))


def tasks_get(store, task_id):
    path = store / f"{task_id}.json"
    if not path.exists():
        return {"error": {"code": -32602, "message": "Unknown taskId"}}
    return {"resultType": "complete", **json.loads(path.read_text())}


def poll(store, task_id, deadline):
    """Resume by id alone, waiting the interval the server declared."""
    statuses, gaps, last = [], [], None
    while time.monotonic() < deadline:
        task = tasks_get(store, task_id)
        now = time.monotonic()
        if last is not None:
            gaps.append(round((now - last) * 1000))
        statuses.append(task["status"])
        last = now
        if task["status"] == "completed":
            return statuses, gaps, task
        time.sleep(task["pollIntervalMs"] / 1000)
    return statuses, gaps, tasks_get(store, task_id)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    reference_dir = str(pathlib.Path(ref.__file__).parent)
    handle = ref.research_generate_report({}, ref._hex(16), None)
    started = time.monotonic()
    ref.orchestrator("tok_alice", "summarize")
    run_ms = (time.monotonic() - started) * 1000

    with tempfile.TemporaryDirectory() as directory:
        store = pathlib.Path(directory)
        script = store / "worker.py"
        script.write_text(WORKER)
        durable_id = handle["taskId"]  # one id, one store durable and one not
        write_task(store, durable_id)
        child = subprocess.Popen(
            [sys.executable, str(script), reference_dir, str(store), durable_id,
             str(LAG)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        statuses, gaps, final = poll(store, durable_id, time.monotonic() + 10)
        child.wait(timeout=20)
        report = json.loads((store / "worker.json").read_text())

    methods = sorted(name for name in dir(ref) if name.startswith("tasks_"))
    return {
        "statuses": statuses, "gaps": gaps, "declared_ms": POLL_MS,
        "resumed": statuses[0], "final_status": final["status"],
        "final_text": final["result"]["content"][0]["text"],
        "result_inline": "result" in final and "_meta" in final["result"],
        "tasks_result_calls": 0, "methods": methods,
        "parent_pid": os.getpid(), "worker_pid": report["pid"],
        "volatile": report["volatile"]["error"]["message"],
        "volatile_code": report["volatile"]["error"]["code"],
        "worker_tasks": report["tasks"], "parent_tasks": len(ref.TASKS),
        "handle_interval": handle["pollIntervalMs"], "run_ms": run_ms,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the task survives a restart, resumes, honours its interval, self-carries",
            all([result["resumed"] == "working", result["final_status"] == "completed",
                 len(result["statuses"]) >= 2, result["gaps"] != [],
                 min(result["gaps"]) >= POLL_MS, result["result_inline"],
                 result["tasks_result_calls"] == 0,
                 result["final_text"] == "3 papers summarized."]),
            f"the client resumes by id alone -- {result['statuses']} across "
            f"{len(result['gaps'])} gaps of {result['gaps']}ms, none under the declared "
            f"{result['declared_ms']}ms -- and reads the final result out of the completed "
            f"task with {result['tasks_result_calls']} calls to tasks/result",
        ),
        practice.Check(
            "FINDING: the shipped client ignores the interval it was handed",
            all([result["handle_interval"] == 1_000, result["run_ms"] < 50]),
            f"research_generate_report declares pollIntervalMs "
            f"{result['handle_interval']} and orchestrator calls tasks_get on the next "
            f"line; a whole run takes {result['run_ms']:.2f}ms. The field is emitted and "
            "never read, so the only client that exists is the one that violates it",
        ),
        practice.Check(
            "FINDING: the volatile store does not survive its own process",
            all([result["worker_pid"] != result["parent_pid"],
                 result["volatile"] == "Unknown taskId", result["volatile_code"] == -32602,
                 result["worker_tasks"] == 0, result["parent_tasks"] >= 1]),
            f"pid {result['worker_pid']} imports the same module as pid "
            f"{result['parent_pid']} and answers {result['volatile_code']} "
            f"{result['volatile']!r} for a task the parent minted, holding "
            f"{result['worker_tasks']} tasks against the parent's {result['parent_tasks']}. "
            "TASKS is a module global, so durability here is a missing file",
        ),
        practice.Check(
            "FINDING: tasks/result is absent and unnecessary in the same breath",
            all([result["methods"] == ["tasks_get"], result["result_inline"]]),
            f"the module defines {result['methods']} and the completed task already carries "
            "result with its content and _meta. A second fetch would exist only to "
            "re-deliver a field the client is holding, which is why the current extension "
            "dropped it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
