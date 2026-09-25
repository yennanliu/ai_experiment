"""Exercise 2 — adding a skill means replacing two module globals.

    Add a second skill to the server (e.g., "summarize"). Update the Agent
    Card. Write a client that picks the skill based on task type.

Reading of the exercise: add the skill without editing the lesson's file --
so everything that has to be overridden or replaced is a measurement of how
extensible the server is -- give the card skills the shape the A2A spec's
AgentSkill has (id, name, description, tags), and let the client choose by
matching the task type against tags rather than by a hardcoded name.

**ANSWER: a registry-driven store, a card with skill objects, and a client
that routes by tag.** `review` tasks go to `review-python` and come back with
its issues; `summarize` tasks go to `summarize` and come back with the first
sentence and a word count. The client never names a skill: it reads the
card, finds the skill whose `tags` contain the task type, and submits.

**FINDING: the server has no seam for a skill -- it takes two global
replacements and an override.** The only dispatch is the `if t["skill"] ==
"review-python"` inside `TaskStore._run`, so a second skill means overriding
`_run` wholesale. And `A2AHandler` reads the module globals `STORE` and
`AGENT_CARD` directly -- it has no reference to either -- so serving the new
store and card means rebinding both in the module. Two agents cannot share
one process.

**FINDING: validation at submit turns a delayed failure into an immediate
400.** Against the shipped handler, an unknown skill is answered 201 and a
task record is created, which reads `failed` only after the worker's 0.2s
sleep. Checking the request against the card's skill ids refuses it with 400
before any task exists: 0 records created against 1.

**FINDING: the shipped card cannot drive this client.** Its `skills` is
`["review-python"]` -- a list of strings, with no description, no tags and no
input shape -- so a client can only succeed by already knowing the skill's
name and that its payload key is `code`, which is what `run_client` does.

Structure: `SkilledStore` overrides `_run` with a registry; `Strict` is the
handler that checks the body's skill against the card before creating a
task; `pick()` is the routing client.
"""

from __future__ import annotations

import copy
import inspect
import json
import threading
import time
import urllib.error
from http.server import HTTPServer

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "12-a2a-protocol"
SKILLS = [
    {"id": "review-python", "name": "Python review", "tags": ["review", "python"],
     "description": "Flags missing def/return in a Python snippet."},
    {"id": "summarize", "name": "Summarize", "tags": ["summarize", "tldr"],
     "description": "Returns the first sentence and a word count."},
]
REGISTRY = {
    "review-python": lambda p: {"issues": [m for k, m in (("def ", "no function definition"),
                                                         ("return", "no return statement"))
                                           if k not in p.get("code", "")]},
    "summarize": lambda p: {"summary": p.get("text", "").split(". ")[0],
                            "words": len(p.get("text", "").split())},
}


def build(ref):
    class SkilledStore(ref.TaskStore):
        def _run(self, tid):
            with self._lock:
                task = self.tasks[tid]
                run = REGISTRY.get(task["skill"])
                task["state"] = "completed" if run else "failed"
                task["artifact"] = {"type": "structured", "data": run(task["payload"])} if run else None

    class Strict(ref.A2AHandler):
        def do_POST(self):
            if self.path != "/tasks":
                return super().do_POST()
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if body.get("skill") not in {s["id"] for s in ref.AGENT_CARD["skills"]}:
                return self._send_json(400, {"error": "unknown skill"})
            tid = ref.STORE.create(body["skill"], body.get("payload", {}))
            return self._send_json(201, {"task_id": tid, "state": ref.STORE.get(tid)["state"]})

    return SkilledStore, Strict


def serve(handler):
    server = HTTPServer(("localhost", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://localhost:{server.server_port}"


def pick(card, task_type):
    return next(s["id"] for s in card["skills"] if task_type in s["tags"])


def submit(ref, base, skill, payload):
    """(HTTP status, task id or None) for one POST /tasks."""
    try:
        return 201, ref.http_json("POST", f"{base}/tasks", {"skill": skill, "payload": payload})["task_id"]
    except urllib.error.HTTPError as exc:
        return exc.code, None


def result_of(ref, base, tid):
    task = ref.http_json("GET", f"{base}/tasks/{tid}")
    while task["state"] not in ("completed", "failed"):
        time.sleep(0.01)
        task = ref.http_json("GET", f"{base}/tasks/{tid}")
    return task["artifact"]["data"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = copy.deepcopy(ref.AGENT_CARD)
    saved = ref.STORE, ref.AGENT_CARD
    store_cls, strict = build(ref)
    ref.STORE, ref.AGENT_CARD = store_cls(), {**shipped, "skills": SKILLS}
    server, base = serve(strict)
    try:
        card = ref.http_json("GET", f"{base}/.well-known/agent.json")
        jobs = {"review": {"code": "x = 1\n"}, "summarize": {"text": "A2A is a protocol. It is new."}}
        routed = {kind: pick(card, kind) for kind in jobs}
        done = {kind: result_of(ref, base, submit(ref, base, routed[kind], payload)[1])
                for kind, payload in jobs.items()}
        before = len(ref.STORE.tasks)
        refused = submit(ref, base, "translate", {})[0]
        created = len(ref.STORE.tasks) - before
    finally:
        server.shutdown()
        ref.STORE, ref.AGENT_CARD = saved
    return {"routed": routed, "done": done, "refused": refused, "created": created,
            "shipped_skills": shipped["skills"], "globals": handler_globals(ref)}


def handler_globals(ref):
    src = inspect.getsource(ref.A2AHandler)
    return {"STORE": src.count("STORE."), "AGENT_CARD": src.count("AGENT_CARD"),
            "self_store": src.count("self.store"),
            "dispatch": inspect.getsource(ref.TaskStore._run).count('t["skill"] ==')}


def verify(result):
    done, g = result["done"], result["globals"]
    return [
        practice.Check(
            "ANSWER: a registry store, a card with skill objects, a client that routes by tag",
            all([result["routed"] == {"review": "review-python", "summarize": "summarize"},
                 done["review"]["issues"] == ["no function definition", "no return statement"],
                 done["summarize"] == {"summary": "A2A is a protocol", "words": 7}]),
            f"the client routes {result['routed']} from the card's tags and gets back "
            f"{done['review']} and {done['summarize']}",
        ),
        practice.Check(
            "FINDING: the server has no seam for a skill",
            all([g["dispatch"] == 1, g["STORE"] >= 2, g["AGENT_CARD"] >= 1, g["self_store"] == 0]),
            f"TaskStore._run holds the only dispatch ({g['dispatch']} skill comparison) and "
            f"A2AHandler touches the module-global STORE {g['STORE']} times and AGENT_CARD "
            f"{g['AGENT_CARD']} with no reference of its own -- both had to be rebound",
        ),
        practice.Check(
            "FINDING: validation at submit turns a delayed failure into an immediate 400",
            result["refused"] == 400 and result["created"] == 0,
            f"an unknown skill is refused with {result['refused']} and {result['created']} "
            "task records, where the shipped handler answers 201 and fails it 0.2s later",
        ),
        practice.Check(
            "FINDING: the shipped card cannot drive this client",
            result["shipped_skills"] == ["review-python"],
            f"its skills are {result['shipped_skills']}: strings with no description, "
            "tags or input shape, so only a client that already knows the name and the "
            "'code' payload key can call it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
