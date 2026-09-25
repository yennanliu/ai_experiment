"""Exercise 1 — the submitted state is already gone when the 201 says it.

    Run `code/main.py`. Confirm the client discovers the server and receives
    the correct artifact.

Reading of the exercise: run the lesson's own `run_client` against the
lesson's own handler, on a free port rather than 8765 so the check cannot
collide with anything -- which immediately shows how much of the client's
"discovery" is literal -- then check the artifact against the input by hand.

**ANSWER: it discovers the server and gets an artifact, but not a correct
one.** The client reads the card, posts to the card's endpoint, polls to
`completed` and receives `{'issues': ['no return statement', 'no function
definition'], 'lines': 3}`. The two issues are right for `x = 1\nprint(x)\n`.
The line count is not: the code is 2 lines, and `code.count("\n") + 1`
counts the empty string after the trailing newline as a third.

**FINDING: the 201's `submitted` is a string literal, and no client can ever
see it.** `do_POST` answers `{"task_id": tid, "state": "submitted"}` without
reading the store. `create()` starts the worker thread before returning, and
the worker's first act is `state = "working"`: in 50 of 50 creates on the
machine this was written on, the store already reads `working` by the time
`create` returns (the check allows 45, since it is a thread race). The
state the response reports is one the task has already left.

**FINDING: discovery supplies 1 of the client's 3 URLs, and that one is a
literal too.** `run_client` hardcodes the card URL, takes the submit URL from
`card["endpoints"]["tasks"]`, and hardcodes the poll URL. The card's endpoint
is itself the literal `http://localhost:8765/tasks`, so serving the handler
on any other port sends the discovered submit to port 8765 -- every one of the
client's 5 requests had to be rewritten to run here.

**FINDING: an unknown skill is accepted, then failed.** Posting
`{"skill": "summarize"}` returns 201 `submitted`; the task only turns `failed`
after the 0.2s the worker sleeps. The card lists the one skill, and nothing
compares the request against it.

Structure: `serve()` runs the reference `A2AHandler` on port 0; `run_client()`
runs the reference client with its `http_json` wrapped to redirect 8765.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import threading
import time
from http.server import HTTPServer

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "12-a2a-protocol"
CODE = "x = 1\nprint(x)\n"


def serve(ref):
    server = HTTPServer(("localhost", 0), ref.A2AHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"localhost:{server.server_port}"


def run_client(ref, host):
    """The lesson's client, with every localhost:8765 it uses redirected and counted."""
    original, rewritten = ref.http_json, []

    def redirect(method, url, body=None):
        rewritten.append(url)
        return original(method, url.replace("localhost:8765", host), body)

    ref.http_json, out = redirect, io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            ref.run_client()
    finally:
        ref.http_json = original
    return out.getvalue(), rewritten


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server, host = serve(ref)
    try:
        transcript, urls = run_client(ref, host)
        tid = ref.STORE.create("review-python", {"code": CODE})
        while ref.STORE.get(tid)["state"] != "completed":
            time.sleep(0.02)
        states = [ref.STORE.get(ref.STORE.create("review-python", {"code": CODE}))["state"]
                  for _ in range(50)]
        unknown = ref.http_json("POST", f"http://{host}/tasks", {"skill": "summarize"})
        time.sleep(0.4)
        later = ref.http_json("GET", f"http://{host}/tasks/{unknown['task_id']}")
    finally:
        server.shutdown()
    client = inspect.getsource(ref.run_client)
    return {
        "transcript": transcript, "artifact": ref.STORE.get(tid)["artifact"]["data"],
        "real_lines": len(CODE.splitlines()), "working": states.count("working"),
        "literal": '"state": "submitted"}' in inspect.getsource(ref.A2AHandler.do_POST),
        "rewritten": sum("localhost:8765" in u for u in urls), "requests": len(urls),
        "from_card": client.count('card["endpoints"]'),
        "hardcoded": client.count("http://localhost:8765"),
        "unknown": (unknown["state"], later["state"]),
    }


def verify(result):
    art = result["artifact"]
    return [
        practice.Check(
            "ANSWER: it discovers the server and gets an artifact, but not a correct one",
            all(["state=completed" in result["transcript"], "name=code-review-agent"
                 in result["transcript"], len(art["issues"]) == 2,
                 art["lines"] == 3, result["real_lines"] == 2]),
            f"the client reaches completed with issues {art['issues']} -- right -- and "
            f"lines={art['lines']} for {result['real_lines']}-line code, because "
            "count('\\n') + 1 counts the empty tail after the trailing newline",
        ),
        practice.Check(
            "FINDING: the 201's submitted is a literal no client can ever see",
            result["literal"] and result["working"] >= 45,
            f"do_POST writes state 'submitted' without reading the store, and in "
            f"{result['working']} of 50 creates the store already reads 'working' when "
            "create() returns -- the worker thread starts before it",
        ),
        practice.Check(
            "FINDING: discovery supplies 1 of the client's 3 URLs, and it is a literal too",
            all([result["from_card"] == 1, result["hardcoded"] == 2,
                 result["rewritten"] == result["requests"] == 5]),
            f"run_client takes {result['from_card']} URL from the card and hardcodes "
            f"{result['hardcoded']}; the card's own endpoint is localhost:8765, so all "
            f"{result['rewritten']} of {result['requests']} requests needed rewriting "
            "to reach a server on another port",
        ),
        practice.Check(
            "FINDING: an unknown skill is accepted, then failed",
            result["unknown"] == ("submitted", "failed"),
            f"POST with skill 'summarize' answers {result['unknown'][0]} and the task "
            f"is {result['unknown'][1]} after the worker's sleep -- the card's skill "
            "list is never consulted",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
