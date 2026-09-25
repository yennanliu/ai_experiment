"""Exercise 3 — one open stream blocks every other request.

    Implement an SSE streaming endpoint: `/tasks/{id}/events` that emits state
    changes. What does the client need to do differently?

Reading of the exercise: build the endpoint as a subclass of the lesson's
handler, serve it on the lesson's own server class, and test it the way a
stream is used -- while other clients are also talking to the agent -- since
a stream is the one request that does not end quickly.

**ANSWER: the client holds one connection open and parses frames instead of
sleeping between GETs.** It reads the body line by line, decodes every
`data:` line as JSON, and closes on a terminal state -- so it needs no poll
interval, gets each change once, and makes 1 request where the lesson's
poller makes one GET per 0.1s until the task ends. What it newly has to
handle is the connection itself: a dropped stream is indistinguishable from a quiet one
unless the client knows the terminal states, and the stream never shows
`submitted` -- it opens on `working`, because the task left `submitted`
before the 201 was sent (exercise 1).

**FINDING: on the lesson's server, one open stream blocks everything.**
`run_server` uses `HTTPServer`, which serves one request at a time. While a
stream is open, a request for the Agent Card waits until the task finishes --
about 0.17s here, against well under 0.05s on `ThreadingHTTPServer` -- and a
second subscriber to the same task receives only `['completed']`, having
been connected after the first stream closed. The A2A spec (§3.5.2) requires
that concurrent streams for a task each receive the same events.

**FINDING: the shipped router answers the new path as a missing task.**
Before the override, `GET /tasks/{id}/events` reaches `STORE.get("{id}/events")`
and returns 404 `not found` -- the task-not-found body, not `route not found`
-- because routing is `path.split("/tasks/", 1)[1]`.

**FINDING: the stream is a poll moved into the server.** `TaskStore` has no
condition variable and no callback -- `_run` assigns `state` and returns --
so the endpoint can only re-read the store in a loop. It moves the polling
from the client to the server, and pays for it with a thread per subscriber.

Structure: `streaming()` builds the handler; `subscribe()` is the client;
`contend()` opens a stream, then asks for the card and opens a second stream
while the first is still open.
"""

from __future__ import annotations

import inspect
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import HTTPServer, ThreadingHTTPServer

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "12-a2a-protocol"
TERMINAL = ("completed", "failed", "canceled")


def streaming(ref):
    class Streaming(ref.A2AHandler):
        def do_GET(self):
            if not self.path.endswith("/events"):
                return super().do_GET()
            tid, last = self.path.split("/")[2], None
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            while last not in TERMINAL:
                task = ref.STORE.get(tid)
                if task["state"] != last:
                    last = task["state"]
                    self.wfile.write(f"data: {json.dumps(task)}\n\n".encode())
                    self.wfile.flush()
                time.sleep(0.005)
    return Streaming


def subscribe(url, into):
    with urllib.request.urlopen(url) as resp:
        for line in resp:
            if line.startswith(b"data:"):
                into.append(json.loads(line[5:])["state"])


def contend(ref, server_cls):
    server = server_cls(("localhost", 0), streaming(ref))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://localhost:{server.server_port}"
    try:
        tid = ref.http_json("POST", f"{base}/tasks", {"skill": "review-python",
                                                      "payload": {"code": "def f(): return 1"}})["task_id"]
        first, second = [], []
        a = threading.Thread(target=subscribe, args=(f"{base}/tasks/{tid}/events", first))
        a.start()
        time.sleep(0.03)
        start = time.perf_counter()
        ref.http_json("GET", f"{base}/.well-known/agent.json")
        waited = time.perf_counter() - start
        subscribe(f"{base}/tasks/{tid}/events", second)
        a.join()
    finally:
        server.shutdown()
    return {"first": first, "second": second, "waited": round(waited, 3)}


def shipped_404(ref):
    server = HTTPServer(("localhost", 0), ref.A2AHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        ref.http_json("GET", f"http://localhost:{server.server_port}/tasks/abc/events")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())["error"]
    finally:
        server.shutdown()


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    store = inspect.getsource(ref.TaskStore)
    return {
        "single": contend(ref, HTTPServer), "threaded": contend(ref, ThreadingHTTPServer),
        "server_cls": "HTTPServer((" in inspect.getsource(ref.run_server),
        "shipped": shipped_404(ref),
        "hooks": sum(store.count(k) for k in ("Condition", "notify", "callback", "Event(")),
    }


def verify(result):
    single, threaded = result["single"], result["threaded"]
    return [
        practice.Check(
            "ANSWER: one open connection, parsed frames, no poll interval",
            threaded["first"] == threaded["second"] == ["working", "completed"],
            f"on a threaded server both subscribers receive {threaded['first']} over one "
            "request each; the stream opens on working, never submitted",
        ),
        practice.Check(
            "FINDING: on the lesson's server, one open stream blocks everything",
            all([result["server_cls"], single["first"] == ["working", "completed"],
                 single["second"] == ["completed"],
                 single["waited"] > 0.1, threaded["waited"] < 0.1]),
            f"run_server uses HTTPServer; during a stream the card GET waits "
            f"{single['waited']}s (threaded: {threaded['waited']}s) and a second "
            f"subscriber receives only {single['second']}",
        ),
        practice.Check(
            "FINDING: the shipped router answers the new path as a missing task",
            result["shipped"] == (404, "not found"),
            f"GET /tasks/abc/events on the shipped handler returns {result['shipped']} "
            "-- STORE.get('abc/events'), not 'route not found'",
        ),
        practice.Check(
            "FINDING: the stream is a poll moved into the server",
            result["hooks"] == 0,
            f"TaskStore has {result['hooks']} condition variables, events or callbacks, "
            "so the endpoint can only re-read the store in a loop",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
