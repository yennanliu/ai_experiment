"""Exercise 6 -- with cursor state in process memory, alternating replicas answer 0 of 75 page reads correctly.

    Route consecutive calls to alternating replicas. Replace hidden process memory with an explicit shared handle wherever the workflow needs persistence.

Reading of the exercise: two replicas are two `build_readonly_server()`
objects behind a round-robin router, 100 consecutive calls. The lesson's
own tools need no cross-call state, so the workflow that does is added with
the lesson's `ToolSchema`/`register`: `rows.open` returns a cursor handle and
`rows.next` reads the next page of 2 rows from 6. It is run 25 times (open +
3 pages = 100 calls) twice: once with the cursor kept in each replica's own
dict (hidden process memory), once in one store both replicas read (the
explicit shared handle; an opaque UUID4, as the spec's stateful-tools note
advises, modelcontextprotocol.io/specification/2026-07-28/server/tools, read
2026-09-29).

**ANSWER: with the shared handle, 75 of 75 page reads are correct under
alternating routing.** With hidden memory, 0 of 75 are: 50 land on the
replica that never saw the cursor and return `isError` "unknown handle", and
the other 25 return page 1 where page 2 was due, because each replica
advanced only its own copy. Pin every workflow to one replica (affinity) and
the hidden version is 75 of 75 again: it was correct only by affinity.

**The lesson's own tools are stateless.** 100 alternating
`postgres.readonly`/`s3.list` calls give results byte-identical to one
replica's, and both replicas' `tools/list` bytes are identical.

**FINDING: the TypeScript model keeps exactly this kind of hidden state.**
`incidents_ack` sets `inc.acked = true` on the process-local incident map
from `makeIncidents()`, so behind two replicas an ack on one is invisible to
`incidents_get` on the other. The Python model has no stateful tool at all.
"""

from __future__ import annotations

import json
import time
import uuid

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "13-mcp-server-with-registry"
ROWS = [[i] for i in range(6)]
OBJ = {"type": "object", "additionalProperties": False}


def add_cursor_tools(ref, server, store):
    """rows.open mints a handle; rows.next reads 2 rows and advances the cursor in `store`."""

    def open_(args):
        handle = str(uuid.uuid4())
        store[handle] = 0
        return {"handle": handle}

    def next_(args):
        if args["handle"] not in store:
            raise LookupError("unknown handle")
        at = store[args["handle"]]
        store[args["handle"]] = at + 2
        return {"rows": ROWS[at : at + 2]}

    server.register(ref.ToolSchema("rows.open", "rows:read", False, "Open a cursor.", OBJ), open_)
    server.register(ref.ToolSchema("rows.next", "rows:read", False, "Read the next page.", OBJ), next_)
    return server


def replicas(ref, shared):
    store = {}
    return [add_cursor_tools(ref, ref.build_readonly_server(), store if shared else {}) for _ in range(2)]


def run_workflows(ref, reps, token, sticky=False):
    tally, n = {"correct": 0, "error": 0, "wrong": 0}, 0
    for w in range(25):
        pick = (lambda: reps[w % 2]) if sticky else (lambda: reps[n % 2])
        out = ref.dispatch(pick(), token, "rows.open", {}, ref.request_meta(), [])
        n += 1
        handle = out["structuredContent"]["handle"]
        for page in range(3):
            out = ref.dispatch(pick(), token, "rows.next", {"handle": handle}, ref.request_meta(), [])
            n += 1
            got = None if out["isError"] else out["structuredContent"]["rows"]
            tally["error" if got is None else "correct" if got == ROWS[2 * page : 2 * page + 2] else "wrong"] += 1
    return tally, n


def stateless(ref, token):
    reps, single = [ref.build_readonly_server() for _ in range(2)], ref.build_readonly_server()
    calls = [("postgres.readonly", {"sql": f"SELECT {i}"}) if i % 2 else ("s3.list", {"bucket": f"b{i}"})
             for i in range(100)]
    same = sum(
        json.dumps(ref.dispatch(reps[i % 2], token, *c, ref.request_meta(), []))
        == json.dumps(ref.dispatch(single, token, *c, ref.request_meta(), []))
        for i, c in enumerate(calls)
    )
    lists = {json.dumps(r.tools_list(ref.request_meta())) for r in reps}
    return same, len(lists)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    probe = ref.build_readonly_server()
    scopes = frozenset({"rows:read", "postgres:query:readonly", "s3:list"})
    token = ref.Token("u42", probe.trusted_issuer, probe.url, scopes, time.time() + 3600)
    ts = (parity.lesson_dir(PHASE, LESSON) / "code" / "ts" / "src" / "tools.ts").read_text()
    py = (parity.lesson_dir(PHASE, LESSON) / "code" / "main.py").read_text()
    return {
        "shared": run_workflows(ref, replicas(ref, True), token),
        "hidden": run_workflows(ref, replicas(ref, False), token),
        "hidden_sticky": run_workflows(ref, replicas(ref, False), token, sticky=True),
        "stateless": stateless(ref, token),
        "ts_mutation": ts.count("inc.acked = true"),
        "py_store_writes": py.count("store["),
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: an explicit shared handle keeps 75/75 page reads correct across alternating replicas",
            r["shared"] == ({"correct": 75, "error": 0, "wrong": 0}, 100)
            and r["hidden"] == ({"correct": 0, "error": 50, "wrong": 25}, 100)
            and r["hidden_sticky"] == ({"correct": 75, "error": 0, "wrong": 0}, 100)
            and r["stateless"] == (100, 1),
            f"shared {r['shared']}; hidden memory {r['hidden']}; hidden memory with affinity {r['hidden_sticky']}; "
            f"lesson tools: {r['stateless'][0]}/100 alternating results byte-identical to one replica, "
            f"{r['stateless'][1]} distinct tools/list encoding(s)",
        ),
        practice.Check(
            "FINDING: the TypeScript model's incidents_ack mutates process-local memory",
            (r["ts_mutation"], r["py_store_writes"]) == (1, 0),
            f"'inc.acked = true' in tools.ts: {r['ts_mutation']}; store writes in main.py: {r['py_store_writes']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
