"""Exercise 5 — affinity is neither necessary nor sufficient for the handle.

    Add an explicit workflow handle to the ping tool. Bind it to an
    authorization subject without using connection affinity.

Reading of the exercise: "without using connection affinity" is a negative, and
a negative is only shown by building the thing it forbids alongside the thing
it asks for. So the same `Replica` runs twice -- once over a shared store and
once over per-replica dicts -- and the second one reproduces the five-step
failure the lesson describes. The binding is then tested from both sides: the
wrong subject on the *right* replica, and the right subject on the *wrong* one.

**ANSWER: `ping` mints an unguessable handle and any replica continues it.**
Replica `a` returns a **32**-character `secrets.token_urlsafe` handle at
`turns=1`; replica `b`, a different process with no memory of the call, takes
the same handle to `turns=2`. Nothing about the first connection is consulted.

**FINDING: per-replica storage reproduces the lesson's failure exactly.** The
same `Replica` code over two separate dicts answers **-32004 Unknown workflow
handle** on `b` — step 4 of the mechanical failure, reached by changing one
argument. The handle is not what breaks; the place it is kept is.

**FINDING: the binding is the subject, and affinity is neither necessary nor
sufficient.** A second subject presenting the same handle to the **minting**
replica is refused **-32003**, so staying on the connection buys nothing; the
owning subject is served by **both** replicas, so leaving it costs nothing.
Authorization is checked on every use rather than established once.

**FINDING: the handle has to be an argument, because the transport has no slot
for it.** A real POST to the reference server comes back with
`Mcp-Session-Id: None` — the revision mints no session header and
`http_headers_for` sends none — so there is nowhere in the envelope to hide it.
Expiry then only makes sense on the record: past its TTL the handle is
**-32005**, while the connection it was minted on is irrelevant either way.

Structure: `Replica` is one server process -- the lesson's own `dispatch` for
every other method, plus a `ping` that mints, authorizes and advances a record
in whatever store it was handed.
"""

from __future__ import annotations

import contextlib
import io
import secrets

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "09-mcp-transports"
OWNER, INTRUDER = "user:alice", "user:mallory"
TTL = 300.0
UNKNOWN, FORBIDDEN, EXPIRED = -32004, -32003, -32005


class Replica:
    """One server process: the lesson's dispatch, plus a handle-minting ping."""

    def __init__(self, ref, store, name):
        self.ref, self.store, self.name = ref, store, name

    def call(self, message, subject, now=0.0):
        params = message["params"]
        if message.get("method") != "tools/call" or params.get("name") != "ping":
            return self.ref.dispatch(message)
        handle = params.get("arguments", {}).get("workflow")
        if handle is None:
            handle = secrets.token_urlsafe(24)
            self.store[handle] = {"subject": subject, "turns": 0, "expires": now + TTL}
        record = self.store.get(handle)
        if record is None:
            return self.ref.rpc_error(message["id"], UNKNOWN, "Unknown workflow handle")
        if record["subject"] != subject:
            return self.ref.rpc_error(message["id"], FORBIDDEN, "Handle bound to another subject")
        if now >= record["expires"]:
            return self.ref.rpc_error(message["id"], EXPIRED, "Workflow handle expired")
        record["turns"] += 1
        return {"jsonrpc": "2.0", "id": message["id"], "result": self.ref.complete(
            {"content": [{"type": "text", "text": "pong"}], "isError": False,
             "workflow": handle, "turns": record["turns"], "servedBy": self.name})}


def ping(ref, request_id, handle=None):
    arguments = {} if handle is None else {"workflow": handle}
    return ref.make_request(request_id, "tools/call", {"name": "ping", "arguments": arguments})


def outcome(response):
    """(handle, turns, servedBy) on success, or the error code."""
    if "error" in response:
        return {"code": response["error"]["code"]}
    result = response["result"]
    return {"handle": result["workflow"], "turns": result["turns"],
            "served_by": result["servedBy"], "code": None}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shared = {}
    alpha, beta = Replica(ref, shared, "a"), Replica(ref, shared, "b")
    minted = outcome(alpha.call(ping(ref, 1), OWNER))
    handle = minted["handle"]
    continued = outcome(beta.call(ping(ref, 2, handle), OWNER))

    local_a, local_b = Replica(ref, {}, "a"), Replica(ref, {}, "b")
    stranded = outcome(local_a.call(ping(ref, 3), OWNER))
    lost = outcome(local_b.call(ping(ref, 4, stranded["handle"]), OWNER))

    server = ref.serve()
    with contextlib.redirect_stderr(io.StringIO()):  # the handler logs every request
        probe = ref.make_request(9, "tools/list")
        sent = ref.http_headers_for(probe)
        _, headers, _ = ref.post(f"http://127.0.0.1:{server.server_port}/mcp", probe, sent)
    server.shutdown()
    server.server_close()
    return {
        "minted": minted, "continued": continued, "handle_length": len(handle),
        "intruder_on_minter": outcome(alpha.call(ping(ref, 5, handle), INTRUDER)),
        "owner_on_other": outcome(beta.call(ping(ref, 6, handle), OWNER)),
        "stranded": stranded, "lost": lost,
        "expired": outcome(alpha.call(ping(ref, 7, handle), OWNER, now=TTL + 1)),
        "session_header": headers.get("Mcp-Session-Id"),
        "sent_session": sent.get("Mcp-Session-Id"),
        "unrelated": "result" in alpha.call(ref.make_request(8, "tools/list"), OWNER),
    }


def verify(result):
    minted, continued = result["minted"], result["continued"]
    return [
        practice.Check(
            "ANSWER: ping mints an unguessable handle and a different replica continues it",
            all([minted["turns"] == 1, minted["served_by"] == "a",
                 continued["turns"] == 2, continued["served_by"] == "b",
                 continued["handle"] == minted["handle"], result["handle_length"] == 32,
                 result["unrelated"]]),
            f"replica {minted['served_by']} returns a {result['handle_length']}-character "
            f"token_urlsafe handle at turns={minted['turns']}, and replica "
            f"{continued['served_by']} -- a different process with no memory of that call -- "
            f"takes the same handle to turns={continued['turns']}. Every other method still "
            "goes to the lesson's own dispatch",
        ),
        practice.Check(
            "FINDING: per-replica storage reproduces the lesson's five-step failure",
            all([result["stranded"]["turns"] == 1, result["lost"]["code"] == UNKNOWN]),
            f"the same Replica code over two separate dicts mints at "
            f"turns={result['stranded']['turns']} on a and answers "
            f"{result['lost']['code']} on b -- step 4 of the mechanical failure, reached by "
            "changing one argument. The handle is not what breaks; the place it is kept is",
        ),
        practice.Check(
            "FINDING: the binding is the subject, so affinity is neither necessary nor sufficient",
            all([result["intruder_on_minter"]["code"] == FORBIDDEN,
                 result["owner_on_other"]["code"] is None,
                 result["owner_on_other"]["turns"] == 3]),
            f"a second subject presenting the handle to the minting replica is refused "
            f"{result['intruder_on_minter']['code']}, so staying on the connection buys "
            f"nothing; the owning subject reaches turns={result['owner_on_other']['turns']} "
            "on the other replica, so leaving it costs nothing. Authorization is checked on "
            "every use rather than established once",
        ),
        practice.Check(
            "FINDING: the handle must be an argument, because the envelope has no slot for it",
            all([result["session_header"] is None, result["sent_session"] is None,
                 result["expired"]["code"] == EXPIRED]),
            f"a real POST to the reference server answers with Mcp-Session-Id "
            f"{result['session_header']} and http_headers_for sends "
            f"{result['sent_session']}, so the revision has nowhere in the envelope to hide "
            f"state. Expiry therefore lives on the record -- past its TTL the handle is "
            f"{result['expired']['code']} -- and the connection it was minted on is "
            "irrelevant either way",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
