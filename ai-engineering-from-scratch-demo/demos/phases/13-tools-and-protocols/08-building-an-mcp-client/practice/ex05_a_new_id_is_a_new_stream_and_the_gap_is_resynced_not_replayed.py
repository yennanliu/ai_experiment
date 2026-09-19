"""Exercise 5 — a new id is a new stream, and the gap is resynced, not replayed.

    Add a finite `subscriptions/listen` simulator. On stream loss, re-listen
    with a new request id and refetch tools.

Reading of the exercise: "finite" is the whole design. A real listen is an
unbounded stream, so the simulator yields a fixed number of events and then
dies, twice, which makes loss a scheduled event rather than a fault to wait
for. The tool list changes exactly while no stream is open, because that is the
only arrangement in which the refetch is doing work rather than confirming what
the events already said.

**ANSWER: three listens, three distinct ascending ids, three refetches, and the
registry tracks the server.** Ids **[2, 4, 6]**, strictly increasing and
allocated by the client's own `_new_id`; the registry goes
`['notes_list']` → `['notes_delete', 'notes_list']` →
`['notes_archive', 'notes_delete', 'notes_list']`, matching the server at every
step. **6** events arrive across the three streams.

**FINDING: the client has nowhere to put a stream, and the type signature says
so.** `MultiServerClient` has **0** members matching listen, subscribe, stream
or notification, and `_send` is annotated `dict[str, Any] | None` — one message
per request, so a multi-event stream cannot pass through `_request` at all. The
lesson's own `ModernFakeServer` answers `subscriptions/listen` with **-32601**.
Everything this exercise builds is in the prose and not in the code.

**FINDING: the new id is load-bearing, because the id is the only correlation
key.** Events carry the listen id in `_meta`. Replaying a stream-1 event while
stream 3 is live is rejected **0 of 1** times accepted; had the re-listen reused
id 2, the same stale event is accepted **1 of 1**. Reusing the id does not
resume the stream — it makes the dead stream's traffic indistinguishable from
the live one's.

**FINDING: the refetch recovers the state and not the events.** The server
dropped **2** `notifications/tools/list_changed` because no stream was open to
carry them, and the client received **0** of them — every one of the 6 events it
did receive was a `resources/updated` from a live stream. The client's tool list
is nonetheless correct, because `tools/list` reports the current state and the
notification reported a transition. That is what "modern streams do not resume
with `Last-Event-ID`" costs: the transitions in the gap are unrecoverable, and
recovery is a resync that does not need them.

Structure: `SubscribableServer` wraps the lesson's `ModernFakeServer` and adds
the listen method; `run_subscription` is the client side, built out of the
lesson's own `modern_request` / `decode_rpc_response` / `_new_id`.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "08-building-an-mcp-client"
LISTEN, ACK = "subscriptions/listen", "notifications/subscriptions/acknowledged"
LIST_CHANGED, UPDATED = ("notifications/tools/list_changed",
                         "notifications/resources/updated")
LISTEN_ID, EVENTS = "io.modelcontextprotocol/listenRequestId", 2
FILTER = ["tools/list_changed", "resources/updated"]
GENERATIONS = [["notes_list"], ["notes_delete", "notes_list"],
               ["notes_archive", "notes_delete", "notes_list"]]


class SubscribableServer:
    """The lesson's `ModernFakeServer` plus a finite `subscriptions/listen`.

    Each listen is acknowledged and backed by a generator of `EVENTS` events
    tagged with that listen's id. The first two streams then advance the tool
    list and die, so the notification the change would have carried has no open
    stream to go to and lands in `dropped` instead.
    """

    def __init__(self, ref, name):
        self.ref, self.name, self.generation = ref, name, 0
        self.listen_ids, self.streams, self.dropped = [], {}, []
        self.server = ref.ModernFakeServer(name, self._tools(0))

    def _tools(self, generation):
        return [self.ref.tool(n, f"{n} on {self.name}.") for n in GENERATIONS[generation]]

    def _events(self, listen_id, lossy):
        for index in range(EVENTS):
            yield {"jsonrpc": "2.0", "method": UPDATED, "params": {
                "uri": f"notes://event/{index}", "_meta": {LISTEN_ID: listen_id}}}
        if lossy:  # the tool list changes with nothing listening to hear it
            self.generation += 1
            self.server.tools = self._tools(self.generation)
            self.dropped.append({"jsonrpc": "2.0", "method": LIST_CHANGED, "params": {}})
            raise ConnectionError(f"stream {listen_id} lost")

    def __call__(self, message, timeout_ms=None):
        if message.get("method") != LISTEN:
            return self.server(message, timeout_ms)
        self.ref.validate_modern_request(message, self.server.supported_versions)
        self.listen_ids.append(request_id := message["id"])
        self.streams[request_id] = self._events(request_id, self.generation + 1 < len(GENERATIONS))
        return {"jsonrpc": "2.0", "id": request_id, "result": self.ref.complete(
            self.server.server_info, {"acknowledged": ACK})}


def correlates(event, listen_id):   # an event belongs to a stream by its id alone
    return event.get("params", {}).get("_meta", {}).get(LISTEN_ID) == listen_id


def run_subscription(ref, client, peer, rounds=4):
    """Listen, resync, drain; on loss re-listen with a new id and refetch."""
    ids, registries, received, losses = [], [], [], 0
    for _ in range(rounds):
        request_id = client._new_id()
        message = ref.modern_request(request_id, LISTEN, {"filter": list(FILTER)},
                                     peer.protocol_version, client.client_capabilities)
        kind, payload = ref.decode_rpc_response(peer.transport(message, None), request_id)
        if kind != "result" or payload.get("acknowledged") != ACK:
            raise RuntimeError(f"{peer.name}: listen was not acknowledged")
        ids.append(request_id)
        client.discover_tools()  # a new stream never resumes the old one
        client.merge()
        registries.append(sorted(client.registry))
        try:
            for event in peer.transport.streams[request_id]:
                if correlates(event, request_id):
                    received.append(event["method"])
        except ConnectionError:  # the stream is gone
            losses += 1
            continue
        break
    return ids, registries, received, losses


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    transport = SubscribableServer(ref, "notes")
    client = ref.MultiServerClient()
    client.add_server("notes", transport)
    client.connect_all()
    ids, registries, received, losses = run_subscription(ref, client, client.peers["notes"])
    stale = {"params": {"_meta": {LISTEN_ID: ids[0]}}}  # replayed from stream 1
    refused = transport.server(ref.modern_request(  # the unwrapped lesson server
        1, LISTEN, {}, ref.PROTOCOL_VERSION, ref.CLIENT_CAPABILITIES), None)
    return {
        "ids": ids, "server_ids": transport.listen_ids, "losses": losses,
        "registries": registries, "dropped": len(transport.dropped),
        "server_tools": sorted(t["name"] for t in transport.server.tools),
        "methods": sorted(received), "refused": refused["error"]["code"],
        "stale": [int(correlates(stale, ids[-1])), int(correlates(stale, ids[0]))],
        "send_returns": ref.MultiServerClient._send.__annotations__.get("return"),
        "members": [n for n in dir(ref.MultiServerClient)
                    if any(k in n.lower() for k in ("listen", "subscri", "stream", "notif"))],
    }


def verify(result):
    ids, methods, seen = result["ids"], result["methods"], result["registries"]
    return [
        practice.Check(
            "ANSWER: three listens with distinct ascending ids, and three refetches that track the server",
            all([len(ids) == 3, len(set(ids)) == 3, ids == sorted(ids), ids == result["server_ids"],
                 result["losses"] == 2, seen == GENERATIONS, len(methods) == 6,
                 result["server_tools"] == GENERATIONS[-1]]),
            f"{result['losses']} lost streams, so the loop listens {len(ids)} times with ids "
            f"{ids} -- distinct, ascending, from the client's own _new_id -- refetching after "
            f"each. The registry goes {' -> '.join(str(r) for r in seen)}, tracking the "
            f"server at every step, across {len(methods)} delivered events",
        ),
        practice.Check(
            "FINDING: the client has nowhere to put a stream, and the type signature says so",
            all([result["members"] == [], result["refused"] == -32601,
                 result["send_returns"] == "dict[str, Any] | None"]),
            f"MultiServerClient has {len(result['members'])} members matching listen, "
            f"subscribe, stream or notification; _send is annotated {result['send_returns']!r}, "
            f"one message per request, so a stream cannot pass through _request; and the "
            f"unwrapped ModernFakeServer answers {LISTEN} with {result['refused']}",
        ),
        practice.Check(
            "FINDING: the new id is load-bearing, because the id is the only correlation key",
            result["stale"] == [0, 1],
            f"events carry the listen id in _meta, so replaying a stream-1 event against the "
            f"live stream is accepted {result['stale'][0]} of 1 times, and {result['stale'][1]} "
            f"of 1 had the re-listen reused id {ids[0]}. Reusing the id does not resume the "
            "stream -- it makes dead traffic indistinguishable from live",
        ),
        practice.Check(
            "FINDING: the refetch recovers the state and not the events",
            all([result["dropped"] == 2, methods.count(LIST_CHANGED) == 0,
                 set(methods) == {UPDATED}, seen[-1] == result["server_tools"]]),
            f"the server dropped {result['dropped']} {LIST_CHANGED} with no stream open and "
            f"the client received {methods.count(LIST_CHANGED)}; every delivered event was "
            f"{sorted(set(methods))}. The tool list is right anyway, because tools/list "
            "reports state where the notification reported a transition -- the price of no "
            "Last-Event-ID resume",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
