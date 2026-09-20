"""Exercise 5 — the eras disagree about who owns the request id.

    Design a 2025-11-25 adapter that maps modern MRTR handler output to legacy
    server-initiated elicitation. Keep it isolated from the current handler.

Reading of the exercise: "isolated" is the testable half, so isolation is
defined as something a run can show -- the adapter reaches the server only
through `dispatch`, touches none of its attributes, and the server's
transcript contains no legacy-shaped message. Designing the mapping then turns
up what actually separates the eras, which is not the message shapes but which
side allocates the correlation id.

**ANSWER: the adapter round-trips, and the note is deleted.** Modern
`input_required` becomes a legacy server-initiated `elicitation/create`
request with the adapter's own id; the legacy response becomes modern
`inputResponses` plus the `requestState` the adapter was holding; the retry
completes with `deleted: True`. **3** messages cross the legacy wire and **2**
cross the modern one.

**FINDING: the eras differ by who owns the id.** On the modern path the
*client* allocates the retry id and the server correlates through
`requestState`. On the legacy path the *server* allocates the elicitation id
and the client answers that id. The adapter therefore mints an id the modern
server never sees -- **0** occurrences of the legacy id in the modern
transcript -- and the two numbering spaces never have to agree.

**FINDING: isolation costs the adapter the state the modern design removed.**
The modern server holds **0** fields per in-flight call; the adapter holds the
`requestState` token between the legacy request and the legacy response,
because there is nowhere in the legacy exchange to put it. Re-adding the era
re-adds the per-connection state, in the only component that is allowed to
have it.

**FINDING: the isolation is real and checkable.** The adapter's only contact
with the server is `dispatch`, and the modern transcript contains **0**
messages carrying an `elicitation/create` at top level. Deleting the adapter
leaves the modern flow working unchanged, which is what "keep it isolated"
buys.

Structure: `LegacyAdapter` sits between a legacy client and the lesson's own
`NotesServer`, translating in both directions and recording both transcripts.
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "12-mcp-roots-and-elicitation"
LEGACY_VERSION = "2025-11-25"


class LegacyAdapter:
    """A 2025-11-25 front end for a modern MRTR server, holding no server state."""

    def __init__(self, ref, server):
        self.ref, self.server = ref, server
        self.ids = itertools.count(9_000)
        self.pending = None  # the modern token, which legacy has nowhere to carry
        self.legacy, self.modern = [], []

    def _dispatch(self, request):
        self.modern.append(request)
        return self.server.dispatch(request)

    def begin(self, title="TPS report"):
        """Modern input_required out, legacy server-initiated request in."""
        result = self._dispatch(self.ref.tool_request(1, title))["result"]
        if result["resultType"] != "input_required":
            return result
        self.pending = result["requestState"]
        modern = result["inputRequests"]["delete_choice"]["params"]
        legacy = {"jsonrpc": "2.0", "id": next(self.ids), "method": "elicitation/create",
                  "params": {"message": modern["message"],
                             "requestedSchema": modern["requestedSchema"]}}
        self.legacy.append(legacy)
        return legacy

    def answer(self, legacy_response):
        """Legacy client response in, modern retry out."""
        self.legacy.append(legacy_response)
        retry = self.ref.tool_request(2)
        retry["params"].update({"requestState": self.pending, "inputResponses": {
            "delete_choice": legacy_response["result"]}})
        self.pending = None
        return self._dispatch(retry)


def legacy_client(request):
    """A 2025-11-25 client: it answers the id the server chose."""
    choices = request["params"]["requestedSchema"]["properties"]["note_id"]["enum"]
    return {"jsonrpc": "2.0", "id": request["id"],
            "result": {"action": "accept",
                       "content": {"note_id": choices[-1], "confirm": True}}}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = ref.NotesServer()
    adapter = LegacyAdapter(ref, server)
    before = sorted(server.notes)
    elicit = adapter.begin()
    held_during = adapter.pending is not None  # the token has nowhere legacy to live
    final = adapter.answer(legacy_client(elicit))
    adapter.legacy.append({"jsonrpc": "2.0", "id": elicit["id"], "result": "acknowledged"})

    plain = ref.NotesServer()
    _, unadapted, _ = ref.run_mrtr()
    return {
        "legacy_method": elicit["method"], "legacy_id": elicit["id"],
        "legacy_messages": len(adapter.legacy), "modern_messages": len(adapter.modern),
        "modern_ids": [m["id"] for m in adapter.modern],
        "legacy_id_in_modern": str(elicit["id"]) in str(adapter.modern),
        "deleted": final["result"]["structuredContent"],
        "before": len(before), "after": len(server.notes),
        "server_fields": sorted(vars(plain)),
        "adapter_holds": adapter.pending, "held_during": held_during,
        "elicitation_in_modern": sum("elicitation/create" == m.get("method")
                                     for m in adapter.modern),
        "unadapted": unadapted["result"]["structuredContent"],
        "schema_carried": sorted(elicit["params"]["requestedSchema"]["properties"]),
        "version": LEGACY_VERSION,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the adapter round-trips and the note is deleted",
            all([result["legacy_method"] == "elicitation/create",
                 result["schema_carried"] == ["confirm", "note_id"],
                 result["deleted"]["deleted"] is True,
                 result["before"] == 5, result["after"] == 4,
                 result["legacy_messages"] == 3, result["modern_messages"] == 2]),
            f"modern input_required becomes a legacy {result['legacy_method']!r} carrying "
            f"{result['schema_carried']}, the legacy answer becomes the modern retry, and "
            f"the flow completes with {result['deleted']} -- notes {result['before']} to "
            f"{result['after']}. {result['legacy_messages']} messages cross the legacy wire "
            f"and {result['modern_messages']} the modern one",
        ),
        practice.Check(
            "FINDING: the eras differ by who owns the request id",
            all([result["legacy_id"] >= 9_000, result["modern_ids"] == [1, 2],
                 not result["legacy_id_in_modern"]]),
            f"the adapter allocates {result['legacy_id']} for the server-initiated legacy "
            f"request, while the modern transcript uses {result['modern_ids']} chosen by the "
            "client and correlated through requestState. The legacy id never appears in the "
            "modern transcript, so the two numbering spaces never have to agree",
        ),
        practice.Check(
            "FINDING: isolation costs the adapter the state the modern design removed",
            all([result["server_fields"] == ["authorized_workspaces", "notes", "replay_store"],
                 result["adapter_holds"] is None, result["held_during"]]),
            f"the modern server's fields are {result['server_fields']} -- none of them per "
            "in-flight call -- while the adapter has to hold the requestState between the "
            "legacy request and the legacy response, because the legacy exchange has "
            "nowhere to carry it. Re-adding the era re-adds per-connection state, in the "
            "one component allowed to have it",
        ),
        practice.Check(
            "FINDING: the isolation is real and checkable",
            all([result["elicitation_in_modern"] == 0,
                 result["unadapted"]["deleted"] is True]),
            f"the modern transcript contains {result['elicitation_in_modern']} "
            f"elicitation/create messages, and the lesson's own run_mrtr still completes "
            f"with {result['unadapted']} without the adapter present. The adapter reaches "
            f"the server only through dispatch, so {result['version']} support is additive",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
