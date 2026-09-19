"""Exercise 4 — a reissue is not a resume, because the old id names nothing.

    Break the finite listen stream before its final response. Reissue it with
    a new JSON-RPC id and refetch tools.

Reading of the exercise: `ref.post` reads a body to EOF, so it cannot break
anything; the break has to be a client that stops reading and closes the
socket mid-stream. Having done that, the interesting question is not whether
the reissue works -- it plainly will -- but what the *old* id is worth
afterwards, since that is the difference between a reissue and a resume.

**ANSWER: the break costs the final response, and the reissue replaces the
whole stream.** Closing after the acknowledgement takes **1** of **3** frames
and never sees the JSON-RPC result. `listen-2` delivers all three --
`acknowledged`, `tools/list_changed`, and the result -- and the refetched
`tools/list` answers `['ping']`. Every frame carries
`io.modelcontextprotocol/subscriptionId` equal to its own listen id.

**FINDING: there is nothing to resume from.** The stream writes **0** `id:`
SSE fields, so `Last-Event-ID` has no anchor to send, exactly as the lesson
says it should not. The broken stream's frames are tagged `listen-1` and the
new ones `listen-2`, so the two can never be confused, but neither can the
second continue the first.

**FINDING: a reissue is not a resume because the server keeps no
subscription.** `subscription_messages` is a pure function of the request --
called twice on the same message it returns equal output -- and the module
holds no registry keyed by a listen id, only the `_meta` key constant and that
function. The old id names nothing on the server, so "with a new request id" is
not a courtesy to the server; it is the only thing a client can do.

**FINDING: the refetch is load-bearing, because the notification carries no
tools.** The `tools/list_changed` frame's params hold `_meta` and nothing else;
the names arrive only from `tools/list`. And the acknowledgement echoes the
*accepted* filter rather than the requested one -- `{"toolsListChanged": true,
"bogus": true}` comes back as `{"toolsListChanged": true}` -- so a client is
told what it subscribed to and still has to ask separately what changed.

Structure: `listen` is the partial reader -- it POSTs, walks `data:` lines, and
closes the connection once `stop_after` frames have arrived.
"""

from __future__ import annotations

import contextlib
import http.client
import io
import json

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "09-mcp-transports"
FILTER = {"toolsListChanged": True}
ACK = "notifications/subscriptions/acknowledged"
CHANGED = "notifications/tools/list_changed"


def listen(ref, port, message, stop_after=None):
    """POST a listen and read SSE frames, closing early after `stop_after`."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    connection.request("POST", "/mcp", json.dumps(message), dict(ref.http_headers_for(message)))
    response = connection.getresponse()
    frames, sse_ids = [], 0
    for line in response.fp:
        sse_ids += line.startswith(b"id:")
        if line.startswith(b"data: "):
            frames.append(json.loads(line[6:]))
            if stop_after is not None and len(frames) >= stop_after:
                break
    connection.close()
    return frames, sse_ids


def tagged(frame):
    """The subscription id a frame carries, wherever the frame keeps its _meta."""
    meta = frame.get("params", frame.get("result", {})).get("_meta", {})
    return meta.get("io.modelcontextprotocol/subscriptionId")


def summary(frames):
    """What one stream delivered, once it has stopped delivering it."""
    return {"count": len(frames), "tags": [tagged(f) for f in frames],
            "methods": [f.get("method") for f in frames],
            "final_id": next((f["id"] for f in frames if "result" in f), None)}


def frame(frames, method):
    """The one frame carrying `method`, which every stream sends exactly once."""
    return next(f for f in frames if f.get("method") == method)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = ref.serve()
    with contextlib.redirect_stderr(io.StringIO()):  # the handler logs every request
        first = ref.make_request("listen-1", "subscriptions/listen", {"notifications": FILTER})
        second = ref.make_request("listen-2", "subscriptions/listen", {"notifications": FILTER})
        broken, _ = listen(ref, server.server_port, first, stop_after=1)
        whole, sse_ids = listen(ref, server.server_port, second)
        refetch = ref.make_request(3, "tools/list")
        url = f"http://127.0.0.1:{server.server_port}/mcp"
        _, _, listing = ref.post(url, refetch, ref.http_headers_for(refetch))
    server.shutdown()
    server.server_close()
    return {
        "broken": summary(broken), "whole": summary(whole), "sse_ids": sse_ids,
        "tools": [tool["name"] for tool in listing["result"]["tools"]],
        "changed_params": sorted(frame(whole, CHANGED)["params"]),
        "acknowledged": frame(whole, ACK)["params"]["notifications"],
        "accepted": ref.accepted_filter({"toolsListChanged": True, "bogus": True}),
        "pure": ref.subscription_messages(second) == ref.subscription_messages(second),
        "registry": [n for n in dir(ref) if "subscription" in n.lower()],
    }


def verify(result):
    broken, whole = result["broken"], result["whole"]
    return [
        practice.Check(
            "ANSWER: the break costs the final response, and the reissue replaces the whole stream",
            all([broken["count"] == 1, broken["final_id"] is None,
                 broken["tags"] == ["listen-1"], whole["count"] == 3,
                 whole["methods"] == [ACK, CHANGED, None],
                 whole["tags"] == ["listen-2"] * 3,
                 whole["final_id"] == "listen-2", result["tools"] == ["ping"]]),
            f"closing after the acknowledgement takes {broken['count']} of {whole['count']} "
            f"frames and never sees the result. listen-2 delivers all three, "
            f"{[m or 'result' for m in whole['methods']]}, each tagged "
            f"{whole['tags'][0]!r}, and the refetched tools/list answers {result['tools']}",
        ),
        practice.Check(
            "FINDING: there is nothing to resume from -- the stream writes no SSE event ids",
            all([result["sse_ids"] == 0, broken["tags"] != whole["tags"][:1]]),
            f"the stream writes {result['sse_ids']} 'id:' fields, so Last-Event-ID has no "
            f"anchor to send, exactly as the lesson says it should not. The broken frames are "
            f"tagged {broken['tags']} and the new ones {whole['tags'][:1]}, so "
            "the two cannot be confused -- and neither can the second continue the first",
        ),
        practice.Check(
            "FINDING: a reissue is not a resume, because the server keeps no subscription",
            all([result["pure"], result["registry"] ==
                 ["SUBSCRIPTION_ID_KEY", "subscription_messages"]]),
            f"subscription_messages returns equal output for the same message twice, and the "
            f"module's only subscription-shaped names are {result['registry']} -- a key "
            "constant and that function. Nothing is keyed by a listen id, so the old id names "
            "nothing and a new one is the only thing a client can send",
        ),
        practice.Check(
            "FINDING: the refetch is load-bearing, because the notification carries no tools",
            all([result["changed_params"] == ["_meta"],
                 result["acknowledged"] == {"toolsListChanged": True},
                 result["accepted"] == {"toolsListChanged": True}]),
            f"the {CHANGED} frame's params are {result['changed_params']} and nothing else, so "
            f"the names arrive only from tools/list. And the acknowledgement echoes the "
            f"accepted filter, {result['acknowledged']}, after accepted_filter drops the "
            "unknown key -- a client is told what it subscribed to and still has to ask what "
            "changed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
