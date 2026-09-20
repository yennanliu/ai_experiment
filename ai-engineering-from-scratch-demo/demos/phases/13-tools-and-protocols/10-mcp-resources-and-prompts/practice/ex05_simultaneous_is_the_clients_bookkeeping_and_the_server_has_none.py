"""Exercise 5 — "simultaneous" is the client's bookkeeping, and the server has none.

    Create two simultaneous subscriptions and prove each event carries the
    correct request ID.

Reading of the exercise: "correct" needs something that could be incorrect, so
the two subscriptions are given *overlapping* filters -- both watch the same
resource -- which is the only arrangement where a mis-tag would produce a
plausible frame rather than an obviously wrong one. Then the harder half: what
in the server makes the tag correct? The answer is nothing, so the exercise is
finished by finding the place where it could go wrong instead.

**ANSWER: every frame carries its own id, and none carries the other's.**
Two streams `sub-1` and `sub-2`, both subscribed to `notes://note-1` and both
to the resource list, produce **4** notification frames plus **2** closes, and
all **6** are tagged with the id of the stream that produced them — **0**
cross-tagged.

**FINDING: the fan-out is the caller's loop, not a broadcast.** One update to
`notes://note-1` becomes two frames only because the transmitter asks both
streams for one. `SubscriptionStream.resource_updated` is a method on a single
subscription and consults only its own filter; nothing in the module iterates
over open streams, because nothing holds them.

**FINDING: two streams may share an id, and the server cannot tell.** The id
is an argument to `subscriptions_listen`, not something it allocates, and
there is no registry to collide with. Built twice with `sub-1`, the two
streams' acknowledgements compare **equal** — a frozen dataclass of the id and
the filter has nothing else to differ by. "The correct request ID" is a
property the client maintains by choosing distinct ids.

**FINDING: the terminal frame's `_meta` has a shape the notifications do
not.** `close()` merges the subscription meta with `response_meta()`, so the
final result carries **2** keys -- `subscriptionId` and `serverInfo` -- while
every notification carries **1**. A client correlating on the presence of
`_meta` keys rather than on the id itself meets two shapes in one stream.

Structure: `stream_for` opens one subscription and `broadcast` is the
transmitter -- the loop over streams that the server does not have.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "10-mcp-resources-and-prompts"
WATCHED = "notes://note-1"
SUBSCRIPTION_ID = "io.modelcontextprotocol/subscriptionId"
SERVER_INFO = "io.modelcontextprotocol/serverInfo"
FILTER = {"resourcesListChanged": True, "resourceSubscriptions": [WATCHED]}


def stream_for(ref, request_id):
    return ref.subscriptions_listen(request_id, {"notifications": dict(FILTER), "_meta": {}})


def broadcast(streams, uri):
    """The transmitter the server does not have: ask every open stream."""
    frames = []
    for stream in streams:
        frames.extend([stream.resource_updated(uri), stream.resources_list_changed()])
    return [frame for frame in frames if frame is not None]


def tag(frame):
    meta = frame.get("params", frame.get("result", {}))["_meta"]
    return meta.get(SUBSCRIPTION_ID)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    first, second = stream_for(ref, "sub-1"), stream_for(ref, "sub-2")
    notifications = broadcast([first, second], WATCHED)
    closes = [first.close(), second.close()]
    duplicate = stream_for(ref, "sub-1")
    unsubscribed = ref.subscriptions_listen("sub-3", {"notifications": {}, "_meta": {}})
    no_id = ref.handle({"jsonrpc": "2.0", "id": None, "method": "subscriptions/listen",
                        "params": {"notifications": {}, "_meta": ref.request_meta()}})
    return {
        "tags": [tag(frame) for frame in notifications + closes],
        "methods": [frame.get("method") for frame in notifications],
        "expected": ["sub-1", "sub-1", "sub-2", "sub-2", "sub-1", "sub-2"],
        "single": [f["method"] for f in broadcast([first], WATCHED)],
        "unsubscribed": broadcast([unsubscribed], WATCHED),
        "duplicate_equal": duplicate.acknowledged() == first.acknowledged(),
        "duplicate_is_other": duplicate is first,
        "frozen": ref.SubscriptionStream.__dataclass_params__.frozen,
        "fields": sorted(ref.SubscriptionStream.__dataclass_fields__),
        "registry": [n for n in dir(ref)
                     if "subscri" in n.lower() and not n.startswith("__")],
        "close_meta": sorted(closes[0]["result"]["_meta"]),
        "notification_meta": sorted(notifications[0]["params"]["_meta"]),
        "no_id_error": no_id["error"]["code"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: every frame carries its own id, and none carries the other's",
            all([result["tags"] == result["expected"], len(result["tags"]) == 6,
                 result["methods"] == ["notifications/resources/updated",
                                       "notifications/resources/list_changed"] * 2]),
            f"two streams with overlapping filters produce {len(result['methods'])} "
            f"notification frames plus 2 closes, tagged {result['tags']} -- each the id of "
            "the stream that produced it, and none cross-tagged. The filters overlap, so a "
            "mis-tag would have been a plausible frame rather than an obvious one",
        ),
        practice.Check(
            "FINDING: the fan-out is the caller's loop, not a broadcast",
            all([len(result["single"]) == 2, result["unsubscribed"] == [],
                 result["registry"] == ["SubscriptionStream", "subscriptions_listen"]]),
            f"asking one stream gives {len(result['single'])} frames and asking a stream with "
            f"an empty filter gives {len(result['unsubscribed'])}, because resource_updated "
            f"consults only its own subscription. The module's subscription-shaped names are "
            f"{result['registry']} -- a class and a constructor, and nothing that holds open "
            "streams to iterate",
        ),
        practice.Check(
            "FINDING: two streams may share an id, and the server cannot tell",
            all([result["duplicate_equal"], not result["duplicate_is_other"],
                 result["frozen"], result["fields"] == ["notifications", "subscription_id"],
                 result["no_id_error"] == -32602]),
            f"the id is an argument to subscriptions_listen rather than something it "
            f"allocates, and a frozen dataclass of {result['fields']} has nothing else to "
            f"differ by -- built twice with sub-1, two distinct objects produce equal "
            f"acknowledgements. Only a null id is refused, {result['no_id_error']}, so "
            "distinct ids are the client's job",
        ),
        practice.Check(
            "FINDING: the terminal frame's _meta has a shape the notifications do not",
            all([result["close_meta"] == sorted([SUBSCRIPTION_ID, SERVER_INFO]),
                 result["notification_meta"] == [SUBSCRIPTION_ID]]),
            f"close() merges the subscription meta with response_meta(), so the final result "
            f"carries {len(result['close_meta'])} keys where every notification carries "
            f"{len(result['notification_meta'])}. A client correlating on which _meta keys "
            "are present, rather than on the id itself, meets two shapes in one stream",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
