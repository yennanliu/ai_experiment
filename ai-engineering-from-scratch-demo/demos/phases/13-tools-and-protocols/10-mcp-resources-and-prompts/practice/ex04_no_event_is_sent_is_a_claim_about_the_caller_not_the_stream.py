"""Exercise 4 — "no event is sent" is a claim about the caller, not the stream.

    Add a prompt-list change subscription and prove no event is sent when the
    filter omits `promptsListChanged`.

Reading of the exercise: `prompts_list_changed()` does not decline to produce
an event, it produces `None`, and `None` is only "no event" if something
downstream drops it. `demo()` does, in a list comprehension. So the proof is
taken at the send site rather than at the method, and the near-miss filters --
a typo and a truthy non-`True` -- are run alongside, because those are the
ways a real client ends up unsubscribed while believing otherwise.

**ANSWER: subscribed carries the change, omitted carries only the
acknowledgement.** With `promptsListChanged: True` the send site emits
`acknowledged` then `notifications/prompts/list_changed`, tagged with the
subscription id; with the key absent the method returns `None` and a
transmitter that filters `None` emits the acknowledgement alone -- **1**
change event against **0**.

**FINDING: the method cannot enforce it — the caller can.** A transmitter
without the filter sends **1** frame for the unsubscribed stream, and that
frame is the JSON literal `null`. The lesson's own `demo()` ends with
`[item for item in transcript if item is not None]`, which is where "no event
is sent" is actually implemented.

**FINDING: only the literal `True` subscribes, and `1` is not it.**
`subscriptions_listen` tests `value is True`, so of `True`, `1` and `"yes"`
exactly **1** is accepted — even though `1 == True`. A JSON client that sent
`1` gets an empty filter and no error.

**FINDING: a typo unsubscribes you silently, and the acknowledgement is the
only place it shows.** `promptListChanged` — one missing `s` — is not in
`SUPPORTED_NOTIFICATION_FIELDS`, so it is dropped with no error and the agreed
filter is `{}`. The agreed filter is also rebuilt in alphabetical order rather
than request order, so the ack is a normalized restatement and not an echo:
reading it back is the check, and comparing it to what you sent is the way to
notice.

Structure: `stream_for` builds one subscription from a filter, and `transmit`
is the send site -- the same frames with and without the `None` filter.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "10-mcp-resources-and-prompts"
PROMPTS_CHANGED = "notifications/prompts/list_changed"
ACK = "notifications/subscriptions/acknowledged"
TYPO = "promptListChanged"


def stream_for(ref, request_id, notifications):
    return ref.subscriptions_listen(request_id, {"notifications": notifications, "_meta": {}})


def transmit(stream, drop_none=True):
    """The send site: every frame the stream offers for a prompts change."""
    frames = [stream.acknowledged(), stream.prompts_list_changed(), stream.close()]
    return [f for f in frames if f is not None] if drop_none else frames


def events(frames):
    return [f["method"] for f in frames if f is not None and "method" in f]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    subscribed = stream_for(ref, "sub-1", {"promptsListChanged": True})
    omitted = stream_for(ref, "sub-2", {"resourcesListChanged": True})
    # a list, not a dict: True and 1 hash equal and would collapse into one key
    truthy = [(repr(value), stream_for(ref, "sub-3", {"promptsListChanged": value}).notifications)
              for value in (True, 1, "yes")]
    mixed = stream_for(ref, "sub-4", {"resourcesListChanged": True,
                                      "promptsListChanged": True,
                                      "resourceSubscriptions": ["b", "a"]})
    return {
        "subscribed": events(transmit(subscribed)),
        "omitted": events(transmit(omitted)),
        "omitted_frames": len(transmit(omitted)),
        "unfiltered": [f for f in transmit(omitted, drop_none=False) if f is None],
        "subscribed_filter": subscribed.notifications,
        "omitted_filter": omitted.notifications,
        "tag": subscribed.prompts_list_changed()["params"]["_meta"],
        "truthy": truthy,
        "typo": stream_for(ref, "sub-5", {TYPO: True}).notifications,
        "fields": sorted(ref.SUPPORTED_NOTIFICATION_FIELDS),
        "agreed_order": list(mixed.notifications),
        "sent_order": ["resourcesListChanged", "promptsListChanged", "resourceSubscriptions"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: subscribed emits one prompts/list_changed, omitted emits none",
            all([result["subscribed"] == [ACK, PROMPTS_CHANGED], result["omitted"] == [ACK],
                 result["subscribed_filter"] == {"promptsListChanged": True},
                 result["omitted_filter"] == {"resourcesListChanged": True},
                 result["tag"] == {"io.modelcontextprotocol/subscriptionId": "sub-1"}]),
            f"with promptsListChanged the stream carries the acknowledgement and "
            f"{PROMPTS_CHANGED}, tagged {result['tag']}; with the key absent it carries the "
            f"acknowledgement alone. The agreed filters differ by exactly the key the "
            f"exercise omits, {result['subscribed_filter']} against "
            f"{result['omitted_filter']}",
        ),
        practice.Check(
            "FINDING: the method cannot enforce it, but the caller can",
            all([len(result["unfiltered"]) == 1, result["omitted_frames"] == 2]),
            f"prompts_list_changed() returns None rather than declining, so a transmitter "
            f"without the filter carries {len(result['unfiltered'])} extra frame -- the JSON "
            f"literal null -- while the filtering one carries {result['omitted_frames']}. The "
            "lesson's own demo() ends with `if item is not None`, which is where 'no event is "
            "sent' is actually implemented",
        ),
        practice.Check(
            "FINDING: only the literal True subscribes, and 1 is not it",
            result["truthy"] == [("True", {"promptsListChanged": True}),
                                 ("1", {}), ("'yes'", {})],
            f"subscriptions_listen tests `value is True`, so of True, 1 and 'yes' exactly one "
            f"is accepted -- {result['truthy']} -- even though 1 == True. A JSON client that "
            "sent 1 gets an empty filter and no error to tell it so",
        ),
        practice.Check(
            "FINDING: a typo unsubscribes you silently, and the ack is where it shows",
            all([result["typo"] == {}, TYPO not in result["fields"],
                 result["agreed_order"] == sorted(result["agreed_order"]),
                 result["agreed_order"] != result["sent_order"]]),
            f"{TYPO!r} is not in {result['fields']}, so it is dropped with no error and the "
            f"agreed filter is {result['typo']}. The filter is also rebuilt alphabetically, "
            f"{result['agreed_order']} against the {result['sent_order']} that was sent, so "
            "the ack is a normalized restatement rather than an echo -- comparing it with "
            "what you sent is the check",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
