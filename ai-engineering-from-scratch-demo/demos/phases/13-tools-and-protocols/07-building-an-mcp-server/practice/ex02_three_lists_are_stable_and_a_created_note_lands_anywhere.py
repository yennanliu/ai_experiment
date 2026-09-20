"""Exercise 2 — three lists are stable, and a created note lands anywhere.

    Reverse the `TOOLS`, `PROMPTS`, and note insertion order. Confirm all list
    results remain stable.

Reading of the exercise: all three collections are reversed and all three lists
re-read, which confirms the claim. The interesting case is the one the exercise
does not name -- `NOTES` is the only collection a *tool* can grow, and sorting
it by key makes the position of a new entry a function of a random id rather
than of when it was created.

**ANSWER: all three lists are unchanged by reversal.** `tools/list` and
`prompts/list` sort by `name`, `resources/list` sorts `NOTES.items()` by id, so
insertion order never reaches the wire. Reversing all three at once changes
nothing.

**FINDING: a created note sorts by a random hex id, so its position is
unpredictable.** `exec_notes_create` names it `note-{uuid4().hex[:6]}`, and
`resources/list` sorts lexicographically -- an id beginning `0`-`9` lands
*before* the seeded `note-1`, and one beginning `a`-`f` lands after. Over **200**
creates the new note appears at position 0 sometimes and last other times. The
order is deterministic given the ids and unreproducible across runs.

**FINDING: stability and freshness are different properties, and the server
advertises the wrong one.** Creating a note takes `resources/list` from **3**
entries to **4**, while `SERVER_CAPABILITIES` declares
`resources.listChanged: False` and the list caches for **10,000 ms**. A client
honouring both will serve a stale list for up to ten seconds and will never be
told. Sorting made the list stable; nothing made it current.

**FINDING: and the sort key differs from the identity key on exactly one list.**
`tools/list` and `prompts/list` sort by the field a client would use to
reference the entry. `resources/list` sorts by note id but returns `uri` and
`name` -- so the field the order is built on is not in the payload, and a client
cannot reproduce the ordering from what it received.

Structure: `listed` reads one list method under a temporarily rearranged
module, `reversed_all` flips all three collections at once, and `positions`
samples where a freshly created note lands.
"""

from __future__ import annotations

import contextlib
import random

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "07-building-an-mcp-server"
LISTS = {"tools/list": ("tools", "name"), "prompts/list": ("prompts", "name"),
         "resources/list": ("resources", "uri")}
SAMPLES = 200


@contextlib.contextmanager
def reversed_all(ref):
    notes = dict(ref.NOTES)
    ref.TOOLS.reverse()
    ref.PROMPTS.reverse()
    ref.NOTES.clear()
    ref.NOTES.update(dict(reversed(list(notes.items()))))
    try:
        yield
    finally:
        ref.TOOLS.reverse()
        ref.PROMPTS.reverse()
        ref.NOTES.clear()
        ref.NOTES.update(notes)


def listed(ref, method):
    key, field = LISTS[method]
    response = ref.dispatch(ref.make_request(1, method))
    return [entry[field] for entry in response["result"][key]]


def all_lists(ref):
    return {method: listed(ref, method) for method in LISTS}


def positions(ref, samples=SAMPLES):
    """Where a freshly created note lands in resources/list, over many ids."""
    seen = set()
    for _ in range(samples):
        ref.reset_notes()
        created = ref.dispatch(ref.make_request(
            1, "tools/call", {"name": "notes_create",
                              "arguments": {"title": "T", "body": "B"}}))
        uri = created["result"]["content"][1]["resource"]["uri"]
        seen.add(listed(ref, "resources/list").index(uri))
    ref.reset_notes()
    return sorted(seen)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    random.seed(0)
    ref.reset_notes()
    forward = all_lists(ref)
    with reversed_all(ref):
        backward = all_lists(ref)
    before = len(listed(ref, "resources/list"))
    ref.dispatch(ref.make_request(2, "tools/call",
                                  {"name": "notes_create",
                                   "arguments": {"title": "T", "body": "B"}}))
    after = len(listed(ref, "resources/list"))
    ref.reset_notes()
    landed = positions(ref)
    ttl = ref.dispatch(ref.make_request(3, "resources/list"))["result"]["ttlMs"]
    return {
        "forward": forward, "backward": backward, "stable": forward == backward,
        "lists": len(LISTS),
        "landed": landed, "distinct_positions": len(landed),
        "samples": SAMPLES,
        "before": before, "after": after,
        "list_changed": ref.SERVER_CAPABILITIES["resources"]["listChanged"],
        "resources_ttl": ttl,
        "sort_fields": {"tools/list": "name", "prompts/list": "name",
                        "resources/list": "note id"},
        "payload_fields": {method: sorted(
            ref.dispatch(ref.make_request(4, method))["result"][key][0])
            for method, (key, _) in LISTS.items()},
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: all three lists are unchanged by reversal",
            all([result["stable"], result["lists"] == 3,
                 result["forward"]["tools/list"] ==
                 ["notes_create", "notes_list", "notes_search"],
                 result["forward"]["prompts/list"] == ["review_note"]]),
            f"reversing TOOLS, PROMPTS and the note insertion order at once changes none of "
            f"the {result['lists']} lists: {result['forward']}. Two sort by name and one "
            "sorts NOTES.items() by id, so insertion order never reaches the wire",
        ),
        practice.Check(
            "FINDING: a created note sorts by a random hex id, so its position varies",
            all([result["distinct_positions"] > 1, 0 in result["landed"],
                 max(result["landed"]) == 3]),
            f"exec_notes_create names it note-{{uuid4().hex[:6]}} and resources/list sorts "
            f"lexicographically, so an id beginning 0-9 lands before the seeded note-1. Over "
            f"{result['samples']} creates the new note appeared at positions "
            f"{result['landed']}. The order is deterministic given the ids and "
            "unreproducible across runs",
        ),
        practice.Check(
            "FINDING: stability and freshness are different, and the wrong one is advertised",
            all([result["before"] == 3, result["after"] == 4,
                 result["list_changed"] is False, result["resources_ttl"] == 10_000]),
            f"creating a note takes resources/list from {result['before']} to "
            f"{result['after']} entries, while the server declares resources.listChanged "
            f"{result['list_changed']} and the list caches for "
            f"{result['resources_ttl']:,} ms. A client honouring both serves a stale list "
            "for up to ten seconds and is never told. Sorting made the list stable; nothing "
            "made it current",
        ),
        practice.Check(
            "FINDING: the sort key is absent from the payload on exactly one list",
            all([result["payload_fields"]["tools/list"] ==
                 ["annotations", "description", "inputSchema", "name"],
                 result["payload_fields"]["prompts/list"] ==
                 ["arguments", "description", "name"],
                 result["payload_fields"]["resources/list"] ==
                 ["mimeType", "name", "uri"]]),
            f"tools/list and prompts/list sort by name and carry name: "
            f"{result['payload_fields']['tools/list']}, "
            f"{result['payload_fields']['prompts/list']}. resources/list sorts by note id "
            f"and returns {result['payload_fields']['resources/list']} -- the field the "
            "order is built on is not in the payload, so a client cannot reproduce the "
            "ordering from what it received",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
