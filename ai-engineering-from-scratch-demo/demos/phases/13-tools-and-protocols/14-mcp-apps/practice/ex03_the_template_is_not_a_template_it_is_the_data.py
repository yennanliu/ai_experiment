"""Exercise 3 — the template is not a template, it is the data.

    Change the resource to `cacheScope: private`. Describe the user-specific
    condition that justifies it.

Reading of the exercise: "describe the condition" is answerable by measurement
rather than assertion, because either the bytes depend on the user or they do
not. So the same resource is read against two note sets and the two responses
are compared. They differ, which settles it -- and then the second half of the
answer is that this did not have to be true.

**THE USER-SPECIFIC CONDITION: `resources/read` returns the user's notes, not
a shell to put them in.** `timeline_html(NOTES)` interpolates every title and
date into the markup, so two users' reads differ in **3** of **3** titles. A
`public` cache is a cache keyed on the URI alone, and the URI is the same for
everybody -- so one user's timeline is served to the next. That is the whole
condition: the response body varies with the caller and the cache key does
not.

**ANSWER: `cacheScope: "private"`, `ttlMs` unchanged.** The read's hints go
from `('public', 60000)` to `('private', 60000)` and nothing else moves.

**FINDING: `resources/list` should stay public, and the difference is
measurable.** Its entry carries `uri`, `name`, `description` and `mimeType`
and **0** note titles, so it is identical for both users -- one scope per
method rather than one per server, because scope is a property of what the
bytes contain.

**FINDING: the data already travels twice, which is what makes `public`
recoverable.** `tools/call` returns the same notes as `structuredContent`, so
a resource that shipped an empty template plus a client that filled it from
the tool result would be genuinely user-independent -- and cacheable by
everyone. The scope follows the design; it is not a property the resource is
stuck with. The tool result also hands back `NOTES` itself rather than a copy,
so the response aliases live server state -- which is why this measurement had
to be taken on a snapshot.

Structure: `read_as` swaps `NOTES` for one read and restores it, so the two
users differ in exactly the data and nothing else.
"""

from __future__ import annotations

import copy

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "14-mcp-apps"
ALICE = [{"id": "note-1", "title": "Alice quarterly", "created": "2026-07-28"},
         {"id": "note-2", "title": "Alice travel", "created": "2026-07-29"},
         {"id": "note-3", "title": "Alice offsite", "created": "2026-07-30"}]
BOB = [{"id": "note-1", "title": "Bob payroll", "created": "2026-08-01"},
       {"id": "note-2", "title": "Bob vendors", "created": "2026-08-02"},
       {"id": "note-3", "title": "Bob review", "created": "2026-08-03"}]


def read_as(ref, server, notes, method="resources/read", params=None):
    """One request with `notes` standing in for the store's own."""
    original = ref.NOTES[:]
    ref.NOTES[:] = notes
    try:
        body, headers = ref.make_request(
            method, 1, params if params is not None else {"uri": ref.RESOURCE_URI})
        # deep-copied: structuredContent hands back the store's own list object
        return copy.deepcopy(server.handle(body, headers)[1]["result"])
    finally:
        ref.NOTES[:] = original


def private(result):
    """The exercise's change: one field, nothing else."""
    return {**result, "cacheScope": "private"}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = ref.McpAppServer()
    alice = read_as(ref, server, ALICE)
    bob = read_as(ref, server, BOB)
    alice_html = alice["contents"][0]["text"]
    bob_html = bob["contents"][0]["text"]
    listed = [read_as(ref, server, notes, "resources/list", {}) for notes in (ALICE, BOB)]
    called = read_as(ref, server, ALICE, "tools/call", {"name": "notes_timeline"})
    live = server.handle(*ref.make_request("tools/call", 2, {"name": "notes_timeline"}))[1]
    changed = private(alice)
    return {
        "before": (alice["cacheScope"], alice["ttlMs"]),
        "after": (changed["cacheScope"], changed["ttlMs"]),
        "only_scope_moved": {k: v for k, v in changed.items() if k != "cacheScope"}
                            == {k: v for k, v in alice.items() if k != "cacheScope"},
        "html_differs": alice_html != bob_html,
        "titles_in_html": sum(n["title"] in alice_html for n in ALICE),
        "bob_titles_absent": sum(n["title"] in alice_html for n in BOB),
        "list_identical": listed[0] == listed[1],
        "list_fields": sorted(listed[0]["resources"][0]),
        "list_scope": listed[0]["cacheScope"],
        "titles_in_list": sum(n["title"] in str(listed[0]) for n in ALICE),
        "tool_notes": called["structuredContent"]["notes"] == ALICE,
        "aliases_store": live["result"]["structuredContent"]["notes"] is ref.NOTES,
    }


def verify(result):
    return [
        practice.Check(
            "THE CONDITION: the read returns the user's notes, and the URI is everyone's",
            all([result["html_differs"], result["titles_in_html"] == 3,
                 result["bob_titles_absent"] == 0]),
            f"timeline_html interpolates every title into the markup, so alice's read "
            f"contains {result['titles_in_html']} of her titles and "
            f"{result['bob_titles_absent']} of bob's, and the two reads differ. A public "
            "cache is keyed on the URI alone and the URI is the same for everybody -- the "
            "body varies with the caller and the key does not",
        ),
        practice.Check(
            "ANSWER: cacheScope private, and nothing else moves",
            all([result["before"] == ("public", 60_000),
                 result["after"] == ("private", 60_000), result["only_scope_moved"]]),
            f"the read's hints go from {result['before']} to {result['after']}, and every "
            f"other field of the result is unchanged: {result['only_scope_moved']}",
        ),
        practice.Check(
            "FINDING: resources/list should stay public, and the difference is measurable",
            all([result["list_identical"], result["titles_in_list"] == 0,
                 result["list_scope"] == "public",
                 result["list_fields"] == ["description", "mimeType", "name", "uri"]]),
            f"the list entry carries {result['list_fields']} and "
            f"{result['titles_in_list']} note titles, so it is identical for both users. One "
            "scope per method rather than one per server, because scope is a property of "
            "what the bytes contain",
        ),
        practice.Check(
            "FINDING: the data already travels twice, which is what makes public recoverable",
            all([result["tool_notes"], result["aliases_store"]]),
            "tools/call returns the same notes as structuredContent, so a resource shipping "
            "an empty template plus a client filling it from the tool result would be "
            "genuinely user-independent, and cacheable by everyone. The scope follows the "
            "design. (The tool result also hands back the store's own list object rather "
            "than a copy, which is why this had to be measured on a snapshot.)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
