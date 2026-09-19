"""Exercise 2 — the cursor has to be the sort key, or a page repeats itself.

    Add pagination to `resources/list` while preserving deterministic order.

Reading of the exercise: with two notes in `NOTES` no paging scheme can be
wrong, so the store is seeded to six and the list is walked two at a time.
"Preserving deterministic order" is then read as the harder of its two
meanings -- not "each page is sorted", which any implementation gives, but
"the pages partition the list", which only holds if the cursor and the sort
agree. The test is a write that lands *between* two pages.

**ANSWER: a cursor that is the last URI served, and pages that partition the
list.** Six notes at two per page give **3** full pages whose concatenation
equals the unpaginated `resources_list` — same items, same order, none
repeated, none missed. Six is a multiple of two, so every page is full and
none of them can say it is the last: a **4th** request returns **0** rows.
Exhaustion is a round trip, not a flag, whenever the total divides evenly.

**FINDING: an offset cursor is wrong under concurrent insertion, and it is
wrong by repeating.** Insert a note sorting before the cursor between page 1
and page 2: the key cursor still yields `note-3, note-4`, while the offset
cursor `2` yields `note-2, note-3` — `note-2` served twice and the tail pushed
back a slot. Nothing about the offset is invalid; the list underneath it moved.

**FINDING: the ordering the exercise asks you to preserve was already there,
and paging is what makes it load-bearing.** `resources_list` sorts
`NOTES.items()`, so reversing the insertion order changes neither the full
list nor any page. With one page that is untestable; with three it is the only
reason page boundaries mean the same thing twice.

**FINDING: the cache hint does not survive being split.** Each page inherits
`ttlMs: 300000, cacheScope: "public"` from the unpaginated result, and no page
carries anything naming the generation it came from. A cache may therefore
hold page 1 from before an insert and page 2 from after it, for up to five
minutes -- individually fresh, jointly a list that never existed.

Structure: `by_key` and `by_offset` are the two cursors over the same sorted
`resources_list` payload; `walk` drives either to exhaustion.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "10-mcp-resources-and-prompts"
PAGE = 2
SEED = ["notes://note-3", "notes://note-4", "notes://note-5", "notes://note-6"]
INSERTED = "notes://note-1a"


def note(name):
    return {"name": name, "description": f"Seeded {name}", "mimeType": "text/plain",
            "text": f"Body of {name}"}


def uris(rows):
    return [row["uri"] for row in rows]


def listing(ref):
    """The lesson's own resources/list result, which is already sorted."""
    return ref.resources_list({})["resources"]


def by_key(rows, cursor):
    """Page after the last URI served -- the cursor is the sort key itself."""
    return [row for row in rows if cursor is None or row["uri"] > cursor][:PAGE]


def by_offset(rows, cursor):
    """Page from an integer offset, which names a position and not an item."""
    return rows[(cursor or 0):(cursor or 0) + PAGE]


def walk(ref, page_of, advance):
    """Drive one cursor to exhaustion, collecting pages and their cursors."""
    pages, cursors, cursor = [], [], None
    while True:
        rows = page_of(listing(ref), cursor)
        if not rows:
            return pages, cursors
        pages.append(uris(rows))
        cursor = advance(cursor, rows)
        cursors.append(cursor if len(rows) == PAGE else None)
        if len(rows) < PAGE:
            return pages, cursors


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    original = dict(ref.NOTES)
    try:
        ref.NOTES.update({uri: note(uri.rsplit("/", 1)[-1]) for uri in SEED})
        whole = uris(listing(ref))
        pages, cursors = walk(ref, by_key, lambda _, rows: rows[-1]["uri"])
        offsets, _ = walk(ref, by_offset, lambda c, rows: (c or 0) + len(rows))

        first_key = by_key(listing(ref), None)
        ref.NOTES[INSERTED] = note("note-1a")  # a write between page 1 and page 2
        second_key = uris(by_key(listing(ref), first_key[-1]["uri"]))
        second_offset = uris(by_offset(listing(ref), PAGE))
        del ref.NOTES[INSERTED]

        reversed_notes = dict(reversed(list(ref.NOTES.items())))
        ref.NOTES.clear()
        ref.NOTES.update(reversed_notes)
        reversed_pages, _ = walk(ref, by_key, lambda _, rows: rows[-1]["uri"])
        hints = {key: ref.resources_list({})[key] for key in ("ttlMs", "cacheScope")}
        fields = sorted(listing(ref)[0])
    finally:
        ref.NOTES.clear()
        ref.NOTES.update(original)
    return {
        "whole": whole, "pages": pages, "cursors": cursors, "offsets": offsets,
        "tail": uris(by_key(listing(ref), cursors[-1])),
        "first": uris(first_key),
        "second_key": second_key, "second_offset": second_offset,
        "reversed_pages": reversed_pages, "hints": hints, "fields": fields,
    }


def verify(result):
    pages = result["pages"]
    flattened = [uri for page in pages for uri in page]
    return [
        practice.Check(
            "ANSWER: a cursor that is the last URI served, and pages that partition the list",
            all([len(pages) == 3, all(len(p) == PAGE for p in pages),
                 flattened == result["whole"],
                 result["cursors"] == [p[-1] for p in pages],
                 result["tail"] == [], result["offsets"] == pages]),
            f"six notes at {PAGE} per page give {len(pages)} full pages, {pages}, whose "
            f"concatenation equals the unpaginated resources_list exactly. Every page is "
            f"full, so none can say it is the last: a fourth request at cursor "
            f"{result['cursors'][-1]!r} returns {len(result['tail'])} rows. An offset cursor "
            "agrees throughout, as long as nothing is written",
        ),
        practice.Check(
            "FINDING: an offset cursor is wrong under insertion, and it is wrong by repeating",
            all([result["second_key"] == ["notes://note-3", "notes://note-4"],
                 result["second_offset"] == ["notes://note-2", "notes://note-3"],
                 result["first"] == ["notes://note-1", "notes://note-2"]]),
            f"inserting {INSERTED} between page 1 {result['first']} and page 2, the key cursor "
            f"still yields {result['second_key']} while the offset cursor {PAGE} yields "
            f"{result['second_offset']} -- note-2 served twice and the tail pushed back a "
            "slot. Nothing about the offset became invalid; the list underneath it moved",
        ),
        practice.Check(
            "FINDING: the order was already deterministic, and paging is what makes it matter",
            all([result["reversed_pages"] == pages,
                 result["whole"] == sorted(result["whole"])]),
            f"reversing the insertion order of NOTES leaves both the full list and every page "
            f"unchanged, {result['reversed_pages']}, because resources_list sorts "
            "NOTES.items(). With one page that is untestable; with three it is the only "
            "reason a page boundary means the same thing twice",
        ),
        practice.Check(
            "FINDING: the cache hint does not survive being split",
            all([result["hints"] == {"ttlMs": 300_000, "cacheScope": "public"},
                 result["fields"] == ["description", "mimeType", "name", "uri"]]),
            f"every page inherits {result['hints']} from the unpaginated result, and a row "
            f"carries {result['fields']} -- nothing naming the generation it came from. A "
            "cache may hold page 1 from before an insert and page 2 from after it for five "
            "minutes: individually fresh, jointly a list that never existed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
