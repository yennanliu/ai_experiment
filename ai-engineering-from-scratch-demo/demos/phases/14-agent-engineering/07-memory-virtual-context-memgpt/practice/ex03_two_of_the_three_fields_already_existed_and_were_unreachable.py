"""Exercise 3 — two of the three fields already existed and were unreachable.

    Add `citation` fields (session_id, turn_id, source_url) to archival
    inserts. Make the agent cite sources on every retrieval-backed answer.

Reading of the exercise: `ArchivalRecord` already declares `session_id` and
`turn_id`. What is missing is not the fields but the path to them --
`MemoryTools.archival_memory_insert` takes `text` and `tags` and forwards
neither, and `archival_memory_search` renders `rid: text`. So the exercise is
wiring, plus one genuinely new field, and the measurement is how far a
citation gets before something drops it.

**ANSWER: a cited tool surface, and answers that resolve.** Inserting three
facts with a session, a turn and a source URL and rendering them in the
search result gives an answer citing **3** sources, **3** of which resolve
back to a stored record. Through the shipped surface the same answer cites
**0**.

**FINDING: the two shipped citation fields are unreachable.**
`archival_memory_insert` has **2** parameters, so every record the demo
writes carries the defaults: **1** distinct `(session_id, turn_id)` pair
across all **5** of them, `('s0', 0)`. A field with no writer is a schema
that documents an intention.

**FINDING: search renders none of them.** `archival_memory_search` formats
`f"  {h.rid}: {h.text}"`, so **0** of the record's **4** non-text fields
reach the observation. Even a correctly stored citation is invisible to the
model that is supposed to repeat it.

**FINDING: `source_url` has nowhere typed to go.** `ArchivalRecord` has
**5** fields and **0** of them mention a source or a URL. Parking the URL in
`tags` type-checks -- `tags` is an untyped `tuple[str, ...]` -- and then a
provenance claim and a topic label are the same kind of thing, with no way to
ask which tag is the source.

Structure: `Cited` wraps the lesson's `ArchivalStore` and `MemoryTools`,
adding only the plumbing; the record type stays the lesson's.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "07-memory-virtual-context-memgpt"
FACTS = (
    ("ava ships agents for a living", "s3", 11, "https://chat/s3#11"),
    ("the sales bot registers twelve tools", "s3", 14, "https://chat/s3#14"),
    ("tool chains drift after twenty steps", "s7", 4, "https://docs/bfcl#drift"),
)
QUERY = "tools drift ava"


class Cited:
    """The lesson's store and tools, with the citation path wired end to end."""

    def __init__(self, ref):
        self.store, self.urls = ref.ArchivalStore(), {}
        self.tools = ref.MemoryTools(ref.MainContext(), self.store)

    def insert(self, text, session_id, turn_id, source_url):
        rid = self.store.insert(text, session_id=session_id, turn_id=turn_id)
        self.urls[rid] = source_url
        return rid

    def search(self, query, top_k=3):
        return [self.render(hit) for hit in self.store.search(query, top_k=top_k)]

    def render(self, hit):
        return (f"{hit.rid} ({hit.session_id}/turn {hit.turn_id}, "
                f"{self.urls[hit.rid]}): {hit.text}")


def shipped_records(ref):
    """What the demo's own writes look like coming out of the shipped tool."""
    store = ref.ArchivalStore()
    tools = ref.MemoryTools(ref.MainContext(), store)
    for text, *_ in FACTS:
        tools.archival_memory_insert(text, tags=("demo",))
    tools.archival_memory_insert("sleep-time compute consolidates memory")
    tools.archival_memory_insert("letta adds a third tier")
    return store


def resolvable(cited, answer):
    return sum(1 for rid in cited.urls if rid in answer)


def field_report(ref, rendered):
    """Which of the record's own fields survive into the observation."""
    fields = list(ref.ArchivalRecord.__dataclass_fields__)
    return {
        "record_fields": fields,
        "source_fields": [f for f in fields if "source" in f or "url" in f],
        "search_shows": sorted({f for f in fields if f != "text" and f in rendered}),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cited = Cited(ref)
    for text, session, turn, url in FACTS:
        cited.insert(text, session, turn, url)
    lines = cited.search(QUERY)
    answer = "based on " + "; ".join(line.split(":")[0] for line in lines)
    store = shipped_records(ref)
    shipped_hits = ref.MemoryTools(ref.MainContext(), store).archival_memory_search(QUERY)
    return {
        "cited": len(lines), "resolvable": resolvable(cited, answer),
        "rendered": lines[0],
        "shipped_citations": sorted({(r.session_id, r.turn_id) for r in store._records}),
        "shipped_records": store.count(),
        "insert_params": [p for p in inspect.signature(
            ref.MemoryTools.archival_memory_insert).parameters if p != "self"],
        "tags_type": str(ref.ArchivalRecord.__dataclass_fields__["tags"].type),
        **field_report(ref, shipped_hits),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: three cited sources, three of them resolvable",
            all([result["cited"] == 3, result["resolvable"] == 3,
                 "turn" in result["rendered"], "https://" in result["rendered"]]),
            f"the wired surface returns {result['cited']} hits rendered as "
            f"{result['rendered']!r}, and {result['resolvable']} of the ids in the answer "
            "resolve back to a stored record. Through the shipped surface the same "
            "answer cites 0, because there is nothing in the observation to cite",
        ),
        practice.Check(
            "FINDING: the two shipped citation fields are unreachable",
            all([result["insert_params"] == ["text", "tags"],
                 result["shipped_citations"] == [("s0", 0)],
                 result["shipped_records"] == 5]),
            f"archival_memory_insert takes {result['insert_params']} and forwards neither "
            f"session nor turn, so all {result['shipped_records']} demo records carry "
            f"{result['shipped_citations']} -- one distinct pair, the defaults. A field "
            "with no writer documents an intention",
        ),
        practice.Check(
            "FINDING: search renders none of them",
            all([result["search_shows"] == [],
                 result["record_fields"] == ["rid", "text", "tags", "session_id",
                                             "turn_id"]]),
            f"ArchivalRecord carries {result['record_fields']} and "
            f"archival_memory_search formats rid and text only, so "
            f"{len(result['search_shows'])} of the 4 non-text fields reach the "
            "observation. A correctly stored citation is still invisible to the model",
        ),
        practice.Check(
            "FINDING: source_url has nowhere typed to go",
            all([result["source_fields"] == [], len(result["record_fields"]) == 5,
                 "str" in result["tags_type"]]),
            f"{len(result['source_fields'])} of the {len(result['record_fields'])} record "
            f"fields mention a source or a URL, and tags is {result['tags_type']} -- "
            "untyped, so a provenance claim and a topic label are the same kind of thing "
            "and nothing can ask which tag is the source",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
