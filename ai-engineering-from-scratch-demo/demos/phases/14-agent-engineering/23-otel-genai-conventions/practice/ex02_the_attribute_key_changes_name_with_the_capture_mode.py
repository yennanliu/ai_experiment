"""Exercise 2 — the attribute key changes name with the capture mode.

    Add content capture in "references only" mode: prompts to SQLite, span
    attributes carry only row IDs.

Reading of the exercise: references-only is already the default -- what is
missing is a store that survives the process. So `ExternalContentStore` is
reimplemented over stdlib `sqlite3` with the same three methods, and the
interesting result is not the swap but what comparing the two capture modes
shows about the attribute names.

**ANSWER: prompts land in SQLite and the spans carry row ids only.** Over a
3-span trace the store holds **6** rows, every attribute whose name ends
`.reference_id` is an integer row id, and **0** attributes contain prose --
grepping the whole trace for the card number in the prompt returns **0** hits
where inline capture returns **3**, while `SELECT` finds it in **3** rows.

**FINDING: the two modes do not produce the same attribute name.** Inline
capture writes `gen_ai.input.messages`; references-only writes
`gen_ai.input.messages.reference_id`. The **2** content keys share **0**
members while the **2** non-content keys are identical, so a dashboard query,
an alert or a sampling rule written against one mode matches nothing under the
other -- and the mode is a constructor flag nobody downstream can see.

**FINDING: inline capture truncates silently at 200 characters.** A 516-char
prompt is stored as **200**, losing **316**, with **0** attributes recording
that anything was dropped. The reference path stores all **516**. Inline
capture is lossy in a way that looks exactly like a short prompt.

**FINDING: the reference id is a per-tracer counter, so two tracers collide.**
Two `Tracer`s each produce `content_001` for different content, and a reader
holding a span from one cannot tell which store to ask -- **2** distinct
prompts under **1** id. SQLite's `rowid` is per-database, which moves the
ambiguity to a place that can be namespaced.

Structure: `SqliteStore` is the swap; `keys_for()` collects the attribute
names each mode emits.
"""

from __future__ import annotations

import sqlite3

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "23-otel-genai-conventions"
SECRET = "card 4111-1111-1111-1111"
PROMPT = f"turn 0: look up the refund for {SECRET}"
LONG = "x" * 512


class SqliteStore:
    """ExternalContentStore's three methods, over a real database."""

    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE content (id INTEGER PRIMARY KEY, body TEXT)")

    def put(self, content):
        cur = self.db.execute("INSERT INTO content (body) VALUES (?)", (content,))
        self.db.commit()
        return cur.lastrowid

    def get(self, cid):
        row = self.db.execute("SELECT body FROM content WHERE id = ?",
                              (cid,)).fetchone()
        return row[0] if row else ""

    def items(self):
        return self.db.execute("SELECT id, body FROM content ORDER BY id").fetchall()


def traced(ref, inline, store=None, prompt=PROMPT):
    tracer = ref.Tracer(capture_inline=inline, content_store=store)
    for turn in range(3):
        chat = tracer.start_span("chat", attributes={
            "gen_ai.operation.name": "chat", "gen_ai.request.model": "claude-opus-4-6"})
        tracer.add_content(chat, "gen_ai.input.messages", f"{prompt} [{turn}]")
        tracer.add_content(chat, "gen_ai.output.messages", "ok")
        tracer.end_span()
    return tracer


def walk(span, rows=None):
    rows = [] if rows is None else rows
    if span.name != "__root__":
        rows.append(span)
    for child in span.children:
        walk(child, rows)
    return rows


def keys_for(tracer, content_only=False):
    keys = {key for span in walk(tracer.root) for key in span.attributes}
    return {key for key in keys if "messages" in key} if content_only else keys


def prose_hits(tracer, needle):
    return sum(needle in str(value) for span in walk(tracer.root)
               for value in span.attributes.values())


def collide(ref):
    first, second = ref.ExternalContentStore(), ref.ExternalContentStore()
    return {"ids": (first.put("prompt A"), second.put("prompt B")),
            "bodies": (first.get("content_001"), second.get("content_001"))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    store = SqliteStore()
    refs = traced(ref, inline=False, store=store)
    inline = traced(ref, inline=True)
    long_inline = traced(ref, inline=True, prompt=LONG)
    long_refs = traced(ref, inline=False, store=SqliteStore(), prompt=LONG)
    stored = store.items()
    ref_keys = keys_for(refs, content_only=True)
    inline_keys = keys_for(inline, content_only=True)
    inline_value = walk(long_inline.root)[0].attributes["gen_ai.input.messages"]
    return {
        "spans": len(walk(refs.root)), "rows": len(stored),
        "id_types": sorted({type(row[0]).__name__ for row in stored}),
        "prose_hits": prose_hits(refs, SECRET),
        "inline_hits": prose_hits(inline, SECRET),
        "in_db": sum(SECRET in body for _, body in stored),
        "ref_keys": sorted(ref_keys), "inline_keys": sorted(inline_keys),
        "shared": sorted(ref_keys & inline_keys),
        "shared_all": sorted(keys_for(refs) & keys_for(inline)),
        "kept": len(inline_value), "dropped": len(LONG) + 4 - len(inline_value),
        "truncation_markers": sum("truncat" in key for key in inline_keys),
        "stored_len": len(long_refs.content_store.items()[0][1]),
        "collision": collide(ref),
    }


def verify(result):
    collision = result["collision"]
    return [
        practice.Check(
            "ANSWER: prompts land in SQLite and the spans carry row ids only",
            all([result["rows"] == 6, result["id_types"] == ["int"],
                 result["prose_hits"] == 0, result["in_db"] == 3,
                 result["inline_hits"] == 3, result["spans"] == 3]),
            f"the {result['spans']}-span trace writes {result['rows']} rows and every "
            f"reference is an {result['id_types'][0]}. Grepping the trace for the card "
            f"number returns {result['prose_hits']} hits where inline capture returns "
            f"{result['inline_hits']}, while SELECT finds it in {result['in_db']} rows",
        ),
        practice.Check(
            "FINDING: the two modes do not produce the same attribute name",
            all([result["shared"] == [], len(result["ref_keys"]) == 2,
                 len(result["inline_keys"]) == 2, len(result["shared_all"]) == 2,
                 "gen_ai.input.messages" in result["inline_keys"],
                 "gen_ai.input.messages.reference_id" in result["ref_keys"]]),
            f"inline capture emits the content keys {result['inline_keys']} and "
            f"references-only emits {result['ref_keys']} -- {len(result['shared'])} in "
            f"common, against {len(result['shared_all'])} shared non-content keys. A "
            "dashboard query written against one mode matches nothing under the other, "
            "and the mode is a constructor flag nobody downstream sees",
        ),
        practice.Check(
            "FINDING: inline capture truncates silently at 200 characters",
            all([result["kept"] == 200, result["dropped"] == 316,
                 result["truncation_markers"] == 0,
                 result["stored_len"] == 516]),
            f"a 512-character prompt is kept as {result['kept']} characters, losing "
            f"{result['dropped']}, with {result['truncation_markers']} attributes "
            f"recording that anything was dropped. The reference path stores all "
            f"{result['stored_len']}: inline capture is lossy in a way that looks exactly "
            "like a short prompt",
        ),
        practice.Check(
            "FINDING: the reference id is a per-tracer counter, so two stores collide",
            all([collision["ids"] == ("content_001", "content_001"),
                 collision["bodies"] == ("prompt A", "prompt B")]),
            f"two ExternalContentStores each mint {collision['ids'][0]!r} for different "
            f"content -- {collision['bodies']} -- so a reader holding a span cannot tell "
            "which store to ask. SQLite's rowid is per-database, which moves the ambiguity "
            "somewhere that can be namespaced",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
