"""Exercise 2 — off the critical path is a convention, not a boundary.

    Implement sleep-time dedup over archival: two records whose text has >90%
    token overlap collapse to one. Do it only in the sleep pass, never on the
    critical path.

Reading of the exercise: the dedup itself is a pairwise Jaccard scan plus
`Archival.invalidate`, which already exists and is a soft delete. The part
worth measuring is the second sentence. "Only in the sleep pass" is a
property of who may call what, and both agents hold the *same* `Archival`
object with the same public methods, so the rule lives in the prose rather
than in the types.

**ANSWER: dedup collapses 3 pairs, leaving 9 of 12 records valid.** The scan
runs inside a sleep pass and keeps the earliest record of each near-duplicate
group. Every collapsed record is still present in `all_records()` with
`valid=False`, so the pass is auditable rather than destructive.

**FINDING: nothing records *why* a record was invalidated.**
`ArchivalRecord` has **3** fields -- `rid`, `text`, `valid` -- so a record
dropped as a duplicate and one dropped as contradicted by the `human` block
are the same object afterwards. There is no `superseded_by`, so the pass
cannot be undone selectively and an operator cannot tell staleness from
redundancy.

**FINDING: 0.9 on token sets is a knife edge.** Against a 10-token record,
one extra word scores **0.909** and collapses, two extra words score
**0.833** and survive. Two sentences a reader would call the same fact sit on
opposite sides of the threshold, so the dedup rate is a property of how
verbose the writer was.

**FINDING: the boundary is a comment.** `PrimaryAgent` holds `.archival` and
`Archival.invalidate` takes a `rid` and no caller, so a primary turn can
invalidate mid-conversation and nothing notices: **0** of the store's **4**
public methods restrict the writer. Running dedup off the critical path is
enforced by where the code was typed.

Structure: `dedup()` is the sleep-pass scan; `RECORDS` is built so three
pairs sit above the threshold and two plausible duplicates sit below it.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "08-memory-blocks-sleep-time-compute"
THRESHOLD = 0.9
BASE = "ava ships agents for a living in lisbon"
RECORDS = (
    BASE,                                                    # kept
    BASE + " today",                                         # 7/8 -> survives
    "the curriculum targets senior and staff engineers not junior developers now",
    "the curriculum targets senior and staff engineers not junior developers",
    "tool chains drift after twenty steps per the bfcl v4 report published",
    "tool chains drift after twenty steps per the bfcl v4 report",
    "sleep time compute consolidates memory blocks between turns off the path",
    "sleep time compute consolidates memory blocks between turns off the path now",
    "mem0 fuses vector key value and graph stores",
    "letta adds a recall tier between core and archival",
    "the digest is sent to the sales channel each friday",
    "citation loss is the third failure mode named in the lesson",
)
TEN = "one two three four five six seven eight nine ten"


def overlap(left, right):
    first, second = set(left.lower().split()), set(right.lower().split())
    return round(len(first & second) / len(first | second), 3)


def dedup(archival, threshold=THRESHOLD):
    """The sleep pass: keep the earliest record of each near-duplicate group."""
    kept, collapsed = [], []
    for record in archival.valid_records():
        twin = next((other for other in kept
                     if overlap(other.text, record.text) > threshold), None)
        if twin is None:
            kept.append(record)
            continue
        archival.invalidate(record.rid)
        collapsed.append((twin.rid, record.rid))
    return collapsed


def build(ref):
    archival = ref.Archival()
    for text in RECORDS:
        archival.insert(text)
    return archival


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    archival = build(ref)
    collapsed = dedup(archival)
    primary = ref.PrimaryAgent(ref.BlockStore(), build(ref))
    before = len(primary.archival.valid_records())
    primary.archival.invalidate("a001")
    public = [name for name in dir(ref.Archival)
              if not name.startswith("_") and callable(getattr(ref.Archival, name))]
    return {
        "records": len(RECORDS), "collapsed": len(collapsed),
        "valid": len(archival.valid_records()), "audited": len(archival.all_records()),
        "pairs": collapsed,
        "record_fields": list(ref.ArchivalRecord.__dataclass_fields__),
        "edge": (overlap(TEN, TEN + " eleven"), overlap(TEN, TEN + " eleven twelve")),
        "edge_collapses": (overlap(TEN, TEN + " eleven") > THRESHOLD,
                           overlap(TEN, TEN + " eleven twelve") > THRESHOLD),
        "primary_can_invalidate": before - len(primary.archival.valid_records()),
        "public_methods": public,
        "gated": [name for name in public if "actor" in name or "writer" in name],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: three pairs collapse, nine of twelve records stay valid",
            all([result["records"] == 12, result["collapsed"] == 3,
                 result["valid"] == 9, result["audited"] == 12,
                 len(result["pairs"]) == 3]),
            f"the sleep pass collapses {result['collapsed']} near-duplicate pairs "
            f"{result['pairs']}, leaving {result['valid']} of {result['records']} valid "
            f"while all {result['audited']} are still present in all_records(). The pass "
            "is auditable rather than destructive",
        ),
        practice.Check(
            "FINDING: nothing records why a record was invalidated",
            all([result["record_fields"] == ["rid", "text", "valid"],
                 len(result["record_fields"]) == 3]),
            f"ArchivalRecord carries {result['record_fields']}, so a record dropped as a "
            "duplicate and one dropped as contradicted by the human block are the same "
            "object afterwards. With no superseded_by, an operator cannot tell staleness "
            "from redundancy and the pass cannot be undone selectively",
        ),
        practice.Check(
            "FINDING: 0.9 on token sets is a knife edge",
            all([result["edge"] == (0.909, 0.833),
                 result["edge_collapses"] == (True, False)]),
            f"against a ten-token record, one extra word scores {result['edge'][0]} and "
            f"collapses while two extra words score {result['edge'][1]} and survive "
            f"{result['edge_collapses']}. Two sentences a reader would call the same "
            "fact sit on opposite sides of the threshold",
        ),
        practice.Check(
            "FINDING: the boundary is a comment",
            all([result["primary_can_invalidate"] == 1, result["gated"] == [],
                 "invalidate" in result["public_methods"],
                 len(result["public_methods"]) == 4]),
            f"PrimaryAgent holds .archival and invalidate takes a rid and no caller, so a "
            f"primary turn removes {result['primary_can_invalidate']} record mid-"
            f"conversation and nothing notices. {len(result['gated'])} of the store's "
            f"{len(result['public_methods'])} public methods restrict the writer",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
