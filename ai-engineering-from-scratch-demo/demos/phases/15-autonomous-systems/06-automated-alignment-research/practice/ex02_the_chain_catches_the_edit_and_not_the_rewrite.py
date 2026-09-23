"""Exercise 2 — the chain catches the edit and not the rewrite.

    Modify the simulator so one agent attempts log tampering. Confirm the
    append-only log detects it. Write one paragraph describing exactly what
    the detection signature looks like in the log.

Reading of the exercise: the modification already ships -- `attempt_tamper`
rewrites AAR-A's worst score upward -- so the work is the confirmation, and a
confirmation is only worth writing if it also establishes what the log does
*not* catch. The tamperer is therefore given a second, strictly easier
attempt: re-chain after editing.

**ANSWER: exactly one record is flagged, and nothing after it.** At the
lesson's own seed, `verify()` returns index **12** out of **15**. The
signature is narrow and specific: at that index the recomputed payload hash
does not equal the stored `my_hash`, while `prev_hash` still equals the
previous record's stored hash -- and every record **13** and **14** verifies
clean, because the walk carries the *stored* hash forward rather than the
recomputed one. The break does not propagate; it is a single mismatched cell.

**FINDING: re-chaining defeats `verify()` completely.** Re-posting the
tampered records into a fresh `Forum` recomputes every hash, and `verify()`
then returns **0** flags on a log whose numbers have been altered. The only
surviving evidence is the head: **11cb5fb4fa631b2c** honest against
**759402faba5d56e3** tampered.

**FINDING: nothing in the module ever compares that head to anything.**
`Forum` has **2** fields, `genesis` is the constant `0000000000000000`, and
the class exposes **3** methods -- `head`, `post`, `verify` -- none of which
takes an expected value. `tamper_demo` prints "an out-of-sandbox backup
reveals the tamper on cross-check" and performs no cross-check, so the
property the lesson's section title rests on is asserted in a string.

**FINDING: the detectable tamper is the one an agent would not make.**
Editing a field in place leaves a mismatched hash next to it; recomputing the
chain is the same edit plus **1** call to `post`. So the append-only log
detects the attempt that does not bother to re-run the code that wrote the
log -- which is the attempt an agent with the repository in its sandbox has no
reason to choose.

Structure: `tampered()` runs the shipped attempt; `rechained()` re-posts its
records so every hash is consistent again.
"""

from __future__ import annotations

import inspect
import random

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "06-automated-alignment-research"

SEED = 3                          # the seed main() sets before the demo


def rebuild(ref, records):
    """Post the given records into a fresh Forum, recomputing every hash."""
    forum = ref.Forum()
    for record in records:
        forum.post(ref.ForumRecord(author=record.author, task=record.task,
                                   regime=record.regime, result=record.result,
                                   prev_hash=""))
    return forum


def honest(ref):
    random.seed(SEED)
    return rebuild(ref, ref.run_regime("free"))


def tampered(ref):
    random.seed(SEED)
    return ref.attempt_tamper(ref.run_regime("free"))


def signature(forum, index):
    """What the flagged record looks like beside its neighbours."""
    record = forum.records[index]
    following = forum.records[index + 1:]
    return {
        "prev_hash_matches": record.prev_hash == forum.records[index - 1].my_hash,
        "following": len(following),
        "following_clean": all(row not in following for row in
                               [forum.records[i] for i in forum.verify()]),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    clean, dirty = honest(ref), tampered(ref)
    flagged = dirty.verify()
    rechained = rebuild(ref, dirty.records)
    return {
        "records": len(dirty.records),
        "flagged": flagged,
        **signature(dirty, flagged[0]),
        "rechained_flags": rechained.verify(),
        "honest_head": clean.head(),
        "tampered_head": rechained.head(),
        "heads_differ": clean.head() != rechained.head(),
        "forum_fields": list(ref.Forum.__dataclass_fields__),
        "genesis": ref.Forum().genesis,
        "methods": sorted(name for name, value in vars(ref.Forum).items()
                          if callable(value) and not name.startswith("_")),
        "verify_takes_expected": len(inspect.signature(ref.Forum.verify).parameters),
        "demo_claims_crosscheck": "cross-check" in inspect.getsource(ref.tamper_demo),
        "demo_compares_heads": "head()" in inspect.getsource(ref.tamper_demo),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one record flagged, and nothing after it",
            all([result["flagged"] == [12], result["records"] == 15,
                 result["prev_hash_matches"], result["following"] == 2,
                 result["following_clean"]]),
            f"verify() returns {result['flagged']} of {result['records']} records: the "
            f"recomputed hash differs from the stored one while prev_hash still matches "
            f"its predecessor, and both of the {result['following']} records after it "
            "verify clean",
        ),
        practice.Check(
            "FINDING: re-chaining defeats verify() completely",
            all([result["rechained_flags"] == [], result["heads_differ"],
                 result["honest_head"] == "11cb5fb4fa631b2c",
                 result["tampered_head"] == "759402faba5d56e3"]),
            f"re-posting the tampered records gives {len(result['rechained_flags'])} "
            f"flags on an altered log; the only surviving evidence is the head, "
            f"{result['honest_head']} honest against {result['tampered_head']}",
        ),
        practice.Check(
            "FINDING: nothing in the module ever compares that head to anything",
            all([len(result["forum_fields"]) == 2, result["genesis"] == "0" * 16,
                 result["methods"] == ["head", "post", "verify"],
                 result["verify_takes_expected"] == 1,
                 result["demo_claims_crosscheck"], not result["demo_compares_heads"]]),
            f"Forum has {len(result['forum_fields'])} fields, a constant genesis and "
            f"{len(result['methods'])} methods {result['methods']}, none taking an "
            "expected value; tamper_demo prints that a backup reveals the tamper and "
            "compares nothing",
        ),
        practice.Check(
            "FINDING: the detectable tamper is the one an agent would not make",
            all([result["flagged"] == [12], result["rechained_flags"] == []]),
            "editing a field in place leaves a mismatched hash beside it; recomputing "
            "the chain is the same edit plus one call to post, and the log that "
            "detected the first detects none of the second",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
