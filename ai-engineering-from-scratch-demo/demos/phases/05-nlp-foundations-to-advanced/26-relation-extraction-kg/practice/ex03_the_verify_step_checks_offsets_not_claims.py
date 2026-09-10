"""Exercise 3 — the verify step checks offsets, not claims.

    **Hard.** Build the AEVS pipeline: extract with LLM + verify spans against
    source. Measure hallucination rate before vs after the verify step on 50
    Wikipedia-style sentences.

Reading of the exercise: no LLM is reachable, so the extractor is the lesson's
own patterns and the "hallucinations" are its false positives. Measured that way
the answer is immediate and unhelpful: over exercise 1's thirteen sentences,
`extract` returns 10 triples and `verify` keeps **10**. The rejection rate is
**0.0000**, and it is 0.0000 for every possible input.

Both of `verify`'s conditions are tautologies for a regex match on the same text.
`text[s:e] != t["evidence"]` compares the string against a slice of itself, since
`span` came from `m.start(), m.end()` and `evidence` from `m.group(0)` of that
match. `t["subject"] not in text` and `t["object"] not in text` are substring
tests on groups of that same match. Nothing extracted can fail them, so the
before-and-after numbers the exercise asks for are equal by construction.

Applied to triples from a different source -- seven hand-written cases of the
kinds an LLM actually produces -- it catches **2 of the 5 defective ones and
wrongly rejects one of the 2 sound ones**. It rejects an entity that is not in
the text and an evidence string that does not match its offsets. It accepts a
subject given as `Tim` where the text says `Tim Cook`, because `in` is a
substring test; it accepts a fabricated relation with real entities and real
evidence, because the relation is never checked against anything; and it accepts
a triple lifted from a clause the sentence denies. The one sound triple it
rejects is a span off by one character, whose claim is correct.

The split is exact: **offsets are checked to the character and content is not
checked at all.** A span one byte off is rejected though the claim is right; a
claim that reverses the relation is kept because the span is right.

And the step cannot be applied to the source it exists for. `verify` reads
`t["span"]`, and a triple that came from a language model has no character
offsets -- calling it raises `KeyError: 'span'`. The AEVS step as written
verifies provenance for triples whose provenance was never in doubt.

Structure: exercise 1's sentences are loaded rather than copied; `CASES` holds
one hand-written triple per defect type; `judge` runs the lesson's `verify` on
one; `rejection_rate` measures the extract-then-verify pipeline; `no_span` calls
`verify` on a triple with no offsets.
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "26-relation-extraction-kg"

EX1 = practice.load_module(
    pathlib.Path(__file__).resolve().parent / "ex01_precision_is_no_higher_than_recall.py")
ROWS = EX1.ROWS
TEXT = "Tim Cook became CEO of Apple in 2011. The board denied that Larry Page founded Acme."
EVIDENCE = "Tim Cook became CEO of Apple"
AT = TEXT.index(EVIDENCE)
DENIED = "Larry Page founded Acme"


def triple(subject, relation, obj, evidence=EVIDENCE, start=AT, shift=0):
    """One candidate triple in the shape `verify` expects."""
    return {"subject": subject, "relation": relation, "object": obj,
            "span": (start + shift, start + len(evidence)), "evidence": evidence}


CASES = {
    "faithful": (triple("Tim Cook", "P169", "Apple"), True),
    "entity absent from the text": (triple("Larry Ellison", "P169", "Oracle"), False),
    "subject given as a partial name": (triple("Tim", "P169", "Apple"), False),
    "relation fabricated, entities real": (triple("Tim Cook", "P19", "Apple"), False),
    "evidence not matching its offsets": (triple("Tim Cook", "P169", "Apple",
                                                 evidence="Tim Cook founded Apple"), False),
    "offsets off by one": (triple("Tim Cook", "P169", "Apple", shift=1), True),
    "lifted from a denied clause": (triple("Larry Page", "P112", "Acme", evidence=DENIED,
                                           start=TEXT.index(DENIED)), False),
}


def judge(ref, candidate):
    """Whether the lesson's `verify` keeps one candidate triple."""
    return bool(ref.verify([candidate], TEXT))


def rejection_rate(ref):
    """Triples extracted and triples surviving verification, over exercise 1's sentences."""
    extracted = kept = 0
    for text, _ in ROWS:
        found = ref.extract(text)
        extracted += len(found)
        kept += len(ref.verify(found, text))
    return {"extracted": extracted, "kept": kept,
            "rate": round(1 - kept / extracted, 4) if extracted else 0.0}


def no_span(ref):
    """What `verify` does with a triple that has no character offsets."""
    try:
        ref.verify([{"subject": "A", "relation": "P19", "object": "B", "evidence": "A"}], TEXT)
    except KeyError as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    verdicts = {name: judge(ref, candidate) for name, (candidate, _) in CASES.items()}
    wrong = {name for name, (_, sound) in CASES.items() if not sound}
    return {
        "pipeline": rejection_rate(ref),
        "verdicts": verdicts,
        "caught": sorted(name for name in wrong if not verdicts[name]),
        "missed": sorted(name for name in wrong if verdicts[name]),
        "defects": len(wrong),
        "offset_rejected": not verdicts["offsets off by one"],
        "faithful_kept": verdicts["faithful"],
        "no_span": no_span(ref),
    }


def verify(result):
    pipeline = result["pipeline"]
    return [
        practice.Check(
            "ANSWER: the rejection rate on the extractor's own output is 0.0000",
            pipeline["rate"] == 0.0,
            f"`extract` returns {pipeline['extracted']} triples over exercise 1's sentences and "
            f"`verify` keeps {pipeline['kept']}. The hallucination rate before and after the "
            "verify step is the same number, so the measurement the exercise asks for cannot "
            "produce two values",
        ),
        practice.Check(
            "MECHANISM: both of its conditions are tautologies for a regex match",
            pipeline["extracted"] == pipeline["kept"],
            "`span` comes from `m.start(), m.end()` and `evidence` from `m.group(0)` of the same "
            "match, so `text[s:e] != evidence` compares a string against a slice of itself; and "
            "`subject not in text` is a substring test on a group of that match. Nothing "
            "extracted can fail either",
        ),
        practice.Check(
            "FINDING: on triples from another source it catches half the defects",
            len(result["caught"]) < result["defects"],
            f"over {len(CASES)} hand-written candidates of the kinds a language model produces, "
            f"it rejects {result['caught']} and accepts {result['missed']} -- "
            f"{len(result['caught'])} of {result['defects']} defects caught, while wrongly "
            "rejecting one of the two sound triples",
        ),
        practice.Check(
            "MECHANISM: the split is offsets checked exactly, content not checked at all",
            result["offset_rejected"] and result["verdicts"]["relation fabricated, entities real"],
            "a span one character off is rejected although the claim is right, while a triple "
            "whose relation was invented is kept because its offsets are right. The relation is "
            "never compared to anything, and the entity check is `in`, which accepts `Tim` for "
            "`Tim Cook`",
        ),
        practice.Check(
            "FINDING: a denied clause verifies, because it really is in the text",
            result["verdicts"]["lifted from a denied clause"],
            "'The board denied that Larry Page founded Acme' contains the span, the subject and "
            "the object, so the triple passes. Verifying provenance is not verifying the "
            "assertion, and exercise 1's four hedged false positives all survive this step",
        ),
        practice.Check(
            "CONTROL: and it cannot be applied to the source it exists for",
            result["no_span"] is not None,
            f"`verify` reads `t['span']`, which a triple from a language model does not have: "
            f"{result['no_span']}. The AEVS step as written verifies offsets for triples whose "
            "offsets were never in doubt",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
