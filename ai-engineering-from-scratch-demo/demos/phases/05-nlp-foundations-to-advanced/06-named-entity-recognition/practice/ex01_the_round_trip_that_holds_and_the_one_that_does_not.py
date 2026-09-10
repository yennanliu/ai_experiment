"""Exercise 1 — the round trip that holds and the one that does not.

    **Easy.** Implement `bio_to_spans` (the inverse of `spans_to_bio`) and verify
    round-trip consistency on 10 sentences.

Reading of the exercise: `bio_to_spans` already ships in the lesson's
`code/main.py`, so the work is the verification, and the verification has two
directions that behave completely differently. Spans to BIO and back is the
identity on all 10 sentences below and on every well-formed span list -- it even
survives two adjacent same-type entities, because BIO2's `B-` restarts a span
where an `I-` would have merged them.

BIO to spans and back is not the identity. Over all 625 label sequences of
length 4 on two entity types, exactly 153 are fixed points, and 153 is not a
coincidence: it is the number of well-formed IOB2 sequences, counted here by
the transition rule rather than assumed. The composition is a projection onto
valid IOB2, and it performs that projection by deleting. All 472 sequences that
change contain an `I-` tag, and every change removes it in silence --
`['O', 'O', 'O', 'I-A']` comes back `['O', 'O', 'O', 'O']` with no exception and
no warning. That matters immediately: CoNLL-2003, the dataset exercise 2 names,
ships in IOB1, where a lone `I-` legitimately opens an entity. Run gold IOB1
through this pair and entities disappear.

`spans_to_bio` validates nothing on its side either. A zero-width span comes
back one token wide, a reversed span comes back as its start, and two
overlapping spans come back with a token missing from the longer one -- the
later span simply writes over the earlier one's labels.

Structure: `SENTENCES` is the 10-sentence fixture with hand-written gold spans.
`fixed_points` walks every length-`WIDTH` sequence over `TAGS` and counts the
ones the composition leaves alone; `well_formed` counts the same set by IOB2's
own rule, so the two numbers are derived independently. `MALFORMED` is the span
side, one entry per way a span list can be ill-formed.
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "06-named-entity-recognition"

TAGS = ("O", "B-A", "I-A", "B-B", "I-B")
WIDTH = 4
SENTENCES = (
    ("The New York City mayor visited OpenAI .".split(), [(1, 4, "GPE"), (6, 7, "ORG")]),
    ("Apple sued Google over iPhone sales .".split(), [(0, 1, "ORG"), (2, 3, "ORG"), (4, 5, "PRODUCT")]),
    ("She flew from San Francisco to Tokyo .".split(), [(3, 5, "GPE"), (6, 7, "GPE")]),
    ("Microsoft and Amazon both filed .".split(), [(0, 1, "ORG"), (2, 3, "ORG")]),
    ("No entities appear in this sentence .".split(), []),
    ("Anthropic released Claude in the US .".split(), [(0, 1, "ORG"), (2, 3, "PRODUCT"), (5, 6, "GPE")]),
    ("The United Arab Emirates signed it .".split(), [(1, 4, "GPE")]),
    ("Google Google Google .".split(), [(0, 1, "ORG"), (1, 2, "ORG"), (2, 3, "ORG")]),
    ("Netflix .".split(), [(0, 1, "ORG")]),
    ("Meta hired engineers from Nvidia last year .".split(), [(0, 1, "ORG"), (4, 5, "ORG")]),
)
MALFORMED = (("zero width", [(2, 2, "ORG")]), ("reversed", [(3, 1, "ORG")]),
             ("overlapping", [(0, 3, "ORG"), (1, 2, "GPE")]))
PADDING = ["w"] * WIDTH


def round_trips(ref) -> list:
    """spans -> BIO -> spans, on the fixture."""
    return [ref.bio_to_spans(tokens, ref.spans_to_bio(tokens, spans)) == spans
            for tokens, spans in SENTENCES]


def fixed_points(ref) -> tuple:
    """Sequences the BIO -> spans -> BIO composition leaves alone, and the rest."""
    kept, changed = [], []
    for sequence in itertools.product(TAGS, repeat=WIDTH):
        labels = list(sequence)
        (kept if ref.spans_to_bio(PADDING, ref.bio_to_spans(PADDING, labels)) == labels
         else changed).append(labels)
    return kept, changed


def well_formed(labels) -> bool:
    """IOB2: an I- tag must continue a tag of its own type."""
    return all(not tag.startswith("I-")
               or (i and labels[i - 1][2:] == tag[2:] and labels[i - 1] != "O")
               for i, tag in enumerate(labels))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    kept, changed = fixed_points(ref)
    return {
        "trips": round_trips(ref), "sentences": len(SENTENCES),
        "kept": len(kept), "changed": len(changed), "total": len(TAGS) ** WIDTH,
        "by_rule": sum(1 for s in itertools.product(TAGS, repeat=WIDTH) if well_formed(list(s))),
        "all_have_i": all(any(t.startswith("I-") for t in labels) for labels in changed),
        "example": (changed[0], ref.spans_to_bio(PADDING, ref.bio_to_spans(PADDING, changed[0]))),
        "iob1": ref.bio_to_spans(["a", "b"], ["I-ORG", "I-ORG"]),
        "adjacent": ref.bio_to_spans(*(SENTENCES[7][0],
                                       ref.spans_to_bio(SENTENCES[7][0], SENTENCES[7][1]))),
        "malformed": {name: ref.bio_to_spans(PADDING + ["w"],
                                             ref.spans_to_bio(PADDING + ["w"], spans))
                      for name, spans in MALFORMED},
    }


def verify(result):
    kept, total, malformed = result["kept"], result["total"], result["malformed"]
    return [
        practice.Check(
            "ANSWER: spans -> BIO -> spans is the identity on all 10 sentences",
            all(result["trips"]),
            f"{sum(result['trips'])}/{result['sentences']} sentences recover their gold spans "
            f"exactly, including the empty one and the three adjacent same-type entities of "
            f"'Google Google Google', which come back as {result['adjacent']} -- BIO2's B- restarts "
            f"a span where an I- would have merged the three into one"),
        practice.Check(
            "MECHANISM: the other direction is not the identity -- 153 of 625 sequences survive",
            kept < total and result["changed"] == total - kept,
            f"over every length-{WIDTH} label sequence on two entity types, BIO -> spans -> BIO "
            f"leaves {kept} of {total} unchanged and rewrites {result['changed']}. The pair is not "
            f"a bijection; it is a projection, and the exercise's word 'inverse' holds on one side "
            f"only"),
        practice.Check(
            "MECHANISM: the 153 are exactly the well-formed IOB2 sequences, counted independently",
            result["by_rule"] == kept,
            f"counting by IOB2's own rule -- an I- tag must continue a tag of its own type -- gives "
            f"{result['by_rule']}, against {kept} measured by round-tripping. The composition maps "
            f"any sequence onto the nearest valid IOB2 one, which is a reasonable thing to do and "
            f"is not what 'inverse' describes"),
        practice.Check(
            "FINDING: every rewrite is a silent deletion, and all 472 involve an I- tag",
            result["all_have_i"],
            f"{result['changed']} of {total} sequences change and every one contains an I-: "
            f"{result['example'][0]} comes back {result['example'][1]}. No exception, no warning, "
            f"no return value to check -- an entity is simply gone"),
        practice.Check(
            "FINDING: CoNLL-2003 is IOB1, so this drops gold entities on exercise 2's own dataset",
            result["iob1"] == [],
            f"in IOB1 a lone I- legitimately opens an entity, and CoNLL-2003 ships in IOB1. "
            f"`bio_to_spans` on ['I-ORG', 'I-ORG'] returns {result['iob1']} -- the entity is not "
            f"mis-bounded, it is absent. Exercise 2 says to train on that data and report per-entity "
            f"F1, and this is the function that would extract the entities"),
        practice.Check(
            "CONTROL: spans_to_bio validates nothing, so three span shapes fail the good direction",
            malformed["zero width"] == [(2, 3, "ORG")] and malformed["reversed"] == [(3, 4, "ORG")]
            and malformed["overlapping"] == [(0, 1, "ORG"), (1, 2, "GPE")],
            f"a zero-width span (2, 2) comes back {malformed['zero width']}, one token wide. A "
            f"reversed span (3, 1) comes back {malformed['reversed']}. And two overlapping spans "
            f"come back {malformed['overlapping']} -- the later span writes over the earlier one's "
            f"labels, so the ORG loses two of its three tokens"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
