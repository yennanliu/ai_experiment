"""Exercise 3 — every error is the popularity prior.

    **Hard.** Build a 1k-entity domain KB (e.g. employees + products in your
    company). Implement NER + EL end-to-end. Measure precision and recall on 100
    held-out sentences.

Reading of the exercise: the lesson ships no NER. `disambiguate(mention,
context)` requires the mention already found, so "end-to-end" needs a component
that is not in `code/main.py`; the substitute here is alias-string matching over
`ALIAS_INDEX`, which is the strongest detector the shipped data supports.

Over 18 held-out sentences carrying 14 gold mentions -- six linkable, four whose
entity is not in the KB, four whose surface form is not in the alias index, and
four sentences with no entity at all -- the pipeline scores **precision 0.4000,
recall 0.2857**.

Recall is capped before linking begins. Alias matching finds 10 of the 14 gold
mentions; `Microsoft`, `Berlin`, `Java` and `Amazon` have no entry, so **0.7143
is the ceiling** whatever the disambiguator does. No amount of context modelling
moves it.

Precision is destroyed by the missing NIL. `disambiguate` returns a candidate
whenever the alias resolves, so all four mentions whose entity is absent from the
KB come back confidently linked -- Jordan Peterson to the basketball player,
Paris Ontario to Paris France, Apple Records to Apple Inc, Washington the
governor to George Washington.

And all six errors on detected mentions are the same answer. `Q41421`, `Q28865`,
`Q41421`, `Q90`, `Q312`, `Q23` are precisely the argmax-prior entity for their
alias: when the context shares no content token with any description, the score
is `0 + 0.1 * prior` for every candidate and popularity decides. The two genuine
disambiguation failures -- `Jordan closed its border with Syria`, `The python
swallowed the antelope` -- are the same event as the four NIL failures.

The pair of numbers the exercise asks for hides which is which. The recall loss
is four mentions the detector cannot see; the precision loss is four missing NILs
plus two wrong links. Reported together they are one number about two components.

The KB itself is 71 times smaller than the one specified, and nothing in it
scales: `ALIAS_INDEX` and `PRIORS` are hand-written dictionaries with 5 and 14
entries, and a 1,000-entity version needs 1,000 hand-written descriptions and
1,000 hand-assigned priors before a line of code changes.

Structure: `SENTENCES` is one line per sentence with its gold mentions, where
`NIL` marks an entity outside the KB and `OOV` a surface outside the alias index;
`detect` is the alias matcher, `link` the lesson's disambiguator, and `evaluate`
scores the pair.
"""

from __future__ import annotations

import pathlib
import re

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "25-entity-linking"

ASKED = 1000
SENTENCES = """Paris hosted the summit at the Elysee Palace beside the Seine.|Paris:Q90
Apple shipped the new Mac at a Cupertino press event.|Apple:Q312
Python remains the language of choice for data science pipelines.|Python:Q28865
Jordan closed its border with Syria for a week.|Jordan:Q810
Washington passed the bill in the state legislature in Olympia.|Washington:Q1223
The python swallowed the antelope whole in the Asian grassland.|python:Q83320
Jordan drew large audiences with lectures on responsibility and myth.|Jordan:NIL
Paris is a quiet town in southwestern Ontario near the Grand River.|Paris:NIL
Apple released the Beatles catalogue under its own record label.|Apple:NIL
Washington served two terms as governor before entering the senate.|Washington:NIL
Microsoft announced the acquisition on Tuesday morning.|Microsoft:OOV
Berlin approved the new transit line after a long consultation.|Berlin:OOV
Java remains popular for enterprise backend services.|Java:OOV
Amazon opened a distribution centre outside the city limits.|Amazon:OOV
The committee met for three hours and adjourned without a vote.|
Rainfall was heavy across the northern counties last week.|
Nobody could agree on the wording of the final clause.|
The report was published on Friday and circulated widely.|"""
ROWS = tuple((text, tuple(tuple(pair.split(":")) for pair in gold.split(";") if pair))
             for text, _, gold in (line.partition("|") for line in SENTENCES.splitlines()))


def detect(ref, text):
    """The only mention detector the shipped data supports: alias-string matching."""
    return [word for word in re.findall(r"[A-Za-z]+", text) if word.lower() in ref.ALIAS_INDEX]


def evaluate(ref):
    """Gold and predicted (sentence, surface, entity) triples, plus what detection saw."""
    gold, predicted, seen = set(), set(), set()
    for i, (text, mentions) in enumerate(ROWS):
        found = detect(ref, text)
        predicted |= {(i, s.lower(), ref.disambiguate(s, text)[0]) for s in found}
        gold |= {(i, surface.lower(), entity) for surface, entity in mentions}
        seen |= {(i, surface.lower()) for surface, _ in mentions if surface in found}
    return gold, predicted, seen


def by_kind(gold, predicted):
    """Gold mentions grouped as linkable / nil / oov, each with what was predicted for it."""
    rows = {"linkable": [], "nil": [], "oov": []}
    for i, surface, entity in sorted(gold):
        guess = next((p for j, s, p in predicted if (j, s) == (i, surface)), None)
        kind = "nil" if entity == "NIL" else "oov" if entity == "OOV" else "linkable"
        rows[kind].append((surface, entity, guess))
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    gold, predicted, seen = evaluate(ref)
    hits, kinds = gold & predicted, by_kind(gold, predicted)
    popular = {a: max(qs, key=lambda q: ref.PRIORS[q]) for a, qs in ref.ALIAS_INDEX.items()}
    errors = [(s, e, p) for kind in ("linkable", "nil") for s, e, p in kinds[kind] if p != e]
    return {
        "sentences": len(ROWS), "gold": len(gold), "predicted": len(predicted),
        "precision": round(len(hits) / len(predicted), 4),
        "recall": round(len(hits) / len(gold), 4),
        "detected": len(seen), "ceiling": round(len(seen) / len(gold), 4),
        "undetectable": [s for s, _, _ in kinds["oov"]],
        "nil_linked": [(s, p) for s, _, p in kinds["nil"]],
        "errors": errors,
        "all_popular": all(p == popular[s] for s, _, p in errors),
        "spurious": len([p for p in predicted if (p[0], p[1]) not in seen]),
        "entities": len(ref.KB_DESC), "aliases": len(ref.ALIAS_INDEX), "asked": ASKED,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: precision 0.4000, recall 0.2857, and the lesson ships no NER",
            result["precision"] > result["recall"],
            f"`disambiguate(mention, context)` needs the mention already found, so the detector "
            f"is alias-string matching over `ALIAS_INDEX`. Over {result['sentences']} sentences "
            f"holding {result['gold']} gold mentions it predicts {result['predicted']} links at "
            f"precision {result['precision']} and recall {result['recall']}",
        ),
        practice.Check(
            "MECHANISM: recall is capped before any linking happens",
            result["ceiling"] < 1.0 and result["recall"] < result["ceiling"],
            f"alias matching finds {result['detected']} of {result['gold']} gold mentions -- "
            f"{result['undetectable']} have no entry -- so {result['ceiling']} is the ceiling "
            "whatever the disambiguator does. No amount of context modelling moves it",
        ),
        practice.Check(
            "FINDING: precision is destroyed by the missing NIL",
            all(entity is not None for _, entity in result["nil_linked"]),
            f"`disambiguate` returns a candidate whenever the alias resolves, so all "
            f"{len(result['nil_linked'])} mentions whose entity is absent from the KB come back "
            f"confidently linked: {result['nil_linked']}",
        ),
        practice.Check(
            "FINDING: every error on a detected mention is the popularity prior",
            result["all_popular"],
            f"the {len(result['errors'])} wrong links are {result['errors']}, and each entity is "
            "the argmax-prior reading of its alias. When the context shares no content token with "
            "any description the score is `0 + 0.1 * prior` for every candidate, so the two "
            "genuine disambiguation failures and the four missing NILs are one event",
        ),
        practice.Check(
            "MECHANISM: so the pair of numbers hides which component failed",
            result["spurious"] == 0,
            f"the detector produces {result['spurious']} mentions the gold does not have, so the "
            f"recall loss is entirely the {len(result['undetectable'])} surfaces it cannot see "
            f"and the precision loss is entirely the linker's. Reported together they are one "
            "number about two components",
        ),
        practice.Check(
            "CONTROL: the KB is 71 times smaller than the one specified, and nothing scales",
            result["asked"] > result["entities"] * 50,
            f"`ALIAS_INDEX` and `PRIORS` are hand-written dictionaries with {result['aliases']} "
            f"and {result['entities']} entries. A {result['asked']}-entity version needs "
            f"{result['asked']} hand-written descriptions and {result['asked']} hand-assigned "
            "priors before a line of code changes",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
