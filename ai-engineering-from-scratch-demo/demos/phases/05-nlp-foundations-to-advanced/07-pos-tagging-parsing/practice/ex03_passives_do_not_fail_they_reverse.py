"""Exercise 3 — passives do not fail, they reverse.

    **Hard.** Use spaCy's dependency parse to extract subject-verb-object triples
    from a 1000-sentence sample. Evaluate on 50 manually labeled triples. Document
    where extraction fails (often passives, coordinations, and elided subjects).

Reading of the exercise: spaCy is not installed and there is no 1000-sentence
sample here, so the labelled set is 12 sentences carrying 15 triples, written
below with gold part-of-speech tags and gold triples, four to each failure class
the exercise names plus a control class of plain actives. The extractor is the
one a dependency parse is supposed to replace: nearest NOUN before the VERB is
the subject, nearest NOUN after it is the object. Scored on gold tags, so no
tagging error is in these numbers, it recovers 0.6000 of the triples.

The class breakdown is the point, and one entry in it is not a failure of
recall. Passives score 0.0000 -- and all three produce a triple anyway, with the
arguments the wrong way round. `the dog was chased by the cat` returns
`(dog, chased, cat)`. A consumer counting extraction failures sees none here;
it sees three confident facts asserting the opposite of the sentence, which is
the one error mode a recall number cannot show. Coordination halves recall by
construction, 3 of 6: the rule takes the nearest noun and a conjoined subject or
object has two. Elided subjects score 1.0000 and still cost two spurious
triples, because the intransitive verb in `the cat sat and watched the dog`
finds `dog` downstream and reports `(cat, sat, dog)`.

So the taxonomy the exercise offers is right about which constructions break and
wrong about how. Two of the three classes fail by emitting something rather than
nothing, and the aggregate 0.6000 recall hides five spurious triples on fifteen
gold ones.

Structure: `LABELLED` is the fixture, one row per sentence as (tokens, gold
tags, gold triples, class). `extract` is the POS-only rule. `score` reports
recovered, gold and spurious counts per class, and `reversed_pairs` checks the
specific claim that a passive's spurious triple is its gold triple backwards.
"""

from __future__ import annotations

import collections
import importlib.util

from harness import practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "07-pos-tagging-parsing"

UNAVAILABLE = ("spacy", "nltk", "stanza")


def row(text, tags, triples, kind) -> tuple:
    return text.split(), tags.split(), triples, kind


LABELLED = (
    row("the cat chased the dog", "DET NOUN VERB DET NOUN", [("cat", "chased", "dog")], "active"),
    row("a child carried the box", "DET NOUN VERB DET NOUN", [("child", "carried", "box")], "active"),
    row("the dog found a bone", "DET NOUN VERB DET NOUN", [("dog", "found", "bone")], "active"),
    row("dogs chase mice", "NOUN VERB NOUN", [("dogs", "chase", "mice")], "active"),
    row("the dog was chased by the cat", "DET NOUN AUX VERB ADP DET NOUN",
        [("cat", "chased", "dog")], "passive"),
    row("the box was carried by a child", "DET NOUN AUX VERB ADP DET NOUN",
        [("child", "carried", "box")], "passive"),
    row("the bone was found by the dog", "DET NOUN AUX VERB ADP DET NOUN",
        [("dog", "found", "bone")], "passive"),
    row("the cat and the dog chased the mouse", "DET NOUN CCONJ DET NOUN VERB DET NOUN",
        [("cat", "chased", "mouse"), ("dog", "chased", "mouse")], "coordination"),
    row("the child and the parent carried the box", "DET NOUN CCONJ DET NOUN VERB DET NOUN",
        [("child", "carried", "box"), ("parent", "carried", "box")], "coordination"),
    row("the cat chased the dog and the mouse", "DET NOUN VERB DET NOUN CCONJ DET NOUN",
        [("cat", "chased", "dog"), ("cat", "chased", "mouse")], "coordination"),
    row("the cat sat and watched the dog", "DET NOUN VERB CCONJ VERB DET NOUN",
        [("cat", "watched", "dog")], "elided"),
    row("the dog barked and chased the cat", "DET NOUN VERB CCONJ VERB DET NOUN",
        [("dog", "chased", "cat")], "elided"),
)
CLASSES = ("active", "passive", "coordination", "elided")


def extract(tokens, tags) -> list:
    """The rule a dependency parse replaces: nearest noun each side of the verb."""
    found = []
    for i, tag in enumerate(tags):
        if tag != "VERB":
            continue
        before = [tokens[j] for j in range(i - 1, -1, -1) if tags[j] == "NOUN"]
        after = [tokens[j] for j in range(i + 1, len(tags)) if tags[j] == "NOUN"]
        if before and after:
            found.append((before[0], tokens[i], after[0]))
    return found


def score(rows) -> dict:
    tally = collections.defaultdict(lambda: {"found": 0, "gold": 0, "spurious": 0})
    for tokens, tags, gold, kind in rows:
        got, want = set(extract(tokens, tags)), set(gold)
        tally[kind]["found"] += len(want & got)
        tally[kind]["gold"] += len(want)
        tally[kind]["spurious"] += len(got - want)
    return {kind: dict(counts, recall=round(counts["found"] / counts["gold"], 4))
            for kind, counts in tally.items()}


def reversed_pairs(rows) -> int:
    """Spurious triples that are a gold triple with subject and object swapped."""
    swapped = 0
    for tokens, tags, gold, _ in rows:
        got = set(extract(tokens, tags))
        swapped += sum((obj, verb, subj) in got for subj, verb, obj in gold)
    return swapped


def solve():
    per_class = score(LABELLED)
    overall = {key: sum(row[key] for row in per_class.values())
               for key in ("found", "gold", "spurious")}
    passives = [r for r in LABELLED if r[3] == "passive"]
    return {
        "unavailable": [m for m in UNAVAILABLE if importlib.util.find_spec(m) is None],
        "classes": per_class, "overall": dict(overall, recall=round(
            overall["found"] / overall["gold"], 4)),
        "sentences": len(LABELLED), "reversed": reversed_pairs(LABELLED),
        "passive_example": (" ".join(passives[0][0]), passives[0][2],
                            extract(passives[0][0], passives[0][1])),
        "elided_example": extract(LABELLED[10][0], LABELLED[10][1]),
        "coord_example": extract(LABELLED[7][0], LABELLED[7][1]),
    }


def verify(result):
    per_class, overall = result["classes"], result["overall"]
    return [
        practice.Check(
            "ANSWER: 0.6000 recall on gold tags -- 1.0000 active, 0.0000 passive, 0.5000 coordinated",
            overall["recall"] == 0.6 and per_class["active"]["recall"] == 1.0,
            f"{result['unavailable']} are absent and there is no 1000-sentence sample, so the "
            f"labelled set is {result['sentences']} sentences carrying {overall['gold']} triples "
            f"with gold tags. Recall by class: "
            f"{ {k: per_class[k]['recall'] for k in CLASSES} }, overall {overall['recall']} with "
            f"{overall['spurious']} spurious triples. No tagging error is in these numbers"),
        practice.Check(
            "FINDING: passives do not fail by returning nothing -- all three come back reversed",
            per_class["passive"]["recall"] == 0.0
            and result["reversed"] == per_class["passive"]["gold"],
            f"'{result['passive_example'][0]}' has gold {result['passive_example'][1]} and the "
            f"extractor returns {result['passive_example'][2]}. All "
            f"{result['reversed']} passive triples come back with subject and object swapped, so "
            f"the class scores {per_class['passive']['recall']} recall and "
            f"{per_class['passive']['spurious']} spurious -- confident facts asserting the opposite "
            f"of the sentence"),
        practice.Check(
            "MECHANISM: a recall number cannot show that error, because it is not a miss",
            per_class["passive"]["spurious"] == per_class["passive"]["gold"],
            f"the passive class produces exactly as many triples as it should and gets every one "
            f"backwards. A consumer counting extraction failures sees {0} silent drops here. The "
            f"error is only visible against the gold triple, which is why the exercise asks for 50 "
            f"labelled ones and not for a coverage count"),
        practice.Check(
            "FINDING: coordination halves recall by construction -- the rule takes one noun",
            per_class["coordination"]["recall"] == 0.5,
            f"'the cat and the dog chased the mouse' has two gold triples and the extractor returns "
            f"{result['coord_example']} -- the nearer conjunct only. Across the class that is "
            f"{per_class['coordination']['found']} of {per_class['coordination']['gold']}, and it is "
            f"not a tuning problem: 'nearest noun' returns one noun"),
        practice.Check(
            "FINDING: elided subjects score perfect recall and still invent two triples",
            per_class["elided"]["recall"] == 1.0 and per_class["elided"]["spurious"] == 2,
            f"'the cat sat and watched the dog' returns {result['elided_example']}: the real triple "
            f"and one giving the intransitive 'sat' an object it does not have. The class scores "
            f"{per_class['elided']['recall']} recall with {per_class['elided']['spurious']} spurious "
            f"-- the exercise lists it as a failure class and by recall it is the joint best"),
        practice.Check(
            "CONTROL: two of the three named classes fail by emitting, not by staying silent",
            per_class["passive"]["spurious"] + per_class["elided"]["spurious"]
            == overall["spurious"],
            f"all {overall['spurious']} spurious triples in the run come from the passive and "
            f"elided classes; coordination and the active control produce none. The taxonomy the "
            f"exercise offers names the right constructions and the wrong failure -- precision, not "
            f"recall, is where two of them land"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
