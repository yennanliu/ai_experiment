"""Exercise 3 — coverage can only go up, including when it is wrong.

    **Hard.** Build a coref-enhanced NER pipeline: NER first, then merge via
    coref clusters. Measure entity-coverage improvement vs NER-only on 100
    articles.

Reading of the exercise: the NER layer is `extract_mentions`' named entities and
the merge is the lesson's own `clusters`, over exercise 1's seven paragraphs. An
entity covers a sentence if any mention of it appears there. Against 18 gold
entity-sentence pairs, NER alone finds 9 and the merged pipeline finds 16:
**recall 0.5000 to 0.8333**, two thirds more coverage.

Precision goes the other way, from **1.0000 to 0.9375**. Of the 7 attributions
the merge adds, 6 are right and one is not -- `Apple` picks up the sentence that
belongs to Google, because `It appealed the ruling` resolved to the leftmost
mention. Every added attribution comes from a pronoun link, so the added coverage
can be no better than the link accuracy underneath it.

The metric cannot see that. Merging only ever adds sentences to an entity, so
entity coverage is monotone: it rises when the link is right and it rises when
the link is wrong. Reported alone -- which is what the exercise asks for -- it
scores a correct merge and a wrong one identically, and the wrong one reads as
the larger improvement because it attaches an entity to a sentence it has no
business in.

The three gold pairs still missing come from the same place. `Google`, `Satya
Nadella` and `Priya Sharma` each lose a sentence to a link that went elsewhere:
two to sentence-initial capitalised nouns treated as entities, one to a name the
gender lists do not contain.

Under all of it the NER layer has no types. 4 of its 13 named entities are months
or plural common nouns that began a sentence, so a pipeline described as "NER,
then merge via coref clusters" is merging dates and common nouns into entity
clusters before anything is measured.

Corpus size does not move any of this: the gain is one pronoun's worth of
coverage per pronoun and its error rate is the resolver's, so 100 articles
reports the same ratio as seven paragraphs with a smaller confidence interval.

Structure: exercises 1 and 2 are loaded rather than copied; `pairs` builds the
gold, NER-only and merged entity-sentence sets; `prf` scores a set against gold;
`ner_types` counts what the entity layer actually extracted.
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "24-coreference-resolution"

HERE = pathlib.Path(__file__).resolve().parent
EX1 = practice.load_module(HERE / "ex01_recency_resolves_to_the_leftmost_mention.py")
EX2 = practice.load_module(HERE / "ex02_the_cluster_metric_is_the_flattering_one.py")
ROWS = EX1.ROWS
ENTITIES = frozenset({"John Smith", "Sarah Patel", "Apple", "Google", "Satya Nadella",
                      "Priya Sharma", "Yusuf Demir", "Nokia"})


def spread(doc, mentions, groups):
    """(document, entity, sentence) for every mention grouped with a tracked name."""
    out = set()
    for cluster in groups:
        named = [mentions[i]["text"] for i in cluster if mentions[i]["text"] in ENTITIES]
        out |= {(doc, named[0], mentions[i]["span"][0]) for i in cluster if named}
    return out


def named_only(doc, mentions):
    """(document, entity, sentence) for the named-entity mentions alone."""
    return {(doc, m["text"], m["span"][0]) for m in mentions
            if m["type"] == "ne" and m["text"] in ENTITIES}


def pairs(ref):
    """Gold, NER-only and coref-merged (document, entity, sentence) attributions."""
    gold, ner, merged = set(), set(), set()
    for doc, (text, links) in enumerate(ROWS):
        mentions = ref.extract_mentions(text)
        gold |= spread(doc, mentions, EX2.gold_clusters(mentions, links))
        ner |= named_only(doc, mentions)
        merged |= spread(doc, mentions, EX2.predicted(ref, mentions))
    return gold, ner, merged


def prf(found, gold):
    """Precision and recall of an attribution set."""
    hits = len(found & gold)
    return {"pairs": len(found), "precision": round(hits / len(found), 4),
            "recall": round(hits / len(gold), 4)}


def ner_types(ref):
    """Every named entity the extractor produced, and how many are the ones tracked."""
    found = [m["text"] for text, _ in ROWS for m in ref.extract_mentions(text)
             if m["type"] == "ne"]
    return {"extracted": len(found), "tracked": sum(1 for t in found if t in ENTITIES),
            "other": sorted({t for t in found if t not in ENTITIES})}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    gold, ner, merged = pairs(ref)
    added = merged - ner
    return {
        "paragraphs": len(ROWS), "gold": len(gold),
        "ner": prf(ner, gold), "merged": prf(merged, gold),
        "added": len(added), "added_right": len(added & gold),
        "wrong": sorted(added - gold), "missing": sorted(gold - merged),
        "dropped": len(ner - merged),
        "layer": ner_types(ref),
        "link_accuracy": round(EX1.score(ref)[0] / sum(len(g) for _, g in ROWS), 4),
    }


def verify(result):
    ner, merged, layer = result["ner"], result["merged"], result["layer"]
    return [
        practice.Check(
            "ANSWER: coverage recall goes 0.5000 to 0.8333, and precision 1.0000 to 0.9375",
            merged["recall"] > ner["recall"] and merged["precision"] < ner["precision"],
            f"against {result['gold']} gold entity-sentence pairs over {result['paragraphs']} "
            f"paragraphs, NER alone finds {ner['pairs']} at recall {ner['recall']} and the merged "
            f"pipeline {merged['pairs']} at {merged['recall']}. Precision falls from "
            f"{ner['precision']} to {merged['precision']}",
        ),
        practice.Check(
            "MECHANISM: the added coverage inherits the pronoun link accuracy",
            result["added_right"] < result["added"],
            f"{result['added']} attributions are added and {result['added_right']} are right. "
            f"Every one comes from a pronoun link, and {result['link_accuracy']} of those links "
            f"are correct, so the added coverage can be no better than the resolver: the miss is "
            f"{result['wrong']}, Apple taking the sentence that belongs to Google",
        ),
        practice.Check(
            "FINDING: coverage is monotone, so the metric cannot see a wrong merge",
            result["dropped"] == 0,
            f"merging only ever adds sentences to an entity -- {result['dropped']} of the "
            f"NER-only attributions are lost -- so entity coverage rises when the link is right "
            "and rises when it is wrong. Reported alone, as the exercise asks, it scores a "
            "correct merge and a false one identically",
        ),
        practice.Check(
            "FINDING: the pairs still missing come from the same four links",
            len(result["missing"]) == 3,
            f"{[name for _, name, _ in result['missing']]} each lose a sentence to a link that "
            "went elsewhere: two to sentence-initial capitalised nouns treated as entities, one "
            "to a name the gender lists do not contain. The false addition and the misses are the "
            "same failure counted twice",
        ),
        practice.Check(
            "MECHANISM: the NER layer under it has no types at all",
            layer["tracked"] < layer["extracted"],
            f"{layer['extracted']} named entities are extracted and {layer['tracked']} are the "
            f"ones tracked; the rest are {layer['other']} -- months and plural common nouns that "
            "began a sentence. 'NER, then merge via coref' merges dates into entity clusters "
            "before anything is measured",
        ),
        practice.Check(
            "CONTROL: all of the precision loss belongs to the merge",
            ner["precision"] == 1.0,
            f"NER-only scores precision {ner['precision']} by construction -- a name in a "
            f"sentence is that entity in that sentence. The {round(1 - merged['precision'], 4)} "
            "the merged pipeline gives up is entirely the coref step's, and it is the number the "
            "exercise does not ask for",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
