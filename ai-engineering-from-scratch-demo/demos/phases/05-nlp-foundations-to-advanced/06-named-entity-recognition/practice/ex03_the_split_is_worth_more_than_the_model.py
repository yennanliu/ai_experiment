"""Exercise 3 — the split is worth more than the model.

    **Hard.** Fine-tune `distilbert-base-cased` on a domain-specific NER dataset
    (medical, legal, or financial). Compare against the spaCy small model.
    Document data leakage checks and write up what surprised you.

Reading of the exercise: `transformers` and `spacy` are both absent, there is no
domain NER corpus in this checkout, and neither is downloadable here -- so two
of the three instructions cannot be carried out and are not pretended at. The
third can, and it turns out to be the one worth the most.

"Document data leakage checks" is measurable on the corpus exercise 2 builds,
because the leak has a switch. Split those 144 sentences by *sentence* and every
entity in the test set also appears in training: the model scores entity F1
**1.0000** and token accuracy 1.0000. Split the identical sentences by *entity*,
so no test name was ever seen, and the same model on the same features scores
**0.0952**. Nothing else changed -- not the data, not the features, not the
classifier. The leak is worth the entire result, and a by-sentence split is the
default any `train_test_split` call gives you.

That is the surprise, and the lesson's own `rule_based_ner` is the same leak in
its purest form. Its F1 is exactly its gazetteer's coverage of the test set:
8 of 12 ORG names here, so it scores 1.0000 on the eight and 0.0000 on the four,
which is a fact about the list rather than about the method. And it is worse
than that structurally -- `rule_based_ner` only ever emits `B-` labels, so its
maximum span width is 1. Twenty-one gazetteer words in a row come out as 21
separate entities. Half the entities in this corpus, and most real place and
company names, are unrepresentable by it at any coverage.

Structure: the corpus, feature function, trainer and scorer are imported from
exercise 2 rather than duplicated. `by_sentence` shuffles the pooled rows with a
fixed seed; `by_entity` uses exercise 2's disjoint halves. `gazetteer` scores
the lesson's rule-based tagger on both halves separately.
"""

from __future__ import annotations

import importlib.util
import pathlib
import random

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "06-named-entity-recognition"

SIBLING = "ex02_token_accuracy_flatters_what_entity_f1_does_not.py"
SEED, SPLIT = 0, 0.7
UNAVAILABLE = ("transformers", "spacy", "torch", "datasets")


def by_sentence(rows) -> tuple:
    """The default split: shuffle sentences, and every entity lands on both sides."""
    pool = list(rows)
    random.Random(SEED).shuffle(pool)
    cut = int(len(pool) * SPLIT)
    return pool[:cut], pool[cut:]


def gazetteer(ref, sibling, rows) -> dict:
    """The lesson's rule-based tagger, scored the same way as the model."""
    hits = true = predicted = 0
    for tokens, tags, _ in rows:
        want = set(ref.bio_to_spans(tokens, tags))
        got = set(ref.bio_to_spans(tokens, ref.rule_based_ner(tokens)))
        hits, true, predicted = hits + len(want & got), true + len(want), predicted + len(got)
    precision = hits / predicted if predicted else 0.0
    recall = hits / true if true else 0.0
    return {"f1": round(2 * precision * recall / (precision + recall), 4) if hits else 0.0,
            "found": hits, "spans": true}


def absent() -> list:
    return [name for name in UNAVAILABLE if importlib.util.find_spec(name) is None]


def shared(arms) -> dict:
    """How many entities appear on both sides of each split -- the leakage check itself."""
    return {name: len({e for _, _, e in tr} & {e for _, _, e in te})
            for name, (tr, te) in arms.items()}


def scored(sibling, ref, arms) -> dict:
    return {name: sibling.evaluate(ref, sibling.train(ref, tr), te)
            for name, (tr, te) in arms.items()}


def coverage(ref, sibling, seen, unseen) -> dict:
    return {"seen": gazetteer(ref, sibling, seen), "unseen": gazetteer(ref, sibling, unseen),
            "multi": gazetteer(ref, sibling, [r for r in seen + unseen if len(r[2]) > 1])}


def widths(ref) -> dict:
    """Every span width `rule_based_ner` can produce, on a run of gazetteer words."""
    tokens = sorted(ref.ORG_GAZETTEER | ref.GPE_GAZETTEER)
    spans = ref.bio_to_spans(tokens, ref.rule_based_ner(tokens))
    return {"widths": sorted({end - start for start, end, _ in spans}), "in_a_row": len(spans)}


def solve():
    try:
        import sklearn                              # noqa: F401
    except ImportError as exc:                      # pragma: no cover - T1 needs sklearn
        raise practice.Skip(f"needs scikit-learn: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    sibling = practice.load_module(pathlib.Path(__file__).with_name(SIBLING))
    seen = sibling.corpus(ref, sibling.ORGS[:sibling.HELD_OUT], sibling.GPES[:sibling.HELD_OUT])
    unseen = sibling.corpus(ref, sibling.ORGS[sibling.HELD_OUT:], sibling.GPES[sibling.HELD_OUT:])
    arms = {"leaky": by_sentence(seen + unseen), "clean": (seen, unseen)}
    return dict(widths(ref), scores=scored(sibling, ref, arms), rows=len(seen) + len(unseen),
                unavailable=absent(), overlap=shared(arms),
                entities=len(sibling.ORGS) + len(sibling.GPES), orgs=len(sibling.ORGS),
                gazetteer=coverage(ref, sibling, seen, unseen),
                covered=sum(w in ref.ORG_GAZETTEER for e in sibling.ORGS for w in e),
                multi=sum(len(entity) > 1 for _, _, entity in seen + unseen))


def verify(result):
    leaky, clean = result["scores"]["leaky"], result["scores"]["clean"]
    gaz, overlap = result["gazetteer"], result["overlap"]
    return [
        practice.Check(
            "ANSWER: the same model on the same data scores 1.0000 or 0.0952, depending on the split",
            leaky["f1"] == 1.0 and clean["f1"] < 0.2,
            f"{result['unavailable']} are all absent, so the fine-tune and the spaCy comparison are "
            f"not attempted. The leakage check is: split the {result['rows']} sentences by sentence "
            f"and entity F1 is {leaky['f1']}; split the identical sentences by entity and it is "
            f"{clean['f1']}. Same features, same classifier, same rows"),
        practice.Check(
            "MECHANISM: a by-sentence split puts every test entity into training",
            overlap["leaky"] > 0 and overlap["clean"] == 0,
            f"{overlap['leaky']} of the {result['entities']} entities appear on both sides of the "
            f"by-sentence split, against {overlap['clean']} for the by-entity split. The model does "
            f"not have to generalise from casing and context; it can memorise the name, and the "
            f"frames repeat, so it does"),
        practice.Check(
            "FINDING: the leak is worth more than everything else in the experiment",
            leaky["f1"] - clean["f1"] > 0.8 and leaky["token"] > clean["token"],
            f"F1 {clean['f1']} -> {leaky['f1']} and token accuracy {clean['token']} -> "
            f"{leaky['token']}. No architecture change, no feature change and no extra data buys "
            f"{leaky['f1'] - clean['f1']:.4f} F1. A default `train_test_split` over sentences is "
            f"what produces the first number"),
        practice.Check(
            "FINDING: the lesson's own gazetteer is the same leak in its purest form",
            gaz["seen"]["f1"] > 0.8 and gaz["unseen"]["f1"] == 0.0,
            f"`rule_based_ner` scores F1 {gaz['seen']['f1']} on the {gaz['seen']['spans']} spans of "
            f"the half its gazetteer was written around, finding {gaz['seen']['found']}, and "
            f"{gaz['unseen']['f1']} on the other {gaz['unseen']['spans']} -- "
            f"{result['covered']} of the {result['orgs']} ORG names are in ORG_GAZETTEER and none of "
            f"the held-out four. It is not 1.0 on the first half either, for the same reason: "
            f"GPE_GAZETTEER is missing two of the six countries there. The score is the coverage"),
        practice.Check(
            "MECHANISM: rule_based_ner emits only B-, so its maximum span width is 1",
            result["widths"] == [1],
            f"every branch of `rule_based_ner` appends a `B-` label and none appends `I-`, so "
            f"feeding it all {result['in_a_row']} gazetteer words in a row yields "
            f"{result['in_a_row']} separate entities and the set of span widths it can produce is "
            f"{result['widths']}. 'New York City' is three GPEs or none"),
        practice.Check(
            "CONTROL: a quarter of this corpus is unrepresentable by the gazetteer at any coverage",
            gaz["multi"]["found"] == 0 and gaz["multi"]["spans"] == result["multi"] > 0,
            f"{result['multi']} of the {result['rows']} sentences carry a multi-token entity, and "
            f"`rule_based_ner` recovers {gaz['multi']['found']} of them. Adding every name in the "
            f"corpus to the gazetteer would leave that at zero, because the limit is the label set "
            f"rather than the vocabulary -- the one failure a leakage check cannot flatter"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
