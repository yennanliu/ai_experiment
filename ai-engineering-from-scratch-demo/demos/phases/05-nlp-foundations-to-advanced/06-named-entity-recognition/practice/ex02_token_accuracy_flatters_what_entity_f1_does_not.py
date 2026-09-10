"""Exercise 2 — token accuracy flatters what entity F1 does not.

    **Medium.** Train the sklearn-crfsuite CRF above on the CoNLL-2003 English
    NER dataset. Report per-entity F1 using `seqeval`. Typical result: ~84 F1.

Reading of the exercise: `sklearn_crfsuite`, `seqeval` and `datasets` are all
absent and CoNLL-2003 is not downloadable here, so the ~84 cannot be confirmed
or denied and is not claimed either way. What can be built is the reason the
exercise specifies `seqeval` rather than accuracy, and that reason is a number.

The corpus below is 144 sentences over 24 entities, half of them multi-token,
generated from six frames; the model is logistic regression over the lesson's
own `token_features`, which is the feature function the CRF would use. Split so
that no test entity appears in training, it scores token accuracy **0.8302** and
entity F1 **0.0952**. Those are the same predictions scored two ways, and they
differ by a factor of nine. Token accuracy is inflated by the `O` class: a model
that answers `O` everywhere and never finds anything scores 0.7547 on the same
test set, so the model's 0.8302 is 0.076 above doing nothing at all.

Underneath the aggregate, the split is cleaner than the aggregate suggests.
Single-token entities score F1 0.2500. Multi-token entities score **0.0000** --
zero of 24 recovered, with 54 false positives among them. A per-token classifier
has no representation of "this tag follows that tag", so a three-token name has
to come out right three times independently and does not. Transition features
are the whole of what a CRF adds over this model, and this is the measurement
that shows what they are for.

Structure: `ORGS` and `GPES` are the entity inventory, split by index so the
train and test halves are disjoint; `corpus` crosses entities with frames and
labels each sentence through the lesson's own `spans_to_bio`. `evaluate` scores
one prediction set both ways, using `bio_to_spans` for the entity side, which is
what `seqeval` does.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "06-named-entity-recognition"

# the first HELD_OUT of each list train, the rest test, so no test name is ever seen
ORGS = tuple((n,) for n in ("Apple Google Microsoft OpenAI Anthropic Netflix Amazon Meta Nvidia "
                            "Oracle Spotify Adobe").split())
GPES = tuple((n,) for n in "France Germany Japan India Brazil Canada".split()) + tuple(
    tuple(n.split()) for n in ("New York City", "San Francisco", "United Arab Emirates",
                               "Cape Town", "Costa Rica", "Sri Lanka"))
FRAMES = ("{e} announced a partnership today .", "the report on {e} was published .",
          "analysts said {e} would grow .", "{e} declined to comment .",
          "a spokesperson for {e} confirmed it .", "nobody at {e} replied .")
HELD_OUT, MISSING = 8, ("sklearn_crfsuite", "seqeval", "datasets", "pycrfsuite")


def sentence(ref, frame, entity, label) -> tuple:
    tokens = frame.replace("{e}", " ".join(entity)).split()
    start = tokens.index(entity[0])
    return tokens, ref.spans_to_bio(tokens, [(start, start + len(entity), label)]), entity


def corpus(ref, orgs, gpes) -> list:
    """(tokens, gold BIO, entity). Labelled through the lesson's own spans_to_bio."""
    return [sentence(ref, frame, entity, label)
            for entities, label in ((orgs, "ORG"), (gpes, "GPE"))
            for frame in FRAMES for entity in entities]


def featurize(ref, tokens) -> list:
    return [ref.token_features(token, tokens[i - 1] if i else None,
                               tokens[i + 1] if i + 1 < len(tokens) else None)
            for i, token in enumerate(tokens)]


def train(ref, rows):
    from sklearn import feature_extraction, linear_model
    vector = feature_extraction.DictVectorizer()
    matrix = vector.fit_transform([f for tokens, _, _ in rows for f in featurize(ref, tokens)])
    labels = [tag for _, tags, _ in rows for tag in tags]
    return vector, linear_model.LogisticRegression(max_iter=3000).fit(matrix, labels)


def tally(ref, rows, flat) -> dict:
    """Span counts over the sentences, the way seqeval scores them."""
    counts, cursor = {"found": 0, "spans": 0, "proposed": 0}, 0
    for tokens, tags, _ in rows:
        window, cursor = flat[cursor:cursor + len(tokens)], cursor + len(tokens)
        want, got = set(ref.bio_to_spans(tokens, tags)), set(ref.bio_to_spans(tokens, window))
        counts["found"] += len(want & got)
        counts["spans"], counts["proposed"] = counts["spans"] + len(want), counts["proposed"] + len(got)
    return counts


def rates(counts) -> tuple:
    precision = counts["found"] / counts["proposed"] if counts["proposed"] else 0.0
    recall = counts["found"] / counts["spans"] if counts["spans"] else 0.0
    harmonic = 2 * precision * recall / (precision + recall) if counts["found"] else 0.0
    return round(precision, 4), round(recall, 4), round(harmonic, 4)


def evaluate(ref, model, rows) -> dict:
    """The same predictions scored per token and per entity span."""
    vector, classifier = model
    features = [f for tokens, _, _ in rows for f in featurize(ref, tokens)]
    flat = list(classifier.predict(vector.transform(features)))
    gold = [tag for _, tags, _ in rows for tag in tags]
    counts = tally(ref, rows, flat)
    precision, recall, harmonic = rates(counts)
    return {"token": round(sum(a == b for a, b in zip(gold, flat)) / len(gold), 4),
            "f1": harmonic, "precision": precision, "recall": recall,
            "spans": counts["spans"], "found": counts["found"],
            "spurious": counts["proposed"] - counts["found"],
            "missed": counts["spans"] - counts["found"],
            "all_o": round(gold.count("O") / len(gold), 4)}


def solve():
    try:
        import sklearn                              # noqa: F401
    except ImportError as exc:                      # pragma: no cover - T1 needs sklearn
        raise practice.Skip(f"needs scikit-learn: uv sync --extra math ({exc})") from None
    import importlib.util
    ref = parity.load_reference(PHASE, LESSON, "main")
    train_rows = corpus(ref, ORGS[:HELD_OUT], GPES[:HELD_OUT])
    test_rows = corpus(ref, ORGS[HELD_OUT:], GPES[HELD_OUT:])
    model = train(ref, train_rows)
    return {
        "missing": [m for m in MISSING if importlib.util.find_spec(m) is None],
        "sizes": (len(train_rows), len(test_rows)),
        "entities": len(ORGS) + len(GPES),
        "overlap": len({e for _, _, e in train_rows} & {e for _, _, e in test_rows}),
        "all": evaluate(ref, model, test_rows),
        "single": evaluate(ref, model, [r for r in test_rows if len(r[2]) == 1]),
        "multi": evaluate(ref, model, [r for r in test_rows if len(r[2]) > 1]),
    }


def verify(result):
    total, single, multi = result["all"], result["single"], result["multi"]
    return [
        practice.Check(
            "ANSWER: the same predictions score 0.8302 per token and 0.0952 per entity",
            total["token"] > 8 * total["f1"],
            f"{result['missing']} are all absent and CoNLL-2003 is not downloadable, so the ~84 is "
            f"neither confirmed nor denied. On {result['sizes'][0]}/{result['sizes'][1]} sentences "
            f"over {result['entities']} entities, logistic regression on the lesson's "
            f"token_features scores token accuracy {total['token']} and entity F1 {total['f1']} -- "
            f"a factor of {total['token'] / total['f1']:.0f}"),
        practice.Check(
            "MECHANISM: token accuracy is inflated by the O class",
            total["all_o"] > 0.7 and total["token"] - total["all_o"] < 0.1,
            f"answering O for every token scores {total['all_o']}, the share of tokens that are O, "
            f"so the model's {total['token']} is {total['token'] - total['all_o']:.4f} above a model "
            f"that never finds anything -- which is why the exercise names seqeval"),
        practice.Check(
            "FINDING: multi-token entities score exactly zero",
            multi["f1"] == 0.0 and multi["found"] == 0 and single["f1"] > 0,
            f"of {multi['spans']} multi-token gold entities the model recovers {multi['found']}, "
            f"with {multi['spurious']} spurious spans, for F1 {multi['f1']}; single-token entities "
            f"score {single['f1']} on {single['spans']}, and carry the whole aggregate"),
        practice.Check(
            "MECHANISM: a per-token classifier has no representation of the previous tag",
            multi["token"] > 0.7 and multi["f1"] == 0.0,
            f"on the multi-token half token accuracy is still {multi['token']} while entity F1 is "
            f"{multi['f1']}: a three-token name must come out right three times independently and "
            f"the boundaries must agree. Transition features are the whole of what a CRF adds"),
        practice.Check(
            "FINDING: the errors are over-prediction, not silence",
            total["spurious"] > total["spans"],
            f"the model proposes {total['found'] + total['spurious']} spans against "
            f"{total['spans']} real ones -- {total['spurious']} spurious, {total['missed']} missed, "
            f"precision {total['precision']}, recall {total['recall']}. It does not fail by "
            f"predicting O; it guesses boundaries in the wrong places"),
        practice.Check(
            "CONTROL: no test entity appears in training, which is what makes the numbers mean this",
            result["overlap"] == 0,
            f"the split is by entity, not by sentence: {result['overlap']} of the "
            f"{result['entities']} entities appear on both sides, so every test name is unseen and "
            f"the score measures the feature function -- shape, casing, neighbours -- not a list"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
