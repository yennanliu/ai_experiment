"""Exercise 1 — the dead entry in NEGATION_WORDS.

    **Easy.** Add `apply_negation` as a preprocessing step in the scikit-learn
    pipeline and measure the F1 delta on a small sentiment dataset.

Reading of the exercise: the delta is +0.6667 and it is not evenly earned. The
dataset below is generated from frames and a fixed polarity lexicon so that
every sentiment word recurs across the split, and it is built so the question
has an answer: training carries each word plain and under spelled-out negation,
so without scoping every word appears equally in both classes and the model
sits at chance -- accuracy 0.500, F1 0.000. With `apply_negation` in the
analyzer, `good` and `NOT_good` are separate features and both the plain and
the spelled-out test items go to 1.000.

The contracted items go the other way. `NEGATION_WORDS` contains `n't`, so the
lesson plainly meant to catch `wasn't` and `didn't`, but `TOKEN_RE` matches
`[A-Za-z]+(?:'[A-Za-z]+)?`, which emits `wasn't` as one token and never emits
`n't` at all. The entry is dead. Un-negated, `wasn't good` keeps `good`, and
because the preprocessing step has just made `good` an unambiguous positive
feature, contracted negation falls from 0.500 -- chance -- to 0.000. The step
does not merely fail on the commonest way English negates; it converts a coin
flip into a confident wrong answer, and the single F1 number the exercise asks
for reports the average of the two effects and hides the sign of one.

Structure: `GOOD`/`BAD` are the polarity lexicon and `FRAMES` the carrier
sentences; `render` writes one row in one of the three negation forms, and
`build` crosses frames with words. Train uses the plain and spelled-out forms
only, test adds the contracted form, so the contracted column is the held-out
question. `pipeline` puts the lesson's own `tokenize`/`apply_negation` in a
`CountVectorizer` analyzer, which is what "as a preprocessing step in the
scikit-learn pipeline" means.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "05-sentiment-analysis"

GOOD = ("good", "great", "brilliant", "beautiful", "wonderful", "funny", "moving", "charming")
BAD = ("bad", "awful", "boring", "dull", "terrible", "tedious", "dreadful", "clumsy")
TRAIN_FRAMES = ("the film was {}", "a movie that is {}", "i found the whole thing {}",
                "the acting was {}")
TEST_FRAMES = ("the ending was {}", "every scene is {}")
FORMS = ("plain", "spelled", "contracted")


def render(frame: str, word: str, form: str) -> str:
    if form == "plain":
        return frame.format(word)
    if form == "spelled":
        return frame.format("not " + word)
    head = frame.partition("{}")[0].rstrip()
    stem, contraction = (head[:-3], "wasn't ") if head.endswith("was") else (head[:-2], "isn't ")
    return stem + contraction + word + frame.partition("{}")[2]


def build(frames, forms) -> list:
    """(text, label, form). A negated row carries the opposite label to its word."""
    return [(render(frame, word, form), int((word in GOOD) == (form == "plain")), form)
            for frame in frames for form in forms for word in GOOD + BAD]


def pipeline(ref, negate: bool):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    analyze = (lambda t: ref.apply_negation(ref.tokenize(t))) if negate else ref.tokenize
    return Pipeline([("vec", CountVectorizer(analyzer=analyze)),
                     ("clf", LogisticRegression(max_iter=2000))])


def by_form(truth, predicted, forms) -> dict:
    hits = {f: [a == b for a, b, g in zip(truth, predicted, forms) if g == f] for f in FORMS}
    return {f: round(float(sum(rows)) / len(rows), 4) for f, rows in hits.items()}


def arm(ref, negate, train, test) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    model = pipeline(ref, negate).fit([t for t, _, _ in train], [y for _, y, _ in train])
    predicted = model.predict([t for t, _, _ in test])
    truth, forms = [y for _, y, _ in test], [f for _, _, f in test]
    return dict(by_form(truth, predicted, forms), f1=round(float(f1_score(truth, predicted)), 4),
                accuracy=round(float(accuracy_score(truth, predicted)), 4),
                vocab=len(model.named_steps["vec"].vocabulary_))


def pick(row, forms) -> dict:
    return {f: row[f] for f in forms}


def emits_nt(ref) -> bool:
    """Does TOKEN_RE ever produce the token NEGATION_WORDS is waiting for?"""
    return any(token == "n't" for word in ("wasn't", "didn't", "isn't", "won't")
               for token in ref.tokenize(word))


def solve():
    try:
        import sklearn                              # noqa: F401
    except ImportError as exc:                      # pragma: no cover - T1 needs sklearn
        raise practice.Skip(f"needs scikit-learn: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    train, test = build(TRAIN_FRAMES, FORMS[:2]), build(TEST_FRAMES, FORMS)
    probe = "the ending wasn't good"
    return {
        "off": arm(ref, False, train, test), "on": arm(ref, True, train, test),
        "forms": list(FORMS),
        "sizes": (len(train), len(test)),
        "spelled_tokens": ref.apply_negation(ref.tokenize("the ending was not good")),
        "contracted_tokens": ref.apply_negation(ref.tokenize(probe)),
        "emits_nt": emits_nt(ref), "listed_nt": "n't" in ref.NEGATION_WORDS,
    }


def verify(result):
    off, on = result["off"], result["on"]
    delta = round(on["f1"] - off["f1"], 4)
    off_forms, on_forms = pick(off, result["forms"]), pick(on, result["forms"])
    return [
        practice.Check(
            "ANSWER: the F1 delta is +0.6667, from 0.0000 to 0.6667",
            delta > 0.6 and off["f1"] == 0.0,
            f"on {result['sizes'][0]} training and {result['sizes'][1]} test rows, the pipeline "
            f"scores F1 {off['f1']} without `apply_negation` and {on['f1']} with it, accuracy "
            f"{off['accuracy']} to {on['accuracy']}, over a vocabulary of {off['vocab']} features "
            f"against {on['vocab']}"),
        practice.Check(
            "MECHANISM: without scoping every sentiment word is equally in both classes",
            set(off_forms.values()) == {0.5},
            f"training carries each word plain and under spelled-out negation, so bag-of-words sees "
            f"`good` as often in one class as the other. Per-form accuracy without the step is "
            f"{off_forms} -- chance on all three. The step separates `good` from "
            f"`NOT_good`: {result['spelled_tokens']}"),
        practice.Check(
            "FINDING: the gain is real on two forms and the third goes to zero",
            on["plain"] == on["spelled"] == 1.0 and on["contracted"] == 0.0,
            f"with the step, per-form accuracy is {on_forms}. Plain and "
            f"spelled-out go to 1.0000; contracted falls from {off['contracted']} -- chance -- to "
            f"{on['contracted']}. The one F1 number the exercise asks for averages a gain and a "
            f"collapse and reports neither"),
        practice.Check(
            "MECHANISM: NEGATION_WORDS lists n't, and TOKEN_RE can never produce it",
            result["listed_nt"] and not result["emits_nt"],
            f"`n't` is in NEGATION_WORDS, so the lesson meant to scope contractions. But TOKEN_RE is "
            f"`[A-Za-z]+(?:'[A-Za-z]+)?`, which matches `wasn't` whole: tokenizing wasn't, didn't, "
            f"isn't and won't yields `n't` "
            f"{'somewhere' if result['emits_nt'] else 'nowhere'}. The entry is unreachable"),
        practice.Check(
            "FINDING: so the preprocessing step turns a coin flip into a confident wrong answer",
            result["contracted_tokens"] == ["the", "ending", "wasn't", "good"]
            and on["contracted"] < off["contracted"],
            f"`the ending wasn't good` scopes to {result['contracted_tokens']} -- unchanged, `good` "
            f"intact. Before the step `good` was ambiguous and the item was a coin flip; after it "
            f"`good` is an unambiguous positive feature, so every contracted negation is "
            f"confidently misread. Accuracy on that form goes {off['contracted']} -> "
            f"{on['contracted']}"),
        practice.Check(
            "CONTROL: the vocabulary grows by less than the number of marked types would suggest",
            on["vocab"] > off["vocab"] and on["vocab"] < 2 * off["vocab"],
            f"the step adds {on['vocab'] - off['vocab']} features to {off['vocab']}, not a doubling: "
            f"only tokens that actually follow a negation word get a `NOT_` twin, and on this data "
            f"that is the polarity lexicon and the frame tails. The cost of the step is small; "
            f"what it buys is uneven"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
