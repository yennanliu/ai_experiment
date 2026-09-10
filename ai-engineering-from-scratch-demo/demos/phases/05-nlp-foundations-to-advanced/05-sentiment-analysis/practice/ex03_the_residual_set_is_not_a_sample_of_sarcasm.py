"""Exercise 3 — the residual set is not a sample of sarcasm.

    **Hard.** Build a sarcasm detector by training a second classifier on the
    residuals of the sentiment model. Document your experimental setup. Warn the
    reader when your accuracy is below chance (chance-level on 2-class sarcasm
    is ~50%, and most first attempts land there).

Reading of the exercise: it warns about the outcome and not about the cause, so
this measures both. Setup, stated in full because the exercise asks for it: 96
rows built from 4 frames and a 16-word polarity lexicon. 64 are sincere, with
the sentiment label following the word. 32 are sarcastic -- positive words,
negative sentiment -- reusing the same frames, and 16 of those 32 carry the cue
`oh sure` while the other 16 are byte-identical to a sincere positive row.
A 70/30 split with a fixed permutation; the analyzer is the lesson's own
`tokenize` + `apply_negation`; the classifier is logistic regression.

The result lands where the exercise says it will, at chance and not below: the
residual-trained sarcasm classifier scores 0.6207 on the test set, which is
exactly the rate of predicting the majority class. It learned nothing. The
cause is that residuals are not a sample of sarcasm. Only 3 of the sentiment
model's 10 training errors are sarcastic, while 21 of its 67 training rows are
-- so as a detector of sarcasm the residual set has precision 0.300 and recall
0.143, and the second classifier is being trained to predict the first model's
mistakes, which is a different target that happens to share a name.

The floor under all of it is the 16 duplicated rows: byte-identical text with
opposite labels. No classifier built on features of that text can separate
them, which is why driving the sentiment model's regularization across six
orders of magnitude leaves the residual count at 10 or 11 out of 67 rather than
at zero. The exercise's "most first attempts land there" is not a failure of
effort.

Structure: `build` emits (text, sentiment, sarcastic) triples; `fit` trains one
pipeline on a chosen label column; `residuals` returns the indices the sentiment
model gets wrong. The second classifier is trained on the whole training set
against the derived "is an error" target, because a classifier trained on the
residual rows alone has no non-sarcastic contrast to learn from.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "05-sentiment-analysis"

GOOD = ("good", "great", "brilliant", "beautiful", "wonderful", "funny", "moving", "charming")
BAD = ("bad", "awful", "boring", "dull", "terrible", "tedious", "dreadful", "clumsy")
FRAMES = ("the film was {}", "i found the whole thing {}", "the acting was {}",
          "every scene is {}")
CUE, SPLIT, SEED, STRENGTHS = "oh sure", 0.7, 0, (1.0, 1e3, 1e6)


def build() -> list:
    """(text, sentiment, sarcastic). Half the sarcastic rows duplicate a sincere positive."""
    rows = [(frame.format(word), int(word in GOOD), 0)
            for frame in FRAMES for word in GOOD + BAD]
    rows += [(f"{CUE} {frame.format(word)}" if (i + j) % 2 == 0 else frame.format(word), 0, 1)
             for i, frame in enumerate(FRAMES) for j, word in enumerate(GOOD)]
    return rows


def fit(ref, rows, column, strength=1.0):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    model = Pipeline([("vec", CountVectorizer(analyzer=lambda t: ref.apply_negation(
        ref.tokenize(t)))), ("clf", LogisticRegression(max_iter=5000, C=strength))])
    labels = column if isinstance(column, list) else [row[column] for row in rows]
    return model.fit([text for text, _, _ in rows], labels)


def residuals(model, rows) -> list:
    predicted = model.predict([text for text, _, _ in rows])
    return [i for i, (row, p) in enumerate(zip(rows, predicted)) if row[1] != p]


def accuracy(truth, predicted) -> float:
    return round(float(sum(a == b for a, b in zip(truth, predicted))) / len(truth), 4)


def split_data(np, data) -> tuple:
    order = np.random.default_rng(SEED).permutation(len(data))
    cut = int(len(data) * SPLIT)
    return [data[i] for i in order[:cut]], [data[i] for i in order[cut:]]


def stage_two(ref, train, bad_train, test) -> dict:
    """The nearest well-posed problem: 'is an error' over the whole training set."""
    from sklearn.metrics import precision_score, recall_score
    target = [int(i in set(bad_train)) for i in range(len(train))]
    truth = [row[2] for row in train]
    predicted = fit(ref, train, target).predict([text for text, _, _ in test])
    return {"accuracy": accuracy([row[2] for row in test], predicted),
            "precision": round(float(precision_score(truth, target, zero_division=0)), 4),
            "recall": round(float(recall_score(truth, target, zero_division=0)), 4)}


def fitted(model, train, test) -> dict:
    return {name: accuracy([row[1] for row in rows], model.predict([t for t, _, _ in rows]))
            for name, rows in (("train", train), ("test", test))}


def duplicates(data) -> int:
    sincere = {text for text, _, sarcastic in data if not sarcastic}
    return sum(1 for text, _, sarcastic in data if sarcastic and text in sincere)


def solve():
    try:
        import numpy as np
        import sklearn                              # noqa: F401
    except ImportError as exc:                      # pragma: no cover - T1 needs sklearn
        raise practice.Skip(f"needs scikit-learn: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = build()
    train, test = split_data(np, data)
    sentiment = fit(ref, train, 1)
    bad_train = residuals(sentiment, train)
    second = stage_two(ref, train, bad_train, test)
    sarcasm = [row[2] for row in test]
    return {
        "rows": len(data), "sarcastic": sum(row[2] for row in data), "duplicated": duplicates(data),
        "sizes": (len(train), len(test)), "sentiment": fitted(sentiment, train, test),
        "residuals": {"train": len(bad_train), "test": len(residuals(sentiment, test))},
        "sarcastic_in_residuals": sum(train[i][2] for i in bad_train),
        "sarcastic_in_train": sum(row[2] for row in train),
        "as_detector": {"precision": second["precision"], "recall": second["recall"]},
        "detector_accuracy": second["accuracy"],
        "majority": round(max(sum(sarcasm), len(sarcasm) - sum(sarcasm)) / len(sarcasm), 4),
        "irreducible": [len(residuals(fit(ref, train, 1, c), train)) for c in STRENGTHS],
    }


def verify(result):
    detector, majority = result["detector_accuracy"], result["majority"]
    res, seen = result["residuals"], result["as_detector"]
    return [
        practice.Check(
            "ANSWER: the residual-trained detector lands exactly at the majority rate",
            detector == majority,
            f"setup: {result['rows']} rows, {result['sarcastic']} sarcastic, split "
            f"{result['sizes'][0]}/{result['sizes'][1]} on a fixed permutation, the lesson's own "
            f"tokenize + apply_negation as the analyzer. The sarcasm classifier scores {detector} "
            f"on test, and answering with the majority class scores {majority}. It learned nothing "
            f"-- at chance, which is where the exercise says most first attempts land"),
        practice.Check(
            "MECHANISM: residuals are not a sample of sarcasm -- only 3 of 10 errors are sarcastic",
            result["sarcastic_in_residuals"] < res["train"] / 2,
            f"the sentiment model gets {res['train']} of {result['sizes'][0]} training rows wrong, "
            f"and {result['sarcastic_in_residuals']} of those are sarcastic -- while "
            f"{result['sarcastic_in_train']} of the {result['sizes'][0]} rows are. Errors have many "
            f"causes and sarcasm is one of them"),
        practice.Check(
            "FINDING: as a sarcasm detector the residual set scores 0.300 precision, 0.143 recall",
            seen["precision"] < 0.5 and seen["recall"] < 0.2,
            f"scoring 'the sentiment model got this wrong' directly against the sarcasm label gives "
            f"precision {seen['precision']} and recall {seen['recall']}. The second classifier is "
            f"trained to predict the first model's mistakes, and that target agrees with sarcasm on "
            f"{seen['precision']:.0%} of what it flags"),
        practice.Check(
            "MECHANISM: the residual rows alone have no contrast, so the target has to be derived",
            res["train"] < result["sizes"][0] / 4,
            f"only {res['train']} of {result['sizes'][0]} rows are residuals, and 'train a "
            f"classifier on the residuals' read literally means fitting on those {res['train']} "
            f"rows with every one of them an error. The target used here is 'is an error' over the "
            f"whole training set, which is the nearest well-posed problem"),
        practice.Check(
            "FINDING: the error floor is irreducible -- 16 rows are byte-identical with opposite labels",
            result["duplicated"] == 16 and len(set(result["irreducible"])) <= 2,
            f"{result['duplicated']} of the {result['sarcastic']} sarcastic rows repeat a sincere "
            f"positive row exactly, so no feature of the text separates them. Sweeping the "
            f"sentiment model's C over {list(STRENGTHS)} leaves the training residual count at "
            f"{result['irreducible']} -- it never approaches zero, because it cannot"),
        practice.Check(
            "CONTROL: the sentiment model itself is the thing that generalises badly here",
            result["sentiment"]["test"] < result["sentiment"]["train"],
            f"sentiment accuracy is {result['sentiment']['train']} on train and "
            f"{result['sentiment']['test']} on test. A second stage built on the first stage's "
            f"errors inherits that gap before it starts: the residual set it trains on was measured "
            f"where the model is strongest, and it is applied where the model is weakest"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
