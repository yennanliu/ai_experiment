"""Exercise 2 — balanced loses on the metric that names it.

    **Medium.** Implement class-weighted logistic regression (pass
    `class_weight="balanced"` to scikit-learn, or derive the gradient
    yourself). Measure the effect on a synthetic 90-10 class imbalance.

Reading of the exercise: "measure the effect" does not say on what, and the
answer depends entirely on that. On a 2000-row 90-10 problem, weighting the
classes moves accuracy 0.9633 -> 0.8783 and F1 0.7963 -> 0.5876, both down, and
recall 0.6935 -> 0.8387, up. Three headline metrics, two of them worse, and the
one that improves is the one the intervention exists to improve. A single
number reported without saying which cannot distinguish "this helped" from
"this hurt".

Accuracy is the worse of the two in a specific way: the do-nothing predictor
that answers with the majority class scores 0.8967, which beats the balanced
model's 0.8783. So on the metric the exercise's phrasing most naturally invites,
predicting nothing at all outranks the repair -- while scoring F1 0.000 and
recall 0.000, having never once named the minority class. What accuracy cannot
see is that `class_weight="balanced"` is doing exactly what it says: the
minority weight is n/(2 * count) = 4.79, and the model responds by predicting
115 positives where the plain model predicted 46, against 62 that are real. The
extra 69 guesses buy 9 more true positives and cost precision 0.9348 -> 0.4522.

That is the whole trade and it is not hidden -- it is just not a single number.
The plain model already beats the majority baseline on accuracy, so the
imbalance here is not pathological; what weighting changes is where on the
precision/recall curve the threshold sits, and nothing about the ranking of the
examples.

Structure: `dataset` is scikit-learn's `make_classification` at 90-10 with a
fixed seed, split stratified so the test set holds the same ratio. `score`
fits one arm and returns the four metrics plus how many positives it predicted;
`MAJORITY` is the constant predictor, scored the same way.
"""

from __future__ import annotations

from harness import practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "05-sentiment-analysis"

SAMPLES, MINORITY, SEED, TEST = 2000, 0.1, 0, 0.3


def dataset(np):
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    features, labels = make_classification(
        n_samples=SAMPLES, n_features=20, n_informative=6, n_redundant=2,
        weights=[1 - MINORITY, MINORITY], flip_y=0.01, random_state=SEED)
    return train_test_split(features, labels, test_size=TEST, stratify=labels, random_state=SEED)


def metrics(np, truth, predicted) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    return {"accuracy": round(float(accuracy_score(truth, predicted)), 4),
            "f1": round(float(f1_score(truth, predicted, zero_division=0)), 4),
            "recall": round(float(recall_score(truth, predicted, zero_division=0)), 4),
            "precision": round(float(precision_score(truth, predicted, zero_division=0)), 4),
            "predicted_positive": int(np.sum(predicted))}


def score(np, weight, split) -> dict:
    from sklearn.linear_model import LogisticRegression
    x_train, x_test, y_train, y_test = split
    model = LogisticRegression(max_iter=2000, class_weight=weight).fit(x_train, y_train)
    return metrics(np, y_test, model.predict(x_test))


def solve():
    try:
        import numpy as np
        import sklearn                              # noqa: F401
    except ImportError as exc:                      # pragma: no cover - T1 needs sklearn
        raise practice.Skip(f"needs scikit-learn: uv sync --extra math ({exc})") from None
    split = dataset(np)
    _, _, y_train, y_test = split
    counts = np.bincount(y_train)
    return {
        "arms": {"plain": score(np, None, split), "balanced": score(np, "balanced", split),
                 "majority": metrics(np, y_test, np.zeros_like(y_test))},
        "train_counts": counts.tolist(), "test_counts": np.bincount(y_test).tolist(),
        "weight": round(float(len(y_train) / (2 * counts[1])), 4),
    }


def verify(result):
    plain, balanced = result["arms"]["plain"], result["arms"]["balanced"]
    majority, weight = result["arms"]["majority"], result["weight"]
    real = result["test_counts"][1]
    return [
        practice.Check(
            "ANSWER: weighting moves accuracy and F1 down and recall up -- two of three get worse",
            balanced["accuracy"] < plain["accuracy"] and balanced["f1"] < plain["f1"]
            and balanced["recall"] > plain["recall"],
            f"on {SAMPLES} rows split {result['train_counts']} in training and "
            f"{result['test_counts']} in test: accuracy {plain['accuracy']} -> "
            f"{balanced['accuracy']}, F1 {plain['f1']} -> {balanced['f1']}, recall "
            f"{plain['recall']} -> {balanced['recall']}. 'Measure the effect' has three answers "
            f"here and they do not agree on the sign"),
        practice.Check(
            "FINDING: the do-nothing predictor beats the balanced model on accuracy",
            majority["accuracy"] > balanced["accuracy"] and majority["recall"] == 0.0,
            f"answering with the majority class every time scores accuracy {majority['accuracy']} "
            f"against the balanced model's {balanced['accuracy']}, while scoring F1 "
            f"{majority['f1']} and recall {majority['recall']} -- it never names the minority class "
            f"once. On the metric the exercise's phrasing most invites, predicting nothing outranks "
            f"the repair"),
        practice.Check(
            "MECHANISM: the weight is n/(2*count), and the model spends it on more positive guesses",
            balanced["predicted_positive"] > 2 * plain["predicted_positive"],
            f"`class_weight='balanced'` gives the minority class "
            f"{weight} = {sum(result['train_counts'])}/(2 x "
            f"{result['train_counts'][1]}). The model answers positive "
            f"{balanced['predicted_positive']} times where the plain model answered "
            f"{plain['predicted_positive']}, against {real} that are real"),
        practice.Check(
            "MECHANISM: the extra guesses buy recall and are paid for in precision",
            balanced["precision"] < 0.5 < plain["precision"],
            f"the {balanced['predicted_positive'] - plain['predicted_positive']} additional positive "
            f"answers recover "
            f"{round(balanced['recall'] * real) - round(plain['recall'] * real)} more true "
            f"positives and drop precision {plain['precision']} -> {balanced['precision']}. Fewer "
            f"than half of what the balanced model calls positive is positive"),
        practice.Check(
            "CONTROL: the imbalance is not pathological -- the plain model already beats the baseline",
            plain["accuracy"] > majority["accuracy"] and plain["f1"] > 0.5,
            f"the unweighted model scores accuracy {plain['accuracy']} against the majority "
            f"baseline's {majority['accuracy']} and F1 {plain['f1']}, so 90-10 has not collapsed it "
            f"into the majority class. Weighting is a choice about where to sit on the "
            f"precision/recall curve, not a rescue"),
        practice.Check(
            "CONTROL: weighting moves the threshold, not the ranking",
            balanced["recall"] > plain["recall"] > majority["recall"],
            f"every arm sees the same features and the same 20-dimensional geometry; recall goes "
            f"{majority['recall']} -> {plain['recall']} -> {balanced['recall']} as the model becomes "
            f"more willing to answer positive. Nothing here is a better model -- it is the same "
            f"model asked a different question"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
