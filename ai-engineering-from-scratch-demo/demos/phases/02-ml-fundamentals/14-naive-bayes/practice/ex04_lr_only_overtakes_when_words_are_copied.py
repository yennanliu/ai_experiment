"""Exercise 4 — on the lesson's generator LR only ties NB; it overtakes once words are correlated.

    **NB vs Logistic Regression.** Train both on text data. Start with 100
    training samples and increase to 10,000. Plot accuracy vs training set
    size for both. At what point does Logistic Regression overtake Naive Bayes?

Reading of the exercise: the text data is the lesson's own `make_text_data`
(14,000 documents, the last 4,000 held out), thinned to 1% of each document's
tokens (3.8 words on average). At full length both models score 1.000, and the curve
would be flat. Naive Bayes is the lesson's `MultinomialNB` (alpha=1), and
logistic regression is sklearn's (default L2, C=1). Each size is averaged over
3 generator seeds. The same race is then run with one change: 10 of the 40
tech words each get 8 exact copies, so those features are strongly
correlated.

**ANSWER: on the lesson's data it never meaningfully does.** NB leads by 1.8
points at 100 documents (0.908 vs 0.890). From 300 on the two are within 0.2
points all the way to 10,000 (0.938 vs 0.939). The generator draws every word
as an independent Poisson count given the class, which is exactly Naive Bayes'
model, so NB is already the right model and LR can only converge to it.

**FINDING: break the independence and LR is ahead from the first size.** With
the copied words LR leads by 1.7 points at 100 (0.889 vs 0.872), 2.6 at 300
and 3.7 at 10,000 (0.940 vs 0.902). NB stops improving after 1,000 documents: it counts every
copy as fresh evidence, and more data does not undo that.

**CONTROL: Ng and Jordan's small-data regime is visible.** On the lesson's data
NB is ahead at 100 training documents, the advantage the doc's
"Small data: Better" row claims.

Structure: `curves` runs both models at every size; `race` is one seed.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "14-naive-bayes"
SIZES, SEEDS, KEEP = (100, 300, 1000, 3000, 10000), (1, 2, 3), 0.01


def race(ref, X, y):
    """{n: (NB accuracy, LR accuracy)} on the held-out last 4000 documents."""
    test = slice(10000, 14000)
    out = {}
    for n in SIZES:
        nb = ref.MultinomialNB(alpha=1.0).fit(X[:n], y[:n]).score(X[test], y[test])
        lr = LogisticRegression(max_iter=2000).fit(X[:n], y[:n]).score(X[test], y[test])
        out[n] = (float(nb), float(lr))
    return out


def curves(ref):
    """Seed-averaged curves on the lesson's data and on a copy with duplicated words."""
    runs = {"lesson": [], "copied": []}
    for seed in SEEDS:
        X, y = ref.make_text_data(n_samples=14000, n_features=200, seed=seed)
        short = np.random.RandomState(seed).binomial(X.astype(int), KEEP).astype(float)
        runs["lesson"].append(race(ref, short, y))
        runs["copied"].append(race(ref, np.hstack([short] + [short[:, :10]] * 8), y))
    return {k: {n: tuple(np.mean([r[n] for r in v], axis=0)) for n in SIZES}
            for k, v in runs.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "naive_bayes")
    return curves(ref)


def verify(result):
    lesson, copied = result["lesson"], result["copied"]
    gaps = {n: lesson[n][1] - lesson[n][0] for n in SIZES}
    first = next(n for n in SIZES if copied[n][1] > copied[n][0] + 0.01)
    table = "; ".join(f"n={n} NB {lesson[n][0]:.3f} LR {lesson[n][1]:.3f}" for n in SIZES)
    return [
        practice.Check(
            "ANSWER: on the lesson's data LR never pulls meaningfully ahead",
            max(gaps.values()) < 0.005,
            f"{table}; LR minus NB is at most {max(gaps.values()):+.4f}",
        ),
        practice.Check(
            "FINDING: with correlated (copied) words LR leads from the smallest size",
            first == 100 and copied[10000][1] > copied[10000][0] + 0.02
            and copied[10000][0] < copied[1000][0] + 0.005,
            f"copied words: LR ahead by >1 point from n={first}; at 10000 LR "
            f"{copied[10000][1]:.3f} vs NB {copied[10000][0]:.3f}, and NB at 1000 was "
            f"{copied[1000][0]:.3f}",
        ),
        practice.Check(
            "CONTROL: NB leads at the smallest training set",
            gaps[100] < -0.01,
            f"n=100 on the lesson's data: NB {lesson[100][0]:.3f}, LR {lesson[100][1]:.3f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
