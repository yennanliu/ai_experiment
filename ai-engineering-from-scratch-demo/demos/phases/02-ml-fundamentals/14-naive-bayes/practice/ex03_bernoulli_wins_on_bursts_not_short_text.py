"""Exercise 3 — Bernoulli wins on bursty counts and loses on the short text the doc suggests.

    **Bernoulli implementation.** Extend the code with a BernoulliNB class.
    Convert bag-of-words to binary (present/absent) and compare accuracy against
    MultinomialNB on text data. When does Bernoulli win?

Reading of the exercise: `BernoulliNB` below follows the lesson's interface
and the doc's formula `(docs containing w + alpha) / (docs + 2 alpha)`, scoring
both presence and absence. It is compared with the lesson's `MultinomialNB`
(alpha = 1 for both) on the lesson's `make_text_data` (3000 documents, the last
1000 held out). "When does it win" is answered by varying the two things the
doc says matter: document length (keep a fraction of each document's tokens)
and how repetitive the counts are (repeat a random 5% of a document's present
words 20 times, independent of class, the burstiness real text has).

**ANSWER: on the lesson's data the two tie at 1.000.** Bernoulli wins only when
counts are bursty. On 7.3-word documents they tie (0.986 and 0.985 with 2000
training documents). Repeat a random 5% of present words 20 times and
Multinomial drops to 0.953 while Bernoulli stays at 0.985. Repetition carries
no class signal, and only Multinomial counts it.

**FINDING: it does not win on short text, the doc's stated use case.** At 3.6
words per document and 20 training documents, Multinomial scores 0.773 and
Bernoulli 0.527. Every one of the ~196 absent words casts a vote, and with 20
documents those votes are mostly noise. With 2000 training documents the
two tie (0.928 and 0.929).

**CONTROL: the implementation matches sklearn.** Its joint log-likelihoods
equal `sklearn.naive_bayes.BernoulliNB(alpha=1)`'s within 5.2e-12, on values
in the hundreds.

Structure: `BernoulliNB` is the extension; `race` scores both models.
"""

from __future__ import annotations

import numpy as np
from sklearn.naive_bayes import BernoulliNB as SkBernoulliNB

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "14-naive-bayes"


class BernoulliNB:
    """Presence/absence Naive Bayes, in the lesson's MultinomialNB interface."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = (X > 0).astype(float)
        self.classes_ = np.unique(y)
        n_c = np.array([np.sum(y == c) for c in self.classes_])
        present = np.array([X[y == c].sum(axis=0) for c in self.classes_])
        p = (present + self.alpha) / (n_c[:, None] + 2 * self.alpha)
        self.log_p_, self.log_q_ = np.log(p), np.log1p(-p)
        self.class_log_prior_ = np.log(n_c / len(y))
        return self

    def predict_log_proba(self, X):
        X = (X > 0).astype(float)
        return X @ self.log_p_.T + (1 - X) @ self.log_q_.T + self.class_log_prior_

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


def race(ref, X, y, n_train):
    """(Multinomial, Bernoulli) held-out accuracy after training on n_train documents."""
    tr, te = slice(0, n_train), slice(2000, 3000)
    multi = ref.MultinomialNB(alpha=1.0).fit(X[tr], y[tr]).score(X[te], y[te])
    return float(multi), BernoulliNB(alpha=1.0).fit(X[tr], y[tr]).score(X[te], y[te])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "naive_bayes")
    X, y = ref.make_text_data(n_samples=3000, n_features=200, seed=5)
    rng = np.random.RandomState(1)
    tiny = rng.binomial(X.astype(int), 0.01).astype(float)
    short = rng.binomial(X.astype(int), 0.02).astype(float)
    bursty = short * np.where(rng.rand(*short.shape) < 0.05, 20, 1)
    mine = BernoulliNB().fit(X[:2000], y[:2000]).predict_log_proba(X[2000:])
    theirs = SkBernoulliNB(alpha=1.0).fit(X[:2000] > 0, y[:2000])
    return {
        "full": race(ref, X, y, 2000),
        "tiny": {n: race(ref, tiny, y, n) for n in (20, 2000)},
        "short": race(ref, short, y, 2000),
        "bursty": race(ref, bursty, y, 2000),
        "lengths": [float(m.sum(axis=1).mean()) for m in (X, tiny, short)],
        "gap": float(np.max(np.abs(mine - theirs.predict_joint_log_proba(X[2000:] > 0)))),
    }


def verify(result):
    full, tiny, short, bursty = (result[k] for k in ("full", "tiny", "short", "bursty"))
    words = result["lengths"]
    return [
        practice.Check(
            "ANSWER: a tie on the lesson's data; Bernoulli wins once counts are bursty",
            full == (1.0, 1.0) and bursty[1] > bursty[0] + 0.02,
            f"lesson data ({words[0]:.0f} words/doc): {full[0]:.3f} vs {full[1]:.3f}; "
            f"{words[2]:.1f}-word docs: Multinomial {short[0]:.3f}, Bernoulli {short[1]:.3f}; "
            f"same docs with 5% of present words repeated 20x: {bursty[0]:.3f} vs {bursty[1]:.3f}",
        ),
        practice.Check(
            "FINDING: on short text with little training data Bernoulli loses",
            tiny[20][0] > tiny[20][1] + 0.1 and abs(tiny[2000][0] - tiny[2000][1]) < 0.02,
            f"{words[1]:.1f}-word docs, 20 training docs: Multinomial {tiny[20][0]:.3f}, "
            f"Bernoulli {tiny[20][1]:.3f}; with 2000 training docs {tiny[2000][0]:.3f} vs "
            f"{tiny[2000][1]:.3f}",
        ),
        practice.Check(
            "CONTROL: this BernoulliNB matches sklearn's",
            result["gap"] < 1e-9,
            f"joint log-likelihood within {result['gap']:.1e} of sklearn BernoulliNB(alpha=1)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
