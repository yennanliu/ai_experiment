"""Exercise 1 — high alpha hurts once word evidence, shrinking as 1/alpha, falls under the prior.

    **Smoothing experiment.** Train MultinomialNB on text data with alpha values
    of 0.01, 0.1, 1.0, 10.0, and 100.0. Plot accuracy vs alpha. Where does
    performance peak? Why does very high alpha hurt?

Reading of the exercise: "text data" is the lesson's own `make_text_data`
with the lesson's `MultinomialNB`, in two settings. (A) The lesson's demo
setup: 1200 documents of about 362 words each, 25% held out. (B) A
harder one built from the same generator: keep 3% of each document's tokens
(binomial thinning, about 11 words per document, like short text), train on 40
documents and test on 1000. The "plot" is the accuracy-per-alpha table below,
extended past 100 to find where accuracy actually breaks.

**ANSWER: in the lesson's setup there is no peak.** Accuracy is 1.000 at all
five alphas and still 1.000 at alpha = 10^6. In setting B it peaks at alpha = 1
(0.990, 0.994, 0.997, 0.996, 0.966 for 0.01 to 100), and at 1000 it falls to
0.498.

**FINDING: high alpha hurts through the prior, not by blurring the words.**
For large alpha, `(count + alpha) / (total + alpha V)` tends to `1/V` for both
classes, so the log-likelihood difference between the classes shrinks as
`1/alpha`: its median times alpha is 21.4, 26.2, 26.9, 27.0 at alpha = 10, 100,
10^3, 10^4. The log prior does not shrink. Setting B's 40 training documents
split 21/19, a prior gap of 0.100 nats. At alpha = 1000 the median evidence is
0.027 nats, and 100% of test documents get the training-majority class. The
lesson's 900 training documents split 452/448 (a gap of 0.0089), so its
documents can absorb a million-fold smoothing.

**CONTROL: the probabilities go before the accuracy does.** In setting A,
held-out log loss is below 1e-4 up to alpha = 10^4 and rises to 0.426 at
10^6 while accuracy stays at 1.000: very high alpha flattens every posterior
towards the prior first.

Structure: `sweep` scores one setting; `evidence` is the prior-free log-odds.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "14-naive-bayes"
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
WIDE = (1e3, 1e4, 1e5, 1e6)


def setting_a(ref):
    X, y = ref.make_text_data(n_samples=1200, n_features=200, seed=42)
    return ref.train_test_split(X, y, test_ratio=0.25, seed=42)


def setting_b(ref):
    """Short documents (3% of tokens kept) and only 40 training documents."""
    X, y = ref.make_text_data(n_samples=4000, n_features=200, seed=7)
    short = np.random.RandomState(0).binomial(X.astype(int), 0.03).astype(float)
    return short[:40], short[3000:], y[:40], y[3000:]


def evidence(model, X):
    """Class-1 minus class-0 log score with the prior removed."""
    scores = model.predict_log_proba(X)
    prior = model.class_log_prior_[1] - model.class_log_prior_[0]
    return scores[:, 1] - scores[:, 0] - prior


def sweep(ref, split, alphas):
    """alpha -> (accuracy, log loss, median |evidence|, share predicted as majority)."""
    X_tr, X_te, y_tr, y_te = split
    majority = np.argmax(np.bincount(y_tr))
    out = {}
    for alpha in alphas:
        model = ref.MultinomialNB(alpha=alpha).fit(X_tr, y_tr)
        p_true = model.predict_proba(X_te)[np.arange(len(y_te)), y_te]
        out[alpha] = (float(model.score(X_te, y_te)),
                      float(-np.mean(np.log(np.clip(p_true, 1e-300, 1.0)))) + 0.0,
                      float(np.median(np.abs(evidence(model, X_te)))),
                      float(np.mean(model.predict(X_te) == majority)))
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "naive_bayes")
    a, b = setting_a(ref), setting_b(ref)
    curve_a, curve_b = sweep(ref, a, ALPHAS + WIDE), sweep(ref, b, ALPHAS + WIDE)
    split_a, split_b = np.bincount(a[2]), np.bincount(b[2])
    return {
        "a": curve_a, "b": curve_b, "split_a": split_a.tolist(), "split_b": split_b.tolist(),
        "words_b": float(b[1].sum(axis=1).mean()),
        "peak": max(ALPHAS, key=lambda al: curve_b[al][0]),
        "flat": min(acc for acc, *_ in curve_a.values()) == 1.0,
        "scaled": [curve_b[al][2] * al for al in (10.0, 100.0, 1e3, 1e4)],
        "gaps": [float(np.log(s.max() / s.min())) for s in (split_a, split_b)],
    }


def accuracies(curve, alphas=ALPHAS):
    return ", ".join(f"{curve[al][0]:.3f}" for al in alphas)


def verify(result):
    a, b, scaled = result["a"], result["b"], result["scaled"]
    gap_a, gap_b = result["gaps"]
    return [
        practice.Check(
            "ANSWER: no peak on the lesson's data; a peak at alpha=1 on short, scarce text",
            result["flat"] and result["peak"] == 1.0 and b[1e3][0] < 0.6,
            f"lesson setup accuracy {accuracies(a)} (and {a[1e6][0]:.3f} at 1e6); short text "
            f"({result['words_b']:.1f} words/doc, 40 train) {accuracies(b)}, then "
            f"{b[1e3][0]:.3f} at alpha=1000",
        ),
        practice.Check(
            "FINDING: word evidence shrinks as 1/alpha until the fixed prior outvotes it",
            max(scaled) / min(scaled) < 1.4 and b[1e3][2] < gap_b and b[1e3][3] == 1.0,
            f"median |log-odds without prior| x alpha = {', '.join(f'{v:.1f}' for v in scaled)} "
            "at alpha 10..1e4; training split "
            f"{result['split_b']} is a prior gap of {gap_b:.3f} nats, and at alpha=1000 the "
            f"evidence is {b[1e3][2]:.3f}, so {b[1e3][3]:.0%} of test documents get the "
            f"majority class. The lesson's split {result['split_a']} has a gap of "
            f"{gap_a:.4f}",
        ),
        practice.Check(
            "CONTROL: very high alpha ruins the probabilities before the accuracy",
            a[1e4][1] < 1e-4 and a[1e6][1] > 0.2 and a[1e6][0] == 1.0,
            f"lesson setup: held-out log loss {a[1e4][1]:.1e} at alpha=1e4 and "
            f"{a[1e6][1]:.3f} at 1e6, with accuracy {a[1e6][0]:.3f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
