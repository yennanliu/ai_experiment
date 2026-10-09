"""Exercise 2 — correlated words co-occur 4-9x too often; it costs confidence, not accuracy.

    **Feature independence test.** Take a real text dataset. Pick two words that
    are obviously correlated ("machine" and "learning"). Compute P(word1 |
    class) * P(word2 | class) and compare to P(word1 AND word2 | class). How
    wrong is the independence assumption? Does it affect classification
    accuracy?

Reading of the exercise: no text dataset ships with scikit-learn offline, so the
real text is this curriculum's own lesson prose: every paragraph of 20+ words
(code blocks removed) in `docs/en.md` of phase 02 (ML fundamentals, class 0)
and phase 03 (deep learning core, class 1). A document is a paragraph, and the
probabilities are document-level presence rates. "machine"/"learning" is the
pair the exercise names; "learning"/"rate" is a second pair with enough
support in both classes. Accuracy uses the lesson's own `MultinomialNB`
(alpha=1) on bag-of-words over a 300-paragraph held-out split.

**ANSWER: the assumption is off by a factor of 4 to 9.** "learning" and "rate"
appear together 6.3x (phase 02) and 4.5x (phase 03) as often as the product of
their rates predicts. "machine" and "learning" co-occur 8.7x as often in phase
02, but only 13 paragraphs there contain "machine".

**FINDING: accuracy does not notice.** Merging every "learning rate" into one
token, so its evidence is counted once, changes 1 of 300 held-out predictions
(accuracy 0.930 to 0.933).

**FINDING: confidence does.** 90% of held-out paragraphs get a posterior above
0.99, and those are right only 97.0% of the time. That is 3x the error rate
their probabilities claim: the double counting, summed over every correlated
pair, pushes posteriors to the extremes, as the doc's "ranking over
calibration" says.

**CONTROL: the lesson's MultinomialNB is sklearn's.** Its `feature_log_prob_`
matches `sklearn.naive_bayes.MultinomialNB(alpha=1)` within 1e-12 on this
corpus.

Structure: `paragraphs` reads the corpus; `cooccur` measures one pair.
"""

from __future__ import annotations

import collections
import re

import numpy as np
from sklearn.naive_bayes import MultinomialNB as SkMultinomialNB

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "14-naive-bayes"
CLASSES = ("02-ml-fundamentals", "03-deep-learning-core")
PAIRS = (("machine", "learning"), ("learning", "rate"))


def paragraphs(phase):
    """Lower-cased word lists of every 20+ word prose paragraph in a phase's docs."""
    out = []
    for doc in sorted((parity.find_reference_root() / "phases" / phase).glob("*/docs/en.md")):
        text = re.sub(r"```.*?```", "", doc.read_text(encoding="utf-8"), flags=re.S)
        out += [w for w in (re.findall(r"[a-z]+", p.lower()) for p in re.split(r"\n\s*\n", text))
                if len(w) >= 20]
    return out


def cooccur(docs, w1, w2):
    """(P(w1), P(w2), P(w1 and w2), lift) over documents, by presence."""
    sets = [set(d) for d in docs]
    a, b = np.array([w1 in s for s in sets]), np.array([w2 in s for s in sets])
    both = float((a & b).mean())
    return float(a.mean()), float(b.mean()), both, both / max(a.mean() * b.mean(), 1e-12)


def merge(doc, pair=("learning", "rate")):
    out, i = [], 0
    while i < len(doc):
        joined = tuple(doc[i:i + 2]) == pair
        out.append("_".join(pair) if joined else doc[i])
        i += 2 if joined else 1
    return out


def bag(docs, train):
    counts = collections.Counter(w for i in train for w in docs[i])
    index = {w: j for j, w in enumerate(sorted(w for w, n in counts.items() if n >= 2))}
    X = np.zeros((len(docs), len(index)))
    for i, doc in enumerate(docs):
        for w in doc:
            if w in index:
                X[i, index[w]] += 1
    return X


def fit(ref, corpus, y, train, test):
    """(held-out posteriors of the lesson's MultinomialNB, max gap to sklearn's log probs)."""
    X = bag(corpus, train)
    model = ref.MultinomialNB(alpha=1.0).fit(X[train], y[train])
    sk = SkMultinomialNB(alpha=1.0).fit(X[train], y[train])
    gap = np.max(np.abs(sk.feature_log_prob_ - model.feature_log_prob_))
    return model.predict_proba(X[test]), float(gap)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "naive_bayes")
    by_class = [paragraphs(p) for p in CLASSES]
    docs = by_class[0] + by_class[1]
    y = np.array([0] * len(by_class[0]) + [1] * len(by_class[1]))
    test, train = np.split(np.random.RandomState(0).permutation(len(docs)), [300])
    proba, gap = fit(ref, docs, y, train, test)
    merged = fit(ref, [merge(d) for d in docs], y, train, test)[0].argmax(axis=1)
    raw, sure = proba.argmax(axis=1), proba.max(axis=1) > 0.99
    return {
        "pairs": {f"{a}/{b}": [cooccur(c, a, b) for c in by_class] for a, b in PAIRS},
        "machine_docs": int(sum("machine" in d for d in by_class[0])),
        "acc": {"raw": float(np.mean(raw == y[test])), "merged": float(np.mean(merged == y[test]))},
        "flips": int(np.sum(raw != merged)),
        "sure_share": float(sure.mean()),
        "sure_acc": float(np.mean(raw[sure] == y[test][sure])),
        "gap": gap, "sizes": [len(c) for c in by_class],
    }


def verify(result):
    lr, ml = result["pairs"]["learning/rate"], result["pairs"]["machine/learning"]
    acc = result["acc"]
    return [
        practice.Check(
            "ANSWER: correlated words co-occur several times more than independence predicts",
            min(lr[0][3], lr[1][3]) > 2 and ml[0][3] > 2,
            f"learning/rate: P(both) {lr[0][2]:.3f} vs P x P {lr[0][0] * lr[0][1]:.4f} in "
            f"phase 02 (lift {lr[0][3]:.1f}), lift {lr[1][3]:.1f} in phase 03; machine/learning "
            f"lift {ml[0][3]:.1f} in phase 02 on {result['machine_docs']} paragraphs "
            f"with 'machine' (of {result['sizes']} paragraphs)",
        ),
        practice.Check(
            "FINDING: counting the pair once barely moves accuracy",
            result["flips"] <= 5 and abs(acc["merged"] - acc["raw"]) < 0.02,
            f"merging 'learning rate' into one token flips {result['flips']} of 300 held-out "
            f"predictions; accuracy {acc['raw']:.3f} -> {acc['merged']:.3f}",
        ),
        practice.Check(
            "FINDING: the posteriors are overconfident",
            result["sure_share"] > 0.5 and result["sure_acc"] < 0.99,
            f"{result['sure_share']:.0%} of held-out paragraphs get P > 0.99 and those are "
            f"right {result['sure_acc']:.1%} of the time",
        ),
        practice.Check(
            "CONTROL: the lesson's MultinomialNB matches sklearn's",
            result["gap"] < 1e-12,
            f"feature_log_prob_ agrees with sklearn MultinomialNB(alpha=1) within "
            f"{result['gap']:.1e}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
