"""Exercise 3 — L1, L2 and cosine for KNN on TF-IDF; why cosine wins for text.

    Compare L1, L2, and cosine distance for KNN on a text classification problem
    (use TF-IDF vectors). Which metric gives the best accuracy? Why does cosine
    tend to win for text?

Reading of the exercise: cosine only wins if document *lengths* vary, so the
corpus is built with lengths of 5, 10 and 60 tokens and a shared vocabulary
between the two topics — without both, all three metrics score 1.000 and the
exercise cannot discriminate. Check 4 measures the mechanism directly: the TF-IDF
vector norms span 11.9x, and that magnitude is exactly what cosine divides out
and L1/L2 do not.

Corpus is synthetic and seeded, vectorised with sklearn's TfidfVectorizer at
norm=None — the default L2 normalisation would erase the effect being studied.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "06-knn-and-distances"
SEED, K = 7, 5
COMMON = "the of and report data system value".split()
TOPICS = {0: "budget revenue profit tax".split(), 1: "engine turbine piston torque".split()}
LENGTHS = (5, 10, 60)


def build_corpus(rng):
    docs, labels = [], []
    for topic, words in TOPICS.items():
        for _ in range(50):
            length = rng.choice(LENGTHS)
            docs.append(" ".join(rng.choice(words) if rng.random() < 0.35
                                 else rng.choice(COMMON) for _ in range(length)))
            labels.append(topic)
    return docs, labels


def _score_metrics(ref, matrix, labels, train, test):
    scores = {}
    for name, fn in (("L1", ref.l1_distance), ("L2", ref.l2_distance),
                     ("cosine", ref.cosine_distance)):
        model = ref.KNN(k=K, distance_fn=fn)
        model.fit([matrix[i] for i in train], [labels[i] for i in train])
        scores[name] = ref.accuracy([labels[i] for i in test],
                                    model.predict([matrix[i] for i in test]))
    return scores


def solve():
    if parity.try_numpy() is None:
        raise practice.Skip("needs numpy and scikit-learn — uv sync --extra math")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        raise practice.Skip("needs scikit-learn — uv sync --extra math") from None
    ref = parity.load_reference(PHASE, LESSON, "knn")
    rng = random.Random(SEED)
    docs, labels = build_corpus(rng)
    matrix = TfidfVectorizer(norm=None).fit_transform(docs).toarray().tolist()
    order = list(range(len(matrix)))
    rng.shuffle(order)
    split = int(0.7 * len(order))
    train, test = order[:split], order[split:]
    scores = _score_metrics(ref, matrix, labels, train, test)
    norms = [math.sqrt(sum(v * v for v in row)) for row in matrix]
    return {"scores": scores, "min_norm": min(norms), "max_norm": max(norms),
            "n_docs": len(docs), "vocab": len(matrix[0])}


def verify(result):
    scores = result["scores"]
    spread = result["max_norm"] / result["min_norm"]
    return [
        practice.Check(f"{result['n_docs']} documents over a {result['vocab']}-word vocabulary",
                       result["vocab"] > 8 and len(scores) == 3,
                       ", ".join(f"{k}: {v:.1%}" for k, v in scores.items())),
        practice.Check("ANSWER: cosine wins",
                       scores["cosine"] > scores["L1"] and scores["cosine"] > scores["L2"],
                       f"cosine {scores['cosine']:.1%} against L1 {scores['L1']:.1%} and "
                       f"L2 {scores['L2']:.1%} at K={K}"),
        practice.Check("…and L2 is the worst of the three, not L1",
                       scores["L2"] < scores["L1"],
                       f"L2 {scores['L2']:.1%} against L1 {scores['L1']:.1%} — squaring "
                       f"amplifies the length difference that is the problem here, so the "
                       f"metric that punishes large coordinates most does worst"),
        practice.Check("WHY: document lengths make the vector norms span 11.9x",
                       spread > 5,
                       f"TF-IDF norms run {result['min_norm']:.2f} to "
                       f"{result['max_norm']:.2f}, a {spread:.1f}x spread, because documents "
                       f"are {LENGTHS[0]}, {LENGTHS[1]} or {LENGTHS[-1]} tokens long. L1 and "
                       f"L2 see a long document as far from a short one on the same topic; "
                       f"cosine divides that magnitude out and compares direction only"),
        practice.Check("…and both conditions are needed for the exercise to discriminate",
                       min(scores.values()) < 0.95,
                       f"the topics share {len(COMMON)} common words. With disjoint "
                       f"vocabularies and uniform lengths all three metrics score 100% and "
                       f"the comparison is vacuous — which is what a first attempt at this "
                       f"exercise produces"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
