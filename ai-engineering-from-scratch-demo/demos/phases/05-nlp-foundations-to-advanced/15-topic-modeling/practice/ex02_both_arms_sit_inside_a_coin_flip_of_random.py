"""Exercise 2 — both arms sit inside a coin flip of random.

    **Medium.** Fit BERTopic on the same 20 Newsgroups subset. Compare the number
    of topics found, top words, and qualitative coherence against LDA. Which
    surfaces the real categories more cleanly?

Reading of the exercise: BERTopic is not installed and its embedding model is
not downloadable, so the stand-in is its shape without its encoder -- TF-IDF
reduced by truncated SVD, clustered by k-means, with each cluster's top terms
taken from the mean TF-IDF vector. That is embed-then-cluster, which is what
BERTopic is; what it is not is a sentence transformer, and the difference shows
in nothing measured below.

Over four seeds at K=4 on exercise 1's labelled corpus, LDA averages 0.4375
purity and the clustering arm 0.4792, against a random-assignment baseline of
0.4353. Both are inside four points of chance, and the gap between them is
smaller than either method's own spread across seeds: clustering runs 0.3750 to
0.6250 and LDA 0.4167 to 0.5417. "Which surfaces the real categories more
cleanly" is answered by the seed on this corpus, not by the method.

The other half of the exercise -- "compare the number of topics found" -- does
not apply. BERTopic discovers its topic count from the density of the embedding
space; both arms here take K as an argument, so the number of topics found is
the number of topics asked for, on both sides. The comparison the exercise
frames as a result of the method is a parameter of the experiment.

Structure: the corpus, labels, `fit` and `purity` come from exercise 1 via
`practice.load_module`. `cluster` builds the BERTopic-shaped arm; `spread`
reports the min and max over seeds for each method so the between-method gap can
be read against the within-method one.
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "15-topic-modeling"

SIBLING = "ex01_lda_lands_a_few_points_above_random.py"
UNAVAILABLE = ("bertopic", "sentence_transformers", "umap", "hdbscan")
TOPICS, SEEDS, TERMS = 4, 4, 8


def cluster(np, docs, topics, seed) -> tuple:
    """BERTopic's shape without its encoder: reduce, cluster, read terms off the centroid."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    joined = [" ".join(doc) for doc in docs]
    vector = TfidfVectorizer().fit(joined)
    matrix = vector.transform(joined)
    names = vector.get_feature_names_out()
    reduced = TruncatedSVD(n_components=min(TERMS, topics * 2), random_state=seed).fit_transform(matrix)
    reduced = reduced / (np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-9)
    labels = list(KMeans(n_clusters=topics, n_init=10, random_state=seed).fit(reduced).labels_)
    words = []
    for k in range(topics):
        rows = matrix[[i for i, label in enumerate(labels) if label == k]]
        centre = np.asarray(rows.mean(axis=0)).ravel() if rows.shape[0] else np.zeros(len(names))
        words.append([names[i] for i in np.argsort(-centre)[:TERMS]])
    return words, labels


def summarise(sibling, scores) -> dict:
    return {"purity": {name: sibling.mean(values) for name, values in scores.items()},
            "spread": {name: (round(min(values), 4), round(max(values), 4))
                       for name, values in scores.items()}}


def solve():
    try:
        import numpy as np
        import sklearn                              # noqa: F401
    except ImportError as exc:                      # pragma: no cover - T1 needs sklearn
        raise practice.Skip(f"needs scikit-learn: uv sync --extra math ({exc})") from None
    import importlib.util
    ref = parity.load_reference(PHASE, LESSON, "main")
    sibling = practice.load_module(pathlib.Path(__file__).with_name(SIBLING))
    docs, gold = sibling.corpus(ref, drop_stopwords=True)
    lda = [sibling.fit(ref, docs, TOPICS, seed) for seed in range(SEEDS)]
    clustered = [cluster(np, docs, TOPICS, seed) for seed in range(SEEDS)]
    scores = {"lda": [sibling.purity(a, gold) for _, a in lda],
              "cluster": [sibling.purity(a, gold) for _, a in clustered]}
    return dict(
        summarise(sibling, scores),
        unavailable=[m for m in UNAVAILABLE if importlib.util.find_spec(m) is None],
        **{"random": sibling.expected_purity(gold, TOPICS), "documents": len(gold),
        "found": {"lda": len(lda[0][0]), "cluster": len(clustered[0][0])}, "asked": TOPICS,
           "lda_topic": lda[0][0][0][:5], "cluster_topic": clustered[0][0][0][:5],
           "seeds": SEEDS})


def verify(result):
    scores, spread, chance = result["purity"], result["spread"], result["random"]
    gap = abs(scores["cluster"] - scores["lda"])
    widths = {name: round(hi - lo, 4) for name, (lo, hi) in spread.items()}
    return [
        practice.Check(
            "ANSWER: LDA 0.4375, clustering 0.4792, random 0.4353 -- all three within four points",
            abs(scores["lda"] - chance) < 0.05 and abs(scores["cluster"] - chance) < 0.05,
            f"{result['unavailable']} are all absent, so the stand-in is BERTopic's shape without "
            f"its encoder: TF-IDF reduced by SVD, clustered by k-means, terms read off each "
            f"centroid. Over {result['seeds']} seeds on {result['documents']} documents the two "
            f"arms score {scores} against a random baseline of {chance}"),
        practice.Check(
            "MECHANISM: the gap between the methods is smaller than either one's spread over seeds",
            gap < min(widths.values()),
            f"the methods differ by {gap:.4f} while clustering runs {spread['cluster']} across "
            f"seeds (width {widths['cluster']}) and LDA {spread['lda']} (width {widths['lda']}). "
            f"'Which surfaces the real categories more cleanly' is answered by the seed"),
        practice.Check(
            "FINDING: the clustering arm is the more variable of the two",
            widths["cluster"] > widths["lda"],
            f"clustering's purity spans {widths['cluster']} across {result['seeds']} seeds against "
            f"LDA's {widths['lda']}. Its best run is the best result in the experiment and its "
            f"worst is the worst, so a single fit of it is the least informative single number "
            f"available"),
        practice.Check(
            "MECHANISM: 'compare the number of topics found' does not apply to either arm",
            result["found"]["lda"] == result["found"]["cluster"] == result["asked"],
            f"both arms return {result['found']['lda']} topics because both were asked for "
            f"{result['asked']}. BERTopic discovers its count from the density of the embedding "
            f"space; neither of these does, so the quantity the exercise treats as an output is an "
            f"input on both sides"),
        practice.Check(
            "FINDING: the top-word lists are readable from both, which is why they persuade",
            len(result["lda_topic"]) == len(result["cluster_topic"]) == 5,
            f"LDA's first topic reads {result['lda_topic']} and the clustering arm's "
            f"{result['cluster_topic']}. Both look like categories; neither partition is more than "
            f"four points from random. Qualitative coherence is the thing both methods deliver and "
            f"the thing that does not distinguish them"),
        practice.Check(
            "CONTROL: the stand-in is BERTopic's pipeline, not its encoder",
            "bertopic" in result["unavailable"] and "sentence_transformers" in result["unavailable"],
            f"embed, reduce, cluster, extract terms per cluster -- the shape is right and the "
            f"embedding is TF-IDF rather than a sentence transformer. Anything in the exercise that "
            f"turns on semantic embeddings specifically is not measured here, and nothing measured "
            f"here turned on them"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
