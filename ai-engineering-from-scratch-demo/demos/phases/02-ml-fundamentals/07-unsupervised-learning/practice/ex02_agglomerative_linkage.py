"""Exercise 2 — agglomerative linkage against K-Means, and cutting the dendrogram.

    Add hierarchical agglomerative clustering to the code. Implement Ward's
    linkage and produce a dendrogram (as a nested list of merges). Cut it at
    different levels and compare to K-Means results.

Reading of the exercise: on separated blobs every linkage and K-Means agree
perfectly (1.000 each), so the comparison says nothing there. It is run on
**moons** instead, where the answer separates sharply — single linkage recovers
the true labels exactly while Ward and K-Means both land near 0.75.

Ward's is the linkage the exercise names, and it is the one that fails here: it
minimises within-cluster variance, which is the same spherical assumption K-Means
makes. Check 4 shows the two agreeing with each other far more than either agrees
with the truth, which is the real content of "compare to K-Means".
"""

from __future__ import annotations

from itertools import permutations

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "07-unsupervised-learning"
SEED, N, NOISE, K = 42, 200, 0.08, 2
LINKAGES = ("ward", "single", "complete")


def agreement(a, b, k=K):
    """Best label agreement over all relabellings — cluster ids are arbitrary."""
    return max(sum(1 for x, y in zip(a, b) if mapping[x] == y) / len(b)
               for mapping in (dict(zip(range(k), p)) for p in permutations(range(k))))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "clustering")
    moons, truth = ref.make_moons(n_samples=N, noise=NOISE, seed=SEED)
    with parity.quiet():
        kmeans_labels, _ = ref.kmeans(moons, K, seed=SEED)
        blobs, blob_truth = ref.make_blobs([(0, 0), (6, 0), (3, 6)],
                                           n_per_cluster=40, spread=1.0, seed=SEED)
        blob_kmeans, _ = ref.kmeans(blobs, 3, seed=SEED)
    # agglomerative_clustering returns (labels, merge_history) — the history is
    # the dendrogram the exercise asks for, as a nested list of merges
    merged = {name: ref.agglomerative_clustering(moons, n_clusters=K, linkage=name)
              for name in LINKAGES}
    linkages = {name: labels for name, (labels, _) in merged.items()}
    blob_ward, blob_history = ref.agglomerative_clustering(blobs, n_clusters=3,
                                                           linkage="ward")
    return {
        "kmeans": agreement(kmeans_labels, truth),
        "linkage": {n: agreement(v, truth) for n, v in linkages.items()},
        "ward_vs_kmeans": agreement(linkages["ward"], kmeans_labels),
        "single_vs_kmeans": agreement(linkages["single"], kmeans_labels),
        "blobs_ward": agreement(blob_ward, blob_truth, 3),
        "blobs_kmeans": agreement(blob_kmeans, blob_truth, 3),
        "n": N, "merges": len(merged["ward"][1]),
        "blob_merges": len(blob_history),
    }


def verify(result):
    link, kmeans = result["linkage"], result["kmeans"]
    return [
        practice.Check(f"the dendrogram records {result['merges']} merges for {N} points",
                       result["merges"] == N - K
                       and result["blob_merges"] == 120 - 3,
                       f"agglomerative_clustering returns (labels, merge_history); merging "
                       f"n points down to k clusters takes exactly n − k merges, and cutting "
                       f"that history at a different depth is what changes k"),
        practice.Check("on separated blobs every method is perfect — nothing to compare",
                       result["blobs_ward"] == 1.0 and result["blobs_kmeans"] == 1.0,
                       f"Ward {result['blobs_ward']:.3f}, K-Means "
                       f"{result['blobs_kmeans']:.3f} on 3 Gaussian blobs. The comparison "
                       f"needs a shape that breaks the spherical assumption"),
        practice.Check("ANSWER: on moons, single linkage recovers the truth exactly",
                       link["single"] == 1.0,
                       f"single {link['single']:.3f} over {result['n']} points — it merges "
                       f"whichever two clusters have the closest *pair*, so it follows a "
                       f"curved chain"),
        practice.Check("…while Ward's, the linkage the exercise names, does not",
                       link["ward"] < 0.8,
                       ", ".join(f"{n}: {v:.3f}" for n, v in link.items())
                       + f" against K-Means {kmeans:.3f} — Ward minimises within-cluster "
                         f"variance, which is the same spherical assumption K-Means makes"),
        practice.Check("ANSWER: Ward agrees with K-Means more than either agrees with truth",
                       result["ward_vs_kmeans"] > max(link["ward"], kmeans),
                       f"Ward vs K-Means {result['ward_vs_kmeans']:.3f}, against "
                       f"{link['ward']:.3f} and {kmeans:.3f} vs the truth. They are not "
                       f"independent methods that happen to agree — they optimise the same "
                       f"objective by different search"),
        practice.Check("…and single linkage is the one that disagrees with K-Means",
                       result["single_vs_kmeans"] < result["ward_vs_kmeans"],
                       f"single vs K-Means {result['single_vs_kmeans']:.3f} against Ward's "
                       f"{result['ward_vs_kmeans']:.3f}. 'Compare to K-Means' has a useful "
                       f"answer only for the linkage that is not a variance criterion"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
