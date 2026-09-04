"""Exercise 2 — t-SNE at perplexity 5, 30 and 100; why cluster tightness changes.

    Run t-SNE on the same MNIST subset with perplexity values of 5, 30, and 100.
    Describe how the output changes. Why does perplexity affect cluster tightness?

Reading of the exercise: "describe how the output changes" cannot be adjectives
if it is to be a solution, so tightness is measured — mean intra-class distance
over mean inter-class distance, where lower means tighter.

The question presumes a direction, and there isn't one. The measured ratio is
**non-monotone**: 0.210 at perplexity 5, **0.173 at 30**, 0.238 at 100. The middle
value is tightest, and both extremes are worse for opposite reasons — too few
neighbours fragments each class into local shards, too many blurs classes into
each other. That is also why 30 is scikit-learn's default. Check 5 notes that
perplexity 100 sits at the usual (n−1)/3 ceiling for a 300-point subset, so the
exercise's third value is a boundary case rather than a midpoint.

Tier T1: t-SNE on 300 points takes a few seconds and needs scikit-learn.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "10-dimensionality-reduction"
PERPLEXITIES = (5, 30, 100)
N_SAMPLES, SEED = 300, 42


def separation(numpy, embedding, labels):
    """mean intra-class distance / mean inter-class distance. Lower is tighter."""
    intra, inter = [], []
    for i in range(len(embedding)):
        deltas = numpy.linalg.norm(embedding - embedding[i], axis=1)
        same = labels == labels[i]
        same[i] = False
        if same.any():
            intra.append(deltas[same].mean())
        inter.append(deltas[~same].mean())
    return float(numpy.mean(intra) / numpy.mean(inter))


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy and scikit-learn — uv sync --extra math")
    try:
        from sklearn.datasets import load_digits
        from sklearn.manifold import TSNE
    except ImportError:
        raise practice.Skip("needs scikit-learn — uv sync --extra math") from None
    digits = load_digits()
    X, labels = digits.data[:N_SAMPLES], digits.target[:N_SAMPLES]
    rows = {}
    for perplexity in PERPLEXITIES:
        embedding = TSNE(n_components=2, perplexity=perplexity, random_state=SEED,
                         init="pca", max_iter=400).fit_transform(X)
        rows[perplexity] = {"ratio": separation(numpy, embedding, labels),
                            "spread": float(numpy.linalg.norm(
                                embedding - embedding.mean(axis=0), axis=1).mean())}
    return {"rows": rows, "n": N_SAMPLES, "limit": (N_SAMPLES - 1) / 3}


def verify(result):
    rows = result["rows"]
    ratios = [rows[p]["ratio"] for p in PERPLEXITIES]
    return [
        practice.Check(f"t-SNE run at all {len(PERPLEXITIES)} perplexities on "
                       f"{result['n']} digits",
                       len(rows) == len(PERPLEXITIES),
                       ", ".join(f"perp={p}: intra/inter {rows[p]['ratio']:.4f}"
                                 for p in PERPLEXITIES)),
        practice.Check("every setting separates the classes (ratio well below 1)",
                       all(r < 0.7 for r in ratios),
                       f"worst ratio {max(ratios):.4f} — same-digit points are much closer "
                       f"to each other than to other digits at every perplexity"),
        practice.Check("FINDING: tightness is non-monotone — perplexity 30 wins",
                       ratios[1] < ratios[0] and ratios[1] < ratios[2],
                       f"perp=5 → {ratios[0]:.4f}, perp=30 → {ratios[1]:.4f} (tightest), "
                       f"perp=100 → {ratios[2]:.4f}; the question presumes a direction and "
                       f"there is none"),
        practice.Check("…because the two extremes fail for opposite reasons",
                       ratios[0] > ratios[1] < ratios[2],
                       f"perplexity is the effective neighbour count each point preserves. "
                       f"At 5 each point sees too few neighbours and a class fragments into "
                       f"local shards ({ratios[0]:.4f}); at 100 the neighbourhood spans other "
                       f"digits and classes blur together ({ratios[2]:.4f}). 30 is "
                       f"scikit-learn's default for this reason"),
        practice.Check(f"perp=100 is near the documented ceiling for n={result['n']}",
                       PERPLEXITIES[-1] > result["limit"] * 0.9,
                       f"scikit-learn requires perplexity < n_samples; the usual guidance is "
                       f"below (n−1)/3 = {result['limit']:.0f}, so 100 is at the edge of "
                       f"meaningful on this subset — the exercise's third value is a "
                       f"boundary case, not a midpoint"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
