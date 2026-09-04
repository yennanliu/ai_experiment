"""Exercise 4 — a 100x50 document-term matrix with 3 topics; do they separate?

    Create a 100x50 document-term matrix with 3 synthetic topics. Each topic has
    5 associated terms. Add noise. Apply SVD and verify that the top 3 singular
    values are much larger than the rest. Project documents into the 3D latent
    space and check that documents from the same topic cluster together.

Reading of the exercise: "much larger" and "cluster together" both need numbers.
The spectral gap is measured as σ₃/σ₄ and the clustering as an intra/inter
distance ratio in the 3D projection, both reported. Checks 4 and 5 go after the part
"cluster together" leaves out: *which* latent dimension is which topic. The
answer is that the three components each load on a distinct topic block, but in
a permuted order — component 0 picks up topic 2, component 2 picks up topic 0 —
so latent dimensions carry no labels and must be read off the loadings.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "11-singular-value-decomposition"
N_DOCS, N_TERMS, N_TOPICS, TERMS_PER_TOPIC = 100, 50, 3, 5
SEED, NOISE = 42, 0.4


def build(numpy, rng):
    matrix = numpy.zeros((N_DOCS, N_TERMS))
    labels = rng.integers(N_TOPICS, size=N_DOCS)
    for doc, topic in enumerate(labels):
        start = topic * TERMS_PER_TOPIC
        matrix[doc, start:start + TERMS_PER_TOPIC] = rng.poisson(6, TERMS_PER_TOPIC)
    matrix += rng.poisson(NOISE, (N_DOCS, N_TERMS))
    return matrix, labels


def separation(numpy, points, labels):
    intra, inter = [], []
    for i in range(len(points)):
        distances = numpy.linalg.norm(points - points[i], axis=1)
        same = labels == labels[i]
        same[i] = False
        intra.append(distances[same].mean())
        inter.append(distances[~same].mean())
    return float(numpy.mean(intra) / numpy.mean(inter))


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "svd")
    rng = numpy.random.default_rng(SEED)
    matrix, labels = build(numpy, rng)
    spectrum = numpy.linalg.svd(matrix, compute_uv=False)
    U, S, Vt = ref.truncated_svd(matrix, N_TOPICS)
    projected = U * S
    # mean |loading| of each component on each planted topic block, and on noise
    blocks = [[float(numpy.abs(Vt[j, t * TERMS_PER_TOPIC:(t + 1) * TERMS_PER_TOPIC]).mean())
               for t in range(N_TOPICS)] for j in range(N_TOPICS)]
    noise_loading = [float(numpy.abs(Vt[j, N_TOPICS * TERMS_PER_TOPIC:]).mean())
                     for j in range(N_TOPICS)]
    return {
        "spectrum": spectrum[:6].tolist(),
        "gap": float(spectrum[N_TOPICS - 1] / spectrum[N_TOPICS]),
        "energy": float((spectrum[:N_TOPICS] ** 2).sum() / (spectrum ** 2).sum()),
        "ratio_3d": separation(numpy, projected, labels),
        "ratio_raw": separation(numpy, matrix, labels),
        "blocks": blocks, "noise_loading": noise_loading,
        "argmax": [max(range(N_TOPICS), key=lambda t: row[t]) for row in blocks],
    }


def verify(result):
    return [
        practice.Check(f"{N_DOCS}x{N_TERMS} matrix, {N_TOPICS} topics of "
                       f"{TERMS_PER_TOPIC} terms, Poisson noise",
                       len(result["spectrum"]) == 6,
                       f"σ₁…σ₆ = {[round(v, 2) for v in result['spectrum']]}"),
        practice.Check(f"the top {N_TOPICS} singular values are much larger than the rest",
                       result["gap"] > 3,
                       f"σ₃/σ₄ = {result['gap']:.2f}, and the top {N_TOPICS} hold "
                       f"{result['energy']:.1%} of the spectral energy"),
        practice.Check("same-topic documents cluster in the 3D projection",
                       result["ratio_3d"] < 0.8,
                       f"intra/inter distance ratio {result['ratio_3d']:.4f} in latent space "
                       f"against {result['ratio_raw']:.4f} in the raw 50-d term space — "
                       f"the projection improves separation, it does not merely preserve it"),
        practice.Check("each component loads on a distinct planted topic block",
                       sorted(result["argmax"]) == list(range(N_TOPICS)),
                       "; ".join(f"component {j} → topic {result['argmax'][j]} "
                                 f"({result['blocks'][j][result['argmax'][j]]:.3f})"
                                 for j in range(N_TOPICS))
                       + " — a bijection, so all three topics are recovered"),
        practice.Check("…and the ordering is permuted, so latent dimensions carry no labels",
                       result["argmax"] != list(range(N_TOPICS))
                       and all(result["blocks"][j][result["argmax"][j]]
                               > 3 * result["noise_loading"][j] for j in range(N_TOPICS)),
                       f"the mapping is {result['argmax']}, not [0, 1, 2] — SVD orders by "
                       f"variance, not by topic index. Topic-block loadings beat noise-term "
                       f"loadings by "
                       f"{min(result['blocks'][j][result['argmax'][j]] / result['noise_loading'][j] for j in range(N_TOPICS)):.0f}x "
                       f"at worst, which is what makes the assignment readable at all"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
