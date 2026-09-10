"""Exercise 1 — most of the gain is filter count.

    **Easy.** Train a TextCNN on a 3-class toy dataset (you invent the data).
    Verify that filter widths (2, 3, 4) outperform a single width (3) on average
    F1.

Reading of the exercise: the claim holds -- 0.5910 macro-F1 against 0.4777 --
and the comparison as posed does not isolate what it names. Three widths at
eight filters each is 24 filters against 8, so the obvious reading changes two
things at once. Held to 24 filters on both sides the gain survives and shrinks
by four fifths, to 0.5910 against 0.5705: most of the apparent effect is filter
count, and a fifth of it is width.

That fifth is real and has a direction. Widths (5, 6, 7), also 24 filters, score
0.4755 -- level with single-width-3 at a third of the filters, and far below the
matched arm -- because the classes here are separated by a two-token and a
three-token pattern and a six-wide window averages both away. Width matters; it
is just worth less than the phrasing implies.

"On average" is also doing work. Over 12 embedding seeds the multi-width set
wins 9 times, and the per-seed gap runs from -0.1655 to +0.3897 -- a spread
three times the mean gap of +0.1133. A single training run answers this exercise
either way.

torch is not installed, so the filters here are frozen random projections and
only the linear readout is trained: `conv1d_over_embeddings` from the lesson
provides each filter's activations and `max_pool` reduces them, which is a
TextCNN's forward pass with the convolution left untrained. That isolates filter
geometry from optimisation, which is what the exercise is asking about, and it
is why the absolute numbers sit near 0.5 rather than near 1.

Structure: `corpus` plants `very good` in class 0, `not very good` in class 1 and
neither in class 2, so only a width-3 window can separate the first two.
`featurize` runs the lesson's own convolution and pooling over a frozen
embedding table; `arm` scores one (widths, filters-per-width) configuration
across seeds.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "08-cnns-rnns-for-text"

FILLER = ("the", "a", "some", "quite", "rather", "film", "movie", "story", "acting",
          "script", "pacing", "scene", "ending", "cast")
MARKERS = ("very", "good", "not")
DIM, SEEDS, BIAS = 16, 12, -0.2
ARMS = {"one width": ((3,), 8), "three widths": ((2, 3, 4), 8),
        "one width, matched": ((3,), 24), "too wide": ((5, 6, 7), 8)}


def corpus(np, per_class=40) -> tuple:
    """Class 0 carries 'very good', class 1 'not very good', class 2 neither."""
    rng, rows = np.random.default_rng(0), []
    for label in (0, 1, 2):
        for _ in range(per_class):
            length = int(rng.integers(8, 12))
            tokens = [str(t) for t in rng.choice(FILLER, size=length)]
            at = int(rng.integers(0, length - 3))
            if label == 0:
                tokens[at:at + 2] = ["very", "good"]
            elif label == 1:
                tokens[at:at + 3] = ["not", "very", "good"]
            else:
                tokens[at:at + 2] = ["good", "very"]
            rows.append((tokens, label))
    return rows, sorted({t for tokens, _ in rows for t in tokens})


def table(np, vocab, seed) -> dict:
    return dict(zip(vocab, np.random.default_rng(seed).normal(size=(len(vocab), DIM))))


def filters(np, widths, per_width, seed) -> list:
    rng = np.random.default_rng(seed + 999)
    return [rng.normal(scale=0.5, size=(w, DIM)) for w in widths for _ in range(per_width)]


def featurize(ref, tokens, embeddings, banks) -> list:
    """One max-pooled activation per filter -- the lesson's own conv and pool."""
    rows = [list(embeddings[t]) for t in tokens]
    return [ref.max_pool(ref.conv1d_over_embeddings(rows, [list(r) for r in bank], bias=BIAS))
            for bank in banks]


def arm(np, ref, rows, vocab, widths, per_width) -> list:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    labels = np.array([label for _, label in rows])
    order = np.random.default_rng(7).permutation(len(labels))
    cut = int(len(labels) * 0.7)
    scores = []
    for seed in range(SEEDS):
        embeddings, banks = table(np, vocab, seed), filters(np, widths, per_width, seed)
        matrix = np.array([featurize(ref, tokens, embeddings, banks) for tokens, _ in rows])
        model = LogisticRegression(max_iter=3000).fit(matrix[order[:cut]], labels[order[:cut]])
        scores.append(float(f1_score(labels[order[cut:]], model.predict(matrix[order[cut:]]),
                                     average="macro")))
    return scores


def solve():
    try:
        import numpy as np
        import sklearn                              # noqa: F401
    except ImportError as exc:                      # pragma: no cover - T1 needs sklearn
        raise practice.Skip(f"needs scikit-learn: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows, vocab = corpus(np)
    runs = {name: arm(np, ref, rows, vocab, widths, per)
            for name, (widths, per) in ARMS.items()}
    gaps = [b - a for a, b in zip(runs["one width"], runs["three widths"])]
    return {
        "mean": {name: round(sum(s) / len(s), 4) for name, s in runs.items()},
        "range": {name: (round(min(s), 4), round(max(s), 4)) for name, s in runs.items()},
        "counts": {name: len(widths) * per for name, (widths, per) in ARMS.items()},
        "wins": sum(g > 0 for g in gaps), "seeds": SEEDS,
        "gap": (round(min(gaps), 4), round(sum(gaps) / len(gaps), 4), round(max(gaps), 4)),
        "rows": len(rows), "vocab": len(vocab),
    }


def verify(result):
    mean, counts, gap = result["mean"], result["counts"], result["gap"]
    matched = mean["three widths"] - mean["one width, matched"]
    naive = mean["three widths"] - mean["one width"]
    return [
        practice.Check(
            "ANSWER: widths (2,3,4) do beat width 3 on average macro-F1, 0.5910 to 0.4777",
            mean["three widths"] > mean["one width"],
            f"on {result['rows']} sentences over {result['vocab']} word types, averaged across "
            f"{result['seeds']} embedding seeds: {mean['three widths']} against "
            f"{mean['one width']}, a gap of {naive:+.4f}. torch is not installed, so the filters are "
            f"frozen random projections and only the readout is trained -- which is why the absolute "
            f"numbers sit near 0.5 rather than near 1"),
        practice.Check(
            "MECHANISM: the comparison changes two things -- three widths is also three times the filters",
            counts["three widths"] == 3 * counts["one width"],
            f"three widths at 8 filters each is {counts['three widths']} filters against "
            f"{counts['one width']}. The readout gets three times as many features, so the "
            f"as-posed comparison cannot separate 'more widths' from 'more filters'"),
        practice.Check(
            "FINDING: held to 24 filters on both sides, four fifths of the gain disappears",
            0 < matched < naive / 4,
            f"width 3 with {counts['one width, matched']} filters scores "
            f"{mean['one width, matched']} against the multi-width set's {mean['three widths']} -- a "
            f"gap of {matched:+.4f} where the uncontrolled comparison showed {naive:+.4f} -- "
            f"{1 - matched / naive:.0%} of the effect was capacity, not width"),
        practice.Check(
            "CONTROL: width still matters, and in the right direction",
            mean["too wide"] < mean["one width, matched"],
            f"widths (5,6,7), also {counts['too wide']} filters, score {mean['too wide']} against "
            f"the matched single-width arm's {mean['one width, matched']} -- and level with "
            f"width 3 at {counts['one width']} filters. The classes are separated by a two-token "
            f"and a three-token pattern, and a six-wide window averages both away, so the extra "
            f"capacity buys nothing. The width effect is small and it is not zero"),
        practice.Check(
            "FINDING: 'on average' is doing work -- the per-seed gap changes sign",
            0 < result["wins"] < result["seeds"] and gap[0] < 0 < gap[2],
            f"the multi-width set wins on {result['wins']} of {result['seeds']} seeds. The per-seed "
            f"gap runs {gap[0]:+.4f} to {gap[2]:+.4f} around a mean of {gap[1]:+.4f} -- a spread "
            f"three times the effect. One training run answers this exercise either way"),
        practice.Check(
            "CONTROL: every arm is above chance, so the ranking is of working models",
            min(mean.values()) > 1 / 3,
            f"macro-F1 for a three-class guess is about {1 / 3:.3f}; the arms score {mean}. Even the "
            f"deliberately-too-wide configuration is above it, so what is being compared is a set of "
            f"models that all work, not one that works against ones that do not"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
