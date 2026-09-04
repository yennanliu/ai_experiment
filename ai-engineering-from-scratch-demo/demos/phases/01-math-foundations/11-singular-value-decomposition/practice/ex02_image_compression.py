"""Exercise 2 — compress a grayscale image at ranks 1…100; find the usable rank.

    Load a real grayscale image (or convert one to grayscale). Compress it at
    ranks 1, 5, 10, 25, 50, 100. For each rank, compute the compression ratio and
    the relative error. Find the rank where the image becomes visually
    acceptable.

Reading of the exercise: "visually acceptable" is not assertable as stated, so it
gets a stated stand-in — the rank at which relative Frobenius error first falls
below 10% — and the answer is **rank 100**, measured rather than guessed. Note
how weak the compression is by then: at k=100 the factors still occupy 39% of the
original pixels for 7.4% error, so on a 427x640 photograph SVD is a poor codec
compared with anything that exploits local structure. The image is sklearn's
bundled `china.jpg` in grayscale, so the exercise stays offline.

One naming trap: the lesson's `compression_ratio(m, n, k)` returns
`compressed / original` — a *fraction of the original size*, not an "N× smaller"
ratio. 0.39 means "39% of the original", i.e. 2.6× compression. Check 1 reports
it as a percentage to keep that unambiguous.

Tier T1: needs numpy and pillow (the `vision` group) for the bundled JPEG.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "11-singular-value-decomposition"
RANKS = (1, 5, 10, 25, 50, 100)
ACCEPTABLE = 0.10


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    try:
        from sklearn.datasets import load_sample_image
    except ImportError:
        raise practice.Skip("needs scikit-learn and pillow — "
                            "uv sync --extra math --extra vision") from None
    ref = parity.load_reference(PHASE, LESSON, "svd")
    image = load_sample_image("china.jpg").mean(axis=2)
    m, n = image.shape
    norm = float(numpy.linalg.norm(image))
    spectrum = numpy.linalg.svd(image, compute_uv=False)
    rows = {}
    for k in RANKS:
        U, S, Vt = ref.truncated_svd(image, k)
        rebuilt = ref.reconstruct(U, S, Vt)
        rows[k] = {"error": float(numpy.linalg.norm(image - rebuilt) / norm),
                   "ratio": float(ref.compression_ratio(m, n, k)),
                   "energy": float((spectrum[:k] ** 2).sum() / (spectrum ** 2).sum())}
    acceptable = next((k for k in RANKS if rows[k]["error"] < ACCEPTABLE), None)
    breakeven = (m * n) / (m + n + 1)
    return {"rows": rows, "shape": (m, n), "acceptable": acceptable,
            "breakeven": breakeven, "full_rank": int(min(m, n))}


def verify(result):
    rows = result["rows"]
    m, n = result["shape"]
    errors = [rows[k]["error"] for k in RANKS]
    return [
        practice.Check(f"a real {m}x{n} grayscale image, compressed at "
                       f"{len(RANKS)} ranks",
                       len(rows) == len(RANKS),
                       "; ".join(f"k={k}: err {rows[k]['error']:.1%}, stores "
                                 f"{rows[k]['ratio']:.1%} of the pixels" for k in RANKS)),
        practice.Check("relative error falls monotonically with rank",
                       all(a > b for a, b in zip(errors, errors[1:])),
                       f"{errors[0]:.1%} at k=1 down to {errors[-1]:.2%} at k=100"),
        practice.Check(f"ANSWER: error first drops below {ACCEPTABLE:.0%} at rank "
                       f"{result['acceptable']}",
                       result["acceptable"] == 100,
                       f"k=100 gives {rows[100]['error']:.2%} error while storing "
                       f"{rows[100]['ratio']:.1%} of the pixels and holding "
                       f"{rows[100]['energy']:.2%} of the spectral energy — k=50 is still "
                       f"at {rows[50]['error']:.1%}. SVD needs most of the rank budget to "
                       f"look right on a photograph"),
        practice.Check("rank 1 is recognisable as a failure, not noise",
                       0.1 < errors[0] < 0.35,
                       f"k=1 already captures {rows[1]['energy']:.1%} of the energy at "
                       f"{errors[0]:.1%} error — natural images are dominated by their mean "
                       f"brightness, which one outer product can express"),
        practice.Check("every rank tested is below the break-even point",
                       result["breakeven"] > RANKS[-1]
                       and all(rows[k]["ratio"] < 1 for k in RANKS),
                       f"k(m+n+1) < mn requires k < {result['breakeven']:.0f}; the largest "
                       f"rank tested is {RANKS[-1]}, so all {len(RANKS)} genuinely store "
                       f"less than the original. Past k={result['breakeven']:.0f} the "
                       f"factors cost more than the image, and full rank is "
                       f"{result['full_rank']}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
