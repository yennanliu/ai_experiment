"""Exercise 2 — fourteen frames do not carry a fourth moment.

    **Medium.** Replace `summarize` with [mean, var, skew, kurtosis]. Does
    4-moment pooling beat mean+var on the same synthetic dataset?

Reading of the exercise: "on the same synthetic dataset" is taken literally --
`main()`'s four tones at sigma=0.05, same 12/5 split, same `featurize`, same
`knn` -- and then, because that dataset answers nothing, repeated at two higher
noise levels to find where an answer could exist.

**No.** On the dataset as shipped both poolings score **1.000**, so the question
has no answer there. Raising the noise does not produce one either: the two arms
differ by one to three clips out of twenty, which is one to three steps of the
test set's own 1/20 resolution.

| sigma | mean+var (26-d) | 4-moment (52-d) |
|---:|---:|---:|
| 0.05 (the lesson's) | 1.00 | 1.00 |
| 1.0 | 0.80 | 0.90 |
| 4.0 | 0.30 | 0.15 |

Two reasons it cannot do better, both fixed by the lesson's own parameters.

**A clip is 14 frames.** At n=14 the standard error of skewness is
`sqrt(6/n) = 0.655` and of excess kurtosis `sqrt(24/n) = 1.309` -- the size of
the quantities being estimated. Measured within-class relative spread across the
four blocks says the same thing: **0.085** for the means, 0.379 for the
variances, **4.13** for the skews, 1.52 for the kurtoses. The two new blocks
vary within a single class by up to 49x what the mean block does.

**`cosine` is norm-weighted, and the new half carries 11% of the norm.** MFCC
means and variances run to tens; standardised third and fourth moments are O(1).
Appending 26 dimensions that hold `0.11` of the vector length can move a cosine
ranking by about that much, whatever they contain.

So the honest answer to "does it beat mean+var" is that this dataset cannot tell,
and the pooling is not the reason.

Structure: `moments` computes both poolings from one pass over the same MFCC
frames, so no arm pays for the other's features; `dataset` rebuilds `main()`'s
corpus at a noise level; `block_spread` is the within-class relative standard
deviation of one slice of the feature vector; `norm_share` is what the new half
contributes to the vector length.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "03-audio-classification"
SR, SECONDS, K, N_TRAIN, N_TEST = 8000, 0.25, 3, 12, 5
CLASSES = {"low": 200, "mid_low": 400, "mid_high": 800, "high": 1600}
SIGMAS = (0.05, 1.0, 4.0)
BLOCKS = {"mean": slice(0, 13), "var": slice(13, 26), "skew": slice(26, 39),
          "kurtosis": slice(39, 52)}


def central(frames, mean, scale, power):
    """The `power`-th central moment of each coefficient, divided by `scale`."""
    return [sum(((f[i] - mean[i]) / scale[i]) ** power for f in frames) / len(frames)
            for i in range(len(mean))]


def moments(frames):
    """(mean+var, mean+var+skew+kurtosis) from one pass over the same frames."""
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(len(frames[0]))]
    var = central(frames, mean, [1.0] * len(mean), 2)
    sd = [math.sqrt(v) or 1e-12 for v in var]
    skew, kurt = central(frames, mean, sd, 3), central(frames, mean, sd, 4)
    return mean + var, mean + var + skew + [k - 3.0 for k in kurt]


def dataset(ref, sigma):
    """`main()`'s corpus at one noise level, pooled both ways from shared frames."""
    random.seed(42)
    train, test = ([], [], []), ([], [], [])
    for name, freq in CLASSES.items():
        for index in range(N_TRAIN + N_TEST):
            frames = ref.featurize(ref.add_noise(ref.sine(freq, SR, SECONDS), sigma), SR)
            two, four = moments(frames)
            bucket = train if index < N_TRAIN else test
            bucket[0].append(two)
            bucket[1].append(four)
            bucket[2].append(name)
    return train, test


def score(ref, train, test, arm):
    return sum(ref.knn(q, train[arm], train[2], K) == g
               for q, g in zip(test[arm], test[2])) / len(test[2])


def block_spread(rows, labels, window):
    """Median within-class relative standard deviation over one feature block."""
    out = []
    for name in CLASSES:
        group = [row[window] for row, label in zip(rows, labels) if label == name]
        for column in zip(*group):
            centre = statistics.mean(column)
            if abs(centre) > 1e-9:
                out.append(statistics.pstdev(column) / abs(centre))
    return statistics.median(out)


def norm_share(vector, start=26):
    """What fraction of the 52-d vector's length lives in the skew+kurtosis half."""
    total = math.sqrt(sum(x * x for x in vector))
    return math.sqrt(sum(x * x for x in vector[start:])) / total


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs, reference = {}, None
    for sigma in SIGMAS:
        train, test = dataset(ref, sigma)
        reference = reference or train
        runs[sigma] = {"two": score(ref, train, test, 0), "four": score(ref, train, test, 1)}
    rows, labels = reference[1], reference[2]
    return {
        "runs": runs, "frames": len(ref.featurize(ref.sine(200, SR, SECONDS), SR)),
        "spread": {name: block_spread(rows, labels, window) for name, window in BLOCKS.items()},
        "share": statistics.median(norm_share(row) for row in rows),
        "test_n": len(CLASSES) * N_TEST,
    }


def verify(result):
    runs, spread, n = result["runs"], result["spread"], result["test_n"]
    shipped = runs[SIGMAS[0]]
    gaps = [abs(row["four"] - row["two"]) * n for row in runs.values()]
    frames = result["frames"]
    return [
        practice.Check(
            "ANSWER: no -- on the dataset as shipped both poolings score 1.000",
            shipped["two"] == shipped["four"] == 1.0,
            f"at the lesson's sigma={SIGMAS[0]}, 26-d mean+var and 52-d 4-moment both return "
            f"{shipped['two']:.3f} over {n} test clips. The question the exercise asks has no "
            "answer on the dataset it asks about",
        ),
        practice.Check(
            "FINDING: where the ceiling lifts, the gap is one to three clips of twenty",
            max(gaps) <= 3,
            f"{runs[SIGMAS[1]]['two']:.2f} against {runs[SIGMAS[1]]['four']:.2f} at "
            f"sigma={SIGMAS[1]} and {runs[SIGMAS[2]]['two']:.2f} against "
            f"{runs[SIGMAS[2]]['four']:.2f} at sigma={SIGMAS[2]} -- differences of "
            f"{[int(g) for g in gaps]} clips, in either direction, on a test set whose "
            f"resolution is 1/{n}",
        ),
        practice.Check(
            "MECHANISM: a clip is 14 frames, and a fourth moment needs more than that",
            frames == 14 and spread["skew"] > 10 * spread["mean"],
            f"at n={frames} the standard error of skewness is sqrt(6/n)="
            f"{math.sqrt(6 / frames):.3f} and of excess kurtosis sqrt(24/n)="
            f"{math.sqrt(24 / frames):.3f}. Measured within-class relative spread: mean "
            f"{spread['mean']:.3f}, var {spread['var']:.3f}, skew {spread['skew']:.2f}, kurtosis "
            f"{spread['kurtosis']:.2f} -- the new blocks wander {spread['skew'] / spread['mean']:.0f}x "
            "further inside one class",
        ),
        practice.Check(
            "MECHANISM: `cosine` is norm-weighted and the new half holds 11% of the norm",
            0.05 < result["share"] < 0.2,
            f"MFCC means and variances run to tens while standardised third and fourth moments "
            f"are O(1), so the appended 26 dimensions carry {result['share']:.4f} of the vector "
            "length. They can move a cosine ranking by about that much, whatever they contain",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
