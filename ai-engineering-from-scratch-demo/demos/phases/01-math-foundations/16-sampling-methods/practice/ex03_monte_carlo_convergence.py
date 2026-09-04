"""Exercise 3 — ∫₀^π sin(x) dx by Monte Carlo; does error scale as O(1/√N)?

    Estimate the integral of sin(x) from 0 to pi using Monte Carlo with 1,000,
    10,000, and 100,000 samples. Compare the error at each level. Verify that the
    error scales as O(1/sqrt(N)).

Reading of the exercise: three single runs cannot verify a *rate* — the error of
one run is a random draw, and three draws could easily come out in the wrong
order. So each sample count is repeated over many independent trials and the
**RMS** error is compared, which is the quantity O(1/√N) actually describes.
Check 5 shows a single-run comparison going the wrong way, which is what the
exercise as literally written would produce some fraction of the time.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "16-sampling-methods"
SEED, TRIALS = 20260904, 200
COUNTS = (1_000, 10_000, 100_000)
EXACT = 2.0                              # ∫₀^π sin x dx = 2


def solve():
    ref = parity.load_reference(PHASE, LESSON, "sampling")
    rng = random.Random(SEED)
    rows = {}
    for n in COUNTS:
        errors = []
        for _ in range(TRIALS):
            random.seed(rng.randrange(1 << 30))
            errors.append(ref.monte_carlo_integral(math.sin, 0.0, math.pi, n) - EXACT)
        rows[n] = {"rms": math.sqrt(sum(e * e for e in errors) / TRIALS),
                   "bias": sum(errors) / TRIALS,
                   "single": abs(errors[0])}
    # the theoretical constant: sd(sin) over [0,π] times (b−a)/√N
    mean_square = 0.5                    # ∫ sin² /π
    variance = mean_square - (2 / math.pi) ** 2
    predicted = {n: math.pi * math.sqrt(variance / n) for n in COUNTS}
    return {"rows": rows, "predicted": predicted}


def _row_text(rows, template) -> str:
    return "; ".join(template(n, rows[n]) for n in COUNTS)


def _unbiased(rows) -> bool:
    return all(abs(rows[n]["bias"]) < rows[n]["rms"] for n in COUNTS)


def _matches_theory(rows, predicted) -> bool:
    return all(0.85 < rows[n]["rms"] / predicted[n] < 1.15 for n in COUNTS)


def verify(result):
    rows, predicted = result["rows"], result["predicted"]
    rms = [rows[n]["rms"] for n in COUNTS]
    ratios = [rms[i] / rms[i + 1] for i in range(len(rms) - 1)]
    singles = [rows[n]["single"] for n in COUNTS]
    out_of_order = not (singles[0] > singles[1] > singles[2])
    return [
        practice.Check(f"{TRIALS} independent trials at each of {len(COUNTS)} sample counts",
                       len(rows) == len(COUNTS),
                       _row_text(rows, lambda n, r: f"N={n:,}: RMS error {r['rms']:.5f}")),
        practice.Check("the estimator is unbiased at every N",
                       _unbiased(rows),
                       "signed bias: "
                       + _row_text(rows, lambda n, r: f"N={n:,}: {r['bias']:+.5f}")),
        practice.Check("10x the samples cuts RMS error by √10 ≈ 3.16",
                       all(2.6 < r < 3.8 for r in ratios),
                       f"measured ratios {[round(r, 3) for r in ratios]} against a "
                       f"predicted {math.sqrt(10):.3f} — this is the O(1/√N) claim, and it "
                       f"needs the RMS over trials to be checkable at all"),
        practice.Check("…and the absolute error matches the closed-form σ(b−a)/√N",
                       _matches_theory(rows, predicted),
                       _row_text(rows, lambda n, r: f"N={n:,}: {r['rms']:.5f} vs "
                                                    f"predicted {predicted[n]:.5f}")),
        practice.Check("a single run per N would not have shown the rate reliably",
                       out_of_order or singles[0] / singles[2] < 10,
                       f"first-trial errors {[round(s, 5) for s in singles]} — one draw "
                       f"each is a random variable, so three of them need not even come "
                       f"out in order, let alone at the right ratio"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
