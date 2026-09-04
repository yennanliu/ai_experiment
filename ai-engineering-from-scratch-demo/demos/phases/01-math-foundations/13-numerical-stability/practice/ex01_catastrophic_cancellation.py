"""Exercise 1 — variance of [1e6, 1e6+1, 1e6+2] in float32, naive vs Welford.

    **Catastrophic cancellation.** Compute the variance of [1000000.0,
    1000001.0, 1000002.0] using the naive formula `E[x^2] - E[x]^2` in float32.
    Then compute it using Welford's online algorithm. Compare the errors against
    the true variance (0.6667).

Reading of the exercise: float32 is named specifically, and forcing it matters —
`numpy.float32` on every intermediate, not just the input array. The true
population variance is exactly 2/3, so checks use the fraction rather than the
exercise's rounded 0.6667.

The comparison comes out more strongly than the exercise implies. float32 does
not merely lose accuracy, it returns **−65536**, a value no variance can take.
And float64 does not rescue the naive formula: it still errs by 4.1e-05, about 11
of 16 significant digits. Welford in *float32* beats naive in *float64* by
2000x — the algorithm matters more than the precision.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "13-numerical-stability"
VALUES = [1000000.0, 1000001.0, 1000002.0]
TRUE_VARIANCE = 2.0 / 3.0


def naive_float32(numpy, values):
    """E[x²] − E[x]², every operation forced through float32."""
    data = numpy.array(values, dtype=numpy.float32)
    mean = data.mean(dtype=numpy.float32)
    mean_square = (data * data).mean(dtype=numpy.float32)
    return float(numpy.float32(mean_square) - numpy.float32(mean) * numpy.float32(mean))


def welford_float32(numpy, values):
    """Online update; each running quantity kept in float32."""
    count, mean, m2 = 0, numpy.float32(0.0), numpy.float32(0.0)
    for raw in values:
        value = numpy.float32(raw)
        count += 1
        delta = numpy.float32(value - mean)
        mean = numpy.float32(mean + delta / numpy.float32(count))
        m2 = numpy.float32(m2 + delta * numpy.float32(value - mean))
    return float(m2 / numpy.float32(count))


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy for float32 — uv sync --extra math")
    ref = parity.load_reference(PHASE, LESSON, "numerical")
    return {
        "naive32": naive_float32(numpy, VALUES),
        "welford32": welford_float32(numpy, VALUES),
        "naive64": ref.variance_naive(VALUES),
        "welford64": ref.welford_variance(VALUES),
        "kahan": ref.kahan_sum(VALUES),
        "exact_sum": sum(VALUES),
    }


def verify(result):
    naive_err = abs(result["naive32"] - TRUE_VARIANCE)
    welford_err = abs(result["welford32"] - TRUE_VARIANCE)
    naive64_err = abs(result["naive64"] - TRUE_VARIANCE)
    return [
        practice.Check("Welford in float32 gets the variance right",
                       welford_err < 1e-5,
                       f"{result['welford32']:.9f} vs the exact 2/3 = "
                       f"{TRUE_VARIANCE:.9f}, error {welford_err:.3g}"),
        practice.Check("the naive formula in float32 does not",
                       naive_err > 0.1,
                       f"{result['naive32']:.6f}, error {naive_err:.4g} — "
                       f"{naive_err / max(welford_err, 1e-12):.0g}x worse than Welford"),
        practice.Check("…and it is not merely inaccurate, it is not even a variance",
                       result["naive32"] < 0 or result["naive32"] == 0.0,
                       f"E[x²] − E[x]² = {result['naive32']} — the two terms agree to ~13 "
                       f"significant digits and float32 holds ~7, so the subtraction "
                       f"returns a value a variance can never take"),
        practice.Check("FINDING: Welford in float32 beats the naive formula in float64",
                       welford_err < naive64_err / 100,
                       f"Welford/float32 error {welford_err:.3g} against naive/float64 "
                       f"{naive64_err:.3g} — {naive64_err / welford_err:.0f}x. The "
                       f"algorithm matters more than the precision: doubling the mantissa "
                       f"does not buy back what the subtraction destroys"),
        practice.Check("…and float64 does not rescue the naive formula either",
                       1e-6 < naive64_err < 1e-3,
                       f"naive in float64 gives {result['naive64']:.9f}, error "
                       f"{naive64_err:.3g} — about 11 of 16 significant digits gone. It "
                       f"fails less visibly than float32, not less"),
        practice.Check("Kahan summation is exact on the same values",
                       result["kahan"] == result["exact_sum"],
                       f"kahan_sum = {result['kahan']:.1f} = plain sum; these values need "
                       f"compensation for the *variance*, not the sum, because the loss "
                       f"comes from squaring them"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
