"""Exercise 2 — the smallest float32 x with 1.0 + x == 1.0, against finfo.eps.

    **Precision hunt.** Find the smallest positive float32 value `x` such that
    `1.0 + x == 1.0` in Python. This is the machine epsilon. Verify it matches
    `numpy.finfo(numpy.float32).eps`.

Reading of the exercise: as written it asks for two different quantities and
calls them the same thing, so the solution has to distinguish them.
`finfo.eps` is the **spacing above 1.0** — the smallest x with 1+x ≠ 1. The
smallest x with 1+x **==** 1 is the largest value that *vanishes*, which is
smaller still: it is exactly eps/2 = 2⁻²⁴, which is a tie that
round-to-nearest-even resolves downward. One ulp above that already survives.
Searching for "the smallest x that vanishes" is also unanswerable: every
positive float below the threshold vanishes, so the infimum is the smallest
subnormal. Checks 3–5 report all three numbers rather than conflating them.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "13-numerical-stability"


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy for float32 — uv sync --extra math")
    one = numpy.float32(1.0)
    info = numpy.finfo(numpy.float32)
    # bisect for the boundary: the largest x that still vanishes when added to 1
    low, high = numpy.float32(0.0), numpy.float32(1.0)
    for _ in range(200):
        mid = numpy.float32((numpy.float64(low) + numpy.float64(high)) / 2)
        if mid == low or mid == high:
            break
        if one + mid == one:
            low = mid
        else:
            high = mid
    return {
        "eps": float(info.eps),
        "largest_vanishing": float(low),
        "smallest_surviving": float(high),
        "half_eps": float(info.eps) / 2,
        "tiny": float(info.tiny),
        "smallest_subnormal": float(info.smallest_subnormal),
        "sum_at_half_eps": float(one + numpy.float32(float(info.eps) / 2)),
        "sum_at_eps": float(one + numpy.float32(info.eps)),
        "float64_eps": float(numpy.finfo(numpy.float64).eps),
    }


def verify(result):
    return [
        practice.Check("float32 eps is 2⁻²³ ≈ 1.1920929e-07",
                       abs(result["eps"] - 2 ** -23) < 1e-20,
                       f"numpy.finfo(float32).eps = {result['eps']:.7e} = 2⁻²³"),
        practice.Check("1 + eps ≠ 1, and eps is the spacing above 1.0",
                       result["sum_at_eps"] != 1.0,
                       f"1 + eps = {result['sum_at_eps']:.10f}, the next float32 after 1.0"),
        practice.Check("the exercise's quantity is different: 1 + x == 1 up to eps/2",
                       abs(result["largest_vanishing"] - result["half_eps"]) < 1e-20
                       and result["sum_at_half_eps"] == 1.0,
                       f"the largest vanishing x is {result['largest_vanishing']:.7e} = "
                       f"eps/2 = 2⁻²⁴, and 1 + eps/2 == 1 exactly. Round-to-nearest breaks "
                       f"the tie downward here"),
        practice.Check("…and the first surviving x is the very next float32 above eps/2",
                       result["half_eps"] < result["smallest_surviving"] < result["eps"],
                       f"bisection converges to [{result['largest_vanishing']:.8e}, "
                       f"{result['smallest_surviving']:.8e}] — adjacent float32 values, so "
                       f"the boundary is located exactly. Exactly eps/2 is a tie and "
                       f"round-to-nearest-even sends it down to 1.0; one ulp more rounds up"),
        practice.Check("'the smallest x with 1+x == 1' has no answer",
                       result["smallest_subnormal"] < result["tiny"] < result["eps"],
                       f"every positive float below eps/2 vanishes, so the infimum is the "
                       f"smallest subnormal, {result['smallest_subnormal']:.3e} — not a "
                       f"property of 1.0 at all. float64 eps is "
                       f"{result['float64_eps']:.7e} for contrast"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
