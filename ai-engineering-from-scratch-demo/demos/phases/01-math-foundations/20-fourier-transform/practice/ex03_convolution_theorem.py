"""Exercise 3 — circular convolution directly and via FFT, then linear.

    **Convolution theorem proof by example.** Create signal x = [1, 2, 3, 4, 0,
    0, 0, 0] and filter h = [1, 1, 1, 0, 0, 0, 0, 0]. Compute their circular
    convolution directly (nested loop). Then compute it via FFT (transform,
    multiply, inverse transform). Verify the results match. Now do linear
    convolution by zero-padding appropriately.

Reading of the exercise: the padding in the given signals is already enough that
circular and linear convolution **coincide** — x has 4 non-zero taps, h has 3,
and 4+3−1 = 6 ≤ 8. So the exercise's final step changes nothing unless the
padding is removed, and check 4 shows the wrap-around it is there to prevent by
running the same convolution at length 4, where the tail folds back onto the head.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "20-fourier-transform"
X = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
H = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
TOL = 1e-10


def circular_direct(x, h):
    n = len(x)
    return [sum(x[k] * h[(i - k) % n] for k in range(n)) for i in range(n)]


def circular_fft(ref, x, h):
    X_, H_ = ref.fft(x), ref.fft(h)
    return [z.real for z in ref.ifft([a * b for a, b in zip(X_, H_)])]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "fourier")
    direct = circular_direct(X, H)
    via_fft = circular_fft(ref, X, H)
    linear = list(ref.convolve_direct(X[:4], H[:3]))
    # length 4: too short for 4+3-1=6 taps, so the tail wraps
    short_x, short_h = X[:4], H[:3] + [0.0]
    wrapped = circular_direct(short_x, short_h)
    return {
        "direct": direct, "fft": via_fft, "linear": linear,
        "gap": max(abs(a - b) for a, b in zip(direct, via_fft)),
        "linear_gap": max(abs(a - b) for a, b in zip(linear, direct[:len(linear)])),
        "wrapped": wrapped,
        "wrap_gap": max(abs(a - b) for a, b in zip(wrapped, linear[:4])),
        "needed": len(X[:4]) + len(H[:3]) - 1,
    }


def verify(result):
    return [
        practice.Check("circular convolution by nested loop and by FFT agree",
                       result["gap"] < TOL,
                       f"worst |Δ| = {result['gap']:.3g}; result "
                       f"{[round(v, 6) for v in result['direct']]}"),
        practice.Check("…which is the convolution theorem: multiply in the frequency domain",
                       result["gap"] < TOL,
                       "transform both, multiply pointwise, transform back — 8 complex "
                       "multiplies plus three transforms, against 64 multiplies for the "
                       "direct loop. At N=2048 exercise 2 measured that trade as 270x"),
        practice.Check(f"linear convolution needs {result['needed']} taps and fits in 8",
                       result["linear_gap"] < TOL,
                       f"convolve_direct on the {4}- and {3}-tap signals gives "
                       f"{[round(v, 4) for v in result['linear']]}, matching the circular "
                       f"result's first {len(result['linear'])} entries to "
                       f"{result['linear_gap']:.3g}"),
        practice.Check("…so the exercise's zero-padding step changes nothing as given",
                       result["linear_gap"] < TOL,
                       f"x has 4 non-zero taps and h has 3, so linear convolution needs "
                       f"{result['needed']} and the signals are already 8 long. The two "
                       f"convolutions coincide, which is what the padding was for"),
        practice.Check("WRAP: at length 4 the tail folds back onto the head",
                       result["wrap_gap"] > 1.0,
                       f"same convolution at N=4 gives "
                       f"{[round(v, 1) for v in result['wrapped']]} against the linear "
                       f"{[round(v, 1) for v in result['linear'][:4]]} — worst difference "
                       f"{result['wrap_gap']:.1f}. Taps 5 and 6 have nowhere to go and are "
                       f"added to positions 1 and 2 instead"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
