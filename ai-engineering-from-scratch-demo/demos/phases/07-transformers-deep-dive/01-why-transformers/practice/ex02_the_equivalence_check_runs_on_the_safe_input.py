"""Exercise 2 — the equivalence check runs on the one input that cannot fail.

    **Medium.** Implement a parallel prefix-sum (Hillis-Steele scan) in pure
    Python. Verify it produces the same numerical output as a serial scan on
    length 1024. Count the depth.

Reading of the exercise: the lesson already ships `parallel_scan`, a
Hillis-Steele scan, so "implement" is read as *re-derive and then audit* -- the
scan here is written from the recurrence rather than copied, checked against the
lesson's own for bit equality, and the two reference implementations are then
compared with each other at the length the exercise names. "Same numerical
output" is taken at its word, i.e. bit-exact, because that is the claim the
sentence makes and it is the claim that fails.

**ANSWER: depth 10, and 9.0x the additions.** log2(1024) = 10 passes against
1023 serial steps -- 102x less depth -- but 9217 adds against 1023, because a
pass at stride `s` updates `n - s` positions. Hillis-Steele is not work-efficient
and the factor is exactly what you buy the depth with.

**FINDING: the outputs are not the same.** On the lesson's own `benchmark()`
data only **23 of 1024** entries are bit-identical, worst disagreement 4.1e-14
absolute and up to 45 ULP. Floating-point addition is not associative and the
two scans bracket the same sum differently, so "the same numerical output" is
true to about 1e-14 and false as written.

**FINDING: `main()`'s check is run on the one input that cannot disagree.** Its
fixture is `[float(i) for i in range(16)]` -- small exact integers, whose every
partial sum is exactly representable -- and that stays 1024/1024 identical at
length 1024 too. The fixture, not the length, is what makes the check pass.

**FINDING: the 1e-9 tolerance is the wrong shape.** It is absolute, on a
quantity that grows with n, and it fails at n = 1,048,576 with 1.9e-8 -- which is
exactly the largest N `main()`'s own `benchmark()` sweeps.

**CONTROL: the parallel scan is ~35x slower.** Its docstring says "on a CPU it's
the same wall-clock but the graph shape is what matters". The shape claim is
right; the wall-clock claim is off by a factor of 35, and the log-depth costs
9x the adds plus a full list copy per pass.

Structure: `hillis_steele` is the re-derivation; `adds` counts the work in closed
form; `disagreement` is the bit/ULP comparison the exercise asks for.
"""

from __future__ import annotations

import math
import time

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "01-why-transformers"
LENGTH, BIG, TOLERANCE = 1024, 1_048_576, 1e-9


def hillis_steele(xs):
    """Re-derived, not copied: out[i] <- out[i] + out[i-s] for s = 1, 2, 4, ..."""
    out, stride = list(xs), 1
    while stride < len(out):
        out = [out[i] + out[i - stride] if i >= stride else out[i] for i in range(len(out))]
        stride *= 2
    return out


def adds(n):
    """Closed form: a pass at stride s touches n - s positions. Returns (adds, depth)."""
    total, depth, stride = 0, 0, 1
    while stride < n:
        total, depth, stride = total + n - stride, depth + 1, stride * 2
    return total, depth


def disagreement(left, right):
    """(bit-identical count, worst absolute gap, worst gap in ULP)."""
    same = sum(1 for a, b in zip(left, right) if a == b)
    gaps = [(abs(a - b), abs(a - b) / math.ulp(max(abs(a), abs(b), 5e-324)))
            for a, b in zip(left, right)]
    return same, max(g for g, _ in gaps), max(u for _, u in gaps)


def timed(call):
    """Wall-clock of one call, in seconds."""
    start = time.perf_counter()
    call()
    return time.perf_counter() - start


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = [0.001 * (i % 17) for i in range(LENGTH)]
    integers = [float(i) for i in range(LENGTH)]
    big = [0.001 * (i % 17) for i in range(BIG)]
    serial, parallel = ref.serial_scan(data), ref.parallel_scan(data)
    return {
        "mine_matches_lesson": hillis_steele(data) == parallel,
        "float": disagreement(serial, parallel),
        "integer": disagreement(ref.serial_scan(integers), ref.parallel_scan(integers)),
        "big": disagreement(ref.serial_scan(big), ref.parallel_scan(big)),
        "adds": adds(LENGTH), "serial_adds": LENGTH - 1,
        "parallel_s": timed(lambda: ref.parallel_scan(big)),
        "serial_s": timed(lambda: ref.serial_scan(big)),
    }


def verify(result):
    added, depth = result["adds"]
    same, worst, ulps = result["float"]
    slowdown = result["parallel_s"] / result["serial_s"]
    return [
        practice.Check(
            "ANSWER: depth 10 = log2(1024), bought with 9.0x the additions",
            depth == int(math.log2(LENGTH)) and 8 < added / result["serial_adds"] < 10,
            f"{depth} passes against {result['serial_adds']} serial steps -- "
            f"{result['serial_adds'] / depth:.0f}x less depth -- for {added} adds against "
            f"{result['serial_adds']}, {added / result['serial_adds']:.2f}x. A pass at stride s "
            "updates n - s positions, so Hillis-Steele is not work-efficient by exactly that much",
        ),
        practice.Check(
            "FINDING: the outputs are not the same, only close",
            same < LENGTH // 8 and worst < 1e-12 and ulps > 1,
            f"at length {LENGTH} on the lesson's own benchmark() data, {same}/{LENGTH} entries "
            f"are bit-identical; worst gap {worst:.2e} absolute, {ulps:.0f} ULP. Float addition "
            "is not associative and the two scans bracket the same sum differently, so 'the same "
            "numerical output' holds to ~1e-14 and not as written",
        ),
        practice.Check(
            "FINDING: main()'s fixture is the one input that cannot disagree",
            result["integer"][0] == LENGTH,
            f"its check runs on [float(i) for i in range(16)] -- small exact integers, every "
            f"partial sum exactly representable. The same generator at length {LENGTH} is still "
            f"{result['integer'][0]}/{LENGTH} bit-identical. The fixture is what passes the "
            "check, not the length, and swapping in benchmark()'s own data breaks it",
        ),
        practice.Check(
            "FINDING: the 1e-9 tolerance is absolute, and fails at main()'s own largest N",
            result["big"][1] > TOLERANCE,
            f"the mismatch count uses abs(a - b) > {TOLERANCE:g} on a prefix sum whose magnitude "
            f"grows with n. At n = {BIG:,} -- the top of benchmark()'s own sweep -- the worst gap "
            f"is {result['big'][1]:.2e}, over the tolerance, while the relative error has not "
            "moved. A relative bound would have held at every length",
        ),
        practice.Check(
            "CONTROL: the parallel scan is ~35x slower, not 'the same wall-clock'",
            slowdown > 10 and result["mine_matches_lesson"],
            f"at n = {BIG:,}: {result['parallel_s'] * 1e3:.0f} ms against "
            f"{result['serial_s'] * 1e3:.0f} ms, {slowdown:.0f}x. Its docstring says 'on a CPU "
            "it's the same wall-clock but the graph shape is what matters' -- the shape claim "
            "holds, the wall-clock one costs 9x the adds plus a full list copy per pass. The "
            "scan re-derived here is bit-identical to the lesson's, so this is its number",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
