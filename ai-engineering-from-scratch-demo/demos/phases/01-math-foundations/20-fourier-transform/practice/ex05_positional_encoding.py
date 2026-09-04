"""Exercise 5 — does the positional-encoding dot product depend only on |p1−p2|?

    **Positional encoding analysis.** Generate the sinusoidal positional
    encodings for d_model = 128 and max_pos = 512. For each pair of positions
    (p1, p2), compute the dot product of their encodings. Show that the dot
    product depends only on |p1 - p2|, not on the absolute positions. What
    happens to the dot product as the distance increases?

Reading of the exercise: the claim is **exactly true**, and for a reason worth
stating — each (sin, cos) pair contributes cos(ω(p₁−p₂)) by the angle-difference
identity, so absolute position cancels term by term. Check 2 measures the spread
across all same-distance pairs rather than spot-checking a few.

The second question needs a dense sweep to answer honestly. Sampled at a handful
of distances (0, 1, 2, 5, 10, 50, 100, 250, 500) the dot product looks
monotonically decreasing. Sampled every 8 positions, **28 of 63 steps rise** — it
falls sharply over the first few positions and then oscillates. And it never
approaches zero: the floor is about 11.5, roughly 18% of the self dot product of
64, because the lowest-frequency components have periods near 57,000 positions and
barely move across 512.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "20-fourier-transform"
D_MODEL, MAX_POS = 128, 512
DISTANCES = (0, 1, 2, 5, 10, 50, 100, 250, 500)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _at_distance(encodings, distance):
    """Every 7th pair at this separation — enough to expose any p-dependence."""
    values = [dot(encodings[p], encodings[p + distance])
              for p in range(0, MAX_POS - distance, 7)]
    return {"mean": sum(values) / len(values),
            "spread": max(values) - min(values), "n": len(values)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "fourier")
    encodings = [ref.positional_encoding(p, D_MODEL) for p in range(MAX_POS)]
    rows = {d: _at_distance(encodings, d) for d in DISTANCES}
    sparse = [rows[d]["mean"] for d in DISTANCES]
    # a dense sweep, because the sparse one hides the oscillation entirely
    dense = [dot(encodings[0], encodings[d]) for d in range(0, MAX_POS, 8)]
    rises = sum(1 for a, b in zip(dense, dense[1:]) if b > a)
    return {"rows": rows, "self_dot": rows[0]["mean"],
            "sparse_monotone": all(a > b for a, b in zip(sparse, sparse[1:])),
            "rises": rises, "steps": len(dense) - 1,
            "floor": min(dense), "late_max": max(dense[8:]),
            "dense_tail": dense[-4:]}


def verify(result):
    rows = result["rows"]
    worst_spread = max(rows[d]["spread"] for d in DISTANCES)
    return [
        practice.Check(f"self dot product is d_model/2 = {D_MODEL // 2}",
                       abs(result["self_dot"] - D_MODEL / 2) < 1e-9,
                       f"{result['self_dot']:.9f} — each of the {D_MODEL // 2} (sin, cos) "
                       f"pairs contributes sin² + cos² = 1"),
        practice.Check("ANSWER: the dot product depends only on the distance — exactly",
                       worst_spread < 1e-9,
                       f"across {sum(rows[d]['n'] for d in DISTANCES)} pairs at "
                       f"{len(DISTANCES)} distances, the worst spread within a distance is "
                       f"{worst_spread:.3g}"),
        practice.Check("…because each (sin, cos) pair contributes cos(ω·Δ) identically",
                       worst_spread < 1e-9,
                       "sin(ωp₁)sin(ωp₂) + cos(ωp₁)cos(ωp₂) = cos(ω(p₁−p₂)) — the "
                       "angle-difference identity, applied term by term. Absolute position "
                       "cancels algebraically, not approximately"),
        practice.Check("ANSWER: it decays sharply, then oscillates — it is not monotone",
                       result["sparse_monotone"] and result["rises"] > 10,
                       f"the sparse distances {list(DISTANCES)} give "
                       + ", ".join(f"{rows[d]['mean']:.1f}" for d in DISTANCES)
                       + f", which looks monotone. Sampled every 8 positions instead, "
                         f"{result['rises']} of {result['steps']} steps *rise*. Spot-checking "
                         f"a handful of distances hides the behaviour entirely"),
        practice.Check("…and it never approaches zero: the floor is ~18% of the self value",
                       result["floor"] > 0.1 * result["self_dot"],
                       f"minimum over the sweep {result['floor']:.2f} against a self dot "
                       f"product of {result['self_dot']:.0f}, and it still reaches "
                       f"{result['late_max']:.1f} after distance 64. The lowest frequency is "
                       f"1/10000^(126/128) ≈ 1.1e-4 — a period near 57,000 positions — so "
                       f"across 512 those components hardly move and keep contributing near "
                       f"+1 each. Distant positions are not close to orthogonal"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
