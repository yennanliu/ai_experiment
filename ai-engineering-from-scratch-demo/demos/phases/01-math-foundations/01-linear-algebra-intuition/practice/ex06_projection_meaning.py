"""Exercise 6 — project [1, 2, 3] onto [1, 1, 1], and say what that means.

    Project the vector [1, 2, 3] onto [1, 1, 1]. What does the result represent
    geometrically?

Reading of the exercise: as in exercise 5, the number is the easy half. The
geometric claim is that the projection is the *closest point on the line* — and
that is a minimisation, so check 4 tests it as one, by sampling the line and
showing nothing beats it.

Answer: [2, 2, 2]. It is the shadow of [1, 2, 3] on the line through the origin
in the direction [1, 1, 1] — the unique point on that line closest to [1, 2, 3].
Equivalently it is the component of [1, 2, 3] that the direction [1, 1, 1] can
explain; the leftover, [-1, 0, 1], is orthogonal to the line and is exactly what
the direction cannot explain. Since [1,1,1] points along the "all coordinates
equal" diagonal, the projection scale 2 is the *mean* of [1, 2, 3] — projecting
onto the diagonal is how averaging looks geometrically.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "01-linear-algebra-intuition"
V, ONTO = [1, 2, 3], [1, 1, 1]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "vectors")
    v, onto = ref.Vector(V), ref.Vector(ONTO)
    projection = v.project_onto(onto)
    residual = v - projection
    samples = []
    for i in range(-40, 41):
        t = i / 10.0
        point = ref.Vector([t * x for x in ONTO])
        samples.append((t, (v - point).magnitude()))
    best_t, best_distance = min(samples, key=lambda row: row[1])
    return {"projection": projection.components, "residual": residual.components,
            "residual_dot": residual.dot(onto), "distance": residual.magnitude(),
            "best_t": best_t, "best_distance": best_distance,
            "mean": sum(V) / len(V)}


def verify(result):
    projection = result["projection"]
    return [
        practice.Check("projection is [2, 2, 2]",
                       all(abs(x - 2) < 1e-12 for x in projection), f"got {projection}"),
        practice.Check("the residual is orthogonal to [1,1,1]",
                       abs(result["residual_dot"]) < 1e-12,
                       f"residual {result['residual']}, dot = {result['residual_dot']:.3g}"),
        practice.Check("residual is what the direction cannot explain",
                       result["residual"] == [-1, 0, 1], f"got {result['residual']}"),
        practice.Check("no point on the line is closer — it is the closest point",
                       result["best_distance"] >= result["distance"] - 1e-12,
                       f"best of 81 sampled points: t={result['best_t']} at "
                       f"{result['best_distance']:.6f} vs projection {result['distance']:.6f}"),
        practice.Check("scale equals the mean — projecting on the diagonal is averaging",
                       abs(projection[0] - result["mean"]) < 1e-12,
                       f"mean of {V} = {result['mean']}, projection scale = {projection[0]}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
