<!-- generated:start -->
# 01-math-foundations / 01-linear-algebra-intuition

Solutions to all 6 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/01-linear-algebra-intuition/) · upstream spec
`phases/01-math-foundations/01-linear-algebra-intuition/docs/en.md`

```bash
uv run demo practice run 01-linear-algebra-intuition --ex 1
uv run demo explain 01-linear-algebra-intuition --ex 1
uv run pytest demos/phases/01-math-foundations/01-linear-algebra-intuition
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement `Vector.angle_between(other)` that returns the angle in degrees between two vectors | code | T0 | `ex01_angle_between.py` |
| 2 | Create a 2D scaling matrix that doubles the x-coordinate and triples the y-coordinate, then a… | code | T0 | `ex02_scaling_matrix.py` |
| 3 | Given 5 random word-like vectors (dimension 50), find the two most similar using cosine simil… | code | T0 | `ex03_most_similar_pair.py` |
| 4 | Verify that the Gram-Schmidt output is truly orthonormal: check that every pair has dot produ… | code | T0 | `ex04_orthonormal_check.py` |
| 5 | Create a 3x3 matrix with rank 2. Verify using the `rank()` method. Then explain what geometri… | code | T0 | `ex05_rank_two_matrix.py` |
| 6 | Project the vector [1, 2, 3] onto [1, 1, 1]. What does the result represent geometrically? | code | T0 | `ex06_projection_meaning.py` |
<!-- generated:end -->

## Prose answers

Exercises 5 and 6 end in a question, so a green test is only half the answer.

**5 — what do the columns span?** A **plane through the origin** in R³: a
2-dimensional subspace. Rank 2 means exactly two columns are linearly
independent; the third, `c₃ = 2c₁ − c₂`, is a combination of them and adds no
new direction. A span is closed under scaling and addition, so it must contain
the origin — it is a plane, not a sheet floating anywhere in space. Its normal is
`c₁ × c₂ = [-1, -1, 1]`, and every vector off the plane, the normal included, is
unreachable by any combination of the columns. That unreachability is what the
solution checks; "rank is 2" alone would not distinguish a plane from a line.

**6 — what does the projection represent?** `[2, 2, 2]`: the shadow of `[1,2,3]`
on the line through the origin in direction `[1,1,1]`, and the unique closest
point on that line. Equivalently it is the part of `[1,2,3]` that the direction
can explain; the leftover `[-1, 0, 1]` is orthogonal to the line and is exactly
what it cannot. Because `[1,1,1]` is the "all coordinates equal" diagonal, the
projection scale is the **mean** of `[1,2,3]` — averaging is what projecting onto
the diagonal looks like geometrically.

## Two findings about the lesson's code

Both came from running it, not from reading it — the argument for execution as
the gate (`DESIGN §6`).

1. **`angle_between` loses all precision near 0°.** It computes
   `degrees(acos(clamp(cosθ)))`. For `[1,0,0]` against `[1,1e-8,0]` the true
   angle is `5.7296e-7°`; the lesson returns `0.0`, an error of 100%. `acos` is
   ill-conditioned as θ→0 because cosθ→1 quadratically. Exercise 1's answer uses
   Kahan's `2·atan2(‖â−b̂‖, ‖â+b̂‖)`, which is exact there and agrees with the
   lesson to 7.1e-15° everywhere else — so the exercise's "implement it" is worth
   doing even though the method already exists.

2. **`gram_schmidt` silently drops vectors.** It skips any residual with
   `magnitude() < 1e-10`, with no error and no warning. Given a Läuchli basis —
   `[[1,1,1], [1e-10,0,0], [0,1e-10,0]]`, whose exact determinant is 1e-20 and so
   is genuinely independent — it returns **one** vector for three. `is_independent`
   shares the same threshold and also calls them dependent, so the two agree with
   each other but not with the algebra. This is why exercise 4 counts the returned
   vectors: "every pair is orthogonal" is vacuously true of a 1-vector basis.

Neither is a bug to fix upstream by this repo's rules — a solution answers the
exercise as posed and files findings rather than silently improving the lesson
(`DESIGN §7`).
