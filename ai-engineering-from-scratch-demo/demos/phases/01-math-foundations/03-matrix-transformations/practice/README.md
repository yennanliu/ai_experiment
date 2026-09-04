<!-- generated:start -->
# 01-math-foundations / 03-matrix-transformations

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/03-matrix-transformations/) · upstream spec
`phases/01-math-foundations/03-matrix-transformations/docs/en.md`

```bash
uv run demo practice run 03-matrix-transformations --ex 1
uv run demo explain 03-matrix-transformations --ex 1
uv run pytest demos/phases/01-math-foundations/03-matrix-transformations
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Apply rotation, scaling, and shearing to a unit square (corners at [0,0], [1,0], [1,1], [0,1]… | code | T0 | `ex01_unit_square_transforms.py` |
| 2 | Find the eigenvalues of the matrix [[4, 2], [1, 3]] by hand using the characteristic equation… | code | T0 | `ex02_eigenvalues_by_hand.py` |
| 3 | Create a composition of three transformations (rotate 30 degrees, scale by [1.5, 0.8], shear… | code | T0 | `ex03_composed_transformations.py` |
<!-- generated:end -->

## Notes

**Why all three maps have determinant 1.** `scaling_2d(2, 0.5)` was chosen so
that rotation, scaling and shearing all preserve area. Only rotation preserves
*distance*. Determinant 1 means "volume-preserving", which is strictly weaker
than "rigid" — a shear turns the unit square into a parallelogram of the same
area, and a 2×/0.5× scaling turns it into a rectangle of the same area. Being an
isometry is the stronger condition, and it is what exercise 1 actually tests.

**Why exercise 3 checks composition order separately.** `det(ABC) =
det(A)det(B)det(C)` holds for every ordering, so it cannot detect a
multiplication written the wrong way round. The solution composes the same three
matrices in reverse and shows the determinant is unchanged while the points move
by up to 0.94 — which is why the order claim needs its own check.
