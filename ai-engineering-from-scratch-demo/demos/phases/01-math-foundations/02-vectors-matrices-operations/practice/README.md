<!-- generated:start -->
# 01-math-foundations / 02-vectors-matrices-operations

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/02-vectors-matrices-operations/) · upstream spec
`phases/01-math-foundations/02-vectors-matrices-operations/docs/en.md`

```bash
uv run demo practice run 02-vectors-matrices-operations --ex 1
uv run demo explain 02-vectors-matrices-operations --ex 1
uv run pytest demos/phases/01-math-foundations/02-vectors-matrices-operations
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Verify the inverse. Multiply `A @ A.inverse_2x2()` and confirm you get the identity matrix. T… | code | T0 | `ex01_verify_inverse.py` |
| 2 | Implement 3x3 inverse. Extend the Matrix class to compute inverses for 3x3 matrices using the… | code | T0 | `ex02_implement_3x3_inverse.py` |
| 3 | Build a two-layer network. Using only your Matrix class (no NumPy), create a two-layer neural… | code | T0 | `ex03_build_two_layer_network.py` |
<!-- generated:end -->

## Answers to the questions the exercises ask

**1 — what happens when the determinant is zero?** `inverse_2x2` raises
`ValueError: Matrix is singular, no inverse exists`. That is the right behaviour
and worth stating positively: the alternative — returning `inf`-filled garbage
that silently poisons everything downstream — is what most from-scratch
implementations do. The solution asserts the raise rather than assuming it.

The near-singular case is the more interesting one. `[[1,1],[1,1+1e-8]]` has
determinant `1e-8`, passes the singularity guard, and still inverts to full
double precision here — because its entries are exactly representable. Ill
*conditioning* and exact singularity are different problems, and only the second
one has a guard.

**2 — why the adjugate, when elimination is better?** The exercise prescribes
the adjugate method, and this repo answers the exercise as posed (`DESIGN §7`).
It is worth knowing the cost: on the Hilbert 3×3, the worst residual is `1.4e-14`
against `1e-16`-scale for the well-conditioned cases, and the gap against
`np.linalg.inv` — which uses LU with partial pivoting — reaches `1.25e-12`.
Cofactor expansion is `O(n!)` and loses accuracy as conditioning worsens; at 3×3
both costs are still invisible, which is exactly why 3×3 is a safe place to teach
it.

**3 — no NumPy.** Enforced mechanically, not by intention: the manifest declares
`deps_group: none`, so `scripts/check_deps.py` fails the build if a numpy import
ever appears in that file.
