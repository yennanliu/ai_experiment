<!-- generated:start -->
# 01-math-foundations / 19-complex-numbers

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/19-complex-numbers/) · upstream spec
`phases/01-math-foundations/19-complex-numbers/docs/en.md`

```bash
uv run demo practice run 19-complex-numbers --ex 1
uv run demo explain 19-complex-numbers --ex 1
uv run pytest demos/phases/01-math-foundations/19-complex-numbers
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Complex arithmetic by hand. Compute (2 + 3i) * (4 - i) and verify with the code. Then compute… | code | T0 | `ex01_complex_arithmetic.py` |
| 2 | Rotation sequence. Start with the point (1, 0). Multiply by e^(i*pi/6) twelve times. Verify t… | code | T0 | `ex02_rotation_sequence.py` |
| 3 | DFT of a known signal. Create a signal that is the sum of sin(2*pi*3*t) and 0.5*sin(2*pi*7*t)… | code | T0 | `ex03_dft_known_signal.py` |
| 4 | Roots of unity visualization. Compute the 8th roots of unity. Verify that they sum to zero. V… | code | T0 | `ex04_roots_of_unity.py` |
| 5 | Rotation matrix equivalence. For 10 random angles and 10 random points, verify that complex m… | code | T0 | `ex05_rotation_matrix_equivalence.py` |
<!-- generated:end -->

## Answers

**1 — (2+3i)(4−i) = 11 + 10i** and **(5+2i)/(1−3i) = −0.1 + 1.7i.** Both derived
by hand (8 + 10i − 3i², and multiply by the conjugate over |1−3i|² = 10), then
confirmed against the lesson's `Complex` *and* Python's built-in `complex` — a
second oracle written by someone else.

"Draw both on the complex plane and check that multiplication rotated and scaled"
is not assertable as drawing, but the claim behind it is: |ab| = |a||b| (3.606 ×
4.123 = 14.866) and arg(ab) = arg(a) + arg(b) mod 2π, both to 1e-12. Plus
z·z̄ = |z|², which is the identity the division relied on.

**2 — it returns to (1,0), but only to within 1.1e-15.** Twelve multiplications
each round, so exact equality would fail; the measured drift is reported instead.
Regularity needs two properties and both are checked: every vertex at radius 1
(worst deviation exactly 0) and every consecutive gap exactly π/6 (worst 5.6e-16).
The 12 vertices are also confirmed **distinct**, so the path is a full 12-gon
rather than a shorter polygon retraced — π/6 generates the whole cyclic group of
order 12.

**3 — peaks at bins 3 and 7, ratio exactly 2.0.** Every other bin is below
3.6e-14, and the round-trip through `idft` holds to 7.6e-15.

The ratio is exact only because 3 and 7 are **integer** numbers of cycles over
the 32-point window, so each sinusoid lands entirely in one bin. Change that and
the claim dissolves: a **3.5**-cycle sinusoid spreads above 10% of its peak across
**8 bins**. That is spectral leakage, and it is why windowing exists.

Also worth noting: each peak has a mirror at N−k, since the input is real. Bins 29
and 25 carry equal magnitude, so half the spectrum is redundant — which a
one-sided reading would miss.

**4 — they sum to zero** (residual 6.0e-16), and multiplying any root by the
primitive gives the next across **all 8 pairs including the wrap-around**. That
last case is the only interesting one: a loop over N−1 pairs would miss it and
still pass.

Two properties the exercise does not ask for but that pin down the name: every
root to the 8th power returns 1 (worst 1.8e-15), and all 8 phases are distinct —
which is what makes e^(2πi/8) *primitive* rather than merely a root. A
non-primitive 8th root such as −1 cycles through 2 of them and never reaches the
other 6.

**5 — maximum numerical difference: exactly 0.0** across all 100 angle-point
pairs. Not "to machine precision" — bit-identical, because `cos·x − sin·y` and
`sin·x + cos·y` *are* the real and imaginary parts of (x+iy)(cos+i·sin): the same
four products in the same order. The agreement is an identity, so it is not
evidence of anything, which is why the remaining checks go elsewhere.

The real content is structural: composing two rotations equals multiplying the two
rotors (gap 6.7e-16), so the unit complex numbers under multiplication **are** the
2D rotation group. And they commute, since both orderings reduce to e^(i(a+b)) —
which is exactly where the analogy stops. 3D rotations do not commute, so no
single complex number can represent one; that needs quaternions.
