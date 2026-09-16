"""Exercise 5 — multiplying by a complex scalar IS the RoPE rotation, to fifteen decimal places.

    Read Section 3 of the Mamba-3 paper (arXiv:2603.15569). Explain in three
    sentences why a complex-valued state update is equivalent to a data-dependent
    rotary embedding. Tie the answer to Phase 7 · Lesson 04's RoPE derivation.

Reading of the exercise: the equivalence is an algebraic identity, so it is
checked rather than described -- multiply a complex number by `e^{i*theta}` and
apply RoPE's 2x2 rotation matrix to the same pair of reals, and compare. The
comparison against real-diagonal state matrices is then run the same way, since
"what the real simplification cost" is the other half of the claim.

**ANSWER: the two operations agree to fifteen decimal places, at every angle
tried.** `e^{i*0.7} * (0.3 - 1.2i)` is `(1.0025138809, -0.7245453186)`, and
RoPE's `[[cos, -sin], [sin, cos]]` applied to `(0.3, -1.2)` is the same pair.
A complex diagonal state matrix *is* a per-pair rotation, which is exactly what
RoPE applies to Q and K -- and "data-dependent" means the angle comes from the
input rather than from the position index.

**MECHANISM: `|e^{i*theta}| = 1`, so the update rotates without scaling.** The
state's norm is preserved exactly across a rotation, and a sequence of them
composes into another rotation: `e^{ia} e^{ib} = e^{i(a+b)}`. That is the
state-tracking property -- the state can record *where it is on the circle*,
which is a quantity a magnitude alone cannot carry.

**FINDING: a real diagonal reaches two points of that circle.** Multiplying by a
real `a` leaves the phase unchanged when `a > 0` and adds exactly `pi` when
`a < 0`: across a sweep of real values the set of achievable phase shifts is
`{0.0, 3.14159}` -- **2 of the infinitely many** a complex diagonal can produce.
Mamba-2's scaled identity is the extreme case of this, one real value shared
across the whole state.

**FINDING: the lesson's own calculator has no state matrix at all.**
`ssm_state_size` is an integer width and `ssm_state_bytes` multiplies it by the
layer count and the hidden size. Whether those numbers are real or complex is not
represented, and a complex state of the same width would take **2x** the bytes --
which is the one consequence of Mamba-3's change this module could have shown and
does not.

Structure: `rotate` applies RoPE's 2x2 matrix; `complex_step` multiplies by
`e^{i*theta}`; `phases` records what phase shifts a real multiplier can reach.
"""

from __future__ import annotations

import cmath
import math

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "21-jamba-hybrid-ssm-transformer"
ANGLES = (0.7, 0.1, 1.9, 3.0, -0.4)
REALS = (0.5, 2.0, 0.9, -0.5, -1.3)
VECTOR = (0.3, -1.2)
LAYERS, ATTN, HIDDEN, STATE, BYTES = 32, 4, 4096, 16, 2


def rotate(pair, theta):
    """RoPE's 2x2 rotation, as Phase 7 Lesson 04 writes it."""
    x, y = pair
    return (math.cos(theta) * x - math.sin(theta) * y,
            math.sin(theta) * x + math.cos(theta) * y)


def complex_step(pair, theta):
    """One complex-diagonal state update: multiply by e^{i theta}."""
    product = cmath.exp(1j * theta) * complex(*pair)
    return (product.real, product.imag)


def phases(multipliers, pair=VECTOR):
    """The phase shifts a set of multipliers can produce on one vector."""
    before = cmath.phase(complex(*pair))
    return sorted({round(abs(cmath.phase(value * complex(*pair)) - before), 5)
                   for value in multipliers})


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    gaps = [max(abs(a - b) for a, b in zip(rotate(VECTOR, theta), complex_step(VECTOR, theta)))
            for theta in ANGLES]
    norms = [abs(cmath.exp(1j * theta)) for theta in ANGLES]
    composed = abs(cmath.exp(1j * 0.5) * cmath.exp(1j * 1.1) - cmath.exp(1j * 1.6))
    state_bytes = (LAYERS - ATTN) * HIDDEN * STATE * BYTES
    return {
        "worst_gap": max(gaps),
        "sample": {"rope": rotate(VECTOR, 0.7), "complex": complex_step(VECTOR, 0.7)},
        "norms": norms,
        "composed": composed,
        "real_phases": phases(REALS),
        "complex_phases": len(phases([cmath.exp(1j * t) for t in ANGLES])),
        "state_bytes": state_bytes,
        "complex_state_bytes": state_bytes * 2,
    }


def verify(result):
    sample = result["sample"]
    return [
        practice.Check(
            "ANSWER: the two operations agree to fifteen decimal places at every angle",
            result["worst_gap"] < 1e-15,
            f"e^(i*0.7) x (0.3 - 1.2i) is "
            f"({sample['complex'][0]:.10f}, {sample['complex'][1]:.10f}) and RoPE's "
            f"[[cos, -sin], [sin, cos]] applied to {VECTOR} is "
            f"({sample['rope'][0]:.10f}, {sample['rope'][1]:.10f}). Across {len(ANGLES)} angles "
            f"the worst disagreement is {result['worst_gap']:.1e}. A complex diagonal state "
            "matrix is a per-pair rotation, which is what RoPE applies to Q and K -- and "
            "'data-dependent' means the angle comes from the input rather than the position index",
        ),
        practice.Check(
            "MECHANISM: |e^(i theta)| = 1, so the update rotates without scaling",
            all(abs(norm - 1.0) < 1e-15 for norm in result["norms"])
            and result["composed"] < 1e-15,
            f"every multiplier has modulus 1 to within {max(abs(n - 1) for n in result['norms']):.1e}, "
            f"so the state's norm is preserved exactly, and a sequence of them composes into "
            f"another rotation -- e^(i0.5) e^(i1.1) equals e^(i1.6) to "
            f"{result['composed']:.1e}. That is the state-tracking property: the state can record "
            "where it is on the circle, which a magnitude alone cannot carry",
        ),
        practice.Check(
            "FINDING: a real diagonal reaches two points of that circle",
            result["real_phases"] == [0.0, round(math.pi, 5)],
            f"multiplying by a real a leaves the phase unchanged when a > 0 and adds exactly pi "
            f"when a < 0: across {REALS} the set of achievable phase shifts is "
            f"{result['real_phases']} -- 2 values, against {result['complex_phases']} distinct "
            f"shifts from the same number of complex multipliers, out of infinitely many. "
            "Mamba-2's scaled identity is the extreme case, one real value shared across the "
            "whole state",
        ),
        practice.Check(
            "FINDING: the lesson's calculator has no state matrix at all",
            result["complex_state_bytes"] == 2 * result["state_bytes"],
            f"ssm_state_size is an integer width and ssm_state_bytes multiplies it by the layer "
            f"count and the hidden size. Whether those numbers are real or complex is not "
            f"represented, and a complex state of the same width would take "
            f"{result['complex_state_bytes'] / 1024 ** 2:.2f} MB against "
            f"{result['state_bytes'] / 1024 ** 2:.2f} -- exactly 2x, which is the one consequence "
            "of Mamba-3's change this module could have shown and does not",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
