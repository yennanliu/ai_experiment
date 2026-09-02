"""Phase 01 / Lesson 02 -- the lesson's Matrix class vs numpy.

The lesson builds `Vector` and `Matrix` from nothing but lists and loops, and
the docs assert that this is what numpy does under the hood. This demo checks
that assertion element by element: same inputs into both implementations, and
the deviation printed rather than asserted away.

Run:  uv run demo run phases/01-math-foundations/02-vectors-matrices-operations
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from harness.explain import explain          # noqa: E402
from harness.parity import assert_close, load_reference, report  # noqa: E402

LESSON = "phases/01-math-foundations/02-vectors-matrices-operations/code/matrices.py"
SEED = 42
ATOL = 1e-12


def as_numpy(matrix):
    """The lesson's Matrix stores a nested list, which numpy takes directly."""
    import numpy as np

    return np.array(matrix.data, dtype=float)


def main() -> int:
    import numpy as np

    ref = load_reference(LESSON)
    rng = random.Random(SEED)
    checks = []

    # --- the operations the lesson hand-rolls -----------------------------
    A = ref.Matrix([[rng.uniform(-2, 2) for _ in range(4)] for _ in range(3)])
    B = ref.Matrix([[rng.uniform(-2, 2) for _ in range(5)] for _ in range(4)])
    npA, npB = as_numpy(A), as_numpy(B)

    checks.append(assert_close(A @ B, npA @ npB, label="matmul (3x4 @ 4x5)", atol=ATOL))
    checks.append(assert_close(A.T, npA.T, label="transpose", atol=ATOL))
    checks.append(
        assert_close(A.scalar_multiply(2.5), npA * 2.5, label="scalar multiply", atol=ATOL)
    )
    checks.append(
        assert_close(A.element_wise_multiply(A), npA * npA,
                     label="element-wise multiply", atol=ATOL)
    )

    # Broadcasting a bias row is the lesson's whole point about why a neural
    # network layer can add one bias vector to a batch of activations.
    activations = ref.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bias = ref.Matrix([[10.0, 20.0, 30.0]])
    checks.append(
        assert_close(
            activations + bias,
            as_numpy(activations) + as_numpy(bias),
            label="broadcast bias row",
            atol=ATOL,
        )
    )

    # --- determinant and inverse ------------------------------------------
    square = ref.Matrix([[4.0, 7.0], [2.0, 6.0]])
    npSquare = as_numpy(square)
    checks.append(
        assert_close(square.determinant(), float(np.linalg.det(npSquare)),
                     label="determinant (2x2)", atol=1e-10)
    )
    checks.append(
        assert_close(square.inverse_2x2(), np.linalg.inv(npSquare),
                     label="inverse (2x2)", atol=1e-10)
    )
    # 3x3 exercises the recursive cofactor expansion, not the closed form.
    cube = ref.Matrix([[6.0, 1.0, 1.0], [4.0, -2.0, 5.0], [2.0, 8.0, 7.0]])
    checks.append(
        assert_close(cube.determinant(), float(np.linalg.det(as_numpy(cube))),
                     label="determinant (3x3, cofactor)", atol=1e-9)
    )

    # --- vectors ----------------------------------------------------------
    v = ref.Vector([3.0, 4.0, 12.0])
    w = ref.Vector([1.0, 2.0, 2.0])
    npv, npw = np.array(v.data), np.array(w.data)
    checks.append(assert_close(v.dot(w), float(npv @ npw), label="dot product", atol=ATOL))
    checks.append(
        assert_close(v.magnitude(), float(np.linalg.norm(npv)),
                     label="magnitude (L2 norm)", atol=ATOL)
    )
    checks.append(
        assert_close(v.normalize(), npv / np.linalg.norm(npv),
                     label="normalise", atol=ATOL)
    )

    report(checks, title="phase 01 / lesson 02: stdlib Matrix vs numpy")
    print("\nThe lesson's Matrix class is not an approximation of numpy.")
    print("On these operations it is the same computation, written out longhand.")
    return 0


if __name__ == "__main__":
    if explain(__file__):
        raise SystemExit(0)
    raise SystemExit(main())
