"""Assertions on what Phase 01 / Lesson 02 claims, not on the demo's plumbing."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(DEMO.parents[3]))

from harness.parity import ParityError, load_reference  # noqa: E402
from run import LESSON, as_numpy  # noqa: E402

ref = load_reference(LESSON)


def random_matrix(rows, cols, rng):
    return ref.Matrix([[rng.uniform(-3, 3) for _ in range(cols)] for _ in range(rows)])


@pytest.mark.parametrize("shape", [(2, 3, 4), (5, 5, 5), (1, 7, 2), (8, 3, 1)])
def test_matmul_matches_numpy_at_many_shapes(shape):
    """The claim: the triple loop *is* matrix multiplication, at any shape."""
    rows, inner, cols = shape
    rng = random.Random(rows * 100 + cols)
    A, B = random_matrix(rows, inner, rng), random_matrix(inner, cols, rng)
    np.testing.assert_allclose(as_numpy(A @ B), as_numpy(A) @ as_numpy(B), atol=1e-12)


def test_matmul_is_not_commutative():
    """The lesson's central warning: A @ B and B @ A are different matrices."""
    A = ref.Matrix([[1.0, 2.0], [3.0, 4.0]])
    B = ref.Matrix([[5.0, 6.0], [7.0, 8.0]])
    assert as_numpy(A @ B).tolist() != as_numpy(B @ A).tolist()
    np.testing.assert_allclose(as_numpy(B @ A), as_numpy(B) @ as_numpy(A), atol=1e-12)


def test_inner_dimension_mismatch_is_rejected():
    """Shape discipline is the lesson's point; a wrong shape must not compute."""
    A, B = random_matrix(3, 4, random.Random(0)), random_matrix(3, 4, random.Random(1))
    with pytest.raises(ValueError, match="inner dimensions"):
        A @ B


def test_inverse_returns_the_identity():
    """A @ A^-1 == I is the claim the lesson makes about `inverse_2x2`."""
    A = ref.Matrix([[4.0, 7.0], [2.0, 6.0]])
    np.testing.assert_allclose(as_numpy(A @ A.inverse_2x2()), np.eye(2), atol=1e-12)


def test_singular_matrix_has_no_inverse():
    singular = ref.Matrix([[1.0, 2.0], [2.0, 4.0]])
    assert np.isclose(np.linalg.det(as_numpy(singular)), 0.0)
    with pytest.raises(ValueError, match="singular"):
        singular.inverse_2x2()


def test_cofactor_determinant_matches_numpy_beyond_2x2():
    """`determinant` recurses for n>2; numpy uses LU. They must still agree."""
    rng = random.Random(7)
    for n in (3, 4, 5):
        M = random_matrix(n, n, rng)
        np.testing.assert_allclose(M.determinant(), np.linalg.det(as_numpy(M)), atol=1e-9)


def test_normalised_vector_has_unit_length():
    v = ref.Vector([3.0, 4.0, 12.0])
    assert np.isclose(v.normalize().magnitude(), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        v.normalize().data, np.array(v.data) / np.linalg.norm(v.data), atol=1e-12
    )


def test_parity_helper_rejects_a_shape_mismatch():
    """A parity check must fail loudly on mismatched shapes, never silently pass."""
    from harness.parity import assert_close

    with pytest.raises(ParityError, match="shape mismatch"):
        assert_close([1.0, 2.0], [1.0, 2.0, 3.0], label="deliberate mismatch")
