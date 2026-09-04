"""Exercise 2 — broadcast_to, and auto-broadcasting elementwise ops.

    **Medium -- Implement broadcasting.** Extend the `Tensor` class with a
    `broadcast_to(shape)` method that expands dimensions of size 1 to match a
    target shape. Then modify `_elementwise_op` to automatically broadcast before
    operating. Test with shapes `(3, 1)` and `(1, 4)` producing `(3, 4)`.

Reading of the exercise: "extend the class" is a subclass, not an edit (D5).
Broadcasting is only correct if it *refuses* the incompatible cases, so check 4
tests (3,2) against (4,2) — a pair a naive implementation mangles because both
are the same rank.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "12-tensor-operations"


def _unravel(position, shape):
    index = []
    for extent in reversed(shape):
        index.append(position % extent)
        position //= extent
    return tuple(reversed(index))


def _ravel(index, shape):
    position, stride = 0, 1
    for extent, i in zip(reversed(shape), reversed(index)):
        position, stride = position + i * stride, stride * extent
    return position


def broadcast_shape(a, b):
    """NumPy's rule: right-align, then each axis must match or be 1."""
    a, b = (1,) * max(0, len(b) - len(a)) + a, (1,) * max(0, len(a) - len(b)) + b
    for x, y in zip(a, b):
        if x != y and 1 not in (x, y):
            raise ValueError(f"cannot broadcast {a} with {b}: {x} vs {y}")
    return tuple(max(x, y) for x, y in zip(a, b))


def make_class(ref):
    class Broadcasting(ref.Tensor):
        """Adds broadcast_to, and auto-broadcasting +, *, -."""

        def broadcast_to(self, shape):
            shape = tuple(shape)
            padded = (1,) * (len(shape) - self.rank) + tuple(self.shape)
            if broadcast_shape(padded, shape) != shape:
                raise ValueError(f"cannot broadcast {self.shape} to {shape}")
            # a size-1 axis always reads element 0 — that *is* broadcasting
            flat = [self._data[_ravel(
                tuple(0 if e == 1 else i
                      for e, i in zip(padded, _unravel(p, shape))), padded)]
                for p in range(math.prod(shape))]
            return Broadcasting(flat, shape=shape)

        def _elementwise_op(self, other, op):
            if not isinstance(other, ref.Tensor) or tuple(other.shape) == tuple(self.shape):
                return super()._elementwise_op(other, op)
            target = broadcast_shape(tuple(self.shape), tuple(other.shape))
            left = self.broadcast_to(target)._data
            right = Broadcasting(list(other._data), shape=other.shape) \
                .broadcast_to(target)._data
            return Broadcasting([op(a, b) for a, b in zip(left, right)], shape=target)

    return Broadcasting


def solve():
    ref = parity.load_reference(PHASE, LESSON, "tensors")
    Broadcasting = make_class(ref)
    column = Broadcasting([1, 2, 3], shape=(3, 1))
    row = Broadcasting([10, 20, 30, 40], shape=(1, 4))
    added = column + row
    try:
        Broadcasting([1] * 6, shape=(3, 2)) + Broadcasting([1] * 8, shape=(4, 2))
        refusal = "accepted (3,2) with (4,2)"
    except ValueError as exc:
        refusal = f"ValueError: {exc}"
    try:
        plain = ref.Tensor([1, 2, 3], shape=(3, 1)) + ref.Tensor([10] * 4, shape=(1, 4))
        original = f"returned shape {tuple(plain.shape)}"
    except Exception as exc:
        original = f"{type(exc).__name__}: {exc}"
    return {"shape": tuple(added.shape), "data": list(added._data),
            "expected": [c + r for c in (1, 2, 3) for r in (10, 20, 30, 40)],
            "refusal": refusal, "original": original,
            "subclass": issubclass(Broadcasting, ref.Tensor),
            "scalar": list((column * 2)._data)}


def verify(result):
    return [
        practice.Check("Tensor is extended, not edited", result["subclass"],
                       "Broadcasting subclasses the lesson's Tensor"),
        practice.Check("(3,1) + (1,4) produces (3,4) with the right 12 values",
                       result["shape"] == (3, 4) and result["data"] == result["expected"],
                       f"{result['data']} — row i holds column[i] + every row value"),
        practice.Check("incompatible shapes are refused, even at equal rank",
                       "ValueError" in result["refusal"],
                       f"(3,2) with (4,2) -> {result['refusal']} — same rank, so a check "
                       f"that only compares ranks would mangle this"),
        practice.Check("scalars still work, and the base class supported none of it",
                       result["scalar"] == [2, 4, 6] and "Error" in result["original"],
                       f"column * 2 -> {result['scalar']}; the unmodified Tensor gives "
                       f"{result['original']}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
