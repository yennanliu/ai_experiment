"""Exercise 1 — reshape (2,3,4) → (6,4) → (24,) → (2,3,4), order preserved.

    **Easy -- Reshape round-trip.** Take a tensor of shape `(2, 3, 4)`. Reshape
    it to `(6, 4)`, then to `(24,)`, then back to `(2, 3, 4)`. Printing the flat
    data verifies element order is preserved at each step.

Reading of the exercise: "printing the flat data" verifies nothing on its own, so
the round-trip is asserted instead — flat data identical at every step, and
element-wise identity through indexing at the end. The property that makes this
work is that reshape only reinterprets strides over an unchanged buffer, and
check 4 tests it the only way that distinguishes it from a copy: a reshape to an
incompatible size must be refused, not padded.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "12-tensor-operations"
SHAPE = (2, 3, 4)
CHAIN = ((6, 4), (24,), SHAPE)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "tensors")
    # _data is the flat buffer; the class exposes no public accessor for it,
    # and "printing the flat data" is precisely what the exercise asks for
    original = ref.Tensor(list(range(24)), shape=SHAPE)
    flats = [list(original._data)]
    shapes = [tuple(original.shape)]
    strides = [tuple(original.strides)]
    current = original
    for shape in CHAIN:
        current = current.reshape(shape)
        flats.append(list(current._data))
        shapes.append(tuple(current.shape))
        strides.append(tuple(current.strides))
    same = all(original[i, j, k] == current[i, j, k]
               for i in range(SHAPE[0]) for j in range(SHAPE[1]) for k in range(SHAPE[2]))
    try:
        original.reshape((5, 5))
        refusal = "accepted a (5, 5) reshape of 24 elements"
    except Exception as exc:
        refusal = f"{type(exc).__name__}: {exc}"
    return {"flats": flats, "shapes": shapes, "strides": strides,
            "elementwise_same": same, "refusal": refusal}


def verify(result):
    flats = result["flats"]
    return [
        practice.Check("the chain runs (2,3,4) → (6,4) → (24,) → (2,3,4)",
                       result["shapes"] == [SHAPE, *CHAIN],
                       " → ".join(str(s) for s in result["shapes"])),
        practice.Check("flat data is byte-identical at every step",
                       all(f == flats[0] for f in flats),
                       f"{len(flats)} snapshots, all equal; first 8 elements "
                       f"{flats[0][:8]}"),
        practice.Check("indexing agrees element-by-element after the round trip",
                       result["elementwise_same"],
                       f"all {SHAPE[0] * SHAPE[1] * SHAPE[2]} positions match the original"),
        practice.Check("only the strides change — reshape reinterprets, it does not copy",
                       len(set(result["strides"])) == len(CHAIN),
                       "strides: " + " → ".join(str(s) for s in result["strides"])
                       + " — row-major, so each is the suffix product of the shape"),
        practice.Check("an incompatible reshape is refused, not padded or truncated",
                       "Error" in result["refusal"] or "error" in result["refusal"],
                       f"reshape((5, 5)) on 24 elements -> {result['refusal']}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
