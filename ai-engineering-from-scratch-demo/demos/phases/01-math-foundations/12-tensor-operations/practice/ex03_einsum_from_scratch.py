"""Exercise 3 — a basic einsum, checked against np.einsum.

    **Hard -- Build einsum from scratch.** Implement a basic `einsum(subscripts,
    *tensors)` function that handles at least: dot product (`i,i->`), matrix
    multiply (`ij,jk->ik`), outer product (`i,j->ij`), and transpose (`ij->ji`).
    Parse the subscript string, identify contracted indices, and loop over all
    index combinations. Compare your results against `np.einsum`.

Reading of the exercise: the four named cases are the floor, not the spec — an
implementation that special-cases each one would pass them and nothing else. So
the loop is written generically (free indices outer, contracted indices summed)
and check 3 runs five *additional* patterns it was never told about, including
batched matmul and a three-operand chain. numpy is the oracle throughout.
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "12-tensor-operations"
NAMED = ("i,i->", "ij,jk->ik", "i,j->ij", "ij->ji")
EXTRA = ("ij->", "ii->i", "bij,bjk->bik", "ij,j->i", "ij,jk,kl->il")


def _index_sizes(terms, operands):
    sizes = {}
    for term, operand in zip(terms, operands):
        sizes.update(zip(term, operand.shape))
    return sizes


def _term_product(terms, operands, assignment):
    product = 1.0
    for term, operand in zip(terms, operands):
        product *= operand[tuple(assignment[c] for c in term)]
    return product


def _space(letters, sizes):
    return list(itertools.product(*[range(sizes[c]) for c in letters]))


def _contract(terms, operands, output, free, contracted, sum_space):
    assignment = dict(zip(output, free))
    total = 0.0
    for summed in sum_space:
        assignment.update(zip(contracted, summed))
        total += _term_product(terms, operands, assignment)
    return total


def einsum(numpy, subscripts, *operands):
    """Free indices form the output loop; every other index is summed over."""
    inputs, _, output = subscripts.replace(" ", "").partition("->")
    terms = inputs.split(",")
    sizes = _index_sizes(terms, operands)
    contracted = sorted(set("".join(terms)) - set(output))
    result = numpy.zeros([sizes[c] for c in output] or [1], dtype=float)
    sum_space = _space(contracted, sizes)
    for free in _space(output, sizes):
        result[free if free else 0] = _contract(terms, operands, output, free,
                                                contracted, sum_space)
    return result if output else float(result[0])


def solve():
    numpy = parity.try_numpy()
    if numpy is None:
        raise practice.Skip("needs numpy — uv sync --extra math")
    rng = numpy.random.default_rng(42)
    shapes = {"i,i->": [(5,), (5,)], "ij,jk->ik": [(3, 4), (4, 2)],
              "i,j->ij": [(3,), (4,)], "ij->ji": [(3, 4)], "ij->": [(3, 4)],
              "ii->i": [(4, 4)], "bij,bjk->bik": [(2, 3, 4), (2, 4, 5)],
              "ij,j->i": [(3, 4), (4,)], "ij,jk,kl->il": [(2, 3), (3, 4), (4, 2)]}
    rows = {}
    for spec, operand_shapes in shapes.items():
        operands = [rng.normal(size=s) for s in operand_shapes]
        mine = einsum(numpy, spec, *operands)
        theirs = numpy.einsum(spec, *operands)
        rows[spec] = float(numpy.abs(numpy.asarray(mine) - numpy.asarray(theirs)).max())
    return {"rows": rows}


def verify(result):
    rows = result["rows"]
    named = {k: rows[k] for k in NAMED}
    extra = {k: rows[k] for k in EXTRA}
    return [
        practice.Check("all four named patterns match np.einsum",
                       max(named.values()) < 1e-12,
                       "; ".join(f"{k}: {v:.2g}" for k, v in named.items())),
        practice.Check("the parser is generic, not four special cases",
                       max(extra.values()) < 1e-12,
                       "; ".join(f"{k}: {v:.2g}" for k, v in extra.items())
                       + " — five patterns the implementation was never told about"),
        practice.Check("…including a batched matmul and a 3-operand chain",
                       rows["bij,bjk->bik"] < 1e-12 and rows["ij,jk,kl->il"] < 1e-12,
                       "batch indices are just free indices that appear in both operands, "
                       "and a chain needs no special handling because every non-output "
                       "index is summed"),
        practice.Check("full reduction and diagonal extraction work",
                       rows["ij->"] < 1e-12 and rows["ii->i"] < 1e-12,
                       "'ij->' has an empty output so the free loop runs once; 'ii->i' "
                       "repeats an index within one operand, which the assignment dict "
                       "handles for free"),
        practice.Check(f"{len(rows)} patterns, worst deviation {max(rows.values()):.2g}",
                       max(rows.values()) < 1e-12,
                       "np.einsum is the oracle for every one of them"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
