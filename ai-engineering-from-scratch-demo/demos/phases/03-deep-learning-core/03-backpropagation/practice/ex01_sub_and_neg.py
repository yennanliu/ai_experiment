"""Exercise 1 — the subtraction the lesson already shipped, and its gradients.

    Add a `__sub__` method to the Value class (a - b = a + (-1 * b)). Then
    implement a `__neg__` method. Verify that the gradients are correct by
    comparing with manual calculation for a simple expression like (a - b)^2.

Reading of the exercise: there is nothing to add — `code/main.py` already defines
both, exactly as the exercise spells them, though docs/en.md shows neither. The
exercise collapses onto its third sentence, so "verify" is read as two oracles
rather than one: the hand-derived 2(a-b), and a central finite difference of the
same expression. (Exercise 3 extends the difference check to a whole network.)
Checks 1, 2, 4 and 5 are what the verification turns up on the way.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "03-backpropagation"
A, B, H = 3.0, 1.25, 1e-5


def walk(node) -> dict:
    """id -> op for every node reachable from `node`."""
    seen, stack = {}, [node]
    while stack:
        item = stack.pop()
        seen.setdefault(id(item), item._op)
        stack += [c for c in item._children if id(c) not in seen]
    return seen


def one_sided(V) -> str:
    """The lesson negates on the left only; ask what the right-hand form does."""
    try:
        1.0 - V(2.0)
        return "no error — __rsub__ exists"
    except TypeError as exc:
        return str(exc)


def central_difference(fn, x, h=H) -> float:
    return (fn(x + h) - fn(x - h)) / (2 * h)


def triangular(g: float) -> list:
    """Pass n adds n*g to the root, so the running total is g times T(n)."""
    return [g * n * (n + 1) / 2 for n in (1, 2, 3)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    V = ref.Value
    a, b = V(A), V(B)
    diff = a - b
    loss = diff * diff
    loss.backward()
    grads, compounding = (a.grad, b.grad), [a.grad]
    for _repeat in range(2):
        loss.backward()
        compounding.append(a.grad)
    ops = walk(diff)
    return {
        "grads": grads, "manual": (2 * (A - B), -2 * (A - B)), "compounding": compounding,
        "fd": (central_difference(lambda v: (v - B) ** 2, A),
               central_difference(lambda v: (A - v) ** 2, B)),
        "nodes": len(ops), "ops": sorted(op for op in ops.values() if op),
        "reuse": len(loss._children), "rsub": one_sided(V),
        "has": [n for n in ("__sub__", "__neg__") if hasattr(V, n)],
    }


def verify(result):
    grads, manual = result["grads"], result["manual"]
    worst = max(abs(g - n) for g, n in zip(grads, result["fd"]))
    trail = ", ".join(f"{g:.1f}" for g in result["compounding"])
    return [
        practice.Check("FINDING: there is nothing to add — the lesson's code ships both",
                       result["has"] == ["__sub__", "__neg__"],
                       "`code/main.py` already defines `__sub__` as `self + (-other)` and "
                       "`__neg__` as `self * -1`, the identity the exercise dictates. "
                       "Neither appears in docs/en.md, so the reader is asked to re-derive "
                       "code they were handed"),
        practice.Check("MECHANISM: that identity costs five nodes, and negates on one side",
                       result["nodes"] == 5 and result["ops"] == ["*", "+"],
                       f"the graph under `a - b` holds {result['nodes']} nodes joined by "
                       f"{result['ops']}: a + (-1 * b) materialises a Value(-1) leaf and a "
                       f"multiply that a primitive subtraction would not need. And "
                       f"`1.0 - Value(2.0)` raises TypeError ({result['rsub']}) — no "
                       f"`__rsub__`, so only the left operand may be a Value"),
        practice.Check("ANSWER: (a-b)^2 gradients are exactly the hand derivation, and a "
                       "central finite difference agrees",
                       grads == manual and worst < 1e-9,
                       f"a = {A}, b = {B} — d/da {grads[0]:+.4f} vs 2(a-b) {manual[0]:+.4f}, "
                       f"d/db {grads[1]:+.4f} vs -2(a-b) {manual[1]:+.4f}, deviation exactly "
                       f"0.0; against h = {H:g} differences, {worst:.3e}, which is the "
                       f"quotient's own O(h^2) truncation and not the engine's error"),
        practice.Check("MECHANISM: `diff` is used twice and the child *set* collapses to one",
                       result["reuse"] == 1 and grads == manual,
                       f"`diff * diff` stores `set((diff, diff))`, length {result['reuse']}, "
                       f"yet the gradient is right: `_backward` captured both operands and "
                       f"runs `self.grad +=` then `other.grad +=` against the same object, "
                       f"accumulating 2·diff. The set decides traversal, not credit"),
        practice.Check("FINDING: a second `backward()` compounds rather than doubles",
                       result["compounding"] == triangular(manual[0]),
                       f"three calls on one graph give d/da = {trail}. `backward` *sets* the "
                       f"root to 1.0 but never clears the interior, so pass n re-propagates "
                       f"diff's already-accumulated grad on top of a fresh {manual[0]:.1f}, "
                       f"and the total tracks the triangular numbers rather than n. "
                       f"`zero_grad()` resets parameters only; training survives because each "
                       f"epoch builds a graph from scratch"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
