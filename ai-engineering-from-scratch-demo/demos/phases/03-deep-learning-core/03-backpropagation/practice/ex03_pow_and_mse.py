"""Exercise 3 — `__pow__`, and what "gradients match" turns out to mean.

    Implement a `__pow__` method on Value for integer powers. Use it to replace
    `mse_loss` with a proper `(predicted - target) ** 2` expression. Verify
    gradients match the original implementation.

Reading of the exercise: "match" is read as bitwise, because anything looser
cannot tell a rewrite bug from rounding. Both losses are built on the lesson's
own `Value`, over the whole [2, 4, 1] XOR network and eight seeds, and every
parameter gradient is compared. Check 2 narrows the graph until the two agree
exactly; check 3 identifies what the remaining difference actually is.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "03-backpropagation"
XOR = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)]
SEEDS = range(8)


def install(ref) -> None:
    """The exercise's edit: the power rule, d(x**k)/dx = k * x**(k-1)."""
    value = ref.Value
    def __pow__(self, k):
        out = value(self.data ** k, (self,), f"**{k}")
        def _backward(): self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out
    value.__pow__ = __pow__


def power(predicted, target):
    """The exercise's replacement for `mse_loss`, spelled exactly as it asks."""
    return (predicted - target) ** 2


def grads(ref, seed, loss, rows) -> tuple:
    """Parameter gradients and total loss for one seeded net over `rows`."""
    random.seed(seed)
    net = ref.Network([2, 4, 1])
    total = ref.Value(0.0)
    for inputs, target in rows:
        total = total + loss(net([ref.Value(i) for i in inputs]), target)
    net.zero_grad()
    total.backward()
    return [p.grad for p in net.parameters()], total.data


worst = lambda left, right: max(abs(a - b) for a, b in zip(left, right))   # noqa: E731


def compare(ref, seed) -> dict:
    """Every deviation this exercise measures, for one seed."""
    old, new = grads(ref, seed, ref.mse_loss, XOR), grads(ref, seed, power, XOR)
    again = grads(ref, seed, ref.mse_loss, XOR)
    per = [grads(ref, seed, ref.mse_loss, [r])[0] for r in XOR]
    return {"batch": worst(old[0], new[0]), "self": worst(old[0], again[0]),
            "loss": abs(old[1] - new[1]), "params": len(old[0]),
            "row": max(worst(a, grads(ref, seed, power, [r])[0]) for a, r in zip(per, XOR)),
            "shared": sum(len(XOR) for _p in old[0])}


def edges(ref) -> dict:
    """Where the power rule as written stops being one."""
    value, out = ref.Value, {}
    try:
        (value(0.0) ** 0).backward()
    except ZeroDivisionError as exc:
        out["zero"] = str(exc)
    try:
        2 ** value(3.0)
    except TypeError as exc:
        out["rpow"] = type(exc).__name__
    out["complex"] = type((value(-8.0) ** 0.5).data).__name__
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    install(ref)
    runs = [compare(ref, seed) for seed in SEEDS]
    peak = {key: max(r[key] for r in runs) for key in ("batch", "self", "loss", "row", "shared")}
    return {**peak, "seeds": len(runs), "params": runs[0]["params"],
            "moved": sum(r["self"] > 0 for r in runs), "edges": edges(ref)}


def verify(result):
    seeds, edge = result["seeds"], result["edges"]
    span = result["params"] * seeds
    return [
        practice.Check("ANSWER: `(predicted - target) ** 2` reproduces `mse_loss`, to within "
                       "one ulp of the gradient",
                       result["batch"] < 1e-16 and result["loss"] == 0.0,
                       f"over {span} parameter gradients ({result['params']} params x {seeds} "
                       f"seeds) the worst disagreement is {result['batch']:.3e}, and the loss "
                       f"values agree exactly ({result['loss']:.1f})"),
        practice.Check("CONTROL: on a one-row graph they agree bitwise, so `__pow__` itself "
                       "is exact",
                       result["row"] == 0.0,
                       f"train on a single XOR row and every parameter matches to "
                       f"{result['row']:.1f} on all {seeds} seeds. `diff * diff` credits "
                       f"`diff` twice by `+=`, `diff ** 2` once by 2*diff — the same float"),
        practice.Check("FINDING: the residue is the engine's, not the rewrite's — rebuilding "
                       "`mse_loss` twice moves it just as far",
                       max(result["self"], result["batch"]) < 1e-16,
                       f"two identical `mse_loss` graphs from one seed differ by up to "
                       f"{result['self']:.3e} — the same order as the rewrite's "
                       f"{result['batch']:.3e}, and both are one ulp. MECHANISM: `_children` is a "
                       f"`set`, so `build_topo` visits children in an order set by object hashes "
                       f"rather than by construction, and every one of this graph's "
                       f"{result['shared']} parameter accumulations ({result['params']} params x "
                       f"{len(XOR)} rows) lands in whatever order that gives"),
        practice.Check("FINDING: 'integer powers' is load-bearing — the rule breaks three "
                       "ways outside them",
                       "zero" in edge and edge["rpow"] == "TypeError"
                       and edge["complex"] == "complex",
                       f"`Value(0.0) ** 0` forwards to 1.0 and then raises ZeroDivisionError "
                       f"in backward ({edge['zero']}), because k * x**(k-1) is 0 * 0.0**-1. "
                       f"`2 ** Value(3.0)` raises {edge['rpow']} — no `__rpow__`, as with "
                       f"`__rsub__`. And `Value(-8.0) ** 0.5` silently stores a "
                       f"{edge['complex']}, which `__repr__`'s `:.4f` cannot even print"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
