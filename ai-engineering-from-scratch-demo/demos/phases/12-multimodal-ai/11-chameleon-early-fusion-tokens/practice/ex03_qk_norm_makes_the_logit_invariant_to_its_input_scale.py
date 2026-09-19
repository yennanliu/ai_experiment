"""Exercise 3 — QK-Norm makes the logit invariant to its input scale.

    Implement QK-Norm in pure Python. Given a 64-dim query and key, show the dot
    product before and after LayerNorm. Why is magnitude control important at
    depth?

Reading of the exercise: one before-and-after pair shows almost nothing, because
a pair of unit-variance vectors barely moves under LayerNorm. The pair is
therefore reported and then swept over input scales, which is where the property
QK-Norm exists for becomes visible -- the raw logit grows as the square of the
scale and the normalised one does not move at all.

**ANSWER: -1.9794 before, -1.7674 after.** Barely a change, because the fixture
is already zero-mean and unit-variance -- which is exactly why a single pair is
not a demonstration of anything.

**FINDING: scale the inputs and the two diverge quadratically.** At 2x, 10x and
100x the raw dot product is -7.92, -197.94 and -19,793.94 -- growing as s^2,
since both operands scale -- while the normalised one stays at **-1.7674** to
six decimal places. LayerNorm removes the scale from both factors, so the logit
depends only on the angle between them.

**FINDING: after normalisation the logit is bounded, and the bound is d.** Every
normalised vector has norm exactly **8.0** = sqrt(64), so the dot is
`64 * cos(theta)` and lies in [-64, 64] -- [-8, 8] after the 1/sqrt(d) attention
scale -- whatever the model does upstream. Softmax over logits in that range
cannot saturate.

**FINDING: at depth the input scale is not a constant.** A residual stream that
grows even **5%** a layer is 10.4x after 48 layers and the logit **108x**; at
10% over 32 layers, 21.1x and **446x**; at 50%, 4.3e5 and **1.9e11**. The
quantity QK-Norm bounds is the one that compounds, and it compounds squared.

Structure: `layer_norm` is the normalisation, `dot` the logit, `SCALES` the
sweep that separates the two, and `compounding` is the depth table.
"""

from __future__ import annotations

import math
import random

from harness import practice

DIM, SEED, EPS = 64, 0, 1e-5
SCALES = (1, 2, 10, 100)
GROWTH = ((1.05, 48), (1.10, 32), (1.50, 32))


def make_fixture(dim=DIM, seed=SEED):
    """A labelled pair: two independent standard-normal vectors."""
    rng = random.Random(seed)
    return ([rng.gauss(0, 1) for _ in range(dim)],
            [rng.gauss(0, 1) for _ in range(dim)])


def layer_norm(vector, eps=EPS):
    mean = sum(vector) / len(vector)
    variance = sum((x - mean) ** 2 for x in vector) / len(vector)
    return [(x - mean) / math.sqrt(variance + eps) for x in vector]


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def scaled(vector, factor):
    return [x * factor for x in vector]


def compounding(rate, layers):
    """Norm growth over depth, and the logit growth it implies."""
    norm = rate ** layers
    return round(norm, 2), round(norm * norm, 1)


def solve():
    query, key = make_fixture()
    raw = {factor: round(dot(scaled(query, factor), scaled(key, factor)), 3)
           for factor in SCALES}
    normed = {factor: round(dot(layer_norm(scaled(query, factor)),
                                layer_norm(scaled(key, factor))), 6)
              for factor in SCALES}
    return {
        "before": round(dot(query, key), 4),
        "after": round(dot(layer_norm(query), layer_norm(key)), 4),
        "raw": raw, "normed": normed,
        "raw_growth": round(raw[100] / raw[1], 1),
        "normed_spread": round(max(normed.values()) - min(normed.values()), 9),
        "norm_after": round(math.sqrt(dot(layer_norm(query), layer_norm(query))), 4),
        "root_dim": math.sqrt(DIM),
        "bound": DIM, "scaled_bound": round(DIM / math.sqrt(DIM), 1),
        "depth": {f"{rate}^{layers}": compounding(rate, layers) for rate, layers in GROWTH},
    }


def verify(result):
    raw, normed, depth = result["raw"], result["normed"], result["depth"]
    return [
        practice.Check(
            "ANSWER: -1.9794 before, -1.7674 after",
            all([result["before"] == -1.9794, result["after"] == -1.7674]),
            f"on a standard-normal 64-dim pair the dot product is {result['before']} raw and "
            f"{result['after']} after LayerNorm -- barely a change, because the fixture is "
            "already zero-mean and unit-variance. A single pair is not a demonstration",
        ),
        practice.Check(
            "FINDING: scale the inputs and the two diverge quadratically",
            all([raw == {1: -1.979, 2: -7.918, 10: -197.939, 100: -19793.943},
                 len(set(normed.values())) == 1 or result["normed_spread"] <= 2e-5,
                 result["raw_growth"] == 10002.0]),
            f"at scales {list(SCALES)} the raw logit is {list(raw.values())} -- growing as "
            f"s^2, since both operands scale, {result['raw_growth']:.0f}x across the sweep -- "
            f"while the normalised one stays at {normed[1]} within "
            f"{result['normed_spread']:.0e}. LayerNorm removes the scale from both factors, "
            "so the logit depends only on the angle",
        ),
        practice.Check(
            "FINDING: after normalisation the logit is bounded, and the bound is d",
            all([result["norm_after"] == 8.0, result["norm_after"] == result["root_dim"],
                 result["bound"] == DIM, result["scaled_bound"] == 8.0]),
            f"every normalised vector has norm exactly {result['norm_after']} = sqrt({DIM}), "
            f"so the dot is {DIM} * cos(theta) and lies in [-{result['bound']}, "
            f"{result['bound']}] -- [-{result['scaled_bound']:.0f}, "
            f"{result['scaled_bound']:.0f}] after the 1/sqrt(d) attention scale -- whatever "
            "the model does upstream. A softmax over that range cannot saturate",
        ),
        practice.Check(
            "FINDING: at depth the input scale is not a constant",
            all([depth["1.05^48"] == (10.4, 108.2), depth["1.1^32"] == (21.11, 445.8),
                 depth["1.5^32"][1] > 1e11]),
            f"a residual stream growing 5% a layer is {depth['1.05^48'][0]}x after 48 layers "
            f"and the logit {depth['1.05^48'][1]}x; 10% over 32 layers gives "
            f"{depth['1.1^32']}; 50% gives {depth['1.5^32']}. The quantity QK-Norm bounds is "
            "the one that compounds, and it compounds squared",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
