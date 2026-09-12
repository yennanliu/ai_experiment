"""Exercise 2 — the expectation is inverted: post-norm cannot explode, pre-norm grows.

    **Medium.** Switch from post-norm to pre-norm. Initialize both and measure
    the activation norm after 12 stacked layers on random input. Post-norm's
    activations should explode; pre-norm's should stay bounded.

Reading of the exercise: the lesson's `encoder_block` is already pre-norm, so the
switch runs the other way -- the post-norm variant, `x = norm(x + sublayer(x))`,
is written here and both stacks are run on the *same* twelve `BlockParams` and
the same random input, so the only difference between the two traces is where the
norm sits.

**ANSWER: post-norm is pinned at exactly 1.0; pre-norm grows 2.84x.**

| layer | 1 | 3 | 6 | 12 |
|---|---:|---:|---:|---:|
| pre-norm row RMS | 1.180 | 1.630 | 2.142 | **2.791** |
| post-norm row RMS | 1.000000 | 1.000000 | 1.000000 | **1.000000** |

Post-norm's twelve values span **1.4e-07**, which is `rms_norm`'s own `eps=1e-6`
and not dynamics.

**FINDING: the exercise has both halves backwards, and post-norm's half cannot be
otherwise.** `rms_norm` is the *last* operation of every post-norm block, so the
output's row RMS is 1 by construction -- at any depth, any initialisation, any
input. There is no experiment in which post-norm's forward activations explode.
The thing that explodes in a post-norm transformer is the *gradient*, and the
forward activation norm is precisely the quantity that cannot show it: it is
1.000000 at layer 1 and 1.000000 at layer 12, so the prescribed measurement has
zero dynamic range.

**FINDING: pre-norm is the one that grows, sub-linearly in depth.** The residual
stream is never normalised -- only each branch's *input* is -- so twelve roughly
independent contributions add in variance. The log-log slope of growth against
depth measures **0.36**, near the 0.5 of variance accumulation and far from the 0
a normalised output gives.

**CONTROL: both stacks share their weights.** The same twelve `BlockParams` and
the same input drive both traces, so nothing in the gap is initialisation.

Structure: `post_norm_block` is the variant the exercise asks for; `row_rms` is
the measurement; `slope` fits the growth exponent.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "05-full-transformer"
D_MODEL, HEADS, EXPANSION, TOKENS, DEPTH = 32, 4, 2.0, 8, 12
MARKS = (1, 3, 6, 12)


def post_norm_block(ref, x, params):
    """The variant the exercise asks for: x = norm(x + sublayer(x)), norm last."""
    attended = ref.multi_head_attention(x, params.Wq, params.Wk, params.Wv,
                                        params.Wo, params.n_heads)
    x = ref.rms_norm(ref.add(x, attended))
    gated = ref.ffn_swiglu(x, params.W1, params.W2, params.W3)
    return ref.rms_norm(ref.add(x, gated))


def row_rms(matrix):
    """Mean row RMS -- the activation norm, per position, independent of d."""
    return sum(math.sqrt(sum(v * v for v in matrix.row(i)) / matrix.cols)
               for i in range(matrix.rows)) / matrix.rows


def trace(step, x, stack):
    """Row RMS after every layer of one stack, starting from the input's own."""
    out = [row_rms(x)]
    for params in stack:
        x = step(x, params)
        out.append(row_rms(x))
    return out


def slope(growth):
    """Least-squares exponent of growth against depth in log-log: 0.5 is a square root."""
    xs = [math.log(depth) for depth in range(1, len(growth))]
    ys = [math.log(growth[depth]) for depth in range(1, len(growth))]
    n, sx, sy = len(xs), sum(xs), sum(ys)
    return (n * sum(a * b for a, b in zip(xs, ys)) - sx * sy) / (n * sum(a * a for a in xs) - sx ** 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(42)
    x = ref.randn(TOKENS, D_MODEL, rng, scale=1.0)
    stack = [ref.BlockParams(D_MODEL, HEADS, EXPANSION, rng) for _ in range(DEPTH)]
    pre = trace(lambda a, p: ref.encoder_block(a, p), x, stack)
    post = trace(lambda a, p: post_norm_block(ref, a, p), x, stack)
    return {
        "pre": pre, "post": post, "input": pre[0],
        "growth": [value / pre[0] for value in pre],
        "exponent": slope([value / pre[0] for value in pre]),
        "post_span": max(post[1:]) - min(post[1:]),
        "shared": len(stack),
    }


def verify(result):
    pre, post = result["pre"], result["post"]
    return [
        practice.Check(
            "ANSWER: post-norm is pinned at 1.0 and pre-norm grows 2.8x over 12 layers",
            result["post_span"] < 1e-6 and pre[DEPTH] / pre[0] > 2.5,
            f"row RMS from an input of {result['input']:.3f}: pre-norm "
            f"{[round(pre[m], 3) for m in MARKS]} at layers {list(MARKS)}, post-norm "
            f"{[round(post[m], 6) for m in MARKS]}. Post-norm's twelve values span "
            f"{result['post_span']:.1e}, which is rms_norm's own eps=1e-6 and not dynamics",
        ),
        practice.Check(
            "FINDING: post-norm's activations cannot explode, at any depth",
            all(abs(value - 1.0) < 1e-6 for value in post[1:]),
            "rms_norm is the last operation of every post-norm block, so the output's row RMS is "
            "1 by construction -- at any depth, initialisation or input. There is no experiment "
            f"in which it explodes: {len(post) - 1} layers, all {post[DEPTH]:.6f}. What explodes "
            "in a post-norm transformer is the gradient, and this measurement cannot see it",
        ),
        practice.Check(
            "FINDING: pre-norm is the one that grows, and it grows like sqrt(depth)",
            0.25 < result["exponent"] < 0.75,
            f"the residual stream is never normalised -- only each branch's input is -- so "
            f"roughly independent contributions add in variance. Growth against depth fits a "
            f"log-log slope of {result['exponent']:.3f}: sub-linear, and near the 0.5 of variance "
            f"accumulation rather than the 0 a normalised output would give. "
            f"{result['growth'][DEPTH]:.2f}x at layer {DEPTH}, sqrt({DEPTH}) = "
            f"{math.sqrt(DEPTH):.2f}",
        ),
        practice.Check(
            "FINDING: the lesson is already pre-norm, so the switch runs the other way",
            result["shared"] == DEPTH,
            "encoder_block is x + sublayer(rms_norm(x)) -- the destination, not the starting "
            "point. The work the exercise describes is writing the post-norm variant, which is "
            "the arm whose behaviour it then predicts wrongly",
        ),
        practice.Check(
            "CONTROL: both stacks share their weights and their input",
            pre[0] == post[0],
            f"the same {DEPTH} BlockParams and the same randn({TOKENS}, {D_MODEL}) drive both "
            f"traces, which start from an identical {pre[0]:.4f}. Nothing in the gap between them "
            "is initialisation; the only difference is where the norm sits",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
