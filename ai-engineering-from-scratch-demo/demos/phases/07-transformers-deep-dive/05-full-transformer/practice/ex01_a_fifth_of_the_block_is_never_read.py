"""Exercise 1 — a fifth of the block is never read, and the prescribed check counts it.

    **Easy.** Count the parameters in your encoder_block at `d_model=512,
    n_heads=8, ffn_expansion=4, swiglu=True`. Validate by implementing the block
    and using `sum(p.numel() for p in block.parameters())`.

Reading of the exercise: the count is derived in closed form and then checked
against the lesson's own `BlockParams`, built at exactly those settings -- but
which weights count is decided mechanically, by parsing `encoder_block`'s source
for the attributes it actually reads, rather than by assuming.

**ANSWER: 4,194,304 -- exactly 2^22.** Attention is `4 * d^2` = 1,048,576 and the
SwiGLU FFN is `3 * d * h` = 3,145,728 at `h = 4d = 2048`. No biases, so that is
all of it.

**FINDING: `BlockParams` holds 5,242,880, and `encoder_block` reads 80% of it.**
The constructor allocates `Wq_x`, `Wk_x`, `Wv_x`, `Wo_x` -- decoder
cross-attention -- unconditionally, for every block. `encoder_block` never
mentions them. The validation the exercise prescribes, `sum(p.numel() for p in
block.parameters())`, counts everything the object holds, so it would "validate"
the answer against **1.25x** the right number and agree with itself.

**FINDING: SwiGLU at `ffn_expansion=4` is 1.5x the FFN it replaces.** A gated FFN
has three matrices where ReLU has two, so at the same expansion it costs
3,145,728 against 2,097,152 -- an extra 1,048,576 parameters, the exact size of
the whole attention sublayer. The usual convention is `8/3` expansion with
SwiGLU so the two match; the exercise's settings silently add 50% to the FFN.

**FINDING: the block has zero normalisation parameters.** `rms_norm` and
`layer_norm` are both written without a learnable scale, so the 2 x 512 gains a
standard pre-norm block carries are simply absent.

**CONTROL: `parameters()` does not exist.** `torch` is absent and `BlockParams`
is a plain object, so the prescribed one-liner cannot run. Summing
`len(Matrix.data)` over its attributes is the equivalent -- and it is exactly the
act that exposes the dead cross-attention weights.

Structure: `used` parses `encoder_block` for the weights it names; `held` reads
what the constructor allocated; `closed_form` is the arithmetic.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "05-full-transformer"
D_MODEL, HEADS, EXPANSION = 512, 8, 4


def used(function):
    """The parameter attributes a block function actually reads, from its own source."""
    tree = ast.parse(inspect.getsource(function))
    return {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)}


def held(params):
    """{attribute: element count} for every Matrix the constructor allocated."""
    return {name: len(getattr(params, name).data) for name in vars(type(params)).get("__slots__", ())
            or [n for n in dir(params) if hasattr(getattr(params, n), "data")]}


def closed_form(d=D_MODEL, expansion=EXPANSION):
    """(attention, SwiGLU FFN, ReLU FFN, cross-attention) at these settings."""
    hidden = int(d * expansion)
    return 4 * d * d, 3 * d * hidden, 2 * d * hidden, 4 * d * d


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    params = ref.BlockParams(D_MODEL, HEADS, EXPANSION, random.Random(0))
    sizes = held(params)
    names = used(ref.encoder_block)
    attention, swiglu, relu, cross = closed_form()
    return {
        "sizes": sizes, "names": sorted(n for n in names if n in sizes),
        "read": sum(v for n, v in sizes.items() if n in names),
        "total": sum(sizes.values()), "predicted": attention + swiglu,
        "attention": attention, "swiglu": swiglu, "relu": relu, "cross": cross,
        "dead": sorted(n for n in sizes if n not in names),
        "norm_params": sum(len(inspect.signature(fn).parameters) - 2
                           for fn in (ref.rms_norm, ref.layer_norm)),
        "torch": importlib.util.find_spec("torch") is None,
        "has_parameters": hasattr(params, "parameters"),
    }


def verify(result):
    hidden = D_MODEL * EXPANSION
    return [
        practice.Check(
            "ANSWER: 4,194,304 parameters -- exactly 2^22",
            result["read"] == result["predicted"] == 2 ** 22,
            f"4 * d^2 = {result['attention']:,} for attention and 3 * d * h = "
            f"{result['swiglu']:,} for the SwiGLU FFN at h = {hidden}, total "
            f"{result['predicted']:,} = 2^22. The lesson's own BlockParams, summed over the "
            f"{len(result['names'])} attributes encoder_block names, gives {result['read']:,}",
        ),
        practice.Check(
            "FINDING: BlockParams holds 5,242,880 and the block reads 80% of it",
            result["total"] - result["read"] == result["cross"] and len(result["dead"]) == 4,
            f"the constructor allocates {result['dead']} -- decoder cross-attention -- for every "
            f"block, and encoder_block's source never mentions them: {result['total']:,} held "
            f"against {result['read']:,} read, {result['cross']:,} dead. "
            f"sum(p.numel() for p in block.parameters()) counts what the object holds, so the "
            f"prescribed validation returns {result['total'] / result['read']:.2f}x the answer",
        ),
        practice.Check(
            "FINDING: SwiGLU at expansion 4 costs 1.5x the FFN it replaces",
            result["swiglu"] == 3 * result["relu"] // 2,
            f"three matrices where ReLU has two: {result['swiglu']:,} against "
            f"{result['relu']:,} at the same expansion, an extra "
            f"{result['swiglu'] - result['relu']:,} -- the exact size of the whole attention "
            f"sublayer, {result['attention']:,}. The convention is 8/3 expansion with SwiGLU so "
            "the two match; these settings add 50% to the FFN without saying so",
        ),
        practice.Check(
            "FINDING: the block carries zero normalisation parameters",
            result["norm_params"] == 0,
            "rms_norm(X, eps) and layer_norm(X, eps) both take an input and a constant and no "
            f"learnable scale, so the {2 * D_MODEL:,} gains a standard pre-norm block carries are "
            "absent. The count above is the whole block, and a real one of this shape is larger",
        ),
        practice.Check(
            "CONTROL: the prescribed one-liner cannot run here",
            result["torch"] and not result["has_parameters"],
            "find_spec('torch') is None and BlockParams is a plain object with no parameters() -- "
            "so `sum(p.numel() for p in block.parameters())` has nothing to call. Summing "
            "len(Matrix.data) over its attributes is the equivalent, and doing it is exactly what "
            "makes the dead cross-attention weights visible",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
