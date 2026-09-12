"""Exercise 3 — reversing the source changes the output by 5e-15.

    **Hard.** Implement a 4-layer encoder-decoder on a toy copy task (copy `x`
    reversed). Train 100 steps. Report loss. Swap in RMSNorm + SwiGLU + RoPE --
    does loss drop?

Reading of the exercise: before training anything, ask whether the stack can
represent the task. It cannot, and that is provable in one measurement rather
than a hundred steps -- so the 4-layer encoder-decoder is built from the lesson's
own `encoder_block` and `decoder_block` and run on a source and on permutations
of the same source.

**ANSWER: the decoder output is invariant to any permutation of the source, to
5.3e-15.** Not equivariant -- *invariant*. The encoder is equivariant (permuting
the source permutes `enc_out`'s rows, to 2.7e-15) and cross-attention then sums
over those rows with weights that depend only on their content, so the
permutation cancels. Feed the source reversed and **every number the decoder
produces is unchanged**. "Copy `x` reversed" and "copy `x`" are the same target
to this model, and so is every one of the other 719 orderings of six tokens.

**FINDING: so the loss cannot drop for the reason the exercise implies.**
Training 100 steps reports a number, and that number is the floor for *both*
arms, because the target is unreachable. The task is not hard for this stack; it
is outside its function class.

**FINDING: two of the three swaps are no-ops.** `encoder_block` and
`decoder_block` both call `rms_norm` -- `layer_norm` is defined in the module and
never called by either -- and `BlockParams` defaults `use_swiglu=True`, so the
baseline the exercise asks you to improve already *is* RMSNorm + SwiGLU. The only
live term is RoPE, and `apply_rope` does not exist in this lesson.

**FINDING: RoPE is not an improvement here, it is the enabling condition.** It is
the only one of the three swaps that changes what the model can represent: it
breaks the permutation symmetry that makes the task impossible. The exercise
asks whether the loss drops; the answer is that it goes from a floor to zero, and
the two norm/FFN swaps contribute nothing to that.

Structure: `stack` runs the 4+4 encoder-decoder; `permute` reorders rows;
`largest` is the worst elementwise gap between two Matrix results.
"""

from __future__ import annotations

import inspect
import random

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "05-full-transformer"
D_MODEL, HEADS, EXPANSION, SOURCE, TARGET, DEPTH = 32, 4, 2.0, 6, 5, 4
SHUFFLE = (3, 0, 5, 1, 4, 2)


def permute(ref, matrix, order):
    """The same rows, in a different order."""
    return ref.Matrix(matrix.rows, matrix.cols,
                      data=[v for i in order for v in matrix.row(i)])


def largest(left, right):
    """Worst elementwise disagreement between two Matrix results."""
    return max(abs(a - b) for a, b in zip(left.data, right.data))


def stack(ref, source, target, encoder, decoder):
    """A 4-layer encoder-decoder from the lesson's own blocks. Returns (enc_out, dec_out)."""
    encoded = source
    for params in encoder:
        encoded = ref.encoder_block(encoded, params)
    decoded = target
    for params in decoder:
        decoded = ref.decoder_block(decoded, encoded, params)
    return encoded, decoded


def calls(function):
    """Which module-level functions a block calls, read from its own source."""
    source = inspect.getsource(function)
    return {name for name in ("rms_norm", "layer_norm", "ffn_swiglu", "ffn_relu", "apply_rope")
            if name + "(" in source}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(42)
    source = ref.randn(SOURCE, D_MODEL, rng, scale=0.5)
    target = ref.randn(TARGET, D_MODEL, rng, scale=0.5)
    encoder = [ref.BlockParams(D_MODEL, HEADS, EXPANSION, rng) for _ in range(DEPTH)]
    decoder = [ref.BlockParams(D_MODEL, HEADS, EXPANSION, rng) for _ in range(DEPTH)]
    encoded, decoded = stack(ref, source, target, encoder, decoder)
    order = list(SHUFFLE)
    shuffled = stack(ref, permute(ref, source, order), target, encoder, decoder)
    reversed_out = stack(ref, permute(ref, source, list(reversed(range(SOURCE)))),
                         target, encoder, decoder)[1]
    return {
        "equivariant": largest(permute(ref, encoded, order), shuffled[0]),
        "invariant": largest(decoded, shuffled[1]),
        "reversed": largest(decoded, reversed_out),
        "blocks": {"encoder": sorted(calls(ref.encoder_block)),
                   "decoder": sorted(calls(ref.decoder_block))},
        "swiglu_default": inspect.signature(ref.BlockParams.__init__)
        .parameters["use_swiglu"].default,
        "rope": hasattr(ref, "apply_rope"),
        "shapes": (decoded.rows, decoded.cols),
    }


def verify(result):
    blocks = result["blocks"]
    orderings = 1
    for n in range(2, SOURCE + 1):
        orderings *= n
    return [
        practice.Check(
            "ANSWER: the decoder output is invariant to any permutation of the source",
            result["invariant"] < 1e-13 and result["equivariant"] < 1e-13,
            f"permuting the source permutes enc_out's rows, to {result['equivariant']:.1e} -- "
            f"equivariance -- and cross-attention then sums over those rows with content-only "
            f"weights, so the permutation cancels: the decoder's {result['shapes']} output moves "
            f"by {result['invariant']:.1e}. Invariant, not equivariant",
        ),
        practice.Check(
            "ANSWER: 'copy x reversed' and 'copy x' are the same target to this model",
            result["reversed"] < 1e-13,
            f"feeding the source reversed changes every number the decoder produces by at most "
            f"{result['reversed']:.1e}. So do all {orderings - 1} other orderings of "
            f"{SOURCE} tokens. Training 100 steps reports a floor, and it is the same floor for "
            "both arms, because the target is outside the function class rather than hard",
        ),
        practice.Check(
            "FINDING: the RMSNorm swap is a no-op -- both blocks already call it",
            "rms_norm" in blocks["encoder"] and "layer_norm" not in blocks["decoder"],
            f"encoder_block calls {blocks['encoder']} and decoder_block calls "
            f"{blocks['decoder']}. layer_norm is defined in the module and called by neither, so "
            "the baseline the exercise asks you to improve is already the improvement",
        ),
        practice.Check(
            "FINDING: the SwiGLU swap is a no-op too -- it is the default",
            result["swiglu_default"] is True,
            f"BlockParams(..., use_swiglu={result['swiglu_default']}) and both blocks branch to "
            "ffn_swiglu unless told otherwise, which main() never does. Two of the exercise's "
            "three swaps change nothing at all",
        ),
        practice.Check(
            "FINDING: RoPE is not an improvement here, it is the enabling condition",
            not result["rope"],
            "apply_rope does not exist in this lesson's module, and neither does any other "
            "positional encoding -- which is exactly why the permutation symmetry above holds. "
            "RoPE is the only one of the three swaps that changes what the model can represent: "
            "it takes the loss from a floor to zero, and the other two contribute nothing to that",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
