"""Exercise 5 — orthogonal initialization, against Kaiming at 50 layers.

    Implement orthogonal initialization (generate a random matrix, compute its
    SVD, use the orthogonal matrix U). Compare to Kaiming for ReLU networks at 50
    layers.

Reading of the exercise: the comparison runs through the lesson's own `forward_deep` at its own
50 layers and width 64, so only the init function changes. "Use the orthogonal matrix U" is
taken literally in check 4, where it turns out to be the wrong matrix for any layer that is not
square; everywhere else the factor U @ Vt is used, which is what the recipe means. Checks 2-3
separate what orthogonality gives from what the ReLU takes away.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "08-weight-initialization"
WIDTH, LAYERS, SAMPLES = 64, 50, 20
GAIN = math.sqrt(2.0)                 # the ReLU correction, the same factor Kaiming carries


def orthogonal(numpy, fan_in, fan_out, gain=1.0) -> list:
    """A random matrix replaced by the nearest orthogonal one: U @ Vt from its SVD."""
    draw = [[random.gauss(0, 1) for _ in range(fan_in)] for _ in range(fan_out)]
    u, _s, vt = numpy.linalg.svd(numpy.array(draw), full_matrices=False)
    return (gain * (u @ vt)).tolist()


def arms(ref, numpy) -> dict:
    return {"kaiming": (ref.kaiming_init, ref.relu),
            "orth1": (lambda a, b: orthogonal(numpy, a, b), ref.relu),
            "orth2": (lambda a, b: orthogonal(numpy, a, b, GAIN), ref.relu),
            "orth_linear": (lambda a, b: orthogonal(numpy, a, b), lambda z: z),
            "kaiming_linear": (ref.kaiming_init, lambda z: z)}


def trace(ref, init, act) -> dict:
    mags = ref.forward_deep(init, act, n_layers=LAYERS, width=WIDTH, n_samples=SAMPLES)
    gains = [mags[i + 1] / mags[i] for i in range(LAYERS - 1)]
    return {"first": mags[0], "last": mags[-1], "gain": statistics.mean(gains),
            "spread": statistics.pstdev(gains)}


def orthonormal(numpy) -> tuple:
    """How orthogonal the factor really is, and whether it preserves a vector's length."""
    random.seed(1)
    q = numpy.array(orthogonal(numpy, WIDTH, WIDTH))
    v = numpy.array([random.gauss(0, 1) for _ in range(WIDTH)])
    return (float(numpy.abs(q.T @ q - numpy.eye(WIDTH)).max()),
            float(abs(numpy.linalg.norm(q @ v) - numpy.linalg.norm(v))))


def literal_u(numpy, fan_in, fan_out) -> tuple:
    """The shape 'use the orthogonal matrix U' actually returns, against the one needed."""
    draw = [[random.gauss(0, 1) for _ in range(fan_in)] for _ in range(fan_out)]
    u, _s, vt = numpy.linalg.svd(numpy.array(draw), full_matrices=False)
    return u.shape, (u @ vt).shape, (fan_out, fan_in)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    numpy = parity.try_numpy()
    exact, kept = orthonormal(numpy)
    return {"runs": {name: trace(ref, init, act) for name, (init, act) in arms(ref, numpy).items()},
            "exact": exact, "kept": kept,
            "shapes": {f"{fi}->{fo}": literal_u(numpy, fi, fo)
                       for fi, fo in ((WIDTH, WIDTH), (16, WIDTH), (WIDTH, 16))}}


def verify(result):
    runs, shapes = result["runs"], result["shapes"]
    kai, one, two = runs["kaiming"], runs["orth1"], runs["orth2"]
    lin, kai_lin = runs["orth_linear"], runs["kaiming_linear"]
    wrong = shapes[f"{WIDTH}->16"]
    return [
        practice.Check("ANSWER: orthogonal at the usual gain of 1 vanishes where Kaiming holds, "
                       "and beats it once given ReLU's factor",
                       one["last"] < 1e-6 < kai["last"] and two["last"] > kai["last"],
                       f"magnitude at layer {LAYERS}, width {WIDTH}: orthogonal(gain=1) "
                       f"{one['last']:.3e} at a per-layer gain of {one['gain']:.4f}, Kaiming "
                       f"{kai['last']:.4f} at {kai['gain']:.4f}, orthogonal(gain=sqrt(2)) "
                       f"{two['last']:.4f} at {two['gain']:.4f}. 0.718 is 1/sqrt(2) — the half a "
                       f"ReLU removes, which an orthogonal matrix does nothing to replace"),
        practice.Check("MECHANISM: the factor is orthonormal to 2e-15, so a linear stack of 50 "
                       "does not move at all",
                       result["exact"] < 1e-13 and abs(lin["gain"] - 1.0) < 0.01,
                       f"||Q^T Q - I||_max is {result['exact']:.1e} and ||Qv|| - ||v|| is "
                       f"{result['kept']:.1e}, so each layer preserves length exactly. With the "
                       f"activation removed, {LAYERS} such layers take the magnitude "
                       f"{lin['first']:.4f} -> {lin['last']:.4f} — a per-layer gain of "
                       f"{lin['gain']:.4f} +/- {lin['spread']:.4f}. Every bit of the decay above "
                       f"is the ReLU"),
        practice.Check("FINDING: Kaiming's sqrt(2) is a ReLU correction, not a general one",
                       kai_lin["last"] > 1e6 and abs(kai_lin["gain"] - GAIN) < 0.05,
                       f"the same Kaiming stack with the activation removed has a per-layer gain "
                       f"of {kai_lin['gain']:.4f} — sqrt(2) to {abs(kai_lin['gain'] - GAIN):.3f} — "
                       f"and reaches {kai_lin['last']:.3e} by layer {LAYERS}. Its variance 2/fan_in "
                       f"is calibrated to a nonlinearity that throws half of it away"),
        practice.Check("FINDING: 'use the orthogonal matrix U' is the wrong matrix off the "
                       "diagonal",
                       wrong[0] != wrong[2] and wrong[1] == wrong[2],
                       f"for a {WIDTH} -> 16 layer `svd(A, full_matrices=False)` returns U of shape "
                       f"{wrong[0]} where the weight matrix must be {wrong[2]}; U @ Vt is "
                       f"{wrong[1]}. The recipe reads correctly only because `forward_deep` is "
                       f"square throughout — at {16}->{WIDTH} U is {shapes[f'16->{WIDTH}'][0]}, "
                       f"which is the right shape but only column-orthonormal"),
        practice.Check("CONTROL: orthogonal fixes the scale exactly, Kaiming draws it",
                       lin["spread"] < 0.3 * kai_lin["spread"],
                       f"per-layer gain across the {LAYERS - 1} transitions, activation removed: "
                       f"{lin['gain']:.4f} +/- {lin['spread']:.4f} orthogonal against "
                       f"{kai_lin['gain']:.4f} +/- {kai_lin['spread']:.4f} Kaiming — "
                       f"{kai_lin['spread'] / lin['spread']:.1f}x tighter. Under ReLU the masking "
                       f"dominates and the two spreads converge, {two['spread']:.4f} against "
                       f"{kai['spread']:.4f}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
