"""Exercise 2 — a `groups` argument, and what depthwise actually saves.

    **(Medium)** Extend `conv2d_naive` and `conv2d_im2col` to accept a `groups`
    argument. Show that `groups=C_in=C_out` reproduces a depthwise convolution and
    that its parameter count is `C * K * K` instead of `C * C * K * K`.

Reading of the exercise: `groups` slices the channel axis of both the input and the weights, so
the extension calls the lesson's own two kernels once per group rather than rewriting either —
check 1 holds that to `F.conv2d(..., groups=g)` at four settings. Checks 2-3 are the parameter
claim and the arithmetic claim the exercise does not make. Check 4 is what "reproduces a
depthwise convolution" means when a dense conv can produce the same numbers.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
    import torch.nn.functional as functional
except ImportError as exc:                       # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "04-computer-vision", "02-convolutions-from-scratch"
CHANNELS, KERNEL, SIDE, PAD = 6, 3, 12, 1
SETTINGS = (1, 2, 3, 6)


def grouped(numpy, ref, x, w, groups, engine) -> "numpy.ndarray":
    """The exercise's extension: same kernels, one call per slice of the channel axis."""
    per_out, per_in = w.shape[0] // groups, w.shape[1]
    kernel = ref.conv2d_im2col if engine == "im2col" else ref.conv2d_naive
    return numpy.concatenate([kernel(x[g * per_in:(g + 1) * per_in],
                                     w[g * per_out:(g + 1) * per_out], None, 1, PAD)
                              for g in range(groups)], axis=0)


def reference(x, w, groups):
    return functional.conv2d(torch.tensor(x).unsqueeze(0), torch.tensor(w),
                             None, 1, PAD, 1, groups)[0].numpy()


def multiplies(groups) -> int:
    """One im2col matmul: (C_out x C_in/g*K*K) @ (C_in/g*K*K x HW), once per group."""
    return groups * (CHANNELS // groups) * ((CHANNELS // groups) * KERNEL ** 2) * SIDE ** 2


def block_diagonal(numpy, depthwise):
    """The same depthwise filters written as a dense weight with zeros off the diagonal."""
    dense = numpy.zeros((CHANNELS, CHANNELS, KERNEL, KERNEL), dtype=numpy.float32)
    for channel in range(CHANNELS):
        dense[channel, channel] = depthwise[channel, 0]
    return dense


def solve():
    numpy = parity.try_numpy()
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = numpy.random.default_rng(0)
    x = rng.standard_normal((CHANNELS, SIDE, SIDE)).astype(numpy.float32)
    agree, params = {}, {}
    for groups in SETTINGS:
        w = rng.standard_normal((CHANNELS, CHANNELS // groups, KERNEL, KERNEL)).astype(numpy.float32)
        im2col = grouped(numpy, ref, x, w, groups, "im2col")
        agree[groups] = (float(numpy.abs(im2col - grouped(numpy, ref, x, w, groups, "naive")).max()),
                         float(numpy.abs(im2col - reference(x, w, groups)).max()))
        params[groups] = int(w.size)
    depthwise = rng.standard_normal((CHANNELS, 1, KERNEL, KERNEL)).astype(numpy.float32)
    dense = block_diagonal(numpy, depthwise)
    return {"agree": agree, "params": params,
            "cost": {g: multiplies(g) for g in SETTINGS},
            "same": float(numpy.abs(grouped(numpy, ref, x, depthwise, CHANNELS, "im2col")
                                    - ref.conv2d_im2col(x, dense, None, 1, PAD)).max()),
            "zeros": int((dense == 0).sum()), "dense_size": int(dense.size)}


def verify(result):
    agree, params, cost = result["agree"], result["params"], result["cost"]
    return [
        practice.Check("ANSWER: both kernels take `groups` and both match torch at every setting",
                       max(max(pair) for pair in agree.values()) < 1e-5,
                       "worst |difference| per setting, im2col against naive and against "
                       "`F.conv2d(..., groups=g)` — "
                       + "; ".join(f"g={g}: {a:.1e} / {b:.1e}" for g, (a, b) in agree.items())
                       + f". Slicing the channel axis is the whole extension: neither of the "
                       f"lesson's kernels changes"),
        practice.Check("ANSWER: at groups = C the parameter count is C*K*K, not C*C*K*K",
                       params[CHANNELS] == CHANNELS * KERNEL ** 2
                       and params[1] == CHANNELS ** 2 * KERNEL ** 2,
                       f"weights per setting: "
                       + ", ".join(f"g={g}: {n}" for g, n in params.items())
                       + f" — {params[1]} = C*C*K*K dense against {params[CHANNELS]} = C*K*K "
                       f"depthwise, a factor of C = {CHANNELS}. Each group only ever sees "
                       f"C_in/g input channels, so the weight tensor loses that axis"),
        practice.Check("MECHANISM: the arithmetic falls by the same factor, which the exercise "
                       "does not say",
                       cost[1] == CHANNELS * cost[CHANNELS],
                       "multiply-accumulates in the im2col matmul — "
                       + ", ".join(f"g={g}: {n:,}" for g, n in cost.items())
                       + f". A group's matmul is (C/g x C/g*K*K) @ (C/g*K*K x HW) and there are g "
                       f"of them, so the total scales as 1/g. Parameters and FLOPs both drop by "
                       f"{CHANNELS}x here — the second is why depthwise is on phones"),
        practice.Check("CONTROL: 'reproduces a depthwise convolution' is about cost, not output",
                       result["same"] < 1e-5,
                       f"the same depthwise filters written into a dense C x C weight with zeros "
                       f"off the diagonal ({result['zeros']} of {result['dense_size']} entries "
                       f"zero) give an identical result — worst difference {result['same']:.1e}. "
                       f"A dense conv can produce every depthwise output; what `groups` changes "
                       f"is that it stops storing and multiplying the zeros"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
