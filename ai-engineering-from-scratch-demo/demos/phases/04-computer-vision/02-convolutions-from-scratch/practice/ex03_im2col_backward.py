"""Exercise 3 — the backward pass of `conv2d_im2col`, and the accumulate the trick turns on.

    **(Hard)** Implement the backward pass of `conv2d_im2col` by hand: given the
    gradient of the output, compute the gradient of `x` and `w`. Verify against
    `torch.autograd.grad` on the same inputs and weights. The trick: the gradient
    of im2col is `col2im`, and it has to accumulate overlapping windows.

Reading of the exercise: forward is one matmul on the lesson's own `im2col`, so backward is the
two matmuls that transpose it plus `col2im` — check 1 verifies both gradients against
`torch.autograd.grad` at two settings. The stated trick is a claim with a failure mode, so
check 2 runs the same code with `=` in place of `+=` and check 3 finds the stride at which that
mistake is invisible. Check 4 is why `col2im` is not the inverse of `im2col`.
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
C_IN, C_OUT, KERNEL, SIDE = 3, 4, 3, 9
SETTINGS = ((1, 1), (3, 0))                                  # (stride, padding)


def col2im(numpy, ref, cols, shape, stride, padding, accumulate=True):
    """The adjoint of `im2col`: every window writes back where it was read from."""
    c_in, height, width = shape
    canvas = numpy.zeros((c_in, height + 2 * padding, width + 2 * padding), dtype=numpy.float32)
    h_out = ref.output_size(height, KERNEL, padding, stride)
    w_out = ref.output_size(width, KERNEL, padding, stride)
    for column, (i, j) in enumerate((i, j) for i in range(h_out) for j in range(w_out)):
        patch = cols[:, column].reshape(c_in, KERNEL, KERNEL)
        window = (slice(None), slice(i * stride, i * stride + KERNEL),
                  slice(j * stride, j * stride + KERNEL))
        if accumulate:
            canvas[window] += patch
        else:
            canvas[window] = patch
    return canvas[:, padding:padding + height, padding:padding + width] if padding else canvas


def backward(numpy, ref, x, w, d_out, stride, padding, accumulate=True) -> tuple:
    cols, _h, _w = ref.im2col(x, KERNEL, KERNEL, stride, padding)
    flat = d_out.reshape(C_OUT, -1)
    d_w = (flat @ cols.T).reshape(w.shape)
    d_cols = w.reshape(C_OUT, -1).T @ flat
    return col2im(numpy, ref, d_cols, x.shape, stride, padding, accumulate), d_w


def autograd(x, w, stride, padding, seed) -> tuple:
    xt = torch.tensor(x, requires_grad=True)
    wt = torch.tensor(w, requires_grad=True)
    out = functional.conv2d(xt.unsqueeze(0), wt, None, stride, padding)
    torch.manual_seed(seed)
    upstream = torch.randn_like(out)
    grads = torch.autograd.grad(out, (xt, wt), upstream)
    return upstream[0].numpy(), grads[0].numpy(), grads[1].numpy()


def solve():
    numpy = parity.try_numpy()
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = numpy.random.default_rng(0)
    x = rng.standard_normal((C_IN, SIDE, SIDE)).astype(numpy.float32)
    w = rng.standard_normal((C_OUT, C_IN, KERNEL, KERNEL)).astype(numpy.float32)
    out = {}
    for stride, padding in SETTINGS:
        upstream, gx, gw = autograd(x, w, stride, padding, seed=stride)
        mine_x, mine_w = backward(numpy, ref, x, w, upstream, stride, padding)
        wrong_x, _ = backward(numpy, ref, x, w, upstream, stride, padding, accumulate=False)
        out[(stride, padding)] = (float(numpy.abs(mine_x - gx).max()),
                                  float(numpy.abs(mine_w - gw).max()),
                                  float(numpy.abs(wrong_x - gx).max()))
    ones = numpy.ones((C_IN, SIDE, SIDE), dtype=numpy.float32)
    cols, _h, _w = ref.im2col(ones, KERNEL, KERNEL, 1, 1)
    counts = col2im(numpy, ref, cols, ones.shape, 1, 1)
    return {"runs": out, "touch": (float(counts.min()), float(counts.max())),
            "cols": int(cols.size), "pixels": int(ones.size)}


def verify(result):
    runs, touch = result["runs"], result["touch"]
    overlap, disjoint = runs[SETTINGS[0]], runs[SETTINGS[1]]
    return [
        practice.Check("ANSWER: both gradients match `torch.autograd.grad`",
                       max(overlap[0], overlap[1], disjoint[0], disjoint[1]) < 1e-5,
                       "worst |difference| against autograd, dx / dw — "
                       + "; ".join(f"stride {s} pad {p}: {runs[(s, p)][0]:.1e} / "
                                   f"{runs[(s, p)][1]:.1e}" for s, p in SETTINGS)
                       + ". Forward is `w_flat @ cols`, so backward is `d @ cols.T` for the "
                       "weights and `col2im(w_flat.T @ d)` for the input — the two transposes of "
                       "one matmul, plus the adjoint of the lowering"),
        practice.Check("FINDING: swap the `+=` for `=` and the input gradient is simply wrong",
                       overlap[2] > 1.0,
                       f"the identical code with `canvas[window] = patch` instead of `+=` misses "
                       f"the true dx by {overlap[2]:.1f} at stride {SETTINGS[0][0]}. Every input "
                       f"pixel is read by several windows, so its gradient is a *sum* of their "
                       f"contributions; assignment keeps only whichever window wrote last"),
        practice.Check("CONTROL: at stride = kernel the same mistake is invisible",
                       disjoint[2] == 0.0,
                       f"at stride {SETTINGS[1][0]} with a {KERNEL}x{KERNEL} kernel and no "
                       f"padding the windows tile without overlapping, and `=` scores "
                       f"{disjoint[2]:.1f} — exactly right. A test written only at that setting "
                       f"would pass the broken implementation, which is why the exercise names "
                       f"the overlap and not the accumulation"),
        practice.Check("MECHANISM: `col2im` is an adjoint, not an inverse",
                       touch == (((KERNEL + 1) // 2) ** 2, float(KERNEL ** 2)),
                       f"lowering a field of ones and folding it straight back counts how many "
                       f"windows read each pixel: {touch[1]:.0f} = K^2 in the interior, falling "
                       f"to {touch[0]:.0f} at a corner, where half the windows in each direction "
                       f"start outside the padded canvas. `im2col` writes "
                       f"{result['cols']} numbers from {result['pixels']} pixels — it is not "
                       f"invertible, and `col2im` returns the transpose of that map rather than "
                       f"undoing it"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
