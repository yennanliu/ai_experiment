"""Exercise 1 — a divide-by-zero before tanh is caught one step late, and only by luck.

    Run `debug_tools.py` and read through each section's output. Modify the dummy
    model to introduce a NaN (hint: divide by zero in the forward pass) and watch the
    detector catch it.

Reading of the exercise: torch is not installed here (nor in CI), so the lesson's
`main()` is run as it is and the dummy model is rebuilt in numpy: the 784-256-10
MLP of `demo_nan_detection`, batch 4, cross-entropy, hand-written backward with
torch's divide rule (`grad_in = grad_out / 0`). Its parameters are fed to the
lesson's own `detect_nan` and `check_gradient_health`, which only duck-type their
arguments; `torch.isnan` is pointed at `numpy.isnan`. "Divide by zero in the
forward pass" is tried in two places: after the ReLU, and before a tanh.

**ANSWER: after the ReLU the detector catches it at step 0.** Loss is nan, and
`detect_nan` names all 4 parameters as NaN gradients. ReLU emits exact zeros,
and `0/0` is nan.

**FINDING: before a tanh it is not caught.** `tanh(z/0)` is `tanh(+-inf) = +-1`,
finite, so the loss is a finite 3.206; the backward pass is `0 * inf = nan` in
`0.weight` and `0.bias`. `detect_nan` returns False because it only looks at the
gradients once the loss is already nan, and `check_gradient_health` prints no
warning at all: `nan > 100` and `nan == 0` are both False, so the one check
that sees the gradients prints "Total gradient norm: nan" and nothing else. The
SGD step writes the nan into the weights and the detector fires at step 1, after
the model is already destroyed.

**FINDING: without torch, `debug_tools.py` runs 2 of its 10 sections and exits
1.** Only memory tracking and logging run; the NaN detector never runs on a
machine without torch.

**CONTROL:** with the divisor at 1 both models are healthy: finite losses 3.408
and 2.663, `detect_nan` False, finite gradient norms.

Structure: `Tensor` gives numpy arrays the three torch methods the lesson calls;
`step` is forward + backward; `inspect` runs the lesson's two detectors.
"""

from __future__ import annotations

import contextlib
import io
import logging
import types

import numpy as np

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "12-debugging-and-profiling"
NAMES = ("0.weight", "0.bias", "2.weight", "2.bias")


class Tensor(np.ndarray):
    """A numpy array with the torch spellings `debug_tools` uses."""

    @property
    def data(self):
        return self

    def norm(self, p=2):
        return np.linalg.norm(np.asarray(self).ravel(), p)


def make_model(act, seed=1):
    rng = np.random.default_rng(seed)
    shapes = ((784, 256), (256,), (256, 10), (10,))
    params = {n: types.SimpleNamespace(value=rng.normal(0, 0.05, s) if len(s) == 2
                                       else np.zeros(s), grad=None)
              for n, s in zip(NAMES, shapes)}
    return types.SimpleNamespace(act=act, p=params, named_parameters=lambda: params.items())


def step(model, x, y, div, lr=0.01):
    """One forward/backward/SGD step with `div` in the forward pass. Returns the loss."""
    w1, b1, w2, b2 = (model.p[n].value for n in NAMES)
    rows = np.arange(len(y))
    with np.errstate(all="ignore"):
        z = x @ w1 + b1
        h = np.maximum(z, 0) / div if model.act == "relu" else np.tanh(z / div)
        logits = h @ w2 + b2
        top = logits.max(1, keepdims=True)
        lse = top + np.log(np.exp(logits - top).sum(1, keepdims=True))
        loss = np.mean(lse[:, 0] - logits[rows, y])
        g = np.exp(logits - lse)
        g[rows, y] -= 1
        g /= len(y)
        dh = g @ w2.T
        dz = (dh / div) * (z > 0) if model.act == "relu" else dh * (1 - h**2) / div
        grads = (x.T @ dz, dz.sum(0), h.T @ g, g.sum(0))
    for name, grad in zip(NAMES, grads):
        model.p[name].grad = grad.view(Tensor)
        model.p[name].value = model.p[name].value - lr * grad
    return np.float64(loss)


def inspect(ref, model, loss, at):
    """The lesson's two detectors on one step, reduced to what they saw and said."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        caught = ref.detect_nan(model, loss, at)
        norm = ref.check_gradient_health(model)
    said = out.getvalue()
    return {"loss": float(loss), "caught": bool(caught), "norm": norm,
            "bad": [n for n, p in model.p.items() if np.isnan(p.grad).any()],
            "named": said.count("NaN gradient"), "warned": "WARNING" in said,
            "finite": bool(np.isfinite(loss) and np.isfinite(norm))}


def run(ref, act, div):
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(4, 784)), rng.integers(0, 10, 4)
    model = make_model(act)
    first = inspect(ref, model, step(model, x, y, div), 0)
    second = inspect(ref, model, step(model, x, y, div), 1)
    return first, second


def solve():
    ref = parity.load_reference(PHASE, LESSON, "debug_tools")
    ref.torch = types.SimpleNamespace(isnan=np.isnan, isinf=np.isinf)
    ref.HAS_TORCH, out = False, io.StringIO()
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(out):
            code = ref.main()
    finally:
        logging.disable(logging.NOTSET)
    sections = [line.strip("- ") for line in out.getvalue().splitlines() if line[:4] == "--- "]
    runs = {(a, d): run(ref, a, d) for a in ("relu", "tanh") for d in (0.0, 1.0)}
    return {"main": (code, sections), "runs": runs}


def verify(result):
    runs = result["runs"]
    relu, (tanh, late) = runs[("relu", 0.0)][0], runs[("tanh", 0.0)]
    code, sections = result["main"]
    healthy = [runs[(a, 1.0)][0] for a in ("relu", "tanh")]
    silent = (tanh["caught"], tanh["warned"], late["caught"]) == (False, False, True)
    return [
        practice.Check(
            "ANSWER: dividing the ReLU output by zero is caught at step 0",
            relu["caught"] and relu["named"] == 4,
            f"loss {relu['loss']}; detect_nan returned {relu['caught']} and named "
            f"{relu['named']} of 4 parameters (ReLU zeros give 0/0)",
        ),
        practice.Check(
            "FINDING: before a tanh the loss stays finite and both detectors stay silent",
            silent and np.isfinite(tanh["loss"]) and tanh["bad"] == ["0.weight", "0.bias"],
            f"tanh(z/0) = +-1: loss {tanh['loss']:.3f} with nan in {tanh['bad']}, detect_nan "
            f"{tanh['caught']}, gradient norm {tanh['norm']} with no warning (nan > 100 and "
            f"nan == 0 are False); caught only at step 1 (loss {late['loss']}), after SGD "
            "wrote the nan into the weights",
        ),
        practice.Check(
            "FINDING: without torch, debug_tools.py runs 2 of 10 sections and exits 1",
            (code, len(sections)) == (1, 2),
            f"main() returned {code} after {len(sections)} sections: " + "; ".join(sections),
        ),
        practice.Check(
            "CONTROL: with the divisor at 1 both models are healthy",
            [(h["finite"], h["caught"]) for h in healthy] == [(True, False)] * 2,
            "losses " + ", ".join(f"{h['loss']:.3f}" for h in healthy)
            + "; gradient norms " + ", ".join(f"{h['norm']:.2f}" for h in healthy),
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
