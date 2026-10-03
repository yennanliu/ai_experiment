"""Exercise 5 — at the lesson's breakpoint the gradient on screen is stale.

    Use `breakpoint()` inside a training loop. Practice inspecting tensor shapes,
    devices, and gradient values from the debugger prompt.

Reading of the exercise: the loop is the lesson's Part 2 `training_step` as
written -- forward, loss, `if loss.item() > 100 or torch.isnan(loss):
breakpoint()`, `loss.backward()`, `optimizer.step()` -- on an 8-to-1 linear
regression in numpy, batch 16, 5 steps, with one NaN put into sample 2 of step 3.
`backward` accumulates into `.grad` as torch does, and like the lesson's
snippet nothing zeroes it. The debugger is real `pdb`, fed a fixed command
script instead of a keyboard so the session is reproducible; the commands
include the lesson's own `debug_print` and `check_gradient_health`.

**ANSWER: the prompt shows what the exercise asks for.** At the first stop (step
3): `inputs.shape` (16, 8), device 'cpu', 1 NaN in `outputs`, located at
`inputs[2][4]`, and `debug_print` reports has_nan=True.

**FINDING: the gradient you inspect there is not this loss's gradient.** The
lesson puts `breakpoint()` before `loss.backward()`, so `.grad` still holds
earlier steps' gradients: finite (norm 15.17 by `check_gradient_health`) while the
loss is nan, whereas this step's own gradient, computed after `c`, is nan in 8 of
8 entries. With no `zero_grad` in the snippet, and torch's `+=` into `.grad`,
it is the sum of steps 0-2's gradients, not even the last one. On a stop at step 0 it would
be None.

**FINDING: one bad sample stops the loop on every later step.** The NaN reaches
the weights through `optimizer.step()`, so steps 3 and 4 both stop: 2 stops for
1 bad sample, and on a long run `c` never gets past it.

**CONTROL: the lesson's two thresholds disagree, and here it matters.** The
prose stops at `loss.item() > 100`, `debug_tools.py`'s printed pattern at `> 10`.
The clean steps' losses are 17.6, 7.1 and 12.7, so the `> 10` version would also
stop on 2 healthy steps; with the prose's 100 only the NaN stops the run, and
after the scripted `c` the loop reaches its last step.

Structure: `Param`/`Tensor` stand in for torch's; `scripted_pdb` replaces
`sys.breakpointhook`; `loop` is the lesson's training step.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import pdb
import re
import sys
import types

import numpy as np

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "12-debugging-and-profiling"
BAD_STEP, STEPS = 3, 5
COMMANDS = [
    "p inputs.shape",
    "p inputs.device",
    "p int(np.isnan(outputs).sum())",
    "p np.argwhere(np.isnan(inputs)).tolist()",
    "!dt.debug_print('outputs', outputs)",
    "!dt.check_gradient_health(model)",
    "c",
]


class Tensor(np.ndarray):
    device = "cpu"

    @property
    def data(self):
        return self

    def norm(self, p=2):
        return np.linalg.norm(np.asarray(self).ravel(), p)

    def isnan(self):
        return np.isnan(self)


def scripted_pdb(stops):
    def hook(*_args, **_kwargs):
        frame = sys._getframe(1)
        stops.append({"step": frame.f_locals["step"],
                      "grad": frame.f_locals["model"].weight.grad.copy()})
        debugger = pdb.Pdb(stdin=io.StringIO("\n".join(COMMANDS) + "\n"),
                           stdout=sys.stdout, nosigint=True)
        debugger.use_rawinput = False
        debugger.set_trace(frame)
    return hook


def loop(dt, history, grads):
    rng = np.random.default_rng(0)
    weight = types.SimpleNamespace(value=rng.normal(size=(8, 1)), grad=None)
    model = types.SimpleNamespace(weight=weight, named_parameters=lambda: [("weight", weight)])
    for step in range(STEPS):
        inputs = rng.normal(size=(16, 8)).view(Tensor)
        if step == BAD_STEP:
            inputs[2, 4] = np.nan
        outputs = inputs @ weight.value + 3.0
        loss = float(np.mean(outputs**2))
        history.append(loss)
        if loss > 100 or np.isnan(loss):
            breakpoint()
        grads.append((2 * inputs.T @ outputs / 16).view(Tensor))  # loss.backward() ...
        weight.grad = grads[-1] if weight.grad is None else weight.grad + grads[-1]  # ... adds
        weight.value = weight.value - 0.01 * weight.grad  # optimizer.step()
    return step, dt, model


def thresholds(dt):
    pattern = r"loss\.item\(\) > (\d+)"
    block = parity.doc_text(PHASE, LESSON).split("def training_step")[1].split("```")[0]
    code = inspect.getsource(dt.demo_conditional_breakpoint)
    return re.findall(pattern, block), re.findall(pattern, code), "zero_grad" in block


def solve():
    dt = parity.load_reference(PHASE, LESSON, "debug_tools")
    stops, history, grads, out = [], [], [], io.StringIO()
    saved, sys.breakpointhook = sys.breakpointhook, scripted_pdb(stops)
    try:
        with contextlib.redirect_stdout(out):
            last, *_ = loop(dt, history, grads)
    finally:
        sys.breakpointhook = saved
    replies = [r.strip() for r in out.getvalue().split("(Pdb) ")[1:len(COMMANDS)]]
    norm = re.search(r"Total gradient norm: (\S+)", replies[5])
    return {
        "last": last, "stops": [s["step"] for s in stops], "replies": replies,
        "norm": float(norm.group(1)) if norm else float("nan"),
        "clean": [h for h in history if np.isfinite(h)],
        "stale_gap": float(np.abs(stops[0]["grad"] - sum(grads[:BAD_STEP])).max()),
        "own_nan": int(np.isnan(grads[BAD_STEP]).sum()),
        "thresholds": thresholds(dt),
    }


def verify(result):
    r, clean = result["replies"], result["clean"]
    doc_t, code_t, zero_grad = result["thresholds"]
    loud, listed = sum(h > 10 for h in clean), ", ".join(f"{h:.1f}" for h in clean)
    return [
        practice.Check(
            "ANSWER: the prompt shows shape, device, the NaN count and where the NaN is",
            r[:4] == ["(16, 8)", "'cpu'", "1", "[[2, 4]]"] and "has_nan=True" in r[4],
            f"step {result['stops'][0]}: shape {r[0]}, device {r[1]}, {r[2]} NaN in outputs "
            f"at inputs{r[3]}; debug_print: {r[4].split('has_nan=')[-1]}",
        ),
        practice.Check(
            "FINDING: .grad at the breakpoint is the stale sum of steps 0-2, finite",
            np.isfinite(result["norm"]) and result["own_nan"] > 0
            and result["stale_gap"] < 1e-12 and not zero_grad,
            f"check_gradient_health at the stop: norm {result['norm']:.2f} while the loss is "
            f"nan and this step's own gradient has {result['own_nan']} of 8 entries nan; "
            f".grad - sum(steps 0..2 gradients) = {result['stale_gap']:.1e}; "
            f"zero_grad in the lesson's training_step: {zero_grad}",
        ),
        practice.Check(
            "FINDING: one bad sample stops the loop on every later step",
            result["stops"] == [3, 4],
            f"stops at steps {result['stops']} for one NaN at step {BAD_STEP}",
        ),
        practice.Check(
            "CONTROL: the lesson's two thresholds disagree; only the NaN stopped this run",
            (doc_t, code_t, result["last"], loud > 0) == (["100"], ["10"], STEPS - 1, True)
            and max(clean) < 100,
            f"thresholds: prose {doc_t}, debug_tools.py {code_t}; clean losses {listed} -- "
            f"the > 10 version would also stop on {loud} clean step(s); after `c` the loop "
            f"reaches step {result['last']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
