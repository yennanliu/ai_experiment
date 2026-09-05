"""Exercise 5 — a transposed weight in backward, and where gradient checking says it is.

    **Debug a real failure.** Take the mini-framework from Lesson 10, introduce a
    subtle bug (e.g., transpose the weight matrix in backward), and use gradient
    checking to locate exactly which parameter has incorrect gradients. Document
    the debugging process.

Reading of the exercise: its two halves do not compose — check 1 runs the lesson's own
`gradient_check` on the lesson-10 framework and reports what happens — so the checker here is
written to the same formula (central differences, relative difference, a 1e-5 threshold) over
lesson 10's own `parameters()`. The bug is the one the exercise names. Checks 3-5 are the
debugging process: what the check points at, why the bug is insertable at all, and what the
lesson's own sampling rule would have missed.
"""

from __future__ import annotations

import random

from harness import parity, practice

try:
    import torch                                  # noqa: F401 - check 1 imports this lesson
except ImportError as exc:                        # pragma: no cover - which needs torch itself
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "03-deep-learning-core", "13-debugging-neural-networks"
SIZES, EPS, TOL, SAMPLED, LATE = (4, 16, 16, 1), 1e-5, 1e-5, 5, 100
ROW = [0.5, -0.3, 0.9, 0.1]


def transposed_backward(self, grad) -> list:
    """The lesson-10 `Linear.backward` with `weights[i][j]` transposed on one line."""
    input_grad = [0.0] * self.fan_in
    for i in range(self.fan_out):
        self.bias_grads[i] += grad[i]
        for j in range(self.fan_in):
            self.weight_grads[i][j] += grad[i] * self.input[j]
            input_grad[j] += grad[i] * self.weights[j][i]        # the bug: [j][i], not [i][j]
    return input_grad


def build(mf, bug=None, seed=0):
    """The lesson-10 stack; `bug` names the module index whose backward is replaced."""
    random.seed(seed)
    model = mf.Sequential(mf.Linear(SIZES[0], SIZES[1]), mf.ReLU(),
                          mf.Linear(SIZES[1], SIZES[2]), mf.ReLU(),
                          mf.Linear(SIZES[2], SIZES[3]), mf.Sigmoid())
    if bug is not None:
        model.modules[bug].backward = transposed_backward.__get__(model.modules[bug])
    return model


def slot(entry) -> tuple:
    container, i, j, grads = entry
    return (container[i], grads[i], j) if j is not None else (container, grads, i)


def analytic(mf, model, target) -> list:
    """One backward pass from a cleared state — the gradients the framework claims."""
    crit = mf.BCELoss()
    for entry in model.parameters():
        slot(entry)[1][slot(entry)[2]] = 0.0
    crit(model.forward(ROW), target)
    model.backward(crit.backward())
    return [slot(e)[1][slot(e)[2]] for e in model.parameters()]


def relative(mf, model, target) -> list:
    """The lesson's own formula, over every parameter rather than the first five."""
    crit, out = mf.BCELoss(), []
    for entry, theirs in zip(model.parameters(), analytic(mf, model, target)):
        weights, _g, index = slot(entry)
        start = weights[index]
        weights[index] = start + EPS
        mine = crit(model.forward(ROW), target)
        weights[index] = start - EPS
        mine = (mine - crit(model.forward(ROW), target)) / (2 * EPS)
        weights[index] = start
        out.append(abs(mine - theirs) / max(abs(mine), abs(theirs), 1e-8))
    return out


def by_module(model, diffs) -> dict:
    """The per-parameter differences regrouped the way `gradient_check` prints them."""
    out, start = {}, 0
    for index, module in enumerate(model.modules):
        width = len(module.parameters())
        if width:
            out[index] = (max(diffs[start:start + width]), diffs[start:start + width])
            start += width
    return out


def their_checker(mf, model) -> str:
    """What the lesson's own `gradient_check` does when handed a lesson-10 model."""
    dbg = parity.load_reference(PHASE, LESSON, "debug_neural_nets")
    try:
        dbg.gradient_check(model, ROW, [1.0], mf.BCELoss())
    except AttributeError as exc:
        return " ".join(str(exc).split())[:64]
    return "no error"


def solve():
    mf = parity.load_reference("03-deep-learning-core", "10-mini-framework", "main")
    target, square = [1.0], 2
    clean, bugged = build(mf), build(mf, bug=square)
    diffs = {"clean": relative(mf, clean, target), "bugged": relative(mf, bugged, target)}
    flat = build(mf, bug=0)
    try:
        analytic(mf, flat, target)
        raised = "no error"
    except IndexError as exc:
        raised = " ".join(str(exc).split())[:56]
    return {"per": {k: by_module(clean, v) for k, v in diffs.items()},
            "worst": {k: max(v) for k, v in diffs.items()}, "n": len(diffs["clean"]),
            "checker": their_checker(mf, clean), "raised": raised, "square": square}


def verify(result):
    per, worst = result["per"], result["worst"]
    bugged, clean = per["bugged"], per["clean"]
    late = max(bugged[0][1][SAMPLED:LATE])
    return [
        practice.Check("FINDING: the exercise's two halves do not compose",
                       "double" in result["checker"] or "named_parameters" in result["checker"],
                       f"this lesson's `gradient_check` on the lesson-10 framework raises "
                       f"AttributeError {result['checker']!r} — it calls `x.double()` and "
                       f"`model.named_parameters()`, and lesson 10 has neither. The checker below "
                       f"is the same formula over all {result['n']} parameters"),
        practice.Check("ANSWER: the check separates the two models by eight orders of magnitude",
                       worst["clean"] < TOL < worst["bugged"],
                       f"worst relative difference over all {result['n']} parameters: "
                       f"{worst['clean']:.3e} clean against {worst['bugged']:.3e} with one line of "
                       f"module {result['square']}'s backward transposed — the lesson's own 1e-5 "
                       f"threshold calls the first OK and the second a MISMATCH"),
        practice.Check("FINDING: it names the layer before the bug, not the layer with it",
                       bugged[result["square"]][0] < TOL < bugged[0][0],
                       "per module, worst relative difference: "
                       + ", ".join(f"module {i} {v[0]:.2e}" for i, v in bugged.items())
                       + f". The transpose is in module {result['square']}, whose *own* weight "
                       f"gradients are still right — it corrupts `input_grad`, the next layer "
                       f"down's error. Gradient checking localises the symptom, not the cause"),
        practice.Check("MECHANISM: the bug is only insertable where the layer is square",
                       "range" in result["raised"] or "index" in result["raised"],
                       f"`self.weights[j][i]` needs a row j, so on the {SIZES[0]}->{SIZES[1]} "
                       f"layer it raises {result['raised']!r} on the first backward pass. Only "
                       f"the {SIZES[1]}->{SIZES[2]} layer accepts it in silence — a transpose is "
                       f"a no-op on shape exactly where it is undetectable by shape"),
        practice.Check("CONTROL: the lesson's own sampling rule would have missed it",
                       max(bugged[0][1][:SAMPLED]) > TOL and late > TOL and clean[0][0] < TOL,
                       f"`gradient_check` tests `min({SAMPLED}, param.numel())` entries per "
                       f"parameter, in flat order. The first {SAMPLED} of module 0 already "
                       f"disagree ({max(bugged[0][1][:SAMPLED]):.2e}) so it is caught — but the "
                       f"entries at {SAMPLED}..{LATE} disagree just as much ({late:.2e}), and a "
                       f"bug confined to those reports OK: {SAMPLED} of {len(bugged[0][1])}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
