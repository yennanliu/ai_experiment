"""Exercise 3 — affine alignment invents the delta.

    **(Hard)** Take five pairs of images that differ only by a known object's
    position (e.g. bottle moved 30 cm closer). Use UniDepth to predict metric
    depth on both. Report the predicted distance delta vs the true 30 cm.

Reading of the exercise: `unidepth` raises ModuleNotFoundError and no weights are
downloaded, so no metric predictor exists here. What the lesson ships instead is
the relative-to-metric route everyone actually uses: predict up to an unknown
scale, then `align_scale_shift` against ground truth. That route cannot answer
this exercise, because it fits its two parameters to the very depths you were
trying to measure -- so the centimetres it reports are a property of the fit.
Both halves of that are measured. The prediction contributes no scale at all:
replacing it with any moderately conditioned `c * pred + d` returns the same
answer to 1e-4 cm, sign flips included. And the fit is the wrong model, because
a relative predictor's natural output is disparity and 1/z is not affine in z:
the same true 30 cm comes back as 90.9 cm at 1.6 m and 15.3 cm at 4.0 m,
crossing the truth between 2.4 and 3.0 m, so one well-chosen pair looks perfect.
Doing the reciprocal the lesson never takes -- align in disparity, then invert --
recovers 30 cm at every distance. The accuracy reported alongside is worse than
useless: delta<1.25 is 1.0000 on all five pairs, blind by construction to a
quantity it can only see closer than 1.5 m.

Structure: `masks` returns the object patch and a real ground plane, one with
1/z linear in image row; the `scene` lambda puts the patch at a chosen distance
in front of it; `predict` is a MiDaS-shaped relative model, an arbitrary affine
map of true disparity plus noise; `recover` runs the lesson's own
`align_scale_shift` in either depth or disparity space and reads the object's
mean; `walk` returns the recovered movement in centimetres over the five
distances; `errs`, `worst` and `flips` hold verify()'s comprehensions so it stays
inside D14's complexity cap, and `status` is imported from exercise 1 rather than
repeated. The patch covers 225 of 4,096 pixels, which is exactly the size of the
delta<1.25 step measured in the last check.
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "26-monocular-depth"
SIZE, MOVE, NOISE, NEAR, FAR, GAIN, BIAS = 64, 0.30, 0.01, 1.2, 8.0, 7.3, -2.1
BASES, STEPS = (1.6, 2.0, 2.4, 3.0, 4.0), (1.40, 1.45, 1.50, 1.55, 1.60)
RESCALES, ILL = ((1.0, 0.0), (5.0, -3.0), (-1.0, 7.0)), (0.02, 100.0)

status = practice.load_module(
    pathlib.Path(__file__).with_name("ex01_grayscale_png_erases_metric_scale.py")).status
cm = lambda values: ", ".join(f"{v:.1f}" for v in values)                                # noqa: E731
errs = lambda values: [v - 100 * MOVE for v in values]                                   # noqa: E731
worst = lambda values: max(abs(v) for v in errs(values))                                 # noqa: E731
flips = lambda steps: [base for base, accuracy in steps if accuracy < 1.0]                # noqa: E731
scene = lambda np, tc, at: tc.from_numpy(np.where(masks(np)[0], at, masks(np)[1]).astype("float32"))
predict = lambda tc, gen, gt: GAIN / gt + BIAS + NOISE * tc.randn(gt.shape, generator=gen)


def masks(np):
    row, col = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
    ground = 1.0 / ((1.0 / NEAR) + (row / (SIZE - 1)) * ((1.0 / FAR) - (1.0 / NEAR)))
    return (np.abs(col - SIZE / 2) < SIZE / 8) & (np.abs(row - SIZE / 2) < SIZE / 8), ground


def recover(ref, mask, truth, guess, disparity=False) -> float:
    if disparity:
        return float((1.0 / ref.align_scale_shift(guess, 1.0 / truth).clamp(min=1e-6))[mask].mean())
    return float(ref.align_scale_shift(guess, truth)[mask].mean())


def walk(np, torch, ref, gen, mask, disparity=False) -> list:
    out = []
    for base in BASES:
        before, after = scene(np, torch, base), scene(np, torch, base - MOVE)
        out.append(100.0 * (recover(ref, mask, before, predict(torch, gen, before), disparity)
                            - recover(ref, mask, after, predict(torch, gen, after), disparity)))
    return out


def solve():
    try:
        import numpy as np
        import torch
    except ImportError as exc:                  # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    mask = torch.from_numpy(masks(np)[0])
    before, after = scene(np, torch, BASES[1]), scene(np, torch, BASES[1] - MOVE)
    gen = torch.Generator().manual_seed(1)
    guesses = (predict(torch, gen, before), predict(torch, gen, after))
    return {"metric": status("unidepth"), "pixels": int(mask.sum()),
            "depth": walk(np, torch, ref, torch.Generator().manual_seed(0), mask),
            "disparity": walk(np, torch, ref, torch.Generator().manual_seed(0), mask, True),
            "rescaled": [100.0 * (recover(ref, mask, before, c * guesses[0] + d)
                                  - recover(ref, mask, after, c * guesses[1] + d))
                         for c, d in RESCALES + (ILL,)],
            "absrel": [ref.abs_rel_error(ref.align_scale_shift(guesses[0], space), space)
                       for space in (before, 1.0 / before)],
            "delta": [ref.delta_accuracy(scene(np, torch, b - MOVE), scene(np, torch, b))
                      for b in BASES],
            "step": [(b, ref.delta_accuracy(scene(np, torch, b - MOVE), scene(np, torch, b)))
                     for b in STEPS]}


def verify(result):
    depth, disp, rescaled = result["depth"], result["disparity"], result["rescaled"]
    fair, absrel = rescaled[:len(RESCALES)], result["absrel"]
    return [
        practice.Check(
            "ANSWER: no metric predictor exists, and the substitute needs the answer as input",
            result["metric"] == "ModuleNotFoundError",
            f"importing unidepth gives {result['metric']} and no weights are fetched. The lesson's only "
            "route from a relative prediction to metres is `align_scale_shift`, a least-squares fit of "
            "a*pred + b against the ground-truth depth map -- the quantity the exercise measures"),
        practice.Check(
            "ANSWER: the same true 30.0 cm is reported as 90.9 cm and as 15.3 cm",
            worst(depth) > 10.0 and min(depth) < 100 * MOVE < max(depth),
            f"five pairs at {BASES} m, each moved exactly {100 * MOVE:.0f} cm closer, come back as "
            f"{cm(depth)} cm -- errors {cm(errs(depth))}, worst error {worst(depth):.1f} cm. "
            "It decays with distance and crosses the truth between 2.4 and 3.0 m, so one pair is exact"),
        practice.Check(
            "MECHANISM: the missing step is the reciprocal, not a bigger fit",
            worst(disp) < 0.5,
            "a relative model predicts disparity, and 1/z is not affine in z. Fitting the same "
            f"`align_scale_shift` against 1/gt and inverting gives {cm(disp)} cm on those pairs, within "
            f"{worst(disp):.2f} cm, and drops absRel {absrel[0]:.4f} -> {absrel[1]:.5f}"),
        practice.Check(
            "FINDING: the prediction's own scale contributes nothing to the reported centimetres",
            max(abs(v - fair[0]) for v in fair) < 1e-3,
            f"replacing pred with c*pred + d for {RESCALES} -- a sign flip included -- returns "
            + ", ".join(f"{v:.4f}" for v in fair)
            + " cm, identical to 1e-4. The fit undoes any affine map by construction, so a model "
            "reporting metres and one reporting an arbitrary index score exactly the same"),
        practice.Check(
            "CONTROL: that identity is exact in real arithmetic and fragile in float32",
            abs(rescaled[-1] - fair[0]) > 1.0,
            f"the ill-conditioned c={ILL[0]}, d={ILL[1]}, which pushes the signal into the last mantissa "
            f"bits, returns {rescaled[-1]:.4f} cm against {fair[0]:.4f}. The invariance belongs to the "
            "least-squares solution, not the float32 lstsq computing it -- asserted at 1e-3, no tighter"),
        practice.Check(
            "FINDING: delta<1.25 scores 1.0000 on every pair, and steps only below 1.5 m",
            all(v == 1.0 for v in result["delta"]) and flips(result["step"]) == [1.40, 1.45, 1.50],
            f"the five pairs score delta<1.25 of {result['delta']} despite a {100 * MOVE:.0f} cm move: "
            f"max(d/d*, d*/d) is a step function and the move is only a "
            f"{BASES[1] / (BASES[1] - MOVE):.4f} ratio at {BASES[1]} m. Sweeping the base over {STEPS} m "
            f"it fires at {flips(result['step'])} and no higher -- where base/(base - {MOVE}) hits 1.25, "
            f"at 1.50 m -- and the drop is {1.0 - dict(result['step'])[1.50]:.4f} = "
            f"{result['pixels']}/{SIZE * SIZE}, the object's pixel share and nothing more"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
