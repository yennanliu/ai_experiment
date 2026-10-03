"""Exercise 2 — only the hue matches; a colourless model beats the fit on MSE.

    **Medium.** Extend to color Gaussians (RGB). Confirm reconstruction matches the target color pattern.

Reading of the exercise: each Gaussian keeps one position and one sigma and
gets an `[r, g, b]` colour. The target recolours the lesson's own two blobs:
the upper-left blob red-orange `(1, 0.5, 0)`, the lower-right blob blue-green
`(0, 0.5, 1)`. The extension swaps exactly two functions -- `render` draws
each channel with the lesson's own `render`, `mse` averages the lesson's own
`mse` over channels -- and the lesson's `finite_diff_step` runs **unchanged**:
its `isinstance(g[key], list)` branch already differentiates a 3-vector
colour. "Matches the colour pattern" is checked two ways: the dominant channel
at each of the 50 pixels where the target is bright (>0.3), and the MSE.

**ANSWER: only the hue matches.** At n=8 (seed 15, 30 steps of lr 0.5) the
dominant channel is right at 50 of 50 bright pixels, but the red blob's centre
renders `(0.24, 0.18, 0.12)` against `(1, 0.5, 0)` -- a quarter of the
brightness -- and the RGB MSE is 0.0398.

**FINDING: hue agreement cannot tell undershoot from overshoot.** n=16 also
scores 50/50 and ends at MSE 0.1404, 3.5x worse, with centres
`(1.00, 0.73, 0.48)` and `(0.49, 0.94, 1.40)`: now too bright and the wrong mix.

**FINDING: a model with no colour at all beats both fits on MSE.** One grey
value per splat, shared by all three channels, on the true geometry, leaves
MSE 0.0245 -- below n=8's 0.0398. The gap is the optimiser, not the
representation: rendering is linear in colour, so with the geometry known a
2x2 least-squares solve through the lesson's `render` returns the true colours
to 6e-17, while 30 lesson steps from grey on that same geometry
reach only `(0.69, 0.46, 0.22)` and `(0.16, 0.46, 0.75)`, MSE 0.0056.

**CONTROL: the hue test does catch missing colour.** That best grey model
scores only 21/50 on hue -- every channel ties, and argmax resolves ties to
red.

Structure: `rgb_render` and `rgb_mse` are the two swapped functions; `hue`
scores dominant channels; `best_colours` is the closed-form solve.
"""

from __future__ import annotations

import math
import operator
import random
from functools import partial

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "12-3d-generation"
STEPS, LR, EPS, BRIGHT = 30, 0.5, 0.2, 0.3
CENTRES = [((3.0, 3.0), math.sqrt(3), [1.0, 0.5, 0.0]), ((8.0, 8.0), 2.0, [0.0, 0.5, 1.0])]


def splats(colours):
    return [{"pos": list(p), "sigma": s, "color": c} for (p, s, _), c in zip(CENTRES, colours)]


def rgb_render(render, gs):
    """The first of the two swapped functions: the lesson's render, once per channel."""
    return [render([{**g, "color": g["color"][c]} for g in gs]) for c in range(3)]


def rgb_mse(mse, a, b):
    """The second: the lesson's mse, averaged over channels."""
    return sum(mse(a[c], b[c]) for c in range(3)) / 3


def flat(img):
    return [v for row in img for v in row]


def hue(img, target):
    """'right/total' over bright target pixels, by dominant channel."""
    pairs = zip(zip(*map(flat, img)), zip(*map(flat, target)))
    hits = [i.index(max(i)) == t.index(max(t)) for i, t in pairs if max(t) > BRIGHT]
    return f"{sum(hits)}/{len(hits)}"


def run(ref, gs, target, rgb):
    """The lesson's finite_diff_step, with render and mse swapped for RGB."""
    saved = ref.render, ref.mse
    ref.render, ref.mse = rgb
    try:
        for _ in range(STEPS):
            ref.finite_diff_step(gs, target, LR, EPS)
    finally:
        ref.render, ref.mse = saved
    return rgb[0](gs)


def best_colours(ref, channels):
    """Least-squares splat colours for the fixed true geometry, by Cramer's rule."""
    basis = [flat(ref.render([g])) for g in splats([1.0, 1.0])]
    (a, b), (_, d) = [[sum(map(operator.mul, p, q)) for q in basis] for p in basis]
    rhs = [[sum(map(operator.mul, p, flat(ch))) for p in basis] for ch in channels]
    det = a * d - b * b
    per_channel = [((r0 * d - r1 * b) / det, (r1 * a - r0 * b) / det) for r0, r1 in rhs]
    return [list(splat) for splat in zip(*per_channel)]


def fitted(ref, n, target, rgb):
    gs = ref.init_gaussians(n, random.Random(7 + n))
    for g in gs:
        g["color"] = [g["color"]] * 3
    img = run(ref, gs, target, rgb)
    centres = [[round(img[c][y][y], 2) for c in range(3)] for y in (3, 8)]
    return {"mse": rgb[1](img, target), "hue": hue(img, target), "centres": centres}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rgb = partial(rgb_render, ref.render), partial(rgb_mse, ref.mse)
    target = rgb[0](splats([c for _, _, c in CENTRES]))
    exact = best_colours(ref, target)
    # least squares is linear in the target, so the best grey is the mean of the best RGB
    grey = rgb[0](splats([[sum(e) / 3] * 3 for e in exact]))
    known = splats([[0.5] * 3, [0.5] * 3])
    run(ref, known, target, rgb)
    return {
        "fits": {n: fitted(ref, n, target, rgb) for n in (8, 16)},
        "exact_err": max(abs(a - b) for e, (_, _, c) in zip(exact, CENTRES) for a, b in zip(e, c)),
        "known": [[round(v, 2) for v in g["color"]] for g in known],
        "known_mse": rgb[1](rgb[0](known), target),
        "grey": (rgb[1](grey, target), hue(grey, target)),
        "every": hue(target, target),
    }


def verify(result):
    eight, sixteen = result["fits"][8], result["fits"][16]
    (grey_mse, grey_hue), every = result["grey"], result["every"]
    return [
        practice.Check(
            "ANSWER: only the hue matches -- every bright pixel, at a quarter of the brightness",
            eight["hue"] == every and eight["centres"][0][0] < 0.5,
            f"n=8: dominant channel right at {eight['hue']} bright pixels, but blob centres "
            f"{eight['centres']} vs [1, 0.5, 0], [0, 0.5, 1]; RGB MSE {eight['mse']:.4f}",
        ),
        practice.Check(
            "FINDING: hue agreement cannot tell undershoot from overshoot",
            sixteen["hue"] == every and sixteen["mse"] > 2 * eight["mse"],
            f"n=16 also scores {sixteen['hue']} at MSE {sixteen['mse']:.4f}, "
            f"centres {sixteen['centres']}",
        ),
        practice.Check(
            "FINDING: a colourless model beats the fit; the optimiser, not the model, is short",
            grey_mse < eight["mse"] and result["exact_err"] < 1e-9 and result["known_mse"] > 1e-3,
            f"best grey per splat, true geometry: MSE {grey_mse:.4f} < n=8's {eight['mse']:.4f}. "
            f"2x2 least squares recovers the colours to {result['exact_err']:.0e}; 30 lesson "
            f"steps from grey reach {result['known']}, MSE {result['known_mse']:.4f}",
        ),
        practice.Check(
            "CONTROL: the hue test does catch a model with no colour",
            grey_hue != every,
            f"the best grey model scores {grey_hue} on hue: channels tie, argmax picks red",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
