"""Exercise 3 — half the capture is mirror images, so "held out" is not held out.

    **Hard.** Using gsplat or Nerfstudio, reconstruct a real object from a 50-photo capture. Report fit time and final SSIM on held-out views.

Reading of the exercise: `gsplat`, `nerfstudio` and `torch` all return None
from `find_spec`, there is no GPU and no capture, so per DESIGN D11 this ships
the scaled-down runnable of the same measurement. The real run is printed in
the CONTROL detail. The object is three isotropic 3D Gaussians, photographed
by 50 orthographic cameras evenly spaced around the vertical axis. An
isotropic 3D Gaussian projects to a 2D Gaussian with the same sigma, so every
"photo" is the lesson's own `render` of the projected splats. The fit is the
lesson's own `finite_diff_step`, unchanged: only `render` (all training
views) and `mse` (averaged over them) are swapped, and its list-valued `pos`
branch fits three coordinates as readily as two. Nine views from the front
arc (cameras 0, 3, ..., 24) are trained on. SSIM is computed over the whole
12x12 image as a single window, with a dynamic range of 1.

**ANSWER: fit time ~2 s (100 steps, 14,400 view renders); held-out SSIM
0.904** on the 16 front-arc views not trained on, up from 0.674 at
initialisation -- marginally above the 0.902 the training views score.

**FINDING: 25 of the 50 photos carry no new information.** The lesson's
`render` sums splats with no depth order, as its docs admit ("Our 2D toy just
sums."). So the camera at angle theta + 180 sees the left-right mirror of the
camera at theta, to 6.7e-16. A held-out view opposite a training view scores
exactly that training view's SSIM, 0.902488 against 0.902488. Hold out by random
index and some of the "held-out" score is a training score.

**FINDING: the back of the object cannot be hallucinated here.** The 25
back-arc views, none trained on, score 0.903. The docs' "back-side
hallucination" pitfall needs occlusion, which the lesson's summing renderer
cannot express, so a front-arc capture already determines the whole object.

**CONTROL: the fit, not the split, earns the score.** The same views score
0.674 before fitting.

Structure: `view` projects and renders one camera; `fit` swaps render/mse into
the lesson's step; `ssim` is the single-window index.
"""

from __future__ import annotations

import importlib.util
import math
import random
import time

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "12-3d-generation"
CAMERAS, STEPS, LR, EPS, NEEDED = 50, 100, 0.5, 0.2, ("gsplat", "nerfstudio", "torch")
TRAIN = tuple(range(0, CAMERAS // 2, 3))
SCENE = [{"pos": [3.5, 4.0, 6.0], "sigma": 1.5, "color": 0.9},
         {"pos": [7.5, 7.0, 4.0], "sigma": 1.8, "color": 0.6},
         {"pos": [6.0, 3.0, 8.0], "sigma": 1.2, "color": 0.5}]
REAL = ("ns-process-data images --data photos/ --output-dir capture && "
        "ns-train splatfacto --data capture && ns-eval --load-config <run>/config.yml")


def view(ref, splats, camera, draw=None):
    """Orthographic photo from `camera`, rotating about the vertical axis."""
    angle, mid = 2 * math.pi * camera / CAMERAS, (ref.SIZE - 1) / 2
    flat = [{"pos": [(g["pos"][0] - mid) * math.cos(angle)
                     + (g["pos"][2] - mid) * math.sin(angle) + mid, g["pos"][1]],
             "sigma": g["sigma"], "color": g["color"]} for g in splats]
    return (draw or ref.render)(flat)


def ssim(a, b):
    """Single-window SSIM over the whole image, dynamic range 1."""
    xs, ys = [v for r in a for v in r], [v for r in b for v in r]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    vx, vy = sum((v - mx) ** 2 for v in xs) / n, sum((v - my) ** 2 for v in ys) / n
    cov = sum((p - mx) * (q - my) for p, q in zip(xs, ys)) / n
    c1, c2 = 0.01**2, 0.03**2
    return (2 * mx * my + c1) * (2 * cov + c2) / ((mx * mx + my * my + c1) * (vx + vy + c2))


def score(ref, splats, photos, cameras):
    return sum(ssim(view(ref, splats, c), photos[c]) for c in cameras) / len(cameras)


def fit(ref, splats, photos):
    """The lesson's finite_diff_step over all training views at once."""
    saved, calls = (ref.render, ref.mse), [0]

    def render_all(gs):
        calls[0] += len(TRAIN)
        return [view(ref, gs, c, saved[0]) for c in TRAIN]

    def mse_all(a, b):
        return sum(saved[1](p, q) for p, q in zip(a, b)) / len(a)

    ref.render, ref.mse = render_all, mse_all
    try:
        for _ in range(STEPS):
            ref.finite_diff_step(splats, [photos[c] for c in TRAIN], LR, EPS)
    finally:
        ref.render, ref.mse = saved
    return calls[0]


def mirror_gap(photos):
    """Worst pixel gap between camera c + 25 and the left-right flip of camera c."""
    half = len(photos) // 2
    return max(abs(a - b) for c in range(half)
               for back, front in zip(photos[c + half], photos[c])
               for a, b in zip(back, front[::-1]))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    photos, half = [view(ref, SCENE, c) for c in range(CAMERAS)], CAMERAS // 2
    rng = random.Random(12)
    splats = [{"pos": [rng.uniform(3, 8) for _ in range(3)], "sigma": rng.uniform(1, 2),
               "color": rng.uniform(0.3, 0.7)} for _ in SCENE]
    between = [c for c in range(half) if c not in TRAIN]
    start, clock = score(ref, splats, photos, between), time.perf_counter()
    renders = fit(ref, splats, photos)
    return {
        "seconds": time.perf_counter() - clock,
        "renders": renders,
        "start": start,
        "train": score(ref, splats, photos, TRAIN),
        "between": score(ref, splats, photos, between),
        "mirror": score(ref, splats, photos, [c + half for c in TRAIN]),
        "back": score(ref, splats, photos, range(half, CAMERAS)),
        "mirror_gap": mirror_gap(photos),
        "missing": [m for m in NEEDED if importlib.util.find_spec(m) is None],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: fit time and held-out SSIM on the unseen front-arc views",
            result["between"] > result["start"] + 0.1,
            f"{STEPS} lesson steps, {result['renders']:,} view renders, "
            f"{result['seconds']:.1f} s wall; SSIM on the 16 untrained front-arc views "
            f"{result['between']:.3f}, training views {result['train']:.3f}",
        ),
        practice.Check(
            "FINDING: 25 of 50 photos are mirror images, so they are not held out",
            result["mirror_gap"] < 1e-12 and abs(result["mirror"] - result["train"]) < 1e-12,
            f"the lesson's render sums with no depth order, so camera c+25 is the left-right "
            f"flip of camera c to {result['mirror_gap']:.1e}; views opposite the training views "
            f"score {result['mirror']:.6f}, the training views {result['train']:.6f}",
        ),
        practice.Check(
            "FINDING: the back side cannot be hallucinated without occlusion",
            abs(result["back"] - result["train"]) < 0.01,
            f"the 25 back-arc views, none trained on, score {result['back']:.3f} -- a front-arc "
            "capture already fixes the whole object when nothing can hide behind anything",
        ),
        practice.Check(
            "CONTROL: the fit earns the score; the real run needs what is missing",
            result["start"] < result["between"] - 0.1 and result["missing"],
            f"the untrained views score {result['start']:.3f} before fitting. Missing here: "
            f"{', '.join(result['missing'])}. The real run: {REAL} (the lesson puts a scene fit "
            "at 5-30 minutes on a consumer GPU)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
