"""Exercise 2 — the lesson's refinement recovers a tenth of the offset.

    **(Medium)** Add sub-pixel refinement: given the argmax position, fit a 1D
    parabola along x and y from the neighbouring pixels. Report the accuracy gain
    vs integer argmax.

Reading of the exercise: the parabola through `(-1, l), (0, c), (1, r)` has its
vertex at `0.5 * (l - r) / (l - 2c + r)`, applied per axis around the lesson's own
`heatmap_to_coords` argmax. The lesson already ships a `subpixel_refine`, so it is
scored alongside, and so is the same parabola fitted to the *log* heatmap, which
is exact for a Gaussian. The gain is measured where sub-pixel truth exists --
200 of the lesson's own `gaussian_heatmap` targets at continuous centres -- and
then on the lesson's own pipeline, whose keypoints are integers.

**ANSWER: 0.400 px -> 0.0117 px, a 34x gain** on Gaussian heatmaps at
continuous centres; the log-parabola reaches **9e-08 px**.

**FINDING: the lesson's `subpixel_refine` is not a parabola, and recovers a tenth
of the offset.** Its `x + 0.25 * (r - l)` is a heatmap *value* difference used as
a pixel distance: on a sigma-2 Gaussian it moves a 0.3 px offset by **10.9%** of
the way, where the parabola moves **96%**. It also scales with heatmap height, so
at the 0.26 peak the lesson's trained net produces it gains almost nothing:
0.400 -> **0.357 px** at height 1, 0.400 -> **0.389 px** at height 0.26. The
parabola's ratio is height-invariant, 0.0117 at both.

**FINDING: on the lesson's own data there is nothing to refine.** Keypoints are
drawn with `rng.integers`, so argmax on the targets is already exact (**0.0 px**)
and sub-pixel gain is capped at the net's own error. On the lesson's trained
pipeline the parabola moves 14.59 px to **14.63 px**, slightly worse: the error is
which square, not where in it (exercise 1).

**CONTROL:** the log-parabola recovers continuous centres to **9e-08** and **2e-07 px** at
the two heights, so the offsets above are measured against a known truth.

The file is 13 lines over D14's 120-line target: `trained` re-runs the lesson's
200-step loop, because `main()` keeps its model to itself and the parabola has
to be scored on a real trained heatmap, not only on ideal ones.

Structure: `parabola` is the exercise's refinement (optionally on log values);
`gaussians` scores four decoders on continuous-centre targets; `trained` runs the
lesson's 200-step pipeline and scores argmax against the parabola.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "21-keypoint-pose"
POINTS, HEIGHTS, OFFSET = 200, (1.0, 0.26), 0.3


def parabola(torch, ref, heatmaps, log=False):
    """Vertex of the 1D parabola through argmax and its two neighbours, per axis."""
    coords = ref.heatmap_to_coords(heatmaps)
    out, values = coords.clone(), heatmaps.clamp_min(1e-12).log() if log else heatmaps
    n_, k_, h_, w_ = heatmaps.shape
    for n in range(n_):
        for k in range(k_):
            x, y = int(coords[n, k, 0]), int(coords[n, k, 1])
            if not (0 < x < w_ - 1 and 0 < y < h_ - 1):
                continue
            g = values[n, k]
            for axis, (lo, mid, hi) in enumerate(
                (
                    (g[y, x - 1], g[y, x], g[y, x + 1]),
                    (g[y - 1, x], g[y, x], g[y + 1, x]),
                )
            ):
                if lo - 2 * mid + hi < 0:
                    out[n, k, axis] += 0.5 * (lo - hi) / (lo - 2 * mid + hi)
    return out


def gaussians(torch, np, ref, height):
    """Mean error of each decoder on POINTS Gaussian targets at continuous centres."""
    pts = np.random.default_rng(1).uniform(10, 54, size=(POINTS, 2))
    hms = [height * ref.gaussian_heatmap(64, x, y) for x, y in pts]
    hm, gt = (
        torch.from_numpy(np.stack(hms))[:, None],
        torch.from_numpy(pts).float()[:, None],
    )
    decoders = {
        "argmax": ref.heatmap_to_coords,
        "lesson": ref.subpixel_refine,
        "parabola": lambda h: parabola(torch, ref, h),
        "log": lambda h: parabola(torch, ref, h, log=True),
    }
    return {name: float((f(hm) - gt).norm(dim=-1).mean()) for name, f in decoders.items()}


def trained(torch, np, ref):
    """The lesson's 200-step pipeline, argmax vs parabola on 64 held-out images."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = ref.TinyKeypointNet(num_keypoints=4)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    stack = lambda b, i: torch.from_numpy(np.stack([s[i] for s in b]))
    for _ in range(200):
        batch = [ref.make_synthetic_sample(rng=rng) for _ in range(16)]
        loss = torch.nn.functional.mse_loss(model(stack(batch, 0)), stack(batch, 1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    batch = [ref.make_synthetic_sample(rng=rng) for _ in range(64)]
    with torch.no_grad():
        pred = model.eval()(stack(batch, 0))
    gt, err = (
        stack(batch, 2),
        lambda c: float((c - stack(batch, 2)).norm(dim=-1).mean()),
    )
    exact = ref.heatmap_to_coords(stack(batch, 1)) - gt
    return {
        "argmax": err(ref.heatmap_to_coords(pred)),
        "parabola": err(parabola(torch, ref, pred)),
        "target": float(exact.abs().max()),
    }


def solve():
    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    torch.set_num_threads(2)
    ref = parity.load_reference(PHASE, LESSON, "main")
    one = torch.from_numpy(ref.gaussian_heatmap(64, 32 + OFFSET, 32))[None, None]
    moved = {"lesson": ref.subpixel_refine(one), "parabola": parabola(torch, ref, one)}
    return {
        "by_height": {h: gaussians(torch, np, ref, h) for h in HEIGHTS},
        "moved": {k: (float(v[0, 0, 0]) - 32) / OFFSET for k, v in moved.items()},
        "trained": trained(torch, np, ref),
    }


def verify(result):
    full, low = result["by_height"][1.0], result["by_height"][0.26]
    moved, run = result["moved"], result["trained"]
    return [
        practice.Check(
            "ANSWER: the parabola cuts sub-pixel error by more than 10x",
            full["parabola"] < full["argmax"] / 10,
            f"on {POINTS} Gaussian targets at continuous centres, integer argmax errs "
            f"{full['argmax']:.3f} px and the parabola {full['parabola']:.4f} px, "
            f"{full['argmax'] / full['parabola']:.0f}x better",
        ),
        practice.Check(
            "FINDING: the lesson's subpixel_refine recovers a tenth of the offset",
            moved["lesson"] < 0.2 and moved["parabola"] > 0.9 and low["lesson"] > full["lesson"],
            f"0.25 * (r - l) moves a {OFFSET} px offset {moved['lesson']:.1%} of the way, the "
            f"parabola {moved['parabola']:.0%}. It scales with heatmap height: argmax "
            f"{full['argmax']:.3f} -> {full['lesson']:.3f} px at height 1 but -> "
            f"{low['lesson']:.3f} px at height 0.26, while the parabola stays at "
            f"{low['parabola']:.4f}",
        ),
        practice.Check(
            "FINDING: on the lesson's integer keypoints there is nothing to refine",
            run["target"] == 0.0 and run["parabola"] > run["argmax"] - 0.5,
            f"argmax on the lesson's own targets is exact ({run['target']} px), and on its "
            f"trained pipeline the parabola moves {run['argmax']:.2f} px to "
            f"{run['parabola']:.2f} px -- the error is which square, not where in it",
        ),
        practice.Check(
            "CONTROL: the log-parabola recovers the true centres exactly",
            full["log"] < 1e-4 and low["log"] < 1e-4,
            f"a Gaussian's log is a parabola, so the fit lands {full['log']:.0e} px and "
            f"{low['log']:.0e} px from truth at heights 1 and 0.26",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
