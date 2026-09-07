"""Exercise 2 — view angle harmonic rank.

    **(Medium)** Extend the 2D rasteriser to support per-Gaussian RGB colours that depend on a scalar "view angle" through a degree-2 harmonic. Train on a pair of target images and verify the model reconstructs both.

Reading of the exercise: a "scalar view angle" is not a direction on the
sphere, so the lesson's `sh_degree_3_basis` cannot be used as written -- it
wants a unit 3-vector. The faithful reading maps the angle onto a great circle,
`dir = (cos t, sin t, 0)`, which lets the lesson's own basis and its own
`eval_sh_degree_3` do the evaluation unchanged. That restriction is not free
and it is exact: on `z = 0` the nine degree-<=2 functions span only five
dimensions. Three of them (`Y_1^0`, `Y_2^-1`, `Y_2^1`) carry a `z` factor and
vanish identically, and `Y_2^0` collapses to the constant `-sqrt(5)/2 * Y_0^0`,
so a "degree-2 harmonic in a scalar angle" is really the five-term Fourier
basis 1, cos t, sin t, cos 2t, sin 2t wearing a spherical name. Training shows
the same deficiency from the other side: the three vanishing slots accumulate a
gradient of exactly 0.0 over 200 steps, and a fourth is dead too at the only two
angles a pair of targets gives you. "Verify the model reconstructs both" then
turns out not to be a test that ranks anything -- degree 1 and degree 2
reconstruct the pair equally well, because two views can pin down at most two of
the five dimensions. So the checks add the control the exercise omits (a
view-independent colour, which provably cannot fit two different targets) and
the probe that does separate the two harmonic orders (an unseen angle).

Structure: `targets` builds the pair -- one disc and one bar that swap colours
between the two views, so no view-independent model can serve both; `circle`
evaluates the lesson's `sh_degree_3_basis` at 256 angles on `z = 0` and is what
the rank measurements read; `render` wraps one forward pass, evaluating the
lesson's `eval_sh_degree_3` at one direction and handing the resulting colours
to the lesson's `rasterise_2d`; `train` fits the lesson's own `Splats2D`
geometry plus a `(splats, 16, 3)` coefficient tensor masked to the first
`bands` slots -- 1, 4 and 9, i.e. degree 0, degree <=1 and degree <=2 -- and
accumulates the absolute gradient each slot receives so the dead dimensions can
be counted rather than argued. The module-level lambdas above it are formatting
and derived tables lifted out of `verify`, which D14's complexity ceiling
otherwise fails. At 145 code lines this is over D14's 120-line target and five
clear of its ceiling: three trained arms plus a 256-angle rank measurement, and
five checks whose evidence carries 30 measured numbers.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "22-3d-gaussian-splatting"

SIZE, SPLATS, STEPS, LR = 32, 40, 200, 0.08
ANGLES, BANDS = (0.0, math.pi / 2, math.pi / 4), (1, 4, 9)   # 2 trained views + 1 unseen; 3 degrees
CIRCLE, RCOND = 256, 1e-5

rank_of = lambda values: int(sum(v > values[0] * RCOND for v in values))          # noqa: E731
floats = lambda values: [float(v) for v in values]                                # noqa: E731
worst_of = lambda views: {b: max(views[b]) for b in BANDS}                        # noqa: E731
dead_of = lambda slots: [j for j, g in enumerate(slots) if g == 0.0]              # noqa: E731
sval_row = lambda values: " ".join(f"{v:.4f}" for v in values)                    # noqa: E731
slot_row = lambda slots: " ".join(f"{j}:{g:.3g}" for j, g in enumerate(slots))    # noqa: E731
row = lambda table, fmt="{:.5f}": "  ".join(                                      # noqa: E731
    f"deg{int(math.sqrt(b)) - 1}=" + "/".join(fmt.format(v) for v in table[b]) for b in BANDS)


def targets(torch):
    rows, cols = torch.meshgrid(torch.arange(SIZE, dtype=torch.float32),
                                torch.arange(SIZE, dtype=torch.float32), indexing="ij")
    first, second = torch.zeros(SIZE, SIZE, 3), torch.zeros(SIZE, SIZE, 3)
    disc = (cols - 10) ** 2 + (rows - 10) ** 2 < 49
    bar = (cols - 22).abs() < 6
    first[disc], second[disc] = torch.tensor([0.9, 0.2, 0.15]), torch.tensor([0.15, 0.3, 0.9])
    first[bar], second[bar] = torch.tensor([0.15, 0.3, 0.9]), torch.tensor([0.9, 0.85, 0.15])
    return torch.stack([first, second])


def circle(torch, ref):
    theta = torch.linspace(0.0, 2 * math.pi, CIRCLE + 1)[:-1]
    return ref.sh_degree_3_basis(
        torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=-1))


def render(torch, ref, model, coeffs, direction):
    colour = torch.sigmoid(ref.eval_sh_degree_3(coeffs, direction.expand(SPLATS, 3)))
    return ref.rasterise_2d(model.means, model.covs(), colour,
                            torch.sigmoid(model.opacity_logit), model.depth, (SIZE, SIZE))


def train(torch, ref, bands, goals, dirs, seed=0) -> dict:
    torch.manual_seed(seed)
    model = ref.Splats2D(num_splats=SPLATS, image_size=SIZE, seed=seed)
    sh = torch.nn.Parameter(torch.randn(SPLATS, 16, 3, generator=torch.Generator().manual_seed(seed)) * 0.3)
    mask = (torch.arange(16) < bands).float()[:, None]
    optimiser = torch.optim.Adam(list(model.parameters()) + [sh], lr=LR)
    slots = torch.zeros(16)
    for _ in range(STEPS):
        loss = sum(((render(torch, ref, model, sh * mask, dirs[v]) - goals[v]) ** 2).mean()
                   for v in range(len(goals)))
        optimiser.zero_grad()
        loss.backward()
        slots += sh.grad.abs().sum(dim=(0, 2))
        optimiser.step()
    with torch.no_grad():
        return {"per_view": [float(((render(torch, ref, model, sh * mask, dirs[v]) - goals[v])
                                    ** 2).mean()) for v in range(len(goals))],
                "slots": [float(v) for v in slots[:bands]],
                "unseen": render(torch, ref, model, sh * mask, dirs[-1])}


def solve():
    try:
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    goals = targets(torch)
    dirs = torch.tensor([[math.cos(t), math.sin(t), 0.0] for t in ANGLES])
    basis = circle(torch, ref)
    arms = {bands: train(torch, ref, bands, goals, dirs) for bands in BANDS}
    unseen = arms[4]["unseen"] - arms[9]["unseen"]
    return {"per_view": {b: arms[b]["per_view"] for b in BANDS},
            "slots": {b: arms[b]["slots"] for b in BANDS},
            "svals": floats(torch.linalg.svdvals(basis[:, :9])),
            "full": floats(torch.linalg.svdvals(basis)),
            "peak": floats(basis[:, :9].abs().max(dim=0).values),
            "collapse": float((basis[:, 6] + math.sqrt(5) / 2 * basis[:, 0]).abs().max()),
            "floor": float(((goals - goals.mean(dim=0)) ** 2).mean()),
            "spread": float(((goals[0] - goals[1]) ** 2).mean()),
            "unseen": (float(unseen.abs().max()), float((unseen ** 2).mean()))}


def verify(result):
    views, slots, svals = result["per_view"], result["slots"], result["svals"]
    dead, worst = dead_of(slots[9]), worst_of(views)
    return [
        practice.Check(
            "ANSWER: the degree-2 model reconstructs both views, two orders below what separates them",
            worst[9] * 100 < result["spread"] and worst[9] < worst[1] / 20,
            f"per-view MSE at the two trained angles: {row(views)}. The two targets differ from each "
            f"other by {result['spread']:.5f}, so the degree-2 fit sits "
            f"{result['spread'] / worst[9]:.0f}x below the thing it has to distinguish, and "
            f"{worst[1] / worst[9]:.0f}x below the view-independent arm. The colours come from the "
            "lesson's own `eval_sh_degree_3`, the pixels from its own `rasterise_2d`"),
        practice.Check(
            "CONTROL: a view-independent colour lands exactly on the mean image, and cannot do better",
            abs(worst[1] - result["floor"]) < 0.05 * result["floor"],
            f"the degree-0 arm scores {views[1][0]:.5f}/{views[1][1]:.5f}, against the closed-form "
            f"floor for *any* model that renders one image for both views — the mean of the pair, at "
            f"{result['floor']:.5f}. It is within {worst[1] / result['floor']:.2f}x of that bound, so "
            "the arm is not under-trained: one colour per Gaussian provably cannot fit two targets, "
            "which is the control the exercise's 'verify it reconstructs both' leaves out"),
        practice.Check(
            "MECHANISM: on a scalar angle the degree-2 basis has rank exactly 5 of 9, not 9",
            rank_of(svals) == 5 and rank_of(result["full"]) == 7 and result["collapse"] < 1e-6,
            "singular values of the lesson's `sh_degree_3_basis` evaluated at "
            f"{CIRCLE} angles on z=0, first nine columns: {sval_row(svals)}"
            + f" — five live directions then a {svals[4] / svals[5]:.0e} collapse. Columns 2, 5 and 7 "
            f"are identically zero (peak magnitude {result['peak'][2]:.1e}/{result['peak'][5]:.1e}/"
            f"{result['peak'][7]:.1e}: every one carries a z factor) and column 6 is exactly "
            f"-sqrt(5)/2 times column 0, residual {result['collapse']:.1e} — one float32 eps, which "
            f"is the tolerance an algebraic identity earns. All 16 degree-3 columns span only "
            f"{rank_of(result['full'])}"),
        practice.Check(
            "FINDING: three of the nine coefficient slots collect bitwise-zero gradient, a fourth only noise",
            dead == [2, 5, 7] and slots[9][4] < 1e-15,
            f"summed |grad| per SH slot over {STEPS} steps: {slot_row(slots[9])}"
            + f". Slots {dead} accumulate a gradient of exactly 0.0 — structurally dead on the circle; slot 4 (sin 2t) adds "
            f"{slots[9][4]:.1e} — nonzero only as float noise, because the two sampled angles "
            f"0 and pi/2 both have sin 2t = 0. So {4 * SPLATS * 3} of the {9 * SPLATS * 3} SH "
            "parameters this arm allocates never move, three of them for any angles at all"),
        practice.Check(
            "FINDING: reconstructing the pair does not rank degree 1 against degree 2 — the unseen angle does",
            worst[4] < 1.3 * worst[9] and result["unseen"][1] > 0.5 * worst[9],
            f"degree 1 reconstructs the pair as well as degree 2 ({row(views)}), a ratio of "
            f"{worst[4] / worst[9]:.2f}x: two views can determine at most two of the five live "
            f"dimensions, so the exercise's own success criterion saturates. Rendered at the "
            f"never-trained angle pi/4 the two arms disagree by max {result['unseen'][0]:.3f} per "
            f"channel, MSE {result['unseen'][1]:.5f} — {result['unseen'][1] / worst[9]:.1f}x their "
            "own training error, and the only place the extra five coefficients are visible"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
