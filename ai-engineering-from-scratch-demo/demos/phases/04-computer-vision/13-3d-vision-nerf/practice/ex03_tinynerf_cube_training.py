"""Exercise 3 — tinynerf cube training.

    **(Hard)** Train a TinyNeRF on a synthetic dataset of rendered views of a coloured cube (generated via differentiable rendering or a simple ray tracer). Report rendering loss at epoch 1, 10, and 100. At what epoch does the model produce recognisable views?

Reading of the exercise: "a simple ray tracer" is taken literally -- `views()`
below is a slab-method ray/box intersection giving a cube with six coloured faces
on a white ground, traced analytically at 12x12 from poses on a ring, so nothing
is downloaded and there is no lego scene to fetch. Everything downstream is the
lesson's own `TinyNeRF` and `volumetric_render` at their default settings, Adam
at lr 5e-3, one shared 32-sample grid for every ray. The question "at what epoch
does the model produce recognisable views?" has no answer until "recognisable" is
pinned to a number, so it is defined against a null model: the best single colour
for the scene scores 9.98 dB on three held-out poses placed at the midpoints of
the training ring, and a render counts as recognisable once it clears that by
3 dB. It does, at epoch 10 here with 0.03 dB to spare -- close enough that the
check allows anything up to epoch 20. The exercise then asks for exactly the
wrong statistic. From epoch 20 to 100 the training loss it wants reported drops
6.4x, ending at 0.00165, while novel-view PSNR moves +0.08 dB and the
training-camera PSNR that loss implies, -10log10(MSE), climbs 19.8 to 27.8 dB:
past epoch 20 the number is measuring memorisation of the rays the model was
shown. The eight-view control makes that stark -- train MSE 0.00017, 37.7 dB on
its own cameras, a visually perfect reconstruction, while its novel views end at
10.28 dB against a 9.97 dB flat-colour baseline and never clear recognisable at
any epoch. The binding variable is view count, not epoch. Two mechanisms behind
those numbers are asserted as closed forms rather than as trained results: the
quadrature telescopes to 1 - cumprod(1 - alpha) at *every* prefix, not merely in
total, to 6.0e-08 in float32, which is why the lesson's `torch.relu(self.sigma(h))`
leaves 5 of 10 seeds rendering an all-zero image at a bitwise-zero gradient,
permanently untrainable at a loss equal to the target's mean square 0.6912; and
the 32-sample grid is coarser than Nyquist for 6 of the 10 positional-encoding
bands the lesson feeds it, the top one turning 20.6 cycles between neighbouring
samples, its encoding dot product -0.98894 matching 3*sum_l cos(2^l pi d) to
1.3e-06.

Structure: `views` ray-traces the cube from a list of ring poses and returns
per-pixel origins, directions and colours; `render` runs the lesson's MLP and
quadrature on one shared sample grid; `train` is one Adam arm scored on the
held-out poses at every mark; `solve` runs both arms and then the two closed-form
probes plus the ten-seed initialisation sweep. At 146 code lines this sits above
D14's 120-line target, with 4 to spare under the ceiling: an analytic ray tracer
and scene generator, two independent 100-epoch training arms, a ten-seed
initialisation sweep and two closed-form identities are five deliverables behind
the exercise's two sentences, and none of them can be shared with another file.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "13-3d-vision-nerf"

SIZE, FOCAL, RADIUS, ELEVATION, HALF = 12, 16.5, 3.0, 0.4, 0.6
NEAR, FAR, SAMPLES, BATCH, LR, EPOCHS = 1.8, 4.3, 32, 512, 5e-3, 100
MARKS, VIEWS, HELD_OUT, LATE = (1, 10, 20, 35, 50, 100), (24, 8), (0.5, 8.5, 16.5), 20
DEAD_SEEDS, RECOGNISABLE, BANDS, STEP = 10, 3.0, 10, (FAR - NEAR) / (SAMPLES - 1)
FACES = ((.90, .20, .20), (.20, .60, .95), (.95, .80, .15), (.15, .75, .35), (.85, .35, .85), (.25, .25, .30))
psnr = lambda a, b: -10 * math.log10(float(((a - b) ** 2).mean()))            # noqa: E731
decibels = lambda mse: -10 * math.log10(mse)                                  # noqa: E731
band = lambda level: 2 ** level * math.pi * STEP                              # noqa: E731
row = lambda hist: "  ".join(f"{e}:{hist[e]['novel']:.2f}" for e in MARKS)     # noqa: E731
ALIASED = tuple(level for level in range(BANDS) if band(level) >= math.pi)
OVERLAP = 3 * sum(math.cos(band(level)) for level in range(BANDS))


def views(torch, faces, indices, count):
    angle = 2 * math.pi * torch.tensor(list(indices), dtype=torch.float32) / count
    back = torch.stack([math.cos(ELEVATION) * angle.sin(), torch.full_like(angle, math.sin(ELEVATION)),
                        math.cos(ELEVATION) * angle.cos()], dim=-1)
    right = torch.stack([angle.cos(), torch.zeros_like(angle), -angle.sin()], dim=-1)
    rows, cols = torch.meshgrid(*(torch.arange(SIZE, dtype=torch.float32),) * 2, indexing="ij")
    grid = torch.stack([(cols + .5 - SIZE / 2) / FOCAL, -(rows + .5 - SIZE / 2) / FOCAL, -torch.ones_like(cols)],
                       dim=-1).reshape(-1, 3)
    rotation = torch.stack([right, torch.linalg.cross(back, right), back], dim=-1).transpose(-1, -2)
    ray = (grid @ rotation).reshape(-1, 3)
    origin = (RADIUS * back)[:, None, :].expand(-1, SIZE * SIZE, -1).reshape(-1, 3)
    lo, hi = (-HALF - origin) / ray, (HALF - origin) / ray
    entry, leave = torch.minimum(lo, hi).max(-1), torch.maximum(lo, hi).min(-1)
    face = faces[2 * entry.indices + (ray.gather(-1, entry.indices[..., None])[..., 0] > 0).long()]
    hit = ((entry.values < leave.values) & (leave.values > 0))[..., None]
    return origin, ray, torch.where(hit, face, torch.ones_like(face))


def render(torch, ref, net, t_vals, origin, ray):
    points = origin[:, None, :] + t_vals.view(1, -1, 1) * ray[:, None, :]
    unit = (ray / ray.norm(dim=-1, keepdim=True))[:, None, :]
    return ref.volumetric_render(*net(points, unit.expand_as(points)), t_vals)


def train(torch, ref, net, t_vals, data, held) -> dict:
    origins, directions, colours = data
    optimiser, history = torch.optim.Adam(net.parameters(), lr=LR), {}
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for pick in torch.randperm(origins.shape[0]).split(BATCH):
            loss = ((render(torch, ref, net, t_vals, origins[pick], directions[pick])[0] - colours[pick]) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        if epoch in MARKS:
            with torch.no_grad():
                history[epoch] = {"mse": sum(losses) / len(losses),
                                  "novel": psnr(held[2], render(torch, ref, net, t_vals, *held[:2])[0])}
    return history


def solve():
    try:
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    faces, t_vals, arms = torch.tensor(FACES), torch.linspace(NEAR, FAR, SAMPLES), {}
    held = views(torch, faces, HELD_OUT, VIEWS[0])
    for count in VIEWS:
        data = views(torch, faces, range(count), count)
        torch.manual_seed(0)
        arms[count] = train(torch, ref, ref.TinyNeRF(), t_vals, data, held)
        arms[count]["baseline"] = psnr(data[2].mean(0).expand_as(held[2]), held[2])
    dead, blank = 0, 0
    for seed in range(DEAD_SEEDS):                  # the same probe on each fresh initialisation
        torch.manual_seed(seed)
        net = ref.TinyNeRF()
        rendered = render(torch, ref, net, t_vals, held[0], held[1])[0]
        ((rendered - held[2]) ** 2).mean().backward()
        dead += math.sqrt(sum(float((p.grad ** 2).sum()) for p in net.parameters())) == 0.0
        blank += float(rendered.detach().abs().max()) == 0.0
    sigma = torch.rand(64, SAMPLES) * 2.0
    weights = ref.volumetric_render(sigma, torch.rand(64, SAMPLES, 3), t_vals)[2]
    alpha = 1 - torch.exp(-sigma * torch.cat([t_vals[1:] - t_vals[:-1], torch.full_like(t_vals[:1], 1e10)]))
    ends = [ref.positional_encoding(torch.full((1, 3), shift), L=BANDS)[0] for shift in (0.0, STEP)]
    floor = arms[VIEWS[0]]["baseline"] + RECOGNISABLE
    return {"arms": arms, "floor": floor, "dead": dead, "blank": blank, "overlap": float(ends[0] @ ends[1]),
            "black": float((held[2] ** 2).mean()), "mass": float(weights.sum(-1).min()),
            "telescope": float((weights.cumsum(-1) - (1 - torch.cumprod(1 - alpha, -1))).abs().max()),
            "first": next((e for e in MARKS if arms[VIEWS[0]][e]["novel"] >= floor), None),
            "peak": max(MARKS, key=lambda e: arms[VIEWS[0]][e]["novel"])}


def verify(result):
    rich, thin, peak = result["arms"][VIEWS[0]], result["arms"][VIEWS[1]], result["peak"]
    return [
        practice.Check(
            "ANSWER: loss 0.201 / 0.049 / 0.002 at epochs 1, 10, 100; novel views clear the null model by 20",
            result["first"] is not None and result["first"] <= LATE,
            f"{VIEWS[0]} ray-traced {SIZE}x{SIZE} views, the lesson's own `TinyNeRF` and `volumetric_render`, Adam "
            f"lr {LR}: train MSE {rich[1]['mse']:.5f} / {rich[10]['mse']:.5f} / {rich[EPOCHS]['mse']:.5f}. "
            f"\"Recognisable\" is the best single colour ({rich['baseline']:.2f} dB, {len(HELD_OUT)} midpoint "
            f"poses) +{RECOGNISABLE:.0f} = {result['floor']:.2f}; novel {row(rich)} clears it at {result['first']}"),
        practice.Check(
            "FINDING: the statistic the exercise asks for stops tracking the question it asks",
            rich[LATE]["mse"] / rich[EPOCHS]["mse"] > 3 and abs(rich[EPOCHS]["novel"] - rich[LATE]["novel"]) < 1,
            f"from epoch {LATE} to {EPOCHS} train MSE drops {rich[LATE]['mse'] / rich[EPOCHS]['mse']:.1f}x while novel "
            f"PSNR moves {rich[EPOCHS]['novel'] - rich[LATE]['novel']:+.2f} dB (best at epoch {peak}, "
            f"{rich[peak]['novel']:.2f} dB) and training-camera PSNR, -10log10(MSE), climbs "
            f"{decibels(rich[LATE]['mse']):.1f} -> {decibels(rich[EPOCHS]['mse']):.1f} dB: it measures memorisation"),
        practice.Check(
            "CONTROL: at 8 views no epoch is the answer — perfect training views, null novel views",
            decibels(thin[EPOCHS]["mse"]) > 30 and thin[EPOCHS]["novel"] < thin["baseline"] + 1.0,
            f"the same model and schedule on {VIEWS[1]} views drives train MSE to {thin[EPOCHS]['mse']:.5f} -- "
            f"{decibels(thin[EPOCHS]['mse']):.1f} dB on its own cameras and "
            f"{rich[EPOCHS]['mse'] / thin[EPOCHS]['mse']:.0f}x below the {VIEWS[0]}-view arm -- yet novel "
            f"{row(thin)} never clears {thin['baseline']:.2f} + {RECOGNISABLE:.0f}. View count binds, not epoch"),
        practice.Check(
            "MECHANISM: the quadrature is exactly 1 - prod(1 - alpha), so a zero density renders pure black",
            result["telescope"] < 5e-7 and result["dead"] >= 3,
            f"the product telescopes at every prefix, not just the total: over 64 random rays "
            f"max_k |cumsum(w)_k - (1 - cumprod(1-alpha)_k)| is {result['telescope']:.1e}, float32 round-off at "
            f"order 1, and the 1e10 final delta pins the total at {result['mass']:.6f} whenever sigma > 0 -- a "
            f"partition of unity with no background term. So where `relu(self.sigma(h))` is 0 everywhere the render "
            f"is black and ReLU's zero derivative closes the only path back: {result['dead']}/{DEAD_SEEDS} seeds sit "
            f"at gradient norm exactly 0.0 on an all-zero image, pinned at mean square {result['black']:.4f}"),
        practice.Check(
            "MECHANISM: 6 of the 10 encoding bands sit above the Nyquist limit for this render's spacing",
            len(ALIASED) == 6 and abs(result["overlap"] - OVERLAP) < 1e-4,
            f"{SAMPLES} samples over [{NEAR}, {FAR}] is a step of {STEP:.5f}, so band l advances 2^l*pi*{STEP:.4f} "
            f"rad per step and bands {list(ALIASED)} exceed pi -- the top turns {band(9) / (2 * math.pi):.1f} cycles "
            f"between neighbours. Two points one step apart encode to a dot product of {result['overlap']:.5f}, "
            f"matching 3*sum_l cos(2^l pi d) = {OVERLAP:.5f} to {abs(result['overlap'] - OVERLAP):.1e} "
            f"({3 * BANDS} for identical points)"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
