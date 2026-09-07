"""Exercise 3 — frechet rf vs ddpm convergence.

    **(Hard)** Compute the Fréchet distance (FID proxy) between generated samples from rectified-flow and DDPM versions of the same-size network trained on the same data for the same number of steps. Report which converges faster.

Reading of the exercise: taken literally the first sentence measures nothing.
The Fréchet distance *between the two generators* is a distance between two wrong
answers -- it cannot say which is closer to the data, and it is reported here only
to show that it is numerically almost DDPM's own distance to the data, because the
rectified-flow arm has already arrived. What answers "which converges faster" is
each arm's distance to the *data*, checkpointed during training, against a floor
measured by scoring two halves of the real data against each other. Lesson 23
ships no DDPM code, so the second arm is lesson 10's own `train_step`, its linear
beta schedule and its `sample_ddim`, driving the *same* TinyDiT -- both of lesson
10's functions take `model(x_t, t)` and nothing else, so the network is literally
identical between arms and only the objective and the sampler change. The
"same-size network" the exercise asks for is then the same network.

The verdict is not close, and the mechanism is arithmetic rather than optimisation:
DDIM's `x0_pred = (x - sqrt(1-a)*eps) / sqrt(a)` divides by sqrt(alpha_bar), which
at t=999 on the linear schedule is a 157x amplification of whatever error the
eps-prediction still has, while a rectified-flow Euler step adds its velocity with
coefficient dt <= 1 and cannot amplify anything. A control rules out the
architecture: lesson 10's own TinyUNet, under the same objective and the same
budget, is equally far off.

Structure: `frechet` pools each image set to a 4x4 RGB feature and returns the
Gaussian Fréchet distance in float64, the matrix square root taken through
`torch.linalg.eigh`; `draw` re-seeds and samples an arm by its own sampler;
`run_arm` trains one model with one objective and records the Fréchet distance to
the data at the training checkpoints, taking the model as an argument so the
TinyUNet control runs through the same path.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "23-diffusion-transformers-rectified-flow"
DDPM = "10-image-generation-diffusion"

DIM, DEPTH, HEADS, SIZE, IMAGES = 64, 2, 4, 16, 256
TRAIN, BATCH, LR, T = 600, 32, 3e-4, 1000
CHECKPOINTS, SWEEP, DRAWN, POOL, EVAL = (200, 400, 600), (1, 2, 4, 10, 20), 96, 4, 20
SETTLED = SWEEP[-1]

listing = lambda curve, keys: "  ".join(f"{n}:{curve[n]:.3g}" for n in keys)                        # noqa: E731
faster = lambda a, b: all(a[n] < b[n] / 100 for n in CHECKPOINTS) and a[TRAIN] < a[CHECKPOINTS[0]]  # noqa: E731
useless = lambda cross, arms: cross > 100 * arms["rf"][TRAIN] and abs(cross / arms["ddpm"][TRAIN] - 1) < 0.5  # noqa: E731
ragged = lambda s: min(s["ddpm"].values()) < s["ddpm"][SETTLED] and max(s["rf"].values()) < 10    # noqa: E731


def frechet(torch, first, second):
    pooled = [torch.nn.functional.adaptive_avg_pool2d(x, POOL).flatten(1).double()
              for x in (first, second)]
    eye = 1e-6 * torch.eye(pooled[0].size(1), dtype=torch.double)
    left, right = torch.cov(pooled[0].T) + eye, torch.cov(pooled[1].T) + eye
    values, vectors = torch.linalg.eigh(left)
    root = vectors @ torch.diag(values.clamp_min(0).sqrt()) @ vectors.T
    overlap = torch.linalg.eigvalsh(root @ right @ root).clamp_min(0).sqrt().sum()
    return float(((pooled[0].mean(0) - pooled[1].mean(0)) ** 2).sum()
                 + left.trace() + right.trace() - 2 * overlap)


def draw(torch, ref, ddpm, kind, model, steps, schedule):
    torch.manual_seed(3)
    shape = (DRAWN, 3, SIZE, SIZE)
    if kind == "rf":
        return ref.rectified_flow_sample(model, shape, steps=steps, device="cpu")
    return ddpm.sample_ddim(model, schedule, shape, steps=steps, T=T, device="cpu")


def run_arm(torch, ref, ddpm, numpy, kind, model, data, schedule):
    optimiser, generator = torch.optim.Adam(model.parameters(), lr=LR), numpy.random.default_rng(0)
    torch.manual_seed(1)
    curve = {}
    for step in range(1, TRAIN + 1):
        batch = data[generator.choice(len(data), BATCH)]
        if kind == "rf":
            ref.rectified_flow_train_step(model, batch, optimiser, "cpu")
        else:
            ddpm.train_step(model, batch, schedule, optimiser, "cpu", T=T)
        if step in CHECKPOINTS:
            curve[step] = frechet(torch, draw(torch, ref, ddpm, kind, model, EVAL, schedule), data)
    return model, curve


def solve():
    try:
        import numpy
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    ddpm = parity.load_reference(PHASE, DDPM, "main")
    torch.set_num_threads(2)
    schedule = ddpm.precompute_schedule(ddpm.linear_beta_schedule(T=T))
    data = ref.synthetic_blobs(num=IMAGES, size=SIZE, seed=0)
    arms, models, sweep, spread = {}, {}, {}, {}
    for kind in ("rf", "ddpm"):
        torch.manual_seed(0)
        fresh = ref.TinyDiT(image_size=SIZE, patch_size=2, in_channels=3, dim=DIM, depth=DEPTH, heads=HEADS)
        models[kind], arms[kind] = run_arm(torch, ref, ddpm, numpy, kind, fresh, data, schedule)
        sweep[kind] = {n: frechet(torch, draw(torch, ref, ddpm, kind, models[kind], n, schedule), data)
                       for n in SWEEP}
        spread[kind] = float(draw(torch, ref, ddpm, kind, models[kind], EVAL, schedule).std())
    torch.manual_seed(0)
    unet = ddpm.TinyUNet(img_channels=3, base=16)
    unet, curve = run_arm(torch, ref, ddpm, numpy, "ddpm", unet, data, schedule)
    return {"arms": arms, "sweep": sweep, "spread": spread, "unet": curve[TRAIN],
            "unet_params": sum(p.numel() for p in unet.parameters()),
            "unet_spread": float(draw(torch, ref, ddpm, "ddpm", unet, EVAL, schedule).std()),
            "params": sum(p.numel() for p in models["rf"].parameters()),
            "floor": frechet(torch, data[:IMAGES // 2], data[IMAGES // 2:]),
            "gain": float(1 / schedule["alphas_cumprod"][T - 1].sqrt()), "data_spread": float(data.std()),
            "cross": frechet(torch, draw(torch, ref, ddpm, "rf", models["rf"], EVAL, schedule),
                             draw(torch, ref, ddpm, "ddpm", models["ddpm"], EVAL, schedule))}


def verify(result):
    arms, sweep, spread = result["arms"], result["sweep"], result["spread"]
    return [
        practice.Check(
            "ANSWER: rectified flow converges faster by two orders of magnitude, at every checkpoint",
            faster(arms["rf"], arms["ddpm"]),
            f"one TinyDiT of {result['params']:,} parameters, one dataset, one optimiser, one seed, "
            f"{TRAIN} steps, {EVAL}-step sampling: Fréchet distance to the data at {CHECKPOINTS} steps is "
            f"{listing(arms['rf'], CHECKPOINTS)} for rectified flow and {listing(arms['ddpm'], CHECKPOINTS)} "
            f"for lesson 10's DDPM objective. RF's *first* checkpoint beats DDPM's last by "
            f"{arms['ddpm'][TRAIN] / arms['rf'][CHECKPOINTS[0]]:,.0f}x"),
        practice.Check(
            "FINDING: the distance the exercise literally asks for ranks nothing",
            useless(result["cross"], arms),
            f"Fréchet between the two *generators* is {result['cross']:.3g} -- within "
            f"{abs(result['cross'] / arms['ddpm'][TRAIN] - 1):.1%} of DDPM's own distance to the data "
            f"({arms['ddpm'][TRAIN]:.3g}), because the RF arm has already arrived ({arms['rf'][TRAIN]:.3g}) "
            f"and is standing where the data is. Two halves of the real data score {result['floor']:.4f}, "
            "so that is the floor every number here is read against"),
        practice.Check(
            "MECHANISM: DDIM divides by sqrt(alpha_bar) -- a 157x amplifier the Euler step has no analogue of",
            result["gain"] > 100 and spread["ddpm"] > 10 * result["data_spread"],
            f"`sample_ddim` forms `x0_pred = (x - sqrt(1-a)*eps) / sqrt(a)`, and 1/sqrt(alpha_bar) on the "
            f"linear schedule reaches {result['gain']:.1f} at t={T - 1}. Whatever the eps-head still gets "
            f"wrong is multiplied by that: DDPM samples come out at std {spread['ddpm']:.3f} against the "
            f"data's {result['data_spread']:.3f}, while RF's Euler step adds v*dt with dt <= 1 and lands at "
            f"{spread['rf']:.3f}. Nothing in `rectified_flow_sample` can amplify"),
        practice.Check(
            "FINDING: RF is flat in the few-step regime while DDIM is not even monotone",
            ragged(sweep),
            f"sampling the finished models at {SWEEP} steps: RF {listing(sweep['rf'], SWEEP)}, DDIM "
            f"{listing(sweep['ddpm'], SWEEP)}. RF's 1-step sample ({sweep['rf'][1]:.3g}) is already inside "
            f"{sweep['rf'][1] / sweep['rf'][SETTLED]:.1f}x of its {SETTLED}-step one and beats DDIM's "
            f"best ({min(sweep['ddpm'].values()):.3g}) by {min(sweep['ddpm'].values()) / sweep['rf'][1]:,.0f}"
            "x. DDIM gets *worse* past its optimum -- more steps of a bad field is more error, not less"),
        practice.Check(
            "CONTROL: it is the objective and the budget, not the transformer",
            result["unet"] > 10 and result["unet_spread"] > 10 * result["data_spread"],
            f"lesson 10's own TinyUNet ({result['unet_params']:,} parameters) under the same objective, "
            f"schedule, data and {TRAIN} steps scores {result['unet']:.3g} at std {result['unet_spread']:.3f} "
            f"-- {result['unet'] / result['floor']:,.0f}x the floor, the same failure mode as the DiT's "
            f"{arms['ddpm'][TRAIN]:.3g}, so the architecture is not what is being measured. DDPM is not "
            f"broken; it is a {T}-step chain asked to converge in {TRAIN}, which is what 'the same number "
            f"of steps' sets up"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
