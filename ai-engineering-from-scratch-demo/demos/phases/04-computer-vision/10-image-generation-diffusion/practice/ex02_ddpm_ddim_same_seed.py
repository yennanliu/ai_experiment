"""Exercise 2 — ddpm ddim same seed.

    **(Medium)** Train the TinyUNet on the synthetic-circles dataset for 20 epochs and sample 16 circles. Compare DDPM (1000 steps) and DDIM (50 steps) sampling — do they produce similar images from the same noise seed?

Reading of the exercise: "the same noise seed" is read strictly -- both samplers
draw x_T with their first `torch.randn`, so reseeding immediately before each call
hands them a bit-identical x_T, and that is checked rather than assumed, because
otherwise any answer about similarity is an answer about seeding. The exercise's
own premise is the interesting part: at 20 epochs neither sampler produces a
circle. The lesson's `main()` already shows this at T=200 -- it prints sample
ranges of [-19, 23] against data in [-1, 1] -- and at the T=1000 this exercise
implies, both samplers leave the data range by two orders of magnitude. So
"similar?" is answered twice: at the budget the exercise names, and at 5x that
budget, where the two answers disagree. The mechanism is a closed-form gain the
reverse chain applies to x, which is asserted exactly. Training is 20 epochs on
200 16x16 images and the control is 100; with two DDPM-1000 sampling passes that
is ~12 s in total, which is why there is no third arm.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "10-image-generation-diffusion"

T, EPOCHS, LONGER, BATCH, LR, IMAGES, SAMPLES, DDIM_STEPS, SEED = 1000, 20, 100, 32, 1e-3, 200, 16, 50, 7
GPU = "at scale `python train.py --data cifar10 --steps 1000 --epochs 500` is ~6 h on an A10G (~$2.50 spot)"

spread = lambda s: f"std {s['std']:.3f}, range [{s['min']:.1f}, {s['max']:.1f}]"     # noqa: E731
describe = lambda x: {"std": float(x.std()), "min": float(x.min()),                  # noqa: E731
                      "max": float(x.max()), "in_range": float(((x >= -1) & (x <= 1)).double().mean())}
correlate = lambda th, a, b: float(th.corrcoef(th.stack(                             # noqa: E731
    [a.flatten().double(), b.flatten().double()]))[0, 1])


def train(torch, ref, loader_cls, dataset_cls, data, schedule, epochs) -> tuple:
    """The lesson's own train_step, unmodified, at the exercise's 20 epochs."""
    torch.manual_seed(0)
    loader = loader_cls(dataset_cls(data), batch_size=BATCH, shuffle=True)
    model, losses = ref.TinyUNet(img_channels=3, base=16), []
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(epochs):
        losses = [ref.train_step(model, b, schedule, optimiser, "cpu", T=T) for (b,) in loader]
    return model, sum(losses) / len(losses)


def arm(torch, ref, model, schedule) -> dict:
    """Both samplers run from a bit-identical x_T, which the seeding control checks separately."""
    shape = (SAMPLES, 3, 16, 16)
    torch.manual_seed(SEED)
    ddpm = ref.sample_ddpm(model, schedule, shape, T=T)
    torch.manual_seed(SEED)      # the same first randn, so both chains open on the same x_T
    ddim = ref.sample_ddim(model, schedule, shape, steps=DDIM_STEPS, T=T)
    return {"ddpm": describe(ddpm), "ddim": describe(ddim), "corr": correlate(torch, ddpm, ddim),
            "per_image": sorted(correlate(torch, ddpm[i], ddim[i]) for i in range(SAMPLES))}


def gains(torch, schedule) -> dict:
    """Both reverse chains multiply x by a fixed factor; both factors are closed forms."""
    bar, grid = schedule["alphas_cumprod"], torch.linspace(T - 1, 0, DDIM_STEPS + 1).long()
    return {"ddpm": float(torch.prod(1.0 / torch.sqrt(schedule["alphas"]))),
            "ddim": float(torch.prod(torch.sqrt(bar[grid[1:]] / bar[grid[:-1]]))),
            "ddpm_form": float(1.0 / torch.sqrt(bar[T - 1])),
            "ddim_form": float(torch.sqrt(bar[0] / bar[T - 1]))}


def solve():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    schedule = ref.precompute_schedule(ref.linear_beta_schedule(T=T))
    data = ref.synthetic_circles(num=IMAGES, size=16, seed=0)
    out = {"data": describe(data), "gains": gains(torch, schedule)}
    for key, epochs in (("short", EPOCHS), ("long", LONGER)):
        model, loss = train(torch, ref, DataLoader, TensorDataset, data, schedule, epochs)
        out[key] = dict(arm(torch, ref, model, schedule), mse=loss, epochs=epochs)
    torch.manual_seed(SEED)                     # the seeding control: is x_T really the same?
    first = torch.randn(SAMPLES, 3, 16, 16)
    torch.manual_seed(SEED)
    out["same_start"] = float((first - torch.randn(SAMPLES, 3, 16, 16)).abs().max())
    return out


def verify(result):
    short, long, gain, data = result["short"], result["long"], result["gains"], result["data"]
    return [
        practice.Check(
            "ANSWER: no -- from the identical x_T the two samplers agree only about half a correlation",
            0.2 < short["corr"] < 0.8,
            f"after {EPOCHS} epochs (final MSE {short['mse']:.4f}) the {SAMPLES} DDPM-{T} and DDIM-{DDIM_STEPS} "
            f"images correlate {short['corr']:.3f} pooled, per image {short['per_image'][0]:.3f} to "
            f"{short['per_image'][-1]:.3f}. Same noise, same weights, same schedule -- the 20x cheaper "
            "sampler is landing somewhere else entirely"),
        practice.Check(
            "CONTROL: the x_T really is identical, so the disagreement is not a seeding artefact",
            result["same_start"] == 0.0,
            f"both `sample_ddpm` and `sample_ddim` open with `torch.randn(shape)`, so reseeding to {SEED} "
            f"before each gives x_T differing by exactly {result['same_start']:.1f}. All that follows is the "
            f"sampler: DDPM injects fresh noise at each of the {T - 1} remaining steps, DDIM at eta=0 none"),
        practice.Check(
            "FINDING: neither output is a circle -- both leave the data range by ~100x",
            short["ddpm"]["std"] > 10 * data["std"] and short["ddim"]["std"] > 10 * data["std"],
            f"the circles live in [-1, 1] with std {data['std']:.3f}. DDPM: {spread(short['ddpm'])}, "
            f"{short['ddpm']['in_range']:.1%} of pixels in range; DDIM: {spread(short['ddim'])}, "
            f"{short['ddim']['in_range']:.1%}. main() shows the same at T=200: two failures. {GPU}"),
        practice.Check(
            "MECHANISM: the reverse chain multiplies x by 1/sqrt(alpha_bar_T), and that is 157",
            abs(gain["ddpm"] - gain["ddpm_form"]) < 1e-3 and abs(gain["ddim"] - gain["ddim_form"]) < 1e-3,
            f"DDPM's mean is sqrt(1/alpha_t)*(x - c*eps), so its gain on x is prod_t 1/sqrt(alpha_t) = "
            f"{gain['ddpm']:.4f} = 1/sqrt(alpha_bar_{T - 1}) = {gain['ddpm_form']:.4f}, to "
            f"{abs(gain['ddpm'] - gain['ddpm_form']):.1e}. DDIM at eta=0 keeps sqrt(a_prev/a_t) a step, "
            f"telescoping to sqrt(a_0/a_{T - 1}) = {gain['ddim_form']:.4f} vs {gain['ddim']:.4f} measured -- "
            "any bias the eps-net has at high t is amplified 157-fold"),
        practice.Check(
            "FINDING: train 5x longer and the answer flips -- DDPM recovers, DDIM does not",
            long["ddpm"]["std"] < 0.2 * short["ddpm"]["std"] and long["ddim"]["std"] > long["ddpm"]["std"],
            f"at {LONGER} epochs (MSE {long['mse']:.4f}) DDPM is {spread(long['ddpm'])}, "
            f"{long['ddpm']['in_range']:.1%} in range against the data's std {data['std']:.3f} -- DDIM is "
            f"still {spread(long['ddim'])}, {long['ddim']['in_range']:.1%}. DDPM's per-step noise injection "
            f"re-randomises the accumulated error; DDIM's {DDIM_STEPS} deterministic steps keep it"),
        practice.Check(
            "CONTROL: 'similar?' has no stable answer at this budget -- the correlation collapses",
            long["corr"] < 0.5 * short["corr"],
            f"at {LONGER} epochs the same measurement gives {long['corr']:.3f} pooled, per image "
            f"{long['per_image'][0]:.3f} to {long['per_image'][-1]:.3f}, against {short['corr']:.3f} at "
            f"{EPOCHS}: 5x the budget flips it from 'roughly half' to 'not at all', so the {EPOCHS}-epoch "
            f"answer is not a property of the two samplers"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
