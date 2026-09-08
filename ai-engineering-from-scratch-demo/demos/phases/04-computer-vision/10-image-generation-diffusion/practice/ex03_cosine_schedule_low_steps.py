"""Exercise 3 — cosine schedule low steps.

    **(Hard)** Implement a cosine noise schedule (Nichol & Dhariwal, 2021): `alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi / 2)`. Train the same model with linear and cosine schedules and show that cosine gives better samples at low step counts.

Reading of the exercise: the formula as quoted is only half the schedule. The
lesson's `precompute_schedule` wants *betas*, so the paper's second half --
`beta_t = 1 - a_bar(t)/a_bar(t-1)`, clipped to 0.999 -- has to be supplied, and
the paper's normalisation `a_bar(t)/a_bar(0)` too, or a_bar(0) is 0.99984 rather
than 1. Both are implemented and checked back against the closed form. "Show that
cosine gives better samples" is a directed verdict, and run literally it is false
here by two orders of magnitude: the 0.999 clip drops a_bar_T to 2.4e-9, and
exercise 2's reverse-chain gain 1/sqrt(a_bar_T) is then 20,291 against linear's
157. What the exercise is really claiming is a property of the schedule's *shape*,
so a third arm clips beta at 0.1 to put cosine's terminal SNR beside linear's, and
only there does the claim come back -- on both seeds and all three step counts.
Quality is a 1-D Wasserstein distance between the sample and data pixel
distributions; five 20-epoch trainings on 200 16x16 images, ~10 s in total. At 149
lines of code this file is over D14's 120-line target and under its 150-line
ceiling: three schedule arms, two seeds, three step counts, a closed-form
reconstruction of the paper's alpha_bar and a matched-log-SNR loss probe are five
deliverables the exercise's single sentence quietly asks for.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "10-image-generation-diffusion"

T, S, EPOCHS, BATCH, LR = 1000, 0.008, 20, 32, 1e-3
IMAGES, SAMPLES, STEPS, SEEDS = 200, 16, (10, 20, 50), (0, 1)
PAPER, MATCHED = 0.999, 0.1        # the paper's beta clip, and the SNR-matched control's
PROBES = (-8.0, -4.0, -2.0, 0.0, 2.0, 4.0)      # log-SNR values the two arms are compared at

listing = lambda w: "  ".join(f"{n}:{w[n]:7.3f}" for n in STEPS)                    # noqa: E731
curve = lambda row: " ".join(f"{g:+.0f}:{v:.3f}" for g, v in zip(PROBES, row))      # noqa: E731
pair = lambda a, s: f"seed {s} {listing(a[f'matched{s}']['w1'])} vs {listing(a[f'linear{s}']['w1'])}"  # noqa: E731
log_snr = lambda th, s: th.log(s["alphas_cumprod"].double().div(                    # noqa: E731
    1 - s["alphas_cumprod"].double()))
wins = lambda a: sum(a[f"matched{s}"]["w1"][n] < a[f"linear{s}"]["w1"][n]           # noqa: E731
                     for n in STEPS for s in SEEDS)
best = lambda a: min(a["matched0"]["w1"][n] / a["linear0"]["w1"][n] for n in STEPS)  # noqa: E731
spread = lambda a: max(abs(x - y) for x, y in                                       # noqa: E731
                       zip(a["matched0"]["probe"], a["linear0"]["probe"]))
shape_of = lambda th, s: {"tail": float(s["alphas_cumprod"][-1]),   # noqa: E731 - what the chain undoes
                          "gain": float(1 / th.sqrt(s["alphas_cumprod"][-1])),
                          "half": int((s["alphas_cumprod"] < 0.5).nonzero()[0]),
                          "dead": float((log_snr(th, s) < -5).double().mean())}


def cosine_betas(torch, clip=PAPER):
    """Nichol & Dhariwal: a_bar from the cosine, then betas from consecutive ratios."""
    grid = torch.arange(T + 1, dtype=torch.float64)
    alpha_bar = (lambda f: f / f[0])(torch.cos((grid / T + S) / (1 + S) * math.pi / 2) ** 2)
    return (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=clip).float(), alpha_bar[1:]


def train(torch, ref, loader_cls, dataset_cls, data, schedule, seed) -> tuple:
    """The lesson's own TinyUNet and train_step, unchanged apart from the schedule."""
    torch.manual_seed(seed)
    loader = loader_cls(dataset_cls(data), batch_size=BATCH, shuffle=True)
    model, losses = ref.TinyUNet(img_channels=3, base=16), []
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(EPOCHS):
        losses = [ref.train_step(model, b, schedule, optimiser, "cpu", T=T) for (b,) in loader]
    return model, sum(losses) / len(losses)


def measure(torch, ref, model, schedule, sorted_data, data) -> dict:
    """DDIM quality as 1-D optimal transport to the data's pixels, plus eps-MSE per log-SNR."""
    def w1(steps):
        torch.manual_seed(7)
        drawn = torch.sort(ref.sample_ddim(model, schedule, (SAMPLES, 3, 16, 16), steps=steps,
                                           T=T).flatten().double()).values
        grid = torch.linspace(0, sorted_data.numel() - 1, drawn.numel()).long()
        return float((drawn - sorted_data[grid]).abs().mean())
    snr, probe, batch = log_snr(torch, schedule), [], data[:64]
    model.eval()
    with torch.no_grad():
        for target in PROBES:      # matched log-SNR, not matched t, so the arms are comparable
            torch.manual_seed(11)
            stamp = torch.full((64,), int((snr - target).abs().argmin()), dtype=torch.long)
            noise = torch.randn_like(batch)
            noisy = ref.q_sample(batch, stamp, noise, schedule)
            probe.append(float(((model(noisy, stamp) - noise) ** 2).mean()))
    return {"w1": {steps: w1(steps) for steps in STEPS}, "probe": probe}


def solve():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    paper, analytic = cosine_betas(torch)
    schedules = {"linear": ref.precompute_schedule(ref.linear_beta_schedule(T=T)),
                 "cosine": ref.precompute_schedule(paper),
                 "matched": ref.precompute_schedule(cosine_betas(torch, MATCHED)[0])}
    data = ref.synthetic_circles(num=IMAGES, size=16, seed=0)
    sorted_data, arms = torch.sort(data.flatten().double()).values, {}
    for name, schedule in schedules.items():
        for seed in (SEEDS if name != "cosine" else SEEDS[:1]):
            model, loss = train(torch, ref, DataLoader, TensorDataset, data, schedule, seed)
            arms[f"{name}{seed}"] = dict(mse=loss, **measure(torch, ref, model, schedule,
                                                             sorted_data, data))
    return {"arms": arms, "shape": {n: shape_of(torch, s) for n, s in schedules.items()},
            "recon": float((schedules["cosine"]["alphas_cumprod"].double() - analytic).abs().max()),
            "unnormalised": float(math.cos(S / (1 + S) * math.pi / 2) ** 2), "wins": wins(arms),
            "best": best(arms), "spread": spread(arms), "data_std": float(data.std()),
            "clipped": int((paper >= PAPER - 1e-9).sum())}


def verify(result):
    arms, shape, won = result["arms"], result["shape"], result["wins"]
    gap, points = arms["matched0"]["mse"] - arms["linear0"]["mse"], len(STEPS) * len(SEEDS)
    top = [arms[n]["probe"][-1] for n in ("linear0", "matched0")]
    return [
        practice.Check(
            "ANSWER: the schedule reproduces the paper's alpha_bar to 3e-7, clip and normalisation included",
            result["recon"] < 1e-5 and result["clipped"] == 1,
            f"betas from `1 - a_bar(t)/a_bar(t-1)` clipped at {PAPER} (binding on exactly {result['clipped']} "
            f"step, the last) go back through the lesson's own `precompute_schedule` to within "
            f"{result['recon']:.1e} of cos^2((t/T+{S})/(1+{S})*pi/2) over a_bar(0); unnormalised the chain "
            f"would start at {result['unnormalised']:.5f}, not 1"),
        practice.Check(
            "FINDING: run as written the claim is false by 200x -- cosine is far worse, not better",
            arms["cosine0"]["w1"][STEPS[0]] > 100 * arms["linear0"]["w1"][STEPS[0]],
            f"same model, same {EPOCHS} epochs, DDIM Wasserstein-1 to the data's pixels at {STEPS} steps: "
            f"linear {listing(arms['linear0']['w1'])}; cosine {listing(arms['cosine0']['w1'])}. Against a "
            f"data std of {result['data_std']:.3f} both are bad; cosine is catastrophic"),
        practice.Check(
            "MECHANISM: the 0.999 clip leaves a_bar_T at 2.4e-9, so the reverse gain is 20,000 not 157",
            shape["cosine"]["gain"] > 100 * shape["linear"]["gain"],
            f"terminal a_bar linear {shape['linear']['tail']:.3g} vs cosine {shape['cosine']['tail']:.3g}, so "
            f"exercise 2's reverse gain 1/sqrt(a_bar_T) is {shape['linear']['gain']:.1f} vs "
            f"{shape['cosine']['gain']:,.0f}. Cosine sends a_bar to 0 at t=T by definition and only the clip "
            "stops it -- that one step, not the schedule's shape, is the 200x"),
        practice.Check(
            "CONTROL: match the terminal SNR and the exercise's claim comes back, on every point",
            won == points and shape["matched"]["gain"] < shape["linear"]["gain"],
            f"clipping cosine's beta at {MATCHED} puts a_bar_T at {shape['matched']['tail']:.3g}, gain "
            f"{shape['matched']['gain']:.1f}, below linear's {shape['linear']['gain']:.1f}, shape unchanged "
            f"-- and it beats linear at {won}/{points} step-count x seed points by up to "
            f"{(1 - result['best']) * 100:.0f}%. {pair(arms, 0)}; {pair(arms, 1)}"),
        practice.Check(
            "MECHANISM: linear burns 30% of its steps on log-SNR below -5, cosine 5%",
            shape["linear"]["dead"] > 5 * shape["matched"]["dead"],
            f"linear's a_bar crosses 0.5 at t={shape['linear']['half']} and spends "
            f"{shape['linear']['dead']:.1%} of the chain at log-SNR < -5, where x_t carries no image; "
            f"cosine crosses at t={shape['matched']['half']} (closed form T*(1-s)/2 = {T * (1 - S) / 2:.0f}) "
            f"and spends {shape['matched']['dead']:.1%} -- at {STEPS[0]} steps a third of linear's DDIM "
            "budget lands where there is nothing to undo"),
        practice.Check(
            "CONTROL: cosine's higher training loss is the schedule, not a worse model",
            gap > 0.05 and result["spread"] < 0.10 and top[1] < top[0],
            f"cosine reports MSE {arms['matched0']['mse']:.4f} against linear's {arms['linear0']['mse']:.4f}"
            f", {gap:+.4f} worse -- but scored at matched log-SNR rather than matched t they agree to "
            f"{result['spread']:.3f}: linear {curve(arms['linear0']['probe'])}; cosine "
            f"{curve(arms['matched0']['probe'])}, the better at the informative end ({top[1]:.3f} vs "
            f"{top[0]:.3f} at +4). eps is near-free below log-SNR -5 and linear puts 6x as many t there"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
