"""Exercise 1 — forward process snr decay.

    **(Easy)** Visualise the forward process: take one image and plot `x_t` at `t in [0, 100, 250, 500, 750, 1000]`. Verify that `x_1000` looks like pure Gaussian noise.

Reading of the exercise: a plot cannot be asserted, so "visualise" is answered by
the numbers a reader of the plot would be reading off it -- the mean, the standard
deviation and the correlation with `x0` at each of the six timesteps -- and each is
compared against the closed form `q(x_t | x_0)` implies. Two things in the
exercise do not survive contact with the code. First, `t = 1000` does not exist:
`linear_beta_schedule(T=1000)` returns 1000 betas indexed 0..999, so `q_sample`
raises IndexError on 1000; the last step is 999 and that is what is measured.
Second, "verify that it looks like pure Gaussian noise" is not one claim but two,
and they come true 500 steps apart -- the marginal *shape* is already Gaussian at
t=500 while the image is still plainly there, and the residual signal at t=999 is
below what one image can resolve. Both are measured rather than asserted, and the
sample size each conclusion needs is reported with it.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "10-image-generation-diffusion"

T, ASKED, IMAGES, DRAWS = 1000, (0, 100, 250, 500, 750, 1000), 200, 40
STEPS = (0, 100, 250, 500, 750, 999)      # the exercise's list, with 1000 clamped into range

row = lambda t, k: " ".join(f"{r[k]:+.4f}" for r in t)                                # noqa: E731
share = lambda t: " ".join(f"{r['corr'] ** 2:.4f}" for r in t)                        # noqa: E731
z_of = lambda c, n: c * math.sqrt(n)                                                  # noqa: E731
corr = lambda th, a, b: float(th.corrcoef(th.stack(                                   # noqa: E731
    [a.flatten().double(), b.flatten().double()]))[0, 1])
# what q(x_t|x_0) = N(sqrt(a_bar) x_0, (1 - a_bar) I) predicts for the pooled pixels
closed_form = lambda ab, m0, v0: {"mean": math.sqrt(ab) * m0,                         # noqa: E731
                                  "std": math.sqrt(ab * v0 + 1 - ab),
                                  "corr": math.sqrt(ab * v0 / (ab * v0 + 1 - ab))}


def moments(x) -> dict:
    """Mean, standard deviation, and the two shape statistics a normality eye-test uses."""
    flat = x.flatten().double()
    mean, std = flat.mean(), flat.std(unbiased=False)
    z = (flat - mean) / std
    return {"mean": float(mean), "std": float(std), "skew": float((z ** 3).mean()),
            "exkurt": float((z ** 4).mean() - 3.0)}


def solve():
    try:
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    schedule = ref.precompute_schedule(ref.linear_beta_schedule(T=T))
    data = ref.synthetic_circles(num=IMAGES, size=16, seed=0)
    alpha_bar, mean0, var0 = (schedule["alphas_cumprod"], float(data.mean()),
                              float(data.var(unbiased=False)))
    torch.manual_seed(0)
    measured, predicted, single = [], [], []
    for step in STEPS:
        stamp = torch.full((data.size(0),), step, dtype=torch.long)
        noisy = ref.q_sample(data, stamp, torch.randn_like(data), schedule)
        measured.append(dict(moments(noisy), corr=corr(torch, noisy, data)))
        predicted.append(closed_form(float(alpha_bar[step]), mean0, var0))
        single.append(corr(torch, noisy[0], data[0]))
    try:                    # the exercise asks for t=1000; the schedule stops at 999
        ref.q_sample(data[:1], torch.full((1,), ASKED[-1], dtype=torch.long), data[:1], schedule)
        overflow = "no error"
    except IndexError as exc:
        overflow = f"IndexError: {exc}"
    torch.manual_seed(1)   # DRAWS independent noise draws of the whole set, pooled, at t=999
    stamp = torch.full((data.size(0),), STEPS[-1], dtype=torch.long)
    big = torch.cat([ref.q_sample(data, stamp, torch.randn_like(data), schedule).flatten()
                     for _ in range(DRAWS)])
    return {"alpha_bar": [float(alpha_bar[s]) for s in STEPS], "x0": moments(data), "n": data.numel(),
            "measured": measured, "predicted": predicted, "single": single[-1],
            "pixels": data[0].numel(), "schedule_len": len(alpha_bar), "overflow": overflow,
            "pooled": corr(torch, big, data.flatten().repeat(DRAWS))}


def verify(result):
    got, want, x0, n = result["measured"], result["predicted"], result["x0"], result["n"]
    pixels, tail, big = result["pixels"], want[-1]["corr"], n * DRAWS
    gaps = [max(abs(a[k] - b[k]) for a, b in zip(got, want)) for k in ("mean", "std", "corr")]
    return [
        practice.Check(
            "ANSWER: the six requested t are five -- t=1000 is off the end of the schedule",
            result["schedule_len"] == T and "IndexError" in result["overflow"],
            f"`linear_beta_schedule(T={T})` gives {result['schedule_len']} betas indexed 0..{T - 1}, so "
            f"t={ASKED[-1]} raises `{result['overflow']}`. Measured at {STEPS}, alpha_bar "
            + " ".join(f"{a:.4g}" for a in result["alpha_bar"])),
        practice.Check(
            "ANSWER: every plotted statistic matches q(x_t|x_0) in closed form",
            max(gaps) < 0.01,
            f"over {n:,} pixels at t={STEPS} -- mean {row(got, 'mean')} vs predicted {row(want, 'mean')}; "
            f"std {row(got, 'std')} vs {row(want, 'std')}; corr {row(got, 'corr')} vs "
            f"sqrt(a_bar*Var/(a_bar*Var+1-a_bar)) {row(want, 'corr')}. Worst gaps {gaps[0]:.4f}/"
            f"{gaps[1]:.4f}/{gaps[2]:.4f}, on a {1 / math.sqrt(n):.5f} standard error"),
        practice.Check(
            "FINDING: the marginal is Gaussian-shaped by t=500, 500 steps before the image is gone",
            max(abs(got[3]["skew"]), abs(got[3]["exkurt"])) < 0.05 < got[3]["corr"],
            f"x_0 is strongly non-normal (skew {x0['skew']:+.3f}, excess kurtosis {x0['exkurt']:+.3f}); at "
            f"t=500 that is {got[3]['skew']:+.4f} and {got[3]['exkurt']:+.4f}, indistinguishable from a "
            f"Gaussian at {n:,} samples -- while corr(x_500, x_0) is still {got[3]['corr']:.3f} and the "
            f"mean {got[3]['mean']:+.4f}, not 0. Shape alone does not say the signal is gone"),
        practice.Check(
            "MECHANISM: what decays is alpha_bar, and the correlation is its square root in SNR",
            result["alpha_bar"][-1] * 1e4 < 1 < result["alpha_bar"][0] * 1.01,
            f"alpha_bar falls {result['alpha_bar'][0]:.4f} -> {result['alpha_bar'][-1]:.3g} over {T} steps, "
            f"so the signal's share of the variance, a_bar*Var/(a_bar*Var+1-a_bar) at Var(x_0) = "
            f"{x0['std'] ** 2:.4f}, runs {share(want)}; its square root is the plotted correlation, "
            f"{tail:.4f} at t={STEPS[-1]}"),
        practice.Check(
            "FINDING: 'pure Gaussian noise' at t=999 is a claim about sample size, not about the image",
            abs(z_of(result["single"], pixels)) < 3 < z_of(result["pooled"], big),
            f"one image is {pixels} pixels, where the predicted correlation {tail:.4f} is worth z = "
            f"{z_of(tail, pixels):.2f}; it measured {result['single']:+.4f}, z = "
            f"{z_of(result['single'], pixels):+.2f}. Pool {DRAWS} draws of all {IMAGES} images ({big:,} "
            f"values): residual {result['pooled']:.6f}, z = {z_of(result['pooled'], big):.1f}. 3 sigma "
            f"needs ~{(3 / tail) ** 2:,.0f} pixels -- x_999 is not noise, it is undetectable"),
        practice.Check(
            "CONTROL: at t=999 the residual mean, not the shape, is the only tell left",
            abs(got[-1]["std"] - 1.0) < 0.01 and abs(got[-1]["mean"]) < 0.02 < abs(got[3]["mean"]),
            f"x_999: mean {got[-1]['mean']:+.4f} (closed form {want[-1]['mean']:+.4f} = sqrt(a_bar)*"
            f"{x0['mean']:+.4f}), std {got[-1]['std']:.4f}, skew {got[-1]['skew']:+.4f}, excess kurtosis "
            f"{got[-1]['exkurt']:+.4f} -- N(0, 1) on all four. At t=500 the shape said Gaussian but the "
            f"mean {got[3]['mean']:+.4f} did not, so the eye-test must cover location and scale too"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
