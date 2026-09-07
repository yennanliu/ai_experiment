"""Exercise 1 — cfg scale artefact onset.

    **(Easy)** Generate the same prompt with `guidance_scale` in `[1, 3, 5, 7.5, 10, 15]`. Describe how the image changes. At what guidance value do artefacts appear?

Reading of the exercise: nothing in this lesson can generate an image here. Its
own `has_diffusers()` is False, `torch.cuda.is_available()` is False, and its
`text_to_image_stub` therefore returns None -- all three are measured below, and
downloading `runwayml/stable-diffusion-v1-5` is out of bounds for these
solutions. So the sweep runs against a surrogate whose noise predictor is *exact*
rather than learnt: prompt embeddings e ~ N(0, M^2 I) and latents x0 | e ~ N(e,
S^2 I) make both halves of `eps_uncond + w * (eps_cond - eps_uncond)` closed
form, and they are fed to lesson 10's own `sample_ddim` on lesson 10's own
`linear_beta_schedule`, at the 25 steps and the guidance scale this lesson's
`text_to_image_stub` itself passes. That buys three things a real pipeline could
not: the reference's own `cfg_sweep_demo` labels become checkable, the whole
sampler is provably affine so its output is a closed form rather than a picture
to squint at, and "at what guidance value do artefacts appear" gets a number.
"Artefact" is the ambiguity: SD's high-guidance failure is oversaturation, values
the decoder cannot represent, so the metric is the fraction of sample values
outside the data's own observed range -- which makes the data's own rate exactly
0 by construction. The onset turns out to be smooth, so the answer is a curve and
naming one scale would be reading noise. The whole file runs in about half a
second: no training happens in it anywhere, because an exact score needs none.

`Guided` is everything a denoiser has to be for `sample_ddim`, which only calls
`.eval()` and `model(x, t)`. For x_t | e ~ N(sqrt(a) e, (a s^2 + 1 - a) I) the
noise prediction is `sqrt(1-a) (x_t - sqrt(a) mu) / v`; the null prompt is mu = 0
with the marginal variance M^2 + S^2, which is what a CFG-trained net learns from
the 10% of steps where the caption is dropped. Because that eps is affine in x,
`chain_gains` can collapse the entire 25-step chain to `x_0 = G x_T + H e`. At
144 lines of code the file is over D14's 120-line target and under its ceiling:
six checks, a second x_T seed, and two closed forms -- the affine chain and the
density guidance is named after -- are what the question's three sentences ask.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "11-stable-diffusion"
DIFFUSION = "10-image-generation-diffusion"

T, STEPS, N, SHAPE = 1000, 25, 64, (3, 16, 16)   # STEPS is the lesson's own stub default
SCALES = (1.0, 3.0, 5.0, 7.5, 10.0, 15.0)        # the exercise's list, and cfg_sweep_demo's
M, S = 0.15, 0.45                                # prompt-mean spread, and latent spread per prompt
XT_SEEDS = (11, 12)

fmt = lambda t, key, spec=".3f": " ".join(f"{w:g}:{t[w][key]:{spec}}" for w in SCALES)   # noqa: E731
spread = lambda a, b, key: max(abs(a[w][key] - b[w][key]) for w in SCALES)               # noqa: E731
climbs = lambda t: all(t[a]["outside"] <= t[b]["outside"] for a, b in zip(SCALES, SCALES[1:]))  # noqa: E731
misfit = lambda t, g: max(abs(t[w]["std"] - (g[w][0]**2 + g[w][1]**2 * M**2)**0.5) for w in SCALES)  # noqa: E731


class Guided:
    """`eps_uncond + w (eps_cond - eps_uncond)`, both halves exact for the surrogate."""
    eval = staticmethod(lambda: None)      # all `sample_ddim` asks of a denoiser, besides __call__

    def __init__(self, prompt, weight, alphas):
        self.prompt, self.weight, self.alphas = prompt, weight, alphas

    def __call__(self, x, t):
        a = self.alphas[t].view(-1, 1, 1, 1)
        cond = (x - a.sqrt() * self.prompt) / (a * S ** 2 + 1 - a)
        uncond = x / (a * (M ** 2 + S ** 2) + 1 - a)
        return (1 - a).sqrt() * (uncond + self.weight * (cond - uncond))


def chain_gains(grid, alphas, weight):
    noise, prompt = 1.0, 0.0
    for i in range(STEPS):
        a, prev = alphas[int(grid[i])].double(), alphas[int(grid[i + 1])].double()
        root, v_cond, v_null = (1 - a).sqrt(), a * S ** 2 + 1 - a, a * (M ** 2 + S ** 2) + 1 - a
        step = (1 - prev).sqrt() - (prev / a).sqrt() * root
        carry = (prev / a).sqrt() + step * root * (1 / v_null + weight * (1 / v_cond - 1 / v_null))
        noise, prompt = noise * carry, prompt * carry - step * root * weight * a.sqrt() / v_cond
    return float(noise), float(prompt)


def sweep(torch, diffusion, schedule, prompt, gamut, seed) -> dict:
    table = {}
    for weight in (0.0,) + SCALES:
        torch.manual_seed(seed)
        out = diffusion.sample_ddim(Guided(prompt, weight, schedule["alphas_cumprod"]),
                                    schedule, (N,) + SHAPE, steps=STEPS, T=T)
        over = out.abs() > gamut
        cosine = torch.nn.functional.cosine_similarity(out.flatten(1), prompt.flatten(1))
        table[weight] = {"std": float(out.std()), "peak": float(out.abs().max()), "adherence":
                         float(cosine.mean()), "count": int(over.sum()),
                         "outside": float(over.double().mean())}
    return table


def solve():
    try:
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    reference = parity.load_reference(PHASE, LESSON, "main")
    diffusion = parity.load_reference(PHASE, DIFFUSION, "main")
    torch.set_num_threads(2)
    schedule = diffusion.precompute_schedule(diffusion.linear_beta_schedule(T=T))
    grid, generator = torch.linspace(T - 1, 0, STEPS + 1).long(), torch.Generator().manual_seed(0)
    prompt = torch.randn((N,) + SHAPE, generator=generator) * M
    data = prompt + torch.randn((N,) + SHAPE, generator=generator) * S
    gamut, printed = float(data.abs().max()), io.StringIO()
    with contextlib.redirect_stdout(printed):          # both of these are pure print statements
        reference.cfg_sweep_demo()
        stub = reference.text_to_image_stub("a dog riding a skateboard in tokyo")
    said = [ln.split("expected:") for ln in printed.getvalue().splitlines() if "expected:" in ln]
    return {"tables": [sweep(torch, diffusion, schedule, prompt, gamut, s) for s in XT_SEEDS],
            "labels": {float(head.split("=")[1]): verdict.strip() for head, verdict in said},
            "gains": {w: chain_gains(grid, schedule["alphas_cumprod"], w) for w in SCALES},
            "sharpened": {w: {"std": (w / S**2 + (1 - w) / (M**2 + S**2))**-0.5} for w in SCALES},
            "adherence": float(torch.nn.functional.cosine_similarity(
                data.flatten(1), prompt.flatten(1)).mean()), "gamut": gamut,
            "runnable": [reference.has_diffusers(), torch.cuda.is_available(), stub]}


def verify(result):
    table, second = result["tables"]
    gains, sharp, labels = result["gains"], result["sharpened"], result["labels"]
    return [
        practice.Check(
            "ANSWER: artefacts have no onset -- 0 at w=1, 4 stray values in 49,152 at w=3, 12.9% at 15",
            all([table[1.0]["count"] == 0, table[5.0]["count"] > 100, climbs(table),
                 table[15.0]["outside"] > 0.05]),
            f"share of sample values outside the data's own range (|x| > {result['gamut']:.3f}, so the data "
            f"scores 0.00% by construction): {fmt(table, 'outside', '.2%')}, counts "
            f"{fmt(table, 'count', 'd')} of {N * 768:,}; w=1 peaks at {table[1.0]['peak']:.3f} and w=3 at "
            f"{table[3.0]['peak']:.3f}, so naming one scale reads noise. cfg_sweep_demo calls 5 and 7.5 both "
            f"{labels[5.0]!r} while the rate rises {table[7.5]['outside'] / table[5.0]['outside']:.0f}x there"),
        practice.Check(
            "FINDING: the lesson's own sweep mislabels w=1 unconditional; it is the plain conditional",
            all(["unconditional" in labels[1.0], abs(table[0.0]["adherence"]) < 0.02,
                 table[1.0]["adherence"] > 0.9 * result["adherence"]]),
            f"cfg_sweep_demo prints {labels[1.0]!r} for w=1.0, but w=1 cancels eps_uncond outright. Cosine "
            f"to the prompt: w=0 {table[0.0]['adherence']:+.4f}, w=1 {table[1.0]['adherence']:+.4f}, against "
            f"the data's own {result['adherence']:.4f}. The prose above it has this right -- 'w=0 is "
            "unconditional, w=1 is plain conditional' -- and the code beside it does not"),
        practice.Check(
            "MECHANISM: guidance buys adherence with diversity at a closed-form exchange rate",
            all([misfit(table, gains) < 5e-3, gains[15.0][0] < 0.5 * gains[1.0][0]]),
            f"an affine eps collapses the 25-step chain to x_0 = G x_T + H e. G, all the diversity one prompt "
            f"has left, falls {gains[1.0][0]:.3f} -> {gains[15.0][0]:.3f} while H climbs "
            f"{gains[1.0][1]:.3f} -> {gains[15.0][1]:.3f}; sqrt(G^2 + H^2 M^2) then reproduces the measured "
            f"std {fmt(table, 'std')} to {misfit(table, gains):.1e}"),
        practice.Check(
            "MECHANISM: oversaturation is CFG *not* sampling the density it is named after",
            all([sharp[15.0]["std"] < sharp[1.0]["std"], table[15.0]["std"] > 3 * sharp[15.0]["std"]]),
            f"p_cond^w p_uncond^(1-w) is Gaussian with std {fmt(sharp, 'std')} -- guidance is meant to "
            f"*shrink* the spread. Applying that score at every t instead grows it: {fmt(table, 'std')}, so "
            f"w=15 samples are {table[15.0]['std'] / sharp[15.0]['std']:.1f}x wider than the density they "
            "name. Guided scores are not a diffusion path, and that gap is the artefact"),
        practice.Check(
            "CONTROL: adherence saturates long before the artefacts do",
            all([table[7.5]["adherence"] > 0.95, table[15.0]["adherence"] - table[7.5]["adherence"] < 0.05]),
            f"cosine to the prompt {fmt(table, 'adherence')}: from w=7.5 to w=15 adherence gains "
            f"{table[15.0]['adherence'] - table[7.5]['adherence']:+.4f} while the out-of-gamut rate goes "
            f"{table[7.5]['outside']:.2%} -> {table[15.0]['outside']:.2%}. That is why the stub's own default "
            f"is {SCALES[3]:g}, and why {labels[15.0]!r} is the right label for 15"),
        practice.Check(
            "CONTROL: no Stable Diffusion ran here, and the sweep does not depend on x_T",
            all([result["runnable"] == [False, False, None], spread(table, second, "std") < 0.01,
                 spread(table, second, "adherence") < 0.01]),
            f"has_diffusers() is {result['runnable'][0]}, torch.cuda.is_available() {result['runnable'][1]}, "
            f"text_to_image_stub returns {result['runnable'][2]} -- every number above is the surrogate's. "
            f"Across x_T seeds {XT_SEEDS} std moves at most {spread(table, second, 'std'):.4f}, adherence "
            f"{spread(table, second, 'adherence'):.4f}, out-of-gamut {spread(table, second, 'outside'):.4f}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
