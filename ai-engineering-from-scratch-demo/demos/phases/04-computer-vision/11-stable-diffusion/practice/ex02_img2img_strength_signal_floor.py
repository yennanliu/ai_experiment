"""Exercise 2 — img2img strength signal floor.

    **(Medium)** Take any real photograph, run it through `StableDiffusionImg2ImgPipeline` at `strength` in `[0.2, 0.4, 0.6, 0.8, 1.0]`. Which strength preserves composition while changing style? Why does 1.0 ignore the input entirely?

Reading of the exercise: `StableDiffusionImg2ImgPipeline` needs weights this
repo may not download and a GPU it does not have (exercise 1 measures both), so
the pipeline is rebuilt out of the parts that decide the answer. diffusers' img2img
is the ordinary reverse loop entered late: `get_timesteps` keeps the last
`int(steps * strength)` of the sampler's grid, and the loop starts from
`add_noise(latent, noise, t_start)` instead of pure noise. So `img2img` below is
lesson 10's own `sample_ddim` body with exactly that one line changed -- its own
`q_sample` in place of `torch.randn` -- and at strength 1.0 the two agree to
0.009, which is checked rather than asserted. The denoiser is exercise 1's exact
Gaussian surrogate with one addition the question demands: a real photograph's
composition is low spatial frequency and its texture is high, and an isotropic
latent cannot tell those apart, so the latent covariance is split into a coarse
4x4-block subspace (SL) and its complement (SH). "Preserves composition" is then
measurable as the regression coefficient of the output on the input's coarse
band. The transferable claim is the *ratio* between the two bands, not the exact
strength: only 48 of this surrogate's 768 dimensions carry the coarse band, and
a 512x512 photograph concentrates far more of its variance at low frequency than
that, so a real photograph will hold composition to a higher strength than the
0.2 measured here. Guidance is 7.5, the lesson's own img2img snippet's value.

`coarse` is the orthogonal projection onto per-4x4-block means -- this
surrogate's "composition". `Guided` is exercise 1's CFG denoiser over the
two-band latent, and like it needs only `.eval()` and a call to satisfy
`sample_ddim`, with the two-band precision applied by `band`. `img2img` is that
sampler's loop entered `int(steps * strength)` steps from the end; `arm` is one
strength sweep at one guidance scale from one fixed noise draw; and `photograph`
builds a latent whose variance splits between the two bands. At 147 lines of
code the file is over D14's 120-line target and
under its ceiling: two guidance arms, two noise seeds and a band decomposition
are what separating "the input is gone" from "the input is drowned" costs.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON, DIFFUSION = "04-computer-vision", "11-stable-diffusion", "10-image-generation-diffusion"

T, STEPS, N, SHAPE, BLOCK = 1000, 25, 64, (3, 16, 16), 4
STRENGTHS = (0.2, 0.4, 0.6, 0.8, 1.0)            # the exercise's list
GUIDANCE, PLAIN = 7.5, 1.0                       # the lesson's own img2img snippet, and no guidance
M, SL, SH = 0.15, 1.2, 0.35                      # prompt spread, coarse-band spread, detail spread
NOISE_SEEDS, CLAIM = (11, 12), "0.5-0.7 is the standard range for style transfer"

fmt = lambda t, key, spec=".3f": " ".join(f"{s:g}:{t[s][key]:{spec}}" for s in STRENGTHS)  # noqa: E731
score = lambda t: [t[s]["coarse"] * t[s]["style"] / t[1.0]["style"] for s in STRENGTHS]    # noqa: E731
best = lambda t: STRENGTHS[score(t).index(max(score(t)))]                                  # noqa: E731
bands = lambda t: max(abs(t[s]["coarse"] - t[s]["entry"]) for s in STRENGTHS)              # noqa: E731
splits = lambda t: [t[s]["coarse"] / t[s]["detail"] for s in STRENGTHS[:3]]                # noqa: E731
coarse = lambda x: x.reshape(x.size(0), 3, 16 // BLOCK, BLOCK, 16 // BLOCK, BLOCK).mean(   # noqa: E731
    dim=(3, 5)).repeat_interleave(BLOCK, 2).repeat_interleave(BLOCK, 3)
band = lambda gap, a, extra: (coarse(gap) / (a * (SL**2 + extra) + 1 - a)                  # noqa: E731
                              + (gap - coarse(gap)) / (a * (SH**2 + extra) + 1 - a))
centre = lambda v: v.flatten(1) - v.flatten(1).mean(1, keepdim=True)                       # noqa: E731
project = lambda o, p: float(((o.flatten(1) * p.flatten(1)).sum(1) / (p.flatten(1)**2).sum(1)).mean())  # noqa: E731


class Guided:
    eval = staticmethod(lambda: None)

    def __init__(self, prompt, weight, alphas):
        self.prompt, self.weight, self.alphas = prompt, weight, alphas

    def __call__(self, x, t):
        a = self.alphas[t].view(-1, 1, 1, 1)
        cond, null = band(x - a.sqrt() * self.prompt, a, 0.0), band(x, a, M**2)
        return (1 - a).sqrt() * (null + self.weight * (cond - null))


def img2img(torch, diffusion, schedule, model, photo, strength, noise):
    alphas, grid = schedule["alphas_cumprod"], torch.linspace(T - 1, 0, STEPS + 1).long()
    start, stamp = STEPS - min(int(STEPS * strength), STEPS), torch.zeros(N, dtype=torch.long)
    x = diffusion.q_sample(photo, stamp + int(grid[start]), noise, schedule)
    with torch.no_grad():
        for i in range(start, STEPS):
            step, prev = int(grid[i]), int(grid[i + 1])
            eps = model(x, stamp + step)
            a, a_prev = alphas[step], alphas[prev]
            x = a_prev.sqrt() * (x - (1 - a).sqrt() * eps) / a.sqrt() + (1 - a_prev).sqrt() * eps
    return x, STEPS - start, float(alphas[int(grid[start])].sqrt())


def arm(torch, diffusion, schedule, scene, weight, seed) -> dict:
    photo, prompt, structure, texture = scene
    torch.manual_seed(seed)
    noise, table = torch.randn((N,) + SHAPE), {}
    cosine = torch.nn.functional.cosine_similarity
    model = Guided(prompt, weight, schedule["alphas_cumprod"])
    for strength in STRENGTHS:
        out, calls, entry = img2img(torch, diffusion, schedule, model, photo, strength, noise)
        table[strength] = {"coarse": project(out, structure), "detail": project(out, texture),
                           "calls": calls, "entry": entry, "out": out, "style": float(cosine(out.flatten(1), prompt.flatten(1)).mean()),
                           "similar": float(cosine(centre(out), centre(photo)).mean())}
    return table


def photograph(torch, seed):
    gen = torch.Generator().manual_seed(seed)
    source, prompt = (torch.randn((N,) + SHAPE, generator=gen) * M for _ in range(2))
    z = torch.randn((N,) + SHAPE, generator=gen)
    structure, texture = SL * coarse(z), SH * (z - coarse(z))
    return source + structure + texture, prompt, structure, texture


def solve():
    try:
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    diffusion = parity.load_reference(PHASE, DIFFUSION, "main")
    torch.set_num_threads(2)
    schedule = diffusion.precompute_schedule(diffusion.linear_beta_schedule(T=T))
    scene, other = photograph(torch, 0), photograph(torch, 99)
    model = Guided(scene[1], GUIDANCE, schedule["alphas_cumprod"])
    main, second = (arm(torch, diffusion, schedule, scene, GUIDANCE, s) for s in NOISE_SEEDS)
    torch.manual_seed(NOISE_SEEDS[0])
    swapped = img2img(torch, diffusion, schedule, model, other[0], 1.0, torch.randn((N,) + SHAPE))[0]
    torch.manual_seed(NOISE_SEEDS[0])
    pure = diffusion.sample_ddim(model, schedule, (N,) + SHAPE, steps=STEPS, T=T)
    gap = lambda other: float((main[1.0]["out"] - other).abs().max())   # noqa: E731
    return {"main": main, "second": second, "documented": CLAIM in parity.doc_text(PHASE, LESSON),
            "plain": arm(torch, diffusion, schedule, scene, PLAIN, NOISE_SEEDS[0]),
            "swap": gap(swapped), "reseed": gap(second[1.0]["out"]), "parity": gap(pure),
            "floor": float(schedule["alphas_cumprod"][-1].sqrt())}


def verify(result):
    main, plain, second = result["main"], result["plain"], result["second"]
    return [
        practice.Check(
            "ANSWER: strength 1.0 ignores the photograph because q_sample scales it by 0.0064",
            all([result["swap"] < 0.02, result["reseed"] > 100 * result["swap"], result["parity"] < 0.02,
                 abs(main[1.0]["entry"] - result["floor"]) < 1e-6]),
            f"at strength 1.0 the loop starts at t=999, where sqrt(alpha_bar) is {result['floor']:.6f} -- the "
            f"photograph enters as 0.6% of one unit of noise. Swap in a different photograph at the same "
            f"noise and the output moves {result['swap']:.4f} at most; change the noise instead and it moves "
            f"{result['reseed']:.3f}, {result['reseed'] / result['swap']:.0f}x more. The same call matches "
            f"lesson 10's text-to-image sample_ddim to {result['parity']:.4f}"),
        practice.Check(
            "ANSWER: only 0.2 keeps composition, and the lesson's own '0.5-0.7' keeps 16% of it",
            all([best(main) == 0.2, best(second) == 0.2, main[0.2]["coarse"] > 0.7, result["documented"],
                 main[0.2]["coarse"] > 3 * main[0.6]["coarse"]]),
            f"the output regressed on the photograph's coarse band {fmt(main, 'coarse')}, against style "
            f"(cosine to the new prompt) {fmt(main, 'style')}. Their product peaks at {best(main)} on both noise "
            f"seeds ({' '.join(f'{v:.3f}' for v in score(main))}), where style already reaches "
            f"{main[0.2]['style'] / main[1.0]['style']:.0%} of its ceiling; docs/en.md says {CLAIM!r}, where "
            f"composition runs {main[0.6]['coarse']:.3f} to {main[0.8]['coarse']:.3f}. A real photograph holds "
            "on further, having more variance in the coarse band -- the image's spectrum talking, not the "
            "pipeline, which the lesson states flat"),
        practice.Check(
            "MECHANISM: the reverse chain loses no composition at all -- q_sample loses all of it",
            all([bands(main) < 0.02, min(splits(main)) > 3.0]),
            f"the coarse-band gain {fmt(main, 'coarse')} tracks sqrt(alpha_bar) at the entry step "
            f"{fmt(main, 'entry')} to {bands(main):.4f}: whatever survives add_noise survives the denoiser "
            f"untouched. The detail band does not -- {fmt(main, 'detail')}, a further {splits(main)[0]:.1f}x "
            f"to {splits(main)[2]:.1f}x down, because the model's prior puts SH={SH} against SL={SL} and "
            "shrinks what it believes is texture"),
        practice.Check(
            "CONTROL: strength is also the compute knob -- 0.2 is 5 U-Net calls, not 25",
            [main[s]["calls"] for s in STRENGTHS] == [5, 10, 15, 20, 25],
            f"diffusers' get_timesteps keeps int(steps * strength) of the grid, so the sweep costs "
            f"{fmt(main, 'calls', 'd')} denoiser calls out of {STEPS}. A strength sweep is not a fixed-cost "
            "sweep, and its low end is cheap for the reason it is faithful: it never visits high t"),
        practice.Check(
            "CONTROL: guidance halves the raw similarity to the input without removing any of it",
            all([plain[0.2]["similar"] > 1.5 * main[0.2]["similar"],
                 abs(plain[0.2]["coarse"] - main[0.2]["coarse"]) < 0.05]),
            f"at guidance {PLAIN:g} the plain cosine to the photograph is {fmt(plain, 'similar')}; at "
            f"{GUIDANCE:g} it is {fmt(main, 'similar')} -- roughly halved. But the coarse-band gain is "
            f"{plain[0.2]['coarse']:.3f} against {main[0.2]['coarse']:.3f} at strength 0.2: nothing was lost, "
            "the prompt grew around it, and scoring composition by raw similarity blames strength for it"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
