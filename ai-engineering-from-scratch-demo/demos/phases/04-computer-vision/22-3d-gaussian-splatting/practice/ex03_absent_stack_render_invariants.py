"""Exercise 3 — absent stack render invariants.

    **(Hard)** Clone `nerfstudio` and train `splatfacto` on a 20-photo capture of any scene you have (desk, plant, face, room). Export to glTF `KHR_gaussian_splatting` and open it in a viewer (Three.js `GaussianSplats3D`, SuperSplat, Babylon.js V9). Report training time, number of Gaussians, and rendered fps.

Reading of the exercise: none of it is reachable here. `nerfstudio`, `gsplat`
and `pycolmap` are all absent, there is no capture to pose, and nothing may be
downloaded -- so "training time, number of Gaussians, rendered fps" cannot be
reported, and the figures the lesson quotes for them stay quoted rather than
measured. What is reachable is the arithmetic that stack rests on, and it is
exact enough to assert tightly instead of approximately. Alpha compositing
telescopes, so the weight a pixel accumulates is exactly one minus its final
transmittance; that transmittance is a product, so it does not care what order
the splats arrive in, while the colour it weights very much does. `covs()`
returns a symmetric matrix whose eigenvalues are exactly the squared scales,
which is what `R S S^T R^T` with an orthogonal `R` means -- and checking it on
the output beats rebuilding `R` from the same cosines the lesson used.
Degree-0 spherical harmonics are constant on the sphere. Those are algebraic identities, so the tolerances
asserted below are float32 epsilon (1.2e-7) times the number of accumulations
-- 1e-6 absolute for a 48-splat composite, 1e-6 relative for a covariance whose
entries reach the hundreds -- and never fitted thresholds; two of them come
back bitwise exact and are asserted as `== 0.0`.

Two measurements say something the lesson does not. The `depth` that decides
the compositing order is an `nn.Parameter` that can never learn, because
`argsort` has no gradient: 48 of the model's 480 parameters are handed to the
optimiser and never move. And `R S S^T R^T` is positive definite only in exact
arithmetic -- past a condition number of about 1e8 the float32 covariance has a
determinant of 0.0 where the algebra fixes it at exactly 1.0, and
`eval_2d_gaussian`'s `torch.linalg.inv` of it is wrong by a factor of ten. That
is the mechanism behind the NaN gradients exercise 1 measures at 256 splats:
the clamp to 0.99 hides the overflowing density in the forward pass, so only
the backward pass reports it.

Structure: `probe_import` records each production package's failure as a
measurement rather than a claim; `composite` is the lesson's own compositing
loop instrumented to return accumulated weight and final transmittance beside
the image, so the identities are read off the numbers the lesson renders from;
`gradient_audit` takes one backward pass and one `Adam.step()` and records
which parameters saw a gradient at all and how far `depth` moved. `solve` then
randomises `rot` and `log_scale`, because the initialiser leaves `R` the
identity and `S` isotropic, which would make the covariance identity true for
the wrong reason; its last two statements re-scale one Gaussian to a condition
number of e^20 and are what the determinant reading comes from, so they run
after every other measurement. At 146 code lines this sits above D14's 120-line
target and four clear of its ceiling: five independent probes -- import
failures, a timed render, an instrumented composite, a gradient audit and a
conditioning sweep -- against a `solve` that no single experiment dominates.
"""

from __future__ import annotations

import importlib
import math
import time

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "22-3d-gaussian-splatting"

STACK = ("nerfstudio", "gsplat", "pycolmap")
SIZE, SPLATS, DIRS = 48, 48, 500
DOC_FPS, DOC_SPLATS, FRAME = 147, 6_000_000, 1920 * 1080
WIDE = 5.0                                  # log-scale giving condition number e^(4*WIDE)

peak_row = lambda peak: " ".join(f"{k} {v:.1e}" for k, v in peak.items() if v is not None)  # noqa: E731


def probe_import(name) -> str:
    try:
        importlib.import_module(name)
        return "importable"                      # pragma: no cover - none of these are installed
    except ModuleNotFoundError as exc:
        return str(exc)


def composite(torch, alpha, colours, order) -> tuple:
    out, weight = torch.zeros(SIZE, SIZE, 3), torch.zeros(SIZE, SIZE)
    trans = torch.ones(SIZE, SIZE)
    for i in order:
        out = out + (trans * alpha[i])[..., None] * colours[i]
        weight, trans = weight + trans * alpha[i], trans * (1.0 - alpha[i])
    return out, weight, trans


def gradient_audit(torch, ref, model) -> tuple:
    optimiser = torch.optim.Adam(model.parameters(), lr=0.08)
    ((model((SIZE, SIZE)) - ref.make_target(SIZE)) ** 2).mean().backward()
    peak = {n: None if p.grad is None else float(p.grad.abs().max())
            for n, p in model.named_parameters()}
    before = model.depth.detach().clone()
    optimiser.step()
    return (peak, float((model.depth.detach() - before).abs().max()),
            model.depth.numel(), sum(p.numel() for p in model.parameters()))


def solve():
    try:
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = ref.Splats2D(num_splats=SPLATS, image_size=SIZE)
    audit = gradient_audit(torch, ref, model)
    with torch.no_grad():
        model.rot[:] = torch.rand(SPLATS) * 2.0 * math.pi     # the initialiser leaves R = I and S
        model.log_scale += torch.randn(SPLATS, 2) * 0.5       # isotropic, so it tests nothing
        axis = torch.arange(SIZE, dtype=torch.float32)
        points = torch.stack(torch.meshgrid(axis, axis, indexing="ij")[::-1], dim=-1)
        cov, colours = model.covs(), torch.sigmoid(model.colour_logits)
        opacity, squared = torch.sigmoid(model.opacity_logit), torch.exp(2 * model.log_scale).sort(-1).values
        alpha = (opacity[:, None, None] * ref.eval_2d_gaussian(model.means, cov, points)).clamp(0.0, 0.99)
        order = torch.argsort(model.depth).tolist()
        front, weight, trans = composite(torch, alpha, colours, order)
        back = composite(torch, alpha, colours, order[::-1])
        args = (model.means, cov, colours, opacity, model.depth, (SIZE, SIZE))
        rendered = ref.rasterise_2d(*args)
        clock = time.perf_counter()
        for _ in range(20):
            ref.rasterise_2d(*args)
        seconds = (time.perf_counter() - clock) / 20
        shuffle = float((ref.rasterise_2d(*[a[order[::-1]] for a in args[:5]],
                                          (SIZE, SIZE)) - rendered).abs().max())
        probe = torch.nn.functional.normalize(torch.randn(DIRS, 3), dim=-1)
        full, dc = torch.randn(1, 16, 3).expand(DIRS, 16, 3), torch.eye(16)[0][:, None]
        spread = [float((c.max(0).values - c.min(0).values).abs().max())
                  for c in (ref.eval_sh_degree_3(full * dc, probe), ref.eval_sh_degree_3(full, probe))]
        model.rot[:], model.log_scale[:] = 0.7, torch.tensor([WIDE, -WIDE])   # last: it mutates
        thin = model.covs()[0]
    return {"absent": {n: probe_import(n) for n in STACK}, "seconds": seconds, "depth": audit,
            "shuffle": shuffle, "span": float(squared.max()), "spread": spread,
            "telescope": float((weight - (1.0 - trans)).abs().max()),
            "product": float((trans - torch.prod(1.0 - alpha, dim=0)).abs().max()),
            "flip": (float((front - back[0]).abs().max()), float((trans - back[2]).abs().max())),
            "symmetric": float((cov - cov.transpose(-1, -2)).abs().max()),
            "eigen": float((torch.linalg.eigvalsh(cov) - squared).abs().max()),
            "wide": (float(thin[0, 0] * thin[1, 1] - thin[0, 1] * thin[1, 0]),
                     float((thin @ torch.linalg.inv(thin) - torch.eye(2)).abs().max()))}


def verify(result):
    flip, (det, residual) = result["flip"], result["wide"]
    peak, moved, dead, total = result["depth"]
    rate = SPLATS * SIZE * SIZE / result["seconds"]
    near = {key: result[key] / result["span"] for key in ("symmetric", "eigen")}
    return [
        practice.Check(
            "ANSWER: the whole production stack is absent, so no time, count or fps can be reported",
            all("No module named" in text for text in result["absent"].values()),
            "; ".join(f"`{k}` -> {v}" for k, v in result["absent"].items())
            + f". Measurable instead is the lesson's own rasteriser: {SPLATS} splats at {SIZE}x{SIZE} take "
            f"{result['seconds'] * 1e3:.2f} ms — {1 / result['seconds']:.0f} fps, {rate / 1e6:.1f}M splat-pixels/s, "
            f"so the {DOC_SPLATS:,} splats the lesson says an RTX 3080 Ti renders at {DOC_FPS} fps "
            f"need {DOC_SPLATS * FRAME / rate / 3600:.0f} hours per 1080p frame here"),
        practice.Check(
            "MECHANISM: the composite telescopes, and its transmittance does not care about order",
            max(result["telescope"], result["product"], flip[1]) < 1e-6 < flip[0]
            and result["shuffle"] == 0.0,
            f"over {SIZE * SIZE} pixels max|sum_i T_i a_i - (1 - T_final)| = {result['telescope']:.2e} and "
            f"max|T_final - prod_i (1 - a_i)| = {result['product']:.2e}, both inside the 1e-6 a {SPLATS}-term "
            f"float32 accumulation earns. Reversing the compositing order moves that transmittance "
            f"{flip[1]:.1e} — multiplication commutes — but the image {flip[0]:.3f} per channel. Handing "
            f"`rasterise_2d` those splats reversed changes no bit ({result['shuffle']:.1e}): `argsort` re-sorts"),
        practice.Check(
            "FINDING: `depth` cannot learn — argsort has no gradient, so a tenth of the model is dead",
            peak["depth"] is None and moved == 0.0 and peak["rot"] == 0.0 < peak["means"],
            "one backward pass of the lesson's forward against its own `make_target` gives max|grad| "
            f"{peak_row(peak)} — and `model.depth.grad` is not 0.0 but None, no graph path at all, so "
            f"an `Adam.step()` moves it by exactly {moved:.1f}. Those {dead} of {total} parameters "
            f"({dead / total:.0%}) can never update. `rot` reads 0.0 for a milder reason: the "
            "initialiser makes every Gaussian isotropic, and rotating a circle changes nothing"),
        practice.Check(
            "MECHANISM: `covs()` is R S S^T R^T under 1 ulp — until float32 loses the definiteness",
            max(near["symmetric"], near["eigen"]) < 1e-6 and residual > 1.0,
            f"on entries reaching {result['span']:.0f}, max|Sigma - Sigma^T| = {result['symmetric']:.1e} and the "
            f"eigenvalues match exp(2*log_scale) to {result['eigen']:.1e} — {near['symmetric']:.0e} and "
            f"{near['eigen']:.0e} relative, inside float32's 1.2e-7 ulp, so Sigma is a rotation conjugating "
            f"diag(s^2). Yet det(Sigma), fixed at exactly 1.0 for every anisotropy by that same algebra, comes "
            f"back {det:.3g} at condition {math.exp(4 * WIDE):.0e}, where Sigma @ inv(Sigma) misses I by {residual:.0f}"),
        practice.Check(
            "CONTROL: degree-0 harmonics are view-independent to the bit, not to a tolerance",
            result["spread"][0] == 0.0 < result["spread"][1],
            f"with only the degree-0 coefficient nonzero, `eval_sh_degree_3` over {DIRS} random unit directions "
            f"spreads exactly {result['spread'][0]:.1f} — every direction returns the same bits, because that "
            f"basis function is the constant C0 = 0.2821. Filling all 16 slots spreads {result['spread'][1]:.2f} "
            "over the same directions, so the measurement is not vacuous: degree 0 declines what is on offer"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
