"""Exercise 1 — euler step count error curve.

    **(Easy)** Train the TinyDiT above on the synthetic blob dataset for 500 steps. Compare samples produced with 10, 20, and 50 Euler steps.

Reading of the exercise: "compare samples" is the one thing `main()` cannot do --
it prints `samples range [min, max]` and the shape, and those agree to two
decimals across 10, 20 and 50 steps while the images behind them differ several
fold, so the comparison here is a distance to a converged 400-step solve from the
*same* starting noise. The deeper question -- why 10 steps is nearly enough --
has an exact answer rather than an empirical one. The interpolant
`x_t = (1-t)x_0 + t*eps` has velocity `eps - x_0` with no `t` in it at all, which
is why `rectified_flow_train_step` writes `target_v = epsilon - x0` and never
uses its own `t`; on a *one-point* dataset the exact velocity field is therefore
constant along every trajectory, and one Euler step is then the whole solve. Both
are checked to float precision. The step-count curve is then measured a second
time against the closed-form marginal field of the real 128-blob dataset -- the
model a perfect fit would be -- which separates "the sampler needs steps" from
"the model is undertrained". Over 1-10 steps they are the same curve.

Structure: `oracle_velocity` is the closed-form marginal `E[eps - x_0 | x_t = x]`
for a finite dataset, a softmax over the K Gaussian components the interpolant
induces; `make_oracle` gives it the duck type the lesson's own
`rectified_flow_sample` needs -- a call plus an `.eval()` -- so the exact field is
integrated by the lesson's sampler rather than by a copy of it; `step_curve`
re-seeds before every solve (the sampler's first op is `randn`, so every step
count starts from one point) and reduces the solves to error-versus-steps against
the converged one, plus the min/max `main()` would have printed; `chord_deviation`
reads a field on the straight line joining a solve's own endpoints, where a
genuinely straight path would return one constant vector; `train` runs the
lesson's `rectified_flow_train_step` unchanged for the 500 steps the exercise asks
for. `solve` reuses `step_curve` for the trained network and for each exact field,
so both curves come out of one code path.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "23-diffusion-transformers-rectified-flow"

TRAIN_STEPS, BATCH, LR, IMAGES, SIZE = 500, 32, 3e-4, 128, 16
STEPS, ASKED, POOLS = (1, 2, 4, 10, 20, 50), (10, 20, 50), (1, 2, IMAGES)
CONVERGED, DRAWN, NOISE_SEED, PROBE_T = 400, 16, 11, (0.1, 0.3, 0.5, 0.7, 0.9)

pct = lambda curve: "  ".join(f"{n}:{curve[n]:.1%}" for n in STEPS)                                 # noqa: E731
listing = lambda values: " ".join(f"{v:.1%}" for v in values)                                       # noqa: E731
ranges = lambda c: " ".join(f"{n}:[{c[n]['lo']:.2f},{c[n]['hi']:.2f}]" for n in ASKED)              # noqa: E731
span = lambda c, k: max(c[n][k] for n in ASKED) - min(c[n][k] for n in ASKED)                       # noqa: E731
drift = lambda a, b: max(abs(a[n] - b[n]) for n in (1, 2, 4, 10))                                   # noqa: E731
halving = lambda rel: ", ".join(f"{rel[a] / rel[b]:.2f}x" for a, b in ((1, 2), (2, 4), (10, 20)))   # noqa: E731
converges = lambda rel: rel[10] < 0.10 and rel[20] < rel[10] and rel[50] < rel[20]                  # noqa: E731
blind = lambda c, rel: span(c, "lo") < 0.10 and span(c, "hi") < 0.10 and rel[10] > 4 * rel[50]      # noqa: E731
one_shot = lambda o: o[1][1] < 1e-5 and o[2][1] > 0.05 and o[2][4] < 1e-5                           # noqa: E731
curved = lambda dev, rel: max(dev) > 0.10 and rel[50] < 0.02                                        # noqa: E731


def oracle_velocity(torch, x, moment, pool):
    delta = x.flatten(1)[:, None, :] - ((1 - moment) * pool)[None]
    weight = torch.softmax(-(delta ** 2).sum(-1) / (2 * moment * moment), dim=-1)
    return (weight[..., None] * (delta / moment - pool[None])).sum(1).view_as(x)


def make_oracle(torch, pool):
    field = lambda x, t: oracle_velocity(torch, x, float(t[0]), pool.flatten(1))    # noqa: E731
    field.eval = lambda: None
    return field


def train(torch, ref, numpy, data):
    torch.manual_seed(0)
    model = ref.TinyDiT(image_size=SIZE, patch_size=2, in_channels=3, dim=96, depth=4, heads=3)
    optimiser, generator = torch.optim.Adam(model.parameters(), lr=LR), numpy.random.default_rng(0)
    history = [ref.rectified_flow_train_step(model, data[generator.choice(len(data), BATCH)], optimiser, "cpu")
               for _ in range(TRAIN_STEPS)]
    return model, history


def step_curve(torch, ref, model):
    def draw(steps):
        torch.manual_seed(NOISE_SEED)         # the sampler's first op is randn, so this pins x_1
        return ref.rectified_flow_sample(model, (DRAWN, 3, SIZE, SIZE), steps=steps, device="cpu")

    settled, curve = draw(CONVERGED), {}
    torch.manual_seed(NOISE_SEED)
    start = torch.randn(DRAWN, 3, SIZE, SIZE)
    travel = float((start - settled).flatten(1).norm(dim=1).mean())  # what one whole solve moves
    for steps in STEPS:
        drawn = draw(steps)
        curve[steps] = {"rel": float((drawn - settled).flatten(1).norm(dim=1).mean()) / travel,
                        "lo": float(drawn.min()), "hi": float(drawn.max())}
    return curve, travel, settled, start


def chord_deviation(torch, model, settled, start):
    chord = start - settled                        # the constant velocity a genuinely straight path has
    scale = chord.flatten(1).norm(dim=1)
    model.eval()
    with torch.no_grad():
        read = [model((1 - m) * settled + m * start, torch.full((DRAWN,), m)) for m in PROBE_T]
    return [float(((seen - chord).flatten(1).norm(dim=1) / scale).mean()) for seen in read]


def solve():
    try:
        import numpy
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    data = ref.synthetic_blobs(num=IMAGES, size=SIZE, seed=0)
    torch.manual_seed(5)
    clean, noise = data.double()[:1].expand(8, -1, -1, -1), torch.randn(8, 3, SIZE, SIZE, dtype=torch.double)
    identity = max(float((((1 - m) * clean + m * noise - clean) / m - (noise - clean)).abs().max())
                   for m in PROBE_T)
    model, history = train(torch, ref, numpy, data)
    curve, travel, settled, start = step_curve(torch, ref, model)
    oracle, memory = {}, 0.0
    for size in POOLS:
        exact, reach, target, _ = step_curve(torch, ref, make_oracle(torch, data[:size]))
        oracle[size] = {n: exact[n]["rel"] for n in STEPS}
        memory = float(torch.cdist(target.flatten(1), data[:size].flatten(1)).min(1).values.mean()) / reach
    return {"curve": curve, "oracle": oracle, "identity": identity, "travel": travel, "memory": memory,
            "chord": chord_deviation(torch, model, settled, start),
            "loss": (history[0], sum(history[-50:]) / 50)}


def verify(result):
    curve, oracle, ideal = result["curve"], result["oracle"], result["oracle"][IMAGES]
    rel = {n: curve[n]["rel"] for n in STEPS}
    return [
        practice.Check(
            "ANSWER: 10, 20 and 50 steps land percent apart -- Euler is first-order, so 5x buys one digit",
            converges(rel),
            f"after {TRAIN_STEPS} steps (loss {result['loss'][0]:.4f} -> {result['loss'][1]:.4f}) one fixed "
            f"noise solved at {STEPS} steps sits this far from the converged {CONVERGED}-step solve, against "
            f"the {result['travel']:.2f} travelled: {pct(rel)}, falling ~2x per doubling ({halving(rel)})"),
        practice.Check(
            "FINDING: the statistic `main()` prints cannot rank the three step counts the exercise names",
            blind(curve, rel),
            f"`main()` prints `samples range [min, max]` and the shape. Over {ASKED} steps that range moves "
            f"{span(curve, 'lo'):.3f} at the bottom and {span(curve, 'hi'):.3f} at the top -- {ranges(curve)} "
            f"-- while the images differ by {rel[10] / rel[50]:.1f}x. A saturated statistic ranks nothing"),
        practice.Check(
            "MECHANISM: the regression target is exactly t-independent, to float64 rounding",
            result["identity"] < 1e-14,
            f"`rectified_flow_train_step` writes `target_v = epsilon - x0` and never touches its own `t` -- an "
            f"identity, not an approximation: recovered from the noised sample alone as `(x_t - x_0)/t` at t in "
            f"{PROBE_T} it returns `eps - x_0` to {result['identity']:.1e} in float64, so every step regresses "
            "one function of (x_0, eps) whatever t it drew -- what DDPM's schedule-scaled target cannot do"),
        practice.Check(
            "MECHANISM: on a one-point dataset the lesson's own sampler is exact in ONE step",
            one_shot(oracle),
            f"the exact marginal field is constant along every trajectory when the dataset is one image, so "
            f"`rectified_flow_sample` at 1 step matches its own {CONVERGED}-step solve to {oracle[1][1]:.0e} of "
            f"the travel -- float32 rounding, the tolerance an exact identity gets in a float32 sampler. Add "
            f"one image: 1 step is {oracle[2][1]:.1%} off, 2 steps {oracle[2][2]:.1%}, exact again from 4 "
            f"({oracle[2][4]:.0e})"),
        practice.Check(
            "FINDING: the trained model's step curve is the *true* field's curve, not undertraining",
            drift(ideal, rel) < 0.05,
            f"the closed-form marginal field of the same {IMAGES} blobs -- what a perfect fit would be -- "
            f"through the same sampler gives {pct(ideal)} against the trained net's {pct(rel)}: one curve to "
            f"{drift(ideal, rel):.1%} over 1-10 steps, so the residual is curvature, not a short run. Past 10 "
            f"the exact field memorises -- endpoints {result['memory']:.1e} of the travel from a real image"),
        practice.Check(
            "CONTROL: the path is straight end to end while the field along it is not constant",
            curved(result["chord"], rel),
            f"read at t = {PROBE_T} on the straight chord joining a solve's own endpoints, the trained velocity "
            f"deviates from that chord's one constant direction by {listing(result['chord'])}, worst "
            f"({max(result['chord']):.1%}) at the noise end -- yet 50 steps land {rel[50]:.1%} from {CONVERGED}. "
            "Straight in aggregate, curved pointwise: that is why one step is not enough here"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
