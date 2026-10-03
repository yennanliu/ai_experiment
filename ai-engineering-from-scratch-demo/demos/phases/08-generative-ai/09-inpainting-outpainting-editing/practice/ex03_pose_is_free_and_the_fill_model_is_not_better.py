"""Exercise 3 — pose control is free, and the "proper" fill model is not better.

    **Hard.** Use Hugging Face diffusers to compare: SD 1.5 Inpaint +
    ControlNet-Openpose vs Flux.1-Fill on 20 face-regeneration tasks. Score pose
    adherence and identity preservation separately.

Reading of the exercise: `diffusers` and `torch` are absent and the two
checkpoints are gigabytes this repo does not ship, so what runs is the
scaled-down comparison `DESIGN D11` asks for, built from the lesson's own code.
A "face" is a 5-D source from `sample_data`; dims 3-4 are the face region and
are masked. A **pose** is a requested tilt `x3 - x4 = +-0.4`; **identity** is
whether the regenerated face lands in the source's cluster. 20 tasks, each run
from 5 seeds.

- *SD 1.5 Inpaint + ControlNet-Openpose*: the lesson's model and its
  reinjection sampler, plus a pose-guidance term that pulls `x3 - x4` toward
  the requested pose at every reverse step -- the side signal ControlNet adds.
- *Flux.1-Fill*: a context-conditioned denoiser -- the doc's "proper inpainting
  model" -- whose input carries the masked source and the mask next to the
  noisy sample. It is trained with the lesson's own `init_net`, `forward`,
  `backward` and `apply`, for the lesson's 5000 steps, and has no pose input.

**ANSWER: each pipeline's score lands where its inputs say it should.** Pose
error: **0.021** with ControlNet against **0.483** for Fill. Identity: **99%**
against **92%**. Scored separately, Inpaint + ControlNet wins pose outright and
does not lose identity -- it keeps the face's cluster more often, not less.

**FINDING: pose control costs identity nothing.** The same inpainting sampler
without the pose term scores pose **0.593** and identity **94%**: guidance cuts
the pose error 28-fold and identity does not fall. Pose is `x3 - x4` and
identity lives along `x3 + x4`; the two directions are orthogonal, so the
"separately" in the exercise is literal here. Identity even rises to 99%,
because tying `x3` to `x4` removes the split faces whose two dims straddle 0.

**FINDING: the "proper" fill model is no better at identity than the naive
trick.** The doc says reinjection "works... badly" and that a model which sees
the context "produces coherent completions". At the lesson's training budget
the conditioned model keeps identity **92%** of the time against reinjection's
**94%**: seeing the context did not buy coherence.

**CONTROL: identity outside the mask cannot rank anything, and the real
pipelines are absent.** Every pipeline pastes the source back over the unmasked
dims (the lesson's `inpaint` ends with `x[i] = clean[i]`), so the unmasked
residual is **0.0** for every run -- which is why identity has to be scored
inside the face region. `diffusers` and `torch` return `None` from `find_spec`,
checked rather than assumed.
"""

from __future__ import annotations

import importlib.util
import math

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "09-inpainting-outpainting-editing"
T, T_DIM, HIDDEN, D, TASKS, SEEDS, PULL = 40, 8, 32, 5, 20, 5, 0.25
MASK, NEEDED = [False, False, False, True, True], ("diffusers", "torch")


def context(clean, mask):
    """The Fill model's extra input: the source with the hole zeroed, then the mask."""
    return [0.0 if m else c for c, m in zip(clean, mask)] + [float(m) for m in mask]


def train_fill(ref, bars, rng, steps=5000):
    """A denoiser that also sees the masked source -- the 9-channel idea, in 5-D."""
    net = ref.init_net(D, T_DIM + 2 * D, HIDDEN, rng)
    for _ in range(steps):
        x0, _ = ref.sample_data(rng, D)
        mask, t = [rng.random() < 0.5 for _ in range(D)], rng.randrange(T)
        eps = [rng.gauss(0, 1) for _ in range(D)]
        x_t = [math.sqrt(bars[t]) * x0[i] + math.sqrt(1 - bars[t]) * eps[i] for i in range(D)]
        out, cache = ref.forward(x_t, ref.sin_embed(t, T, T_DIM) + context(x0, mask), net)
        ref.apply(net, ref.backward(eps, out, cache, net), 0.01)
    return net


def reinject(x, clean, mask, bar, rng):
    """The lesson's reinjection: pinned dims become a freshly noised copy of the source."""
    keep, fresh = math.sqrt(bar), math.sqrt(1 - bar)
    return [v if m else keep * c + fresh * rng.gauss(0, 1) for v, c, m in zip(x, clean, mask)]


def predict(ref, nets, sched, x, t, extra):
    """The lesson's reverse-step mean from x_t, by the base net or (with `extra`) Fill."""
    alphas, bars = sched
    eps, _ = ref.forward(x, ref.sin_embed(t, T, T_DIM) + extra, nets[bool(extra)])
    pull = (1 - alphas[t]) / math.sqrt(1 - bars[t])
    return [(v - pull * e) / math.sqrt(alphas[t]) for v, e in zip(x, eps)]


def regenerate(ref, nets, sched, clean, pose, rng, pipeline):
    """One face: 'control' or 'inpaint' use reinjection, 'fill' uses the Fill model."""
    x, fill = [rng.gauss(0, 1) for _ in range(D)], pipeline == "fill"
    free = [True] * D if fill else MASK  # Fill sees the context; it is not reinjected
    for t in range(T - 1, -1, -1):
        x = reinject(x, clean, free, sched[1][t], rng)
        mean = predict(ref, nets, sched, x, t, context(clean, MASK) if fill else [])
        miss = (mean[3] - mean[4] - pose) * PULL * (pipeline == "control")
        mean[3], mean[4] = mean[3] - miss, mean[4] + miss
        noise = math.sqrt(1 - sched[0][t]) * (t > 0)
        x = [m + noise * rng.gauss(0, 1) for m in mean]
    return [x[i] if MASK[i] else clean[i] for i in range(D)]


def score(ref, nets, sched, tasks, pipeline, random):
    """(pose error, identity kept, worst unmasked residual) over every task and seed."""
    pose_err, kept, outside = 0.0, 0, 0.0
    for n, (clean, cluster, pose) in enumerate(tasks):
        for seed in range(SEEDS):
            out = regenerate(ref, nets, sched, clean, pose, random.Random(100 * n + seed), pipeline)
            pose_err += abs(out[3] - out[4] - pose)
            kept += all((out[i] > 0) == bool(cluster) for i in (3, 4))
            outside = max(outside, max(abs(out[i] - clean[i]) for i in range(3)))
    runs = len(tasks) * SEEDS
    return pose_err / runs, kept / runs, outside


def solve():
    import random

    ref = parity.load_reference(PHASE, LESSON, "main")
    rng, sched = random.Random(5), ref.make_schedule(T)
    base = ref.init_net(D, T_DIM, HIDDEN, rng)
    ref.train(base, sched[1], T, steps=5000, lr=0.01, t_dim=T_DIM, d=D, rng=rng)
    nets = (base, train_fill(ref, sched[1], random.Random(6)))
    pick = random.Random(9)
    tasks = [(*ref.sample_data(pick, D), pick.choice([-0.4, 0.4])) for _ in range(TASKS)]
    rows = {p: score(ref, nets, sched, tasks, p, random) for p in ("control", "fill", "inpaint")}
    return {"rows": rows, "absent": [m for m in NEEDED if importlib.util.find_spec(m) is None]}


def verify(result):
    ctl, fill, plain = (result["rows"][p] for p in ("control", "fill", "inpaint"))
    return [
        practice.Check(
            "ANSWER: Inpaint+ControlNet wins pose outright and does not lose identity",
            ctl[0] < 0.1 and fill[0] > 0.3 and ctl[1] >= fill[1],
            f"{TASKS} tasks x {SEEDS} seeds: pose error {ctl[0]:.3f} vs {fill[0]:.3f} (Fill), "
            f"identity kept {ctl[1]:.0%} vs {fill[1]:.0%}",
        ),
        practice.Check(
            "FINDING: pose control costs identity nothing",
            plain[0] > 10 * ctl[0] and ctl[1] >= plain[1],
            f"without the pose term: pose {plain[0]:.3f}, identity {plain[1]:.0%} -- a "
            f"{plain[0] / ctl[0]:.0f}-fold pose gain, as pose is x3 - x4 and identity x3 + x4",
        ),
        practice.Check(
            "FINDING: the 'proper' fill model is no better at identity than the naive trick",
            fill[1] <= plain[1],
            f"at the lesson's 5000-step budget the context-seeing model keeps identity "
            f"{fill[1]:.0%} of the time against the reinjection the doc calls bad, {plain[1]:.0%}",
        ),
        practice.Check(
            "CONTROL: identity outside the mask cannot rank, and the real pipelines are absent",
            ctl[2] == fill[2] == plain[2] == 0.0 and result["absent"] == list(NEEDED),
            f"every pipeline pastes the source back, so unmasked residual is {ctl[2]}, {fill[2]}, "
            f"{plain[2]}; {result['absent']} return None from find_spec",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
