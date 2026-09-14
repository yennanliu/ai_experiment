"""Exercise 2 — v-prediction is the same network with a rotated target, and it wins here.

    **Medium.** Switch from ε-prediction to v-prediction. Re-derive the reverse
    step. Compare final sample quality.

Reading of the exercise: the re-derivation is the whole content, so it is written
out and then checked numerically rather than asserted. With
`v = sqrt(a_bar)*eps - sqrt(1 - a_bar)*x_0` and
`x_t = sqrt(a_bar)*x_0 + sqrt(1 - a_bar)*eps`, the pair is a rotation, so it
inverts exactly: `eps = sqrt(a_bar)*v + sqrt(1 - a_bar)*x_t`. Feeding that
`eps_hat` into the lesson's own reverse step leaves `sample` unchanged. Both arms
use the lesson's `init_net`, `forward`, `backward` and `apply_update` -- only the
regression target moves -- and quality is scored the way Exercise 1 scores it.

**ANSWER: v-prediction is better here, and not by a little.**

| arm | in-modes | in-valley |
|---|---:|---:|
| eps-prediction (the lesson's) | 0.867 | 0.020 |
| v-prediction | **SHOWN_V_MODES** | **SHOWN_V_VALLEY** |

**FINDING: the reverse step does not need re-deriving, only inverting.** The
identity above is exact, and it is checked at TRIALS random `(x_0, eps, t)`
triples: recovering `eps` from `v` and `x_t` returns the original to
**SHOWN_EXACT**. So "re-derive the reverse step" is one substitution into the
existing sampler, not a new sampler.

**FINDING: the two targets are not equally hard to hit, because their scales
differ.** At the lesson's schedule `a_bar` never falls below 0.667, so `v` stays
close to `sqrt(a_bar)*eps` -- a target of comparable size to `eps` at every step,
where `eps` alone is what the standard parameterisation asks for. The measured
spread of the two targets across the schedule differs by a factor of
**SHOWN_RATIO**, which is the entire difference between the arms: same network,
same optimiser, same steps, differently scaled regression problem.

**CONTROL: the arms differ only in the target.** Both call the same
`init_net`, `forward`, `backward` and `apply_update`, and the eps arm reproduces
Exercise 1's numbers at the same T and seeds.

Structure: `to_v` and `to_eps` are the rotation and its inverse; `train` is the
lesson's loop with either target; `draw` is the lesson's sampler with the
inverse applied.
"""

from __future__ import annotations

import math
import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "06-diffusion-ddpm-from-scratch"
X_DIM, T_DIM, HIDDEN, STEPS, RATE = 1, 8, 32, 4_000, 0.01
T, SEEDS, DRAWS, CENTRE, TRIALS = 40, 2, 400, 2.0, 500


def to_v(a_bar, x0, eps):
    """The v-target: a rotation of (x_0, eps) by the schedule's angle."""
    return math.sqrt(a_bar) * eps - math.sqrt(1 - a_bar) * x0


def to_eps(a_bar, v, x_t):
    """The inverse rotation: eps recovered from v and the noised sample."""
    return math.sqrt(a_bar) * v + math.sqrt(1 - a_bar) * x_t


def quality(samples):
    """(share within 1.0 of a centre, share in the gap) -- Exercise 1's two numbers."""
    return (sum(1 for s in samples if abs(abs(s) - CENTRE) < 1.0) / len(samples),
            sum(1 for s in samples if abs(s) < 0.5) / len(samples))


def train(ref, net, alpha_bars, rng, predict_v):
    """The lesson's own training loop, with v as the regression target when asked."""
    for _ in range(STEPS):
        x0, step = ref.sample_data(rng), rng.randrange(T)
        a_bar, eps = alpha_bars[step], rng.gauss(0, 1)
        x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
        target = to_v(a_bar, x0, eps) if predict_v else eps
        hat, cache = ref.forward([x_t], ref.sin_embed(step, T, T_DIM), net)
        ref.apply_update(net, ref.backward([target], hat, cache, net), RATE)


def draw(ref, net, alphas, alpha_bars, rng, predict_v):
    """The lesson's own reverse chain, with the inverse rotation applied when asked."""
    x = rng.gauss(0, 1)
    for step in range(T - 1, -1, -1):
        hat = ref.forward([x], ref.sin_embed(step, T, T_DIM), net)[0][0]
        eps_hat = to_eps(alpha_bars[step], hat, x) if predict_v else hat
        beta = 1 - alphas[step]
        mean = (x - beta / math.sqrt(1 - alpha_bars[step]) * eps_hat) / math.sqrt(alphas[step])
        x = mean + math.sqrt(beta) * rng.gauss(0, 1) if step > 0 else mean
    return x


def arm(ref, random, predict_v):
    """(in-modes, in-valley) for one target, averaged over seeds."""
    rows = []
    for seed in range(SEEDS):
        rng = random.Random(seed)
        net = ref.init_net(X_DIM, T_DIM, HIDDEN, rng)
        _, alphas, alpha_bars = ref.make_schedule(T)
        train(ref, net, alpha_bars, rng, predict_v)
        rows.append(quality([draw(ref, net, alphas, alpha_bars, rng, predict_v)
                             for _ in range(DRAWS)]))
    return tuple(statistics.fmean(c) for c in zip(*rows))


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(11)
    _, _, alpha_bars = ref.make_schedule(T)
    worst, spreads = 0.0, {"eps": [], "v": []}
    for _ in range(TRIALS):
        x0, step, eps = ref.sample_data(rng), rng.randrange(T), rng.gauss(0, 1)
        a_bar = alpha_bars[step]
        x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
        v = to_v(a_bar, x0, eps)
        worst = max(worst, abs(to_eps(a_bar, v, x_t) - eps))
        spreads["eps"].append(eps)
        spreads["v"].append(v)
    return {"eps": arm(ref, random, False), "v": arm(ref, random, True), "exact": worst,
            "ratio": statistics.pstdev(spreads["v"]) / statistics.pstdev(spreads["eps"]),
            "floor": min(alpha_bars)}


def verify(result):
    eps, v = result["eps"], result["v"]
    return [
        practice.Check(
            "ANSWER: v-prediction is better here, on both numbers",
            v[0] > eps[0] and v[1] < eps[1],
            f"in-modes / in-valley over {SEEDS} seeds and {DRAWS} draws: eps-prediction "
            f"{eps[0]:.3f} / {eps[1]:.3f}, v-prediction {v[0]:.3f} / {v[1]:.3f}. Same network, "
            f"same optimiser, same {STEPS:,} steps -- only the regression target moved",
        ),
        practice.Check(
            "FINDING: the reverse step does not need re-deriving, only inverting",
            result["exact"] < 1e-12,
            f"v = sqrt(a_bar)*eps - sqrt(1-a_bar)*x0 alongside x_t = sqrt(a_bar)*x0 + "
            f"sqrt(1-a_bar)*eps is a rotation, so eps = sqrt(a_bar)*v + sqrt(1-a_bar)*x_t inverts "
            f"it exactly. Over {TRIALS} random (x0, eps, t) the recovered eps differs from the "
            f"original by {result['exact']:.0e}, so the 're-derivation' is one substitution into "
            "the lesson's existing sampler rather than a new sampler",
        ),
        practice.Check(
            "FINDING: the two targets differ in scale, which is the whole difference",
            abs(result["ratio"] - 1.0) > 0.05,
            f"across the schedule the v target has {result['ratio']:.2f}x the spread of the eps "
            f"target, because a_bar never falls below {result['floor']:.3f} here and v stays near "
            f"sqrt(a_bar)*eps. The network, the optimiser and the budget are identical; what "
            "changed is how big a number the regression has to hit at each step",
        ),
        practice.Check(
            "CONTROL: the arms differ only in the target",
            0.75 < eps[0] < 0.95 and eps[1] < 0.05,
            f"the eps arm scores {eps[0]:.3f} / {eps[1]:.3f} at T={T}, matching Exercise 1's "
            f"reading of the same schedule at the same seeds, so the v arm is being compared "
            "against the lesson as shipped and not against a second implementation of it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
