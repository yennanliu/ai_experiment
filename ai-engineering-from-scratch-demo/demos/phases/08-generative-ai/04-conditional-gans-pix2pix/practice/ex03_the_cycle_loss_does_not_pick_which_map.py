"""Exercise 3 — it learns a map with no pairs, and the objective cannot say which one.

    **Hard.** Sketch a CycleGAN in the 1-D setting: two distributions, two
    generators, cycle loss. Show that it learns to map between them with no
    paired data.

Reading of the exercise: "sketch" is taken at its word, so each generator is the
smallest thing that can map a line to a line, `x -> w*x + b`. That is not a
simplification for convenience: it makes the exercise's claim *provable* rather
than merely observed, because the optima can be written down. The two
discriminators are the lesson's own `init_mlp` and `d_forward` at
`num_classes = 1`, trained by the lesson's own `update_d`, and the generators
carry the lesson's non-saturating adversarial gradient plus an L1 cycle term.
`A = N(-2, 0.5)`, `B = N(+2, 0.5)`, and no sample is ever paired with another.

**ANSWER: yes -- the marginals match and the cycle closes.** Over SEEDS seeds
`G_AB` sends A to a mean of **+2.00** with sd **0.47**, against B's **+2.0** and
**0.5**, and `G_BA(G_AB(a))` returns to `a` within **0.16** on average. No pair
was ever shown.

**FINDING: which map it learns is not determined, and both answers appear.**

| | seeds | w | b |
|---|---:|---:|---:|
| order-reversing | **6 of 8** | -0.93 | +0.14 |
| order-preserving | **2 of 8** | +0.95 | +3.78 |

Same objective, same data, same code -- the seed decides. One run maps the
*largest* a to the largest b; the other maps it to the smallest.

**FINDING: that is exact, not a training artefact.** For a linear map, matching
the marginal forces `|w| = sd_B / sd_A = 1` and `b = mean_B - w * mean_A`, and the
cycle constraint forces `w_BA = 1 / w_AB`. Both signs satisfy both constraints,
so the objective has **two** global optima with `b = 0` and `b = +4`. Measured:
**+0.14** and **+3.78**. Nothing in "two distributions, two generators, cycle
loss" distinguishes them, and no amount of training will.

**CONTROL: both arms match the marginal, so this is two solutions and not one
failure.** Every seed lands within **0.15** of B's mean and **0.06** of its sd,
whichever sign it chose.

Structure: `pull` is the adversarial gradient through the lesson's critic;
`step` is one generator update carrying adversarial and cycle terms; `run`
trains one seed.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "04-conditional-gans-pix2pix"
HIDDEN, BATCH, G_RATE, D_RATE, STEPS, SEEDS = 16, 32, 0.02, 0.01, 4_000, 8
MEAN_A, SD_A, MEAN_B, SD_B, CYCLE, PROBE = -2.0, 0.5, 2.0, 0.5, 1.0, 400


def pull(ref, critic, value):
    """d(-log D(x))/dx through the lesson's own critic, at num_classes=1."""
    prob, _, pre1, _, _ = ref.d_forward([value], 0, critic, 1)
    seeded = [critic["W2"][0][j] * (prob - 1.0) * ref.leaky_grad(pre1[j]) for j in range(HIDDEN)]
    return sum(critic["W1"][j][0] * seeded[j] for j in range(HIDDEN))


def critique(ref, critic, real, fake):
    """One discriminator step through the lesson's own update_d, at num_classes=1."""
    ref.update_d([([v], 0) for v in real], [([v], 0) for v in fake], critic, 1, D_RATE)


def step(ref, forward, backward, critic, source):
    """Gradient on `forward` from the adversary, and on both maps from the cycle."""
    out, back = [0.0, 0.0], [0.0, 0.0]
    for x in source:
        mapped = forward[0] * x + forward[1]
        adversarial = pull(ref, critic, mapped)
        out[0] += adversarial * x
        out[1] += adversarial
        sign = 1.0 if backward[0] * mapped + backward[1] > x else -1.0
        out[0] += CYCLE * sign * backward[0] * x
        out[1] += CYCLE * sign * backward[0]
        back[0] += CYCLE * sign * mapped
        back[1] += CYCLE * sign
    return out, back


def measure(ab, ba, probe):
    """(mapped mean, mapped sd, mean cycle error) for one trained pair of maps."""
    mapped = [ab[0] * v + ab[1] for v in probe]
    return (statistics.fmean(mapped), statistics.pstdev(mapped),
            statistics.fmean(abs(ba[0] * m + ba[1] - v) for v, m in zip(probe, mapped)))


def run(ref, random, seed):
    """One CycleGAN; returns (w, b, mapped mean, mapped sd, cycle error) for A->B."""
    rng = random.Random(seed)
    critic_a, critic_b = ref.init_mlp(2, HIDDEN, 1, rng), ref.init_mlp(2, HIDDEN, 1, rng)
    ab = [rng.gauss(0, 0.5), rng.gauss(0, 0.5)]
    ba = [rng.gauss(0, 0.5), rng.gauss(0, 0.5)]
    for _ in range(STEPS):
        left = [rng.gauss(MEAN_A, SD_A) for _ in range(BATCH)]
        right = [rng.gauss(MEAN_B, SD_B) for _ in range(BATCH)]
        critique(ref, critic_b, right, [ab[0] * v + ab[1] for v in left])
        critique(ref, critic_a, left, [ba[0] * v + ba[1] for v in right])
        g_ab, cycle_a = step(ref, ab, ba, critic_b, left)
        g_ba, cycle_b = step(ref, ba, ab, critic_a, right)
        for i in (0, 1):
            ab[i] -= G_RATE * (g_ab[i] + cycle_b[i]) / BATCH
            ba[i] -= G_RATE * (g_ba[i] + cycle_a[i]) / BATCH
    return (ab[0], ab[1]) + measure(ab, ba, [rng.gauss(MEAN_A, SD_A) for _ in range(PROBE)])


def summarise(rows):
    """(count, mean w, mean b) for one group of runs."""
    return (len(rows), statistics.fmean(r[0] for r in rows), statistics.fmean(r[1] for r in rows))


def overall(rows):
    """The across-seed numbers the checks quote."""
    return {
        "mean": statistics.fmean(r[2] for r in rows),
        "sd": statistics.fmean(r[3] for r in rows),
        "cycle": statistics.fmean(r[4] for r in rows),
        "scale": statistics.fmean(abs(r[0]) for r in rows),
        "worst_mean": max(abs(r[2] - MEAN_B) for r in rows),
        "worst_sd": max(abs(r[3] - SD_B) for r in rows),
    }


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = [run(ref, random, s) for s in range(SEEDS)]
    return {
        "reversing": summarise([r for r in rows if r[0] < 0]),
        "preserving": summarise([r for r in rows if r[0] > 0]),
        "predicted": (MEAN_B - MEAN_A, MEAN_B + MEAN_A), **overall(rows),
    }


def verify(result):
    rev, pre = result["reversing"], result["preserving"]
    want_up, want_down = result["predicted"]
    return [
        practice.Check(
            "ANSWER: yes -- the marginals match and the cycle closes, with no pairs",
            abs(result["mean"] - MEAN_B) < 0.15 and abs(result["sd"] - SD_B) < 0.1
            and result["cycle"] < 0.3,
            f"over {SEEDS} seeds G_AB sends A to mean {result['mean']:+.2f} sd {result['sd']:.2f}, "
            f"against B's {MEAN_B:+.1f} and {SD_B}, and G_BA(G_AB(a)) returns within "
            f"{result['cycle']:.2f} of a. No sample was ever paired with another: the only "
            "signals are two marginals and a cycle",
        ),
        practice.Check(
            "FINDING: which map it learns is not determined, and both answers appear",
            rev[0] > 0 and pre[0] > 0,
            f"{rev[0]} of {SEEDS} seeds settle on w = {rev[1]:+.2f}, b = {rev[2]:+.2f}, and "
            f"{pre[0]} on w = {pre[1]:+.2f}, b = {pre[2]:+.2f}. Same objective, same data, same "
            "code: one run sends the largest a to the largest b and the other to the smallest",
        ),
        practice.Check(
            "FINDING: the ambiguity is exact -- the objective has two global optima",
            abs(result["scale"] - SD_B / SD_A) < 0.25
            and abs(pre[2] - want_up) < 0.4 and abs(rev[2] - want_down) < 0.4,
            f"for a linear map the marginal forces |w| = sd_B/sd_A = {SD_B / SD_A:.0f} and "
            f"b = mean_B - w*mean_A, and the cycle forces w_BA = 1/w_AB. Both signs satisfy both, "
            f"giving b = {want_up:+.0f} at w=+1 and {want_down:+.0f} at w=-1. Measured "
            f"{pre[2]:+.2f} and {rev[2]:+.2f}, at |w| = {result['scale']:.2f}. Training cannot "
            "break a tie the loss cannot see",
        ),
        practice.Check(
            "CONTROL: both arms match the marginal, so this is two solutions and not one failure",
            result["worst_mean"] < 0.2 and result["worst_sd"] < 0.1,
            f"every seed lands within {result['worst_mean']:.2f} of B's mean and "
            f"{result['worst_sd']:.2f} of its sd, whichever sign it chose. Neither arm is a run "
            "that failed to converge; both converged, to maps the objective scores identically",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
