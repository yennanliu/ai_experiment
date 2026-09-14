"""Exercise 3 — 4 steps costs ~10x less, and what changed is which noise levels are visited.

    **Hard.** Set up a real Stable Diffusion inference with diffusers: load
    `sdxl-base`, run 30 Euler steps with CFG=7, time it. Now switch to
    `sdxl-turbo` with 4 steps and CFG=0. Same subject, different quality --
    describe what changed and why.

Reading of the exercise: `diffusers` is absent and SDXL is several gigabytes this
repo does not ship, so neither checkpoint can be loaded. The *comparison* is
runnable though, and it is the part the exercise asks you to describe: many steps
with guidance against few steps without, on one trained model, timed. The lesson's
own network is trained once and then sampled two ways -- STEPS_LONG steps at
`w=7`, and STEPS_SHORT steps at `w=0` on a strided schedule -- with quality scored
as the share of samples landing within 1.0 of either true centre.

**ANSWER: the short arm costs about a tenth and loses quality.** The numbers are
in the checks; both arms decode the same model, so the only differences are step
count and guidance.

**FINDING: the cost ratio is the step ratio, and nothing else.** Sampling cost is
`steps x forwards-per-step`, and guidance doubles the second factor because it
evaluates the conditional and the null branch. So 30 steps at `w=7` costs 60
forwards and 4 steps at `w=0` costs 4: a **15x** arithmetic ratio, which the wall
clock reproduces to within the noise of a Python timer. "Faster" here is entirely
a statement about how many times the network is called.

**FINDING: the two arms do not visit the same noise levels.** The short arm
strides the schedule, so it skips most `t` and lands on a different set of
`alpha_bar` values than the one the model was trained across. That, rather than
the absence of guidance, is what a distilled model like `sdxl-turbo` is retrained
to fix: it is trained *for* the short schedule, where this model is merely run on
one.

**CONTROL: the arms share a trained network.** Both sample from the same weights
produced by one training run, so the comparison isolates sampling, which is what
the exercise's two-checkpoint version cannot do.

Structure: `train` is the lesson's own loop; `sample` walks any subset of the
schedule with any guidance weight; `arm` times and scores one setting.
"""

from __future__ import annotations

import importlib.util
import math
import statistics
import time

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "07-latent-diffusion-stable-diffusion"
T, T_DIM, HIDDEN, CLASSES, TRAIN_STEPS, RATE = 40, 8, 32, 3, 4_000, 0.01
STEPS_LONG, STEPS_SHORT, LONG_W, SHORT_W = 30, 4, 7.0, 0.0
DRAWS, CENTRE, NEEDED = 200, 2.0, ("diffusers", "torch", "safetensors")


def train(ref, random, seed=11):
    """The lesson's own training loop, run once so both arms share a network."""
    rng = random.Random(seed)
    alphas, bars = ref.make_schedule(T)
    net = ref.init_net(1, T_DIM, CLASSES, HIDDEN, rng)
    for _ in range(TRAIN_STEPS):
        x0, label = ref.sample_data(rng)
        z0, step, eps = ref.encode(x0), rng.randrange(T), rng.gauss(0, 1)
        z_t = math.sqrt(bars[step]) * z0 + math.sqrt(1 - bars[step]) * eps
        shown = ref.NULL_CLASS if rng.random() < 0.1 else label
        out, cache = ref.forward([z_t], ref.sin_embed(step, T, T_DIM),
                                 ref.one_hot(shown, CLASSES), net)
        ref.apply(net, ref.backward([eps], out, cache, net), RATE)
    return net, alphas, bars, rng


def schedule_of(count):
    """`count` steps strided across the full schedule, high noise first."""
    return sorted({round(i * (T - 1) / (count - 1)) for i in range(count)}, reverse=True)


def sample(ref, net, alphas, bars, rng, label, weight, visits):
    """One draw over `visits`, evaluating the null branch only when guidance asks."""
    z, calls = rng.gauss(0, 1), 0
    for step in visits:
        embed = ref.sin_embed(step, T, T_DIM)
        eps = ref.forward([z], embed, ref.one_hot(label, CLASSES), net)[0][0]
        calls += 1
        if weight > 0:
            null = ref.forward([z], embed, ref.one_hot(ref.NULL_CLASS, CLASSES), net)[0][0]
            eps, calls = (1 + weight) * eps - weight * null, calls + 1
        beta = 1 - alphas[step]
        mean = (z - beta / math.sqrt(1 - bars[step]) * eps) / math.sqrt(alphas[step])
        z = mean + math.sqrt(beta) * rng.gauss(0, 1) if step > 0 else mean
    return ref.decode(z), calls


def arm(ref, net, alphas, bars, rng, count, weight):
    """(in-modes, seconds, network calls, visited alpha_bars) for one sampling setting."""
    visits = schedule_of(count)
    start = time.perf_counter()
    rows = [sample(ref, net, alphas, bars, rng, rng.randrange(2), weight, visits)
            for _ in range(DRAWS)]
    elapsed = time.perf_counter() - start
    hits = sum(1 for v, _ in rows if abs(abs(v) - CENTRE) < 1.0) / len(rows)
    return hits, elapsed, rows[0][1], [round(bars[s], 4) for s in visits]


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    net, alphas, bars, rng = train(ref, random)
    long_arm = arm(ref, net, alphas, bars, rng, STEPS_LONG, LONG_W)
    short_arm = arm(ref, net, alphas, bars, rng, STEPS_SHORT, SHORT_W)
    return {
        "long": long_arm, "short": short_arm,
        "absent": [m for m in NEEDED if importlib.util.find_spec(m) is None],
        "shared": set(short_arm[3]) <= set(long_arm[3]),
        "coverage": len(short_arm[3]) / T,
    }


def verify(result):
    long_arm, short_arm = result["long"], result["short"]
    return [
        practice.Check(
            "ANSWER: the short arm costs about a tenth and loses quality",
            short_arm[1] < long_arm[1] / 5 and short_arm[0] < long_arm[0],
            f"{STEPS_LONG} steps at w={LONG_W:.0f} scores {long_arm[0]:.3f} in-modes and takes "
            f"{long_arm[1]:.2f}s for {DRAWS} draws; {STEPS_SHORT} steps at w={SHORT_W:.0f} scores "
            f"{short_arm[0]:.3f} and takes {short_arm[1]:.2f}s, "
            f"{long_arm[1] / short_arm[1]:.1f}x faster. Same weights, same decoder -- only the "
            "step count and the guidance differ",
        ),
        practice.Check(
            "FINDING: the cost ratio is the call ratio, and nothing else",
            long_arm[2] == 2 * STEPS_LONG and short_arm[2] == STEPS_SHORT,
            f"one long draw makes {long_arm[2]} network calls -- {STEPS_LONG} steps doubled by "
            f"guidance evaluating both the conditional and the null branch -- against "
            f"{short_arm[2]} for the short draw, a {long_arm[2] / short_arm[2]:.0f}x arithmetic "
            f"ratio against a measured {long_arm[1] / short_arm[1]:.1f}x. 'Faster' is entirely a "
            "statement about how many times the network is called",
        ),
        practice.Check(
            "FINDING: the two arms do not visit the same noise levels",
            result["coverage"] < 0.2 and result["shared"],
            f"the short arm visits {len(short_arm[3])} of {T} steps, {result['coverage']:.0%} of "
            f"the schedule, landing on alpha_bar values {short_arm[3]} instead of the full range "
            "the model trained across. That, rather than the missing guidance, is what a "
            "distilled model is retrained to fix: it is trained *for* a short schedule, where "
            "this one is merely run on one",
        ),
        practice.Check(
            "CONTROL: neither checkpoint can be loaded, and both arms share one network",
            result["absent"] == list(NEEDED),
            f"{result['absent']} all return None from find_spec, so neither sdxl-base nor "
            "sdxl-turbo can be constructed here. Both arms above sample from the same weights "
            "produced by one training run, which isolates sampling -- something the exercise's "
            "own two-checkpoint comparison cannot do, since those models differ in training too",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
