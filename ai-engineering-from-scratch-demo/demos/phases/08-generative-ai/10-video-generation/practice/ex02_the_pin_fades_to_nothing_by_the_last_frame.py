"""Exercise 2 — the pin fades to nothing by the last frame.

    **Medium.** Add a first-frame condition: pin frame 0 to a given value and
    sample the rest. Measure how the pinned value propagates.

Reading of the exercise: main()'s model is trained exactly as the lesson does
(seed 21, T=40, 3000 steps), and the conditioning is added *at sampling time*
with the lesson's own `forward`, `patchify_with_pos`, `flatten` and `sin_embed`
inside a reverse loop identical to `sample_joint` apart from the pin -- no
retraining, which is how image-to-video is usually bolted on first. Two pins are
tried: hard-writing the clean value at every step, and the replacement method
(RePaint-style), which writes the pin noised to the current step's level.
"Propagation" is measured as the regression slope of E[frame k] on the pinned
value over PINS = -2..2, SAMPLES clips each. The answer is known in advance: in
`make_video` every frame shares `base`, so E[frame k | frame 0 = v] = v / 1.0025
-- a slope of ~1.0 at every frame.

**ANSWER: it propagates, and decays to nothing.** Slope by frame, clean pin:
1.00, 0.46, 0.34, 0.28, 0.14, 0.01. The data's own slopes are 1.00 -> 1.04.
Frame 5 keeps 1% of the pin; the data says it should keep all of it.

**FINDING: the pin method is not the problem.** The replacement method gives
1.00, 0.47, 0.33, 0.27, 0.12, -0.01 -- within 0.02 of the clean pin everywhere --
and the *unpinned* model's own lag regression on frame 0 falls 0.75 -> -0.10.
The decay is in the model; any conditioning trick inherits it.

**FINDING: sampling starts far from pure noise, and the start survives.**
`make_schedule(40)` ends at alpha_bar = **0.667**, so the reverse chain starts
from a step where the model saw 0.82 x clip + 0.58 x noise -- mostly signal,
frames strongly correlated -- yet `sample_joint` hands it iid N(0, 1). Each
output frame keeps a correlation of **0.37-0.76** with its *own* independent
starting draw: per-frame noise that was never coupled survives the 40 steps, and
it is what a pin on frame 0 cannot reach.

**CONTROL:** the pinned frame comes back as exactly the value written, and
`make_video`'s own regression slopes (2000 clips) are 1.00-1.04.

Structure: `pinned_sample` is the reverse loop with a pin; `slopes` regresses
frame means on the pin; `start_memory` reads sample_joint's own starting noise.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "10-video-generation"
T, T_DIM, HIDDEN, PINS, SAMPLES = 40, 8, 48, (-2, -1, 0, 1, 2), 80


def pinned_sample(ref, model, value, noised, rng):
    """sample_joint's reverse loop, with frame 0 overwritten before every step."""
    net, alphas, bars = model
    x = [rng.gauss(0, 1) for _ in range(ref.T_FRAMES)]
    for t in range(T - 1, -1, -1):
        keep = bars[t] if noised else 1.0
        x[0] = math.sqrt(keep) * value + math.sqrt(1 - keep) * rng.gauss(0, 1)
        eps, _ = ref.forward(ref.flatten(ref.patchify_with_pos(x)), ref.sin_embed(t, T_DIM), net)
        beta = 1 - alphas[t]
        x = [(v - beta / math.sqrt(1 - bars[t]) * e) / math.sqrt(alphas[t]) for v, e in zip(x, eps)]
        if t > 0:
            x = [v + math.sqrt(beta) * rng.gauss(0, 1) for v in x]
    x[0] = value
    return x


def regress(xs, clips):
    """Slope of each frame on xs."""
    mx = statistics.fmean(xs)
    den = sum((a - mx) ** 2 for a in xs)
    return [sum((a - mx) * c[k] for a, c in zip(xs, clips)) / den for k in range(len(clips[0]))]


def slopes(ref, model, noised, rng):
    """Regression slope of E[frame k] on the pinned value, one per frame."""
    xs = [value for value in PINS for _ in range(SAMPLES)]
    return regress(xs, [pinned_sample(ref, model, value, noised, rng) for value in xs])


def start_memory(ref, model, rng, n=300):
    """Correlation of each output frame with sample_joint's own starting draw for it."""
    net, alphas, bars = model
    starts, outs = [], []
    for _ in range(n):
        probe = random.Random()
        probe.setstate(rng.getstate())
        starts.append([probe.gauss(0, 1) for _ in range(ref.T_FRAMES)])
        outs.append(ref.sample_joint(net, alphas, bars, T, T_DIM, rng))
    lag = regress([c[0] for c in outs], outs)
    return [statistics.correlation(s, o) for s, o in zip(zip(*starts), zip(*outs))], lag


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(21)
    alphas, bars = ref.make_schedule(T)
    net = ref.init_net(ref.T_FRAMES * (1 + ref.POS_DIM) + T_DIM, HIDDEN, ref.T_FRAMES, rng)
    ref.train_joint(net, bars, T, T_DIM, steps=3000, lr=0.01, rng=rng)
    model = net, alphas, bars
    memory, lag = start_memory(ref, model, rng)
    data = [ref.make_video(random.Random(i)) for i in range(2000)]
    return {
        "clean": slopes(ref, model, False, rng),
        "noised": slopes(ref, model, True, rng),
        "memory": memory,
        "lag": lag,
        "alpha_bar_end": bars[-1],
        "data": regress([c[0] for c in data], data),
        "pin_kept": pinned_sample(ref, model, 1.25, True, rng)[0],
    }


def fmt(values):
    return ", ".join(f"{v:.2f}" for v in values)


def verify(result):
    clean, noised, data = result["clean"], result["noised"], result["data"]
    gap = max(abs(a - b) for a, b in zip(clean, noised))
    return [
        practice.Check(
            "ANSWER: the pin propagates, and decays to nothing by the last frame",
            clean[1] < 0.7 and clean[-1] < 0.2 and min(data) > 0.95,
            f"slope of E[frame k] on the pinned value, clean pin: {fmt(clean)}. The data's "
            f"own slopes are {fmt(data)}: frame 5 should keep all of the pin and keeps "
            f"{clean[-1]:.2f} of it",
        ),
        practice.Check(
            "FINDING: the pin method is not the problem",
            gap < 0.15 and result["lag"][-1] < 0.2,
            f"the replacement (noised) pin gives {fmt(noised)}, within {gap:.2f} of the clean "
            f"pin, and the unpinned model's own lag regression on frame 0 runs "
            f"{fmt(result['lag'])}. The decay is in the model; any conditioning trick inherits it",
        ),
        practice.Check(
            "FINDING: sampling starts far from pure noise, and the per-frame start survives",
            result["alpha_bar_end"] > 0.5 and min(result["memory"]) > 0.25,
            f"make_schedule(40) ends at alpha_bar = {result['alpha_bar_end']:.3f}, so the chain "
            f"starts where training showed {math.sqrt(result['alpha_bar_end']):.2f} x clip + "
            f"{math.sqrt(1 - result['alpha_bar_end']):.2f} x noise, yet sample_joint starts from "
            f"iid N(0,1). Each output frame keeps r = {fmt(result['memory'])} with its own "
            "independent starting draw -- uncoupled noise a pin on frame 0 cannot reach",
        ),
        practice.Check(
            "CONTROL: the pin is honoured, and the data's propagation is ~1 everywhere",
            result["pin_kept"] == 1.25 and max(abs(s - 1) for s in data) < 0.06,
            f"the pinned frame returns {result['pin_kept']} for a pin of 1.25, and make_video's "
            f"own slopes over 2000 clips are {fmt(data)} -- every frame shares `base`, so "
            "E[frame k | frame 0 = v] = v / 1.0025",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
