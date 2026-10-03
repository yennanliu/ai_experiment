"""Exercise 1 — joint sampling wins, but is still half again rougher than the data.

    **Easy.** In `code/main.py`, compare frame-to-frame delta for (a) independent
    per-frame sampling, (b) joint sequence sampling. Report the mean and variance
    of the deltas.

Reading of the exercise: the lesson's own `main()` is reproduced exactly -- same
seed 21, T=40, hidden 48, 3000 training steps -- and then asked for more than its
five clips: CLIPS=200 joint clips from the lesson's `sample_joint` and 200 from
its `independent_per_frame`, 1000 deltas each, through its `frame_deltas`. Two
references the lesson does not print are added: the deltas of the training data
itself (`make_video`), and a *fair* independent baseline -- the same trained
model's frames shuffled across clips, so each frame keeps its exact marginal and
only the coupling between frames is removed.

**ANSWER: joint mean 0.367 / variance 0.113; independent 1.085 / 0.729.** Joint
is 3.0x smaller in mean and 6.5x smaller in variance.

| sampler | mean delta | variance |
|---|---:|---:|
| (a) `independent_per_frame` | 1.085 | 0.729 |
| (b) `sample_joint` | **0.367** | **0.113** |
| same model, frames decoupled | 0.938 | 0.498 |
| the training data | 0.243 | 0.034 |

**FINDING: the lesson's baseline is not a sampler of this data.**
`independent_per_frame` is hand-written `gauss(0, 1) + 0.3 t`: its frame means
climb 0.05 -> 1.43 while the data's stay within 0.04 of zero. The fair baseline
still lands at 0.938, so the gap is real -- coupling, not the baseline's
construction, buys most of it.

**FINDING: joint is still 1.5x the data's own roughness, and its shape is wrong.**
The data moves 0.243 per frame; the model 0.367, with 3.3x the variance. Per-frame
spread is U-shaped, 0.97 -> 0.69 -> 1.11, where the data fans out 0.92 -> 1.75:
the model pinches every clip in the middle instead of learning a slope.

**CONTROL:** the first five joint clips reproduce `main()`'s printed 0.61 exactly
-- which also shows that headline rests on 25 deltas and overstates the joint
mean by 1.7x -- and the data's 0.243 matches the closed form
`E|N(0, 0.3^2 + 2*0.05^2)|` = 0.246.

Structure: `train` rebuilds main()'s model; `summary` gives mean, variance and
per-frame spread; `solve` draws the four samplers.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "10-video-generation"
CLIPS, T, T_DIM, HIDDEN = 200, 40, 8, 48


def train(ref):
    """main()'s exact model and rng stream, ready to sample."""
    rng = random.Random(21)
    alphas, bars = ref.make_schedule(T)
    net = ref.init_net(ref.T_FRAMES * (1 + ref.POS_DIM) + T_DIM, HIDDEN, ref.T_FRAMES, rng)
    ref.train_joint(net, bars, T, T_DIM, steps=3000, lr=0.01, rng=rng)
    return net, alphas, bars, rng


def summary(ref, clips):
    """(mean delta, variance of deltas, per-frame means, per-frame spreads)."""
    deltas = [d for clip in clips for d in ref.frame_deltas(clip)]
    frames = list(zip(*clips))
    return (
        statistics.fmean(deltas),
        statistics.pvariance(deltas),
        [statistics.fmean(f) for f in frames],
        [statistics.pstdev(f) for f in frames],
    )


def shuffle(clips):
    """Frame i of clip k taken from clip k + 7i: marginals kept, coupling removed."""
    return [
        [clips[(k + 7 * i) % len(clips)][i] for i in range(len(clips[0]))]
        for k in range(len(clips))
    ]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    net, alphas, bars, rng = train(ref)
    joint = [ref.sample_joint(net, alphas, bars, T, T_DIM, rng) for _ in range(CLIPS)]
    indep = [ref.independent_per_frame(ref.T_FRAMES, rng) for _ in range(CLIPS)]
    data = [ref.make_video(rng) for _ in range(CLIPS)]
    decoupled = shuffle(joint)
    first_five = [d for clip in joint[:5] for d in ref.frame_deltas(clip)]
    return {
        "joint": summary(ref, joint),
        "indep": summary(ref, indep),
        "data": summary(ref, data),
        "decoupled": summary(ref, decoupled),
        "main_print": statistics.fmean(first_five),
        "closed_form": math.sqrt(0.3**2 + 2 * 0.05**2) * math.sqrt(2 / math.pi),
        "data_drift": max(abs(m) for m in summary(ref, data)[2]),
    }


def verify(result):
    joint, indep, data, fair = result["joint"], result["indep"], result["data"], result["decoupled"]
    spread, drift = ", ".join(f"{s:.2f}" for s in joint[3]), result["data_drift"]
    return [
        practice.Check(
            "ANSWER: joint is ~3x smaller in mean and ~6x in variance",
            joint[0] * 2 < indep[0] and joint[1] * 4 < indep[1],
            f"over {CLIPS} clips each (1000 deltas): joint mean {joint[0]:.3f} variance "
            f"{joint[1]:.3f}; independent mean {indep[0]:.3f} variance {indep[1]:.3f} -- "
            f"{indep[0] / joint[0]:.1f}x in mean, {indep[1] / joint[1]:.1f}x in variance",
        ),
        practice.Check(
            "FINDING: the lesson's baseline is not a sampler of this data, yet the gap survives",
            indep[2][-1] > 1.0 and drift < 0.1 and fair[0] > 2 * joint[0],
            f"independent_per_frame is gauss(0,1) + 0.3t, so its frame means climb "
            f"{indep[2][0]:.2f} -> {indep[2][-1]:.2f} while the data's stay within "
            f"{drift:.2f} of zero. The fair baseline -- the same model's "
            f"frames shuffled across clips -- still gives {fair[0]:.3f} (variance {fair[1]:.3f}), "
            "so coupling, not the baseline's construction, buys most of the gap",
        ),
        practice.Check(
            "FINDING: joint is still 1.5x the data's own roughness, and pinched in the middle",
            joint[0] > 1.3 * data[0] and min(joint[3]) < joint[3][0] < data[3][-1],
            f"the data moves {data[0]:.3f} per frame (variance {data[1]:.3f}); the model "
            f"{joint[0]:.3f} ({joint[1]:.3f}). Its per-frame spread is U-shaped ({spread}) "
            f"where the data fans out {data[3][0]:.2f} -> {data[3][-1]:.2f}: the model pinches "
            "every clip in the middle instead of learning a slope",
        ),
        practice.Check(
            "CONTROL: main()'s own printout is reproduced, and the data matches its closed form",
            abs(result["main_print"] - 0.61) < 0.005
            and abs(data[0] - result["closed_form"]) < 0.02,
            f"the first five joint clips give {result['main_print']:.2f}, main()'s printed "
            f"joint figure -- a headline from 25 deltas that overstates the 1000-delta mean by "
            f"{result['main_print'] / joint[0]:.1f}x. The data's {data[0]:.3f} matches "
            f"E|N(0, 0.3^2 + 2*0.05^2)| = {result['closed_form']:.3f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
