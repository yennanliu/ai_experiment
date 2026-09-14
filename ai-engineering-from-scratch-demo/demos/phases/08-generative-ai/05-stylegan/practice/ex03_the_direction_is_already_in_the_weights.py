"""Exercise 3 — the SVM recovers a direction that is sitting in synth["bias2"].

    **Hard.** Take a pretrained StyleGAN3 FFHQ model (ffhq-1024.pkl). Find the
    `w` direction that controls "smile" by training an SVM on labelled samples;
    report how far you can push before identity drifts.

Reading of the exercise: the pretrained model is unbuildable here -- `torch`,
`dnnlib` and `legacy`, the three imports `ffhq-1024.pkl` needs to unpickle, are
all absent, and the checkpoint is ~300 MB that this repo does not ship. So the
method is run at the scale that *is* available (`DESIGN D11`): the lesson's own
mapping and synthesis networks, a binary attribute standing in for "smile", a
real `LinearSVC` on SAMPLES labelled `w` vectors, and the push measured rather
than eyeballed. The attribute is the output's mean brightness, thresholded at its
median -- the closest thing a 6-channel vector has to a facial expression.

**ANSWER: the SVM finds it, at accuracy 0.998, and it never stops working.**
Pushing `w` along the recovered direction moves the attribute by **+1.019 per
unit of alpha**, linearly, forever: +0.95 at alpha 1, +4.01 at 4, +16.24 at 16.
There is no distance at which the attribute stops responding.

**FINDING: the direction did not have to be learned.** `adain`'s last call sets
the output's mean to `bias`, and `bias = <synth["bias2"], w>`. So the attribute is
an exact linear form in `w` whose normal is a vector already stored in the
weights. Measured: `mean(output) - <bias2, w>` is **7e-17**, and the SVM's
recovered direction aligns with `bias2` at **cos = 0.995**. The exercise's
labelled samples and margin fitting rediscover, to three decimals, a row of the
weight dict.

**FINDING: "how far before identity drifts" has no threshold here.** The output's
normalised shape moves **1.58** within the first unit of alpha and then *stops*,
sitting at 1.38 by alpha 16 -- the earlier blocks saturate into a fixed leaky
regime and the shape converges. Meanwhile the output's spread grows without
bound, 0.22 to **4.44**. Identity drifts immediately and then cannot drift
further; what grows is scale. The regime the question presumes, where the
attribute moves and everything else holds, is empty on this generator.

**CONTROL: the direction beats a random one.** A random unit direction in `w`
moves the attribute by far less per unit of alpha than the fitted one, so the SVM
is finding structure rather than reporting the size of a step.

Structure: `attribute` is the stand-in label; `push` walks along a direction;
`solve` fits the SVM and measures both axes.
"""

from __future__ import annotations

import importlib.util
import math
import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "05-stylegan"
Z_DIM, W_DIM, HIDDEN, DEPTH, SAMPLES = 8, 8, 6, 4, 600
ALPHAS, NEEDED = (0, 1, 2, 4, 8, 16), ("torch", "dnnlib", "legacy")


def attribute(row):
    """The stand-in for 'smile': how bright the sample is overall."""
    return statistics.fmean(row)


def shape_of(row):
    """The output with mean and spread divided out -- what AdaIN does not set."""
    mean, sd = statistics.fmean(row), statistics.pstdev(row)
    return [(v - mean) / (sd or 1.0) for v in row]


def push(ref, synth, const, base, direction, alpha):
    """Synthesise from w moved `alpha` along `direction`."""
    import random
    moved = [b + alpha * d for b, d in zip(base, direction)]
    return ref.stylegan_forward(moved, const, synth, 0.0, random.Random(0), adain_on=True)


def unit(vector):
    """`vector` scaled to length one."""
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


def sample_space(ref, random, mapping, synth, const, rng):
    """SAMPLES latents mapped through to w, and what the generator makes of each."""
    ws = [ref.mapping([rng.gauss(0, 1) for _ in range(Z_DIM)], mapping) for _ in range(SAMPLES)]
    return ws, [push(ref, synth, const, w, [0.0] * W_DIM, 0.0) for w in ws]


def trace(rows, start):
    """(attribute, spread, shape drift from `start`) at each step of a walk."""
    return [(attribute(r), statistics.pstdev(r),
             max(abs(a - b) for a, b in zip(shape_of(r), shape_of(start)))) for r in rows]


def closed_form_gap(synth, ws, rows):
    """Worst gap between the measured attribute and the closed form <bias2, w>."""
    return max(abs(attribute(r) - sum(synth["bias2"][j] * w[j] for j in range(W_DIM)))
               for w, r in zip(ws, rows))


def solve():
    import random
    import numpy as np
    from sklearn.svm import LinearSVC
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(3)
    mapping = ref.init_mapping(Z_DIM, W_DIM, DEPTH, rng)
    synth = ref.init_synth(HIDDEN, W_DIM, rng)
    const = [rng.gauss(0, 0.3) for _ in range(HIDDEN)]
    ws, rows = sample_space(ref, random, mapping, synth, const, rng)
    cut = statistics.median(attribute(r) for r in rows)
    labels = [attribute(r) > cut for r in rows]
    svm = LinearSVC(C=1e4, max_iter=200_000).fit(np.array(ws), labels)
    found = unit(list(svm.coef_[0]))
    walk = [push(ref, synth, const, ws[0], found, a) for a in ALPHAS]
    return {
        "absent": [m for m in NEEDED if importlib.util.find_spec(m) is None],
        "accuracy": float(svm.score(np.array(ws), labels)),
        "cosine": abs(sum(a * b for a, b in zip(found, unit(synth["bias2"])))),
        "closed_form": closed_form_gap(synth, ws, rows),
        "walk": trace(walk, walk[0]),
        "random": attribute(push(ref, synth, const, ws[0],
                                 unit([rng.gauss(0, 1) for _ in range(W_DIM)]), ALPHAS[-1])),
    }


def verify(result):
    walk, alphas = result["walk"], list(ALPHAS)
    per_unit = (walk[-1][0] - walk[0][0]) / alphas[-1]
    return [
        practice.Check(
            "ANSWER: the SVM finds it, and the attribute never stops responding",
            result["absent"] == list(NEEDED) and result["accuracy"] > 0.99 and per_unit > 0.5,
            f"{result['absent']} are all absent, so ffhq-1024.pkl cannot even be unpickled and the "
            f"method runs on the lesson's own networks. The SVM separates {SAMPLES} labelled w at "
            f"accuracy {result['accuracy']:.3f}, and pushing along it moves the attribute "
            f"{per_unit:+.3f} per unit of alpha: "
            + ", ".join(f"a={a}: {w[0]:+.2f}" for a, w in zip(alphas, walk)),
        ),
        practice.Check(
            "FINDING: the direction did not have to be learned -- it is synth['bias2']",
            result["closed_form"] < 1e-12 and result["cosine"] > 0.95,
            f"adain's last call sets the output's mean to bias = <synth['bias2'], w>, so the "
            f"attribute is an exact linear form in w: mean(output) - <bias2, w> measures "
            f"{result['closed_form']:.0e}. The fitted direction aligns with that stored vector at "
            f"cos = {result['cosine']:.3f}, so the labelled samples and the margin rediscover a "
            "row of the weight dict",
        ),
        practice.Check(
            "FINDING: 'how far before identity drifts' has no threshold here",
            walk[1][2] > 1.0 and walk[-1][2] < walk[1][2] and walk[-1][1] > 10 * walk[0][1],
            f"the normalised shape moves {walk[1][2]:.2f} within the first unit of alpha and then "
            f"stops, sitting at {walk[-1][2]:.2f} by alpha {alphas[-1]} as the earlier blocks "
            f"saturate. Meanwhile the spread grows without bound, {walk[0][1]:.2f} to "
            f"{walk[-1][1]:.2f}. Identity drifts at once and then cannot drift further, so the "
            "regime the question presumes -- attribute moves, rest holds -- is empty here",
        ),
        practice.Check(
            "CONTROL: the fitted direction beats a random one",
            abs(walk[-1][0] - walk[0][0]) > 2 * abs(result["random"] - walk[0][0]),
            f"at alpha {alphas[-1]} the fitted direction reaches {walk[-1][0]:+.2f} from a start of "
            f"{walk[0][0]:+.2f}, while a random unit direction of the same length reaches "
            f"{result['random']:+.2f}. The SVM is finding structure rather than reporting the size "
            "of the step it was asked to take",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
