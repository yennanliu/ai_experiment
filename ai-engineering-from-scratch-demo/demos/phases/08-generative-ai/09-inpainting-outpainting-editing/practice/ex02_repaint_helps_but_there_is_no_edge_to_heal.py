"""Exercise 2 — RePaint helps, but there is no edge for it to heal.

    **Medium.** Implement RePaint: at every 10th reverse step, jump back 5 steps
    (add noise) and re-denoise. Measure whether it reduces boundary residual at
    the mask edge.

Reading of the exercise: the model is the one `main()` trains, and the mask is
`main()`'s inpainting mask -- pin dims 0-2, regenerate dims 3-4 -- so the "mask
edge" is dim 3, the regenerated dim next to a pinned one, and dim 4 is the
interior. The reverse step is the lesson's own (its `forward`, `sin_embed` and
schedule, reinjection included); RePaint adds only the jump: after reverse steps
t = 30, 20 and 10 the sample is re-noised 5 steps forward with the closed-form
`sqrt(abar_hi / abar_lo)` kernel and denoised again. Residual is
`|generated - source|`. The same comparison is repeated with 4 of 5 dims masked,
where the plain sampler is weakest.

**ANSWER: yes, modestly.** Edge residual falls **0.344 -> 0.307** with dims 3-4
masked and **0.793 -> 0.696** with dims 1-4 masked. At T = 40, "every 10th step"
is only 3 resamplings, costing 55 network calls instead of 40.

**FINDING: there is no edge in this data, so "boundary residual" is just
residual.** `sample_data` draws every dim independently around the same cluster
centre; nothing makes dim 3 a neighbour of dim 2 rather than of dim 0. The edge
and interior residuals of the plain sampler are **0.344** and **0.367** -- the edge is no
worse than the interior (across three more seeds it is 0.01-0.04 *better*), so
the seam the exercise asks about cannot form here.

**FINDING: what RePaint actually fixes is the cluster.** The fraction of
samples whose regenerated dims land in the source's cluster rises from **96.0%**
to **98.6%** (dims 3-4) and from **65.8%** to **75.0%** (dims 1-4). RePaint's gain in this toy is global coherence -- getting the
semantics right -- which is the mechanism Lugmayr et al. describe for real
images too: the seam is a symptom of the content disagreeing with its context.

**CONTROL: the sampler is the lesson's.** With no jumps the reverse loop here
reproduces the lesson's `inpaint` from one seed to **0.0**, so the RePaint
numbers differ from the baseline only by the resampling.

Structure: `plan` lays out the reverse schedule with its jumps; `reinject` and
`denoise` are the lesson's step; `sample` walks the plan;
`measure` scores edge, interior and cluster match over the trials.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "09-inpainting-outpainting-editing"
T, T_DIM, HIDDEN, D, TRIALS, EVERY, BACK = 40, 8, 32, 5, 500, 10, 5


def plan(every):
    """Reverse times T-1..0; after each `every`-th step, a jump of BACK and a re-run."""
    seq = []
    for t in range(T - 1, -1, -1):
        seq.append(t)
        if every and t % every == 0 and t > 0:
            top = min(t - 1 + BACK, T - 1)
            seq.append(("jump", t - 1, top))
            seq.extend(range(top, t - 1, -1))
    return seq


def reinject(x, clean, mask, bar, rng):
    """The lesson's reinjection: pinned dims become a freshly noised copy of the source."""
    return [
        v if m else math.sqrt(bar) * c + math.sqrt(1 - bar) * rng.gauss(0, 1)
        for v, c, m in zip(x, clean, mask)
    ]


def denoise(ref, model, x, t, rng):
    """The lesson's reverse step from x_t: predicted mean, plus noise unless t == 0."""
    net, alphas, bars = model
    eps, _ = ref.forward(x, ref.sin_embed(t, T, T_DIM), net)
    beta = 1 - alphas[t]
    mean = [
        (x[i] - beta / math.sqrt(1 - bars[t]) * eps[i]) / math.sqrt(alphas[t]) for i in range(D)
    ]
    return [m + math.sqrt(beta) * rng.gauss(0, 1) for m in mean] if t > 0 else mean


def sample(ref, model, clean, mask, rng, seq):
    """Walk `seq`: reinject and denoise at each time, re-noise at each jump."""
    bars = model[2]
    x = [rng.gauss(0, 1) for _ in range(D)]
    for t in seq:
        if isinstance(t, tuple):
            keep = bars[t[2]] / bars[t[1]]
            x = [math.sqrt(keep) * v + math.sqrt(1 - keep) * rng.gauss(0, 1) for v in x]
        else:
            x = denoise(ref, model, reinject(x, clean, mask, bars[t], rng), t, rng)
    return [x[i] if mask[i] else clean[i] for i in range(D)]


def measure(ref, model, masked, every, rng):
    """(edge residual, interior residual, cluster match) over TRIALS sources."""
    mask, edge = [i >= D - masked for i in range(D)], D - masked
    totals, seq = [0.0, 0.0, 0], plan(every)
    for _ in range(TRIALS):
        clean, cluster = ref.sample_data(rng, D)
        out = sample(ref, model, clean, mask, rng, seq)
        totals[0] += abs(out[edge] - clean[edge])
        totals[1] += abs(out[-1] - clean[-1])
        totals[2] += all((out[i] > 0) == bool(cluster) for i in range(edge, D))
    return [v / TRIALS for v in totals]


def solve():
    import random

    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(5)
    alphas, bars = ref.make_schedule(T)
    net = ref.init_net(D, T_DIM, HIDDEN, rng)
    ref.train(net, bars, T, steps=5000, lr=0.01, t_dim=T_DIM, d=D, rng=rng)
    model = (net, alphas, bars)
    clean, mask = ref.sample_data(random.Random(1), D)[0], [False] * 3 + [True] * 2
    mine = sample(ref, model, clean, mask, random.Random(3), plan(0))
    theirs = ref.inpaint(net, alphas, bars, T, T_DIM, D, clean, mask, random.Random(3))
    keys = [(2, 0), (2, EVERY), (4, 0), (4, EVERY)]
    runs = {key: measure(ref, model, *key, random.Random(11)) for key in keys}
    return {"runs": runs, "parity": max(abs(a - b) for a, b in zip(mine, theirs))}


def verify(result):
    runs, calls = result["runs"], sum(isinstance(t, int) for t in plan(EVERY))
    two, two_rp, four, four_rp = runs[2, 0], runs[2, EVERY], runs[4, 0], runs[4, EVERY]
    return [
        practice.Check(
            "ANSWER: yes, modestly -- RePaint lowers the edge residual",
            two_rp[0] < two[0] and four_rp[0] < four[0],
            f"edge residual {two[0]:.3f} -> {two_rp[0]:.3f} with dims 3-4 masked and "
            f"{four[0]:.3f} -> {four_rp[0]:.3f} with dims 1-4 masked; at T={T} that is 3 "
            f"resamplings, {calls} network calls instead of {T}",
        ),
        practice.Check(
            "FINDING: there is no edge in this data, so boundary residual is just residual",
            two[0] < two[1] + 0.02,
            f"sample_data draws every dim independently around one centre, so plain-sampler edge "
            f"and interior residuals are {two[0]:.3f} and {two[1]:.3f}: the edge is no worse",
        ),
        practice.Check(
            "FINDING: what RePaint actually fixes is the cluster",
            two_rp[2] > two[2] and four_rp[2] > four[2],
            f"regenerated dims landing in the source's cluster: {two[2]:.1%} -> {two_rp[2]:.1%} "
            f"(dims 3-4) and {four[2]:.1%} -> {four_rp[2]:.1%} (dims 1-4): global coherence, not a seam",
        ),
        practice.Check(
            "CONTROL: with no jumps the sampler is the lesson's inpaint",
            result["parity"] == 0.0,
            f"from one seed the jump-free loop and the lesson's inpaint agree to {result['parity']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
