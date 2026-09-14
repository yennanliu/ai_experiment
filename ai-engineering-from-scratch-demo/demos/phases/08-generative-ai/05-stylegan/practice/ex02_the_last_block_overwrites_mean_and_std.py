"""Exercise 2 — mixing is disentangled here by algebra, and no decoder learns anything.

    **Medium.** Implement mixing regularization: for a training batch, compute
    `w_a`, `w_b`, and apply `w_a` for the first half of synthesis and `w_b` for
    the second half. Does the decoder learn disentangled styles?

Reading of the exercise: mixing is implemented as the exercise describes -- `w_a`
drives blocks 0 and 1, `w_b` drives block 2 -- reusing the lesson's own `matmul`,
`add`, `leaky` and `adain` so that only the routing of `w` is new. "Disentangled"
is then measured rather than eyeballed: hold one half fixed, vary the other over
PAIRS latents, and ask which properties of the output move.

**ANSWER: perfectly disentangled, and not because anything learned it.**

| varying | output mean, sd across latents | output sd, sd across latents | distinct shapes |
|---|---:|---:|---:|
| `w_b` (last block) | 0.049 | 0.092 | **2** |
| `w_a` (first two) | **0.000000** | **0.000023** | **40** |

`w_a` moves the *shape* and cannot move the mean or the spread at all; `w_b` moves
the mean and the spread and barely moves the shape. The split is exact to six
decimal places.

**FINDING: `adain` is the whole mechanism, as an identity.** The last operation of
the last block is `adain(h, scale, bias)`, which returns
`scale * (f - mean) / sd + bias`. So the output's mean **is** `bias` and its sd
**is** `|scale|`, and both are linear forms in `w_b`. Measured over PAIRS latents,
`mean(output) - <bias2, w>` is **2e-17** and `sd(output) - |<scale2, w>|` is
**9e-08**, the second being `adain`'s own epsilon. Nothing earlier in the network
can affect either number, because the last line overwrites both.

**FINDING: there is no decoder that learns.** The lesson has no loss, no
gradient and no optimiser -- `main()` calls `init_mapping` and `init_synth` and
then only reads. So "does the decoder learn disentangled styles" has no referent
here: the disentanglement above is a property of where `adain` sits, present at
initialisation and unchanged by a training run that does not exist.

**CONTROL: the mix really is a mix.** Driving both halves with the same `w`
reproduces the lesson's own `stylegan_forward` on that `w`, bit for bit, so the
mixing path differs from the reference only in routing.

Structure: `mix` is the two-`w` synthesis; `spread` reports what moved.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "05-stylegan"
Z_DIM, W_DIM, HIDDEN, DEPTH, PAIRS, SPLIT = 8, 8, 6, 4, 40, 2


def mix(ref, synth, const, w_a, w_b):
    """The lesson's synthesis with w_a driving blocks < SPLIT and w_b driving the rest."""
    h = list(const)
    for i in range(3):
        w = w_a if i < SPLIT else w_b
        h = [ref.leaky(x) for x in ref.add(ref.matmul(synth[f"W{i}"], h), synth[f"b{i}"])]
        h = ref.adain(h, sum(synth[f"scale{i}"][j] * w[j] for j in range(W_DIM)),
                      sum(synth[f"bias{i}"][j] * w[j] for j in range(W_DIM)))
    return h


def shape_of(row):
    """The output with its mean and spread divided out -- what AdaIN does not set."""
    mean, sd = statistics.fmean(row), statistics.pstdev(row)
    return tuple(round((v - mean) / (sd or 1.0), 6) for v in row)


def spread(rows):
    """(sd of the outputs' means, sd of their sds, number of distinct shapes)."""
    return (statistics.pstdev([statistics.fmean(r) for r in rows]),
            statistics.pstdev([statistics.pstdev(r) for r in rows]),
            len({shape_of(r) for r in rows}))


def linear_gap(synth, key, ws, rows, absolute=False):
    """Worst gap between a measured statistic of the output and its closed form <key, w>."""
    forms = [sum(synth[key][j] * w[j] for j in range(W_DIM)) for w in ws]
    got = [statistics.pstdev(r) if absolute else statistics.fmean(r) for r in rows]
    return max(abs(g - (abs(f) if absolute else f)) for g, f in zip(got, forms))


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(3)
    mapping = ref.init_mapping(Z_DIM, W_DIM, DEPTH, rng)
    synth = ref.init_synth(HIDDEN, W_DIM, rng)
    const = [rng.gauss(0, 0.3) for _ in range(HIDDEN)]
    ws = [ref.mapping([rng.gauss(0, 1) for _ in range(Z_DIM)], mapping) for _ in range(PAIRS)]
    base = ws[0]
    rows_b = [mix(ref, synth, const, base, w) for w in ws]
    return {
        "vary_b": spread(rows_b),
        "vary_a": spread([mix(ref, synth, const, w, base) for w in ws]),
        "mean_gap": linear_gap(synth, "bias2", ws, rows_b),
        "sd_gap": linear_gap(synth, "scale2", ws, rows_b, absolute=True),
        "same": mix(ref, synth, const, base, base) == ref.stylegan_forward(
            base, const, synth, 0.0, random.Random(0), adain_on=True),
        "trains": [n for n in ("backward", "update", "loss", "apply_grads") if hasattr(ref, n)],
    }


def verify(result):
    vary_a, vary_b = result["vary_a"], result["vary_b"]
    return [
        practice.Check(
            "ANSWER: perfectly disentangled -- w_a cannot touch the mean or the spread",
            vary_a[0] < 1e-9 and vary_a[1] < 1e-3 and vary_b[0] > 0.01 and vary_a[2] > vary_b[2],
            f"varying w_b over {PAIRS} latents moves the output's mean by {vary_b[0]:.3f} and its "
            f"spread by {vary_b[1]:.3f} across {vary_b[2]} distinct shapes; varying w_a moves them "
            f"by {vary_a[0]:.6f} and {vary_a[1]:.6f} across {vary_a[2]}. One half owns the shape, "
            "the other owns the mean and the spread, and the split is exact",
        ),
        practice.Check(
            "FINDING: adain is the whole mechanism, and it is an identity not a tendency",
            result["mean_gap"] < 1e-12 and result["sd_gap"] < 1e-3,
            f"the last line of the last block is adain(h, scale, bias) = scale*(f-mean)/sd + bias, "
            f"so the output's mean IS bias and its sd IS |scale|, both linear in w_b. Measured: "
            f"mean(output) - <bias2, w> is {result['mean_gap']:.0e} and sd(output) - |<scale2, w>| "
            f"is {result['sd_gap']:.0e}, the second being adain's own epsilon. Nothing earlier can "
            "affect either, because the last line overwrites both",
        ),
        practice.Check(
            "FINDING: there is no decoder that learns",
            result["trains"] == [],
            f"the module exposes {result['trains']} of backward, update, loss and apply_grads: it "
            "has no loss, no gradient and no optimiser, and main() only reads. 'Does the decoder "
            "learn disentangled styles' has no referent -- the split above is a property of where "
            "adain sits, present at initialisation and unchanged by training that does not exist",
        ),
        practice.Check(
            "CONTROL: the mix really is a mix",
            result["same"],
            "driving both halves with the same w reproduces the lesson's own stylegan_forward on "
            "that w bit for bit, so the mixing path differs from the reference in routing alone "
            "and the numbers above are not measuring a second implementation",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
