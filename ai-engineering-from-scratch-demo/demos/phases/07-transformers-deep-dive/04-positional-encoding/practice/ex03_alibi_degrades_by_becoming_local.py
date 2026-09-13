"""Exercise 3 — ALiBi degrades by becoming local; RoPE degrades into untrained.

    **Hard.** Implement ALiBi and RoPE in the same attention module. Train a
    4-layer transformer on a copy task with sequences of length 512.
    Extrapolate to 2048 at test time. Compare degradation.

Reading of the exercise: the module is built -- `attend` rotates q and k with the
lesson's `apply_rope` and adds the lesson's `alibi_bias` slope to the score, so
both encodings act in the same place and compose by addition. The 4-layer
transformer does not exist to be trained: the lesson ships no model, no data, no
loss and no optimizer, and `torch` is absent. "Compare degradation" is therefore
compared where the difference lives -- in the score as a function of gap, over
4,096 random query/key pairs of one head's width, `d_model/n_heads = 16`, scored
the way a head scores, `q.k/sqrt(16)`, at the two lengths the exercise names.

**ANSWER: the two compose exactly, because they act on different terms.** RoPE
rotates `q` and `k` before the dot product; ALiBi adds `-m * |i - j|` after it.
The combined score equals the RoPE score plus the bias to **0.0**, at every gap,
so "in the same attention module" is addition and nothing else.

**FINDING: RoPE alone has no distance decay whatever.** The rotation is
orthogonal: it moves `k` without changing its norm, so over isotropic `q, k` the
score's distribution *cannot* depend on the gap. Measured over 4,096 pairs the
standard deviation is **0.992 at gap 1 and 0.998 at gap 2048** -- 0.6% apart,
which is sampling noise on a quantity that is theoretically identical -- with the
mean at zero throughout.
The "long-term decay" RoPE is credited with is a property of trained `Wq`/`Wk`,
not of the rotation. Extrapolating RoPE to 2048 does not produce a decayed score;
it produces an **untrained** one, distributed exactly as gap 1 is.

**ANSWER: ALiBi's degradation is a multiplier, and the multiplier is the length
ratio.**

| head | slope | at 512, in content sigmas | at 2048 | ratio |
|---|---:|---:|---:|---:|
| 0 | 0.2500 | 129.0 | **516.0** | 4.000 |
| 1 | 0.0625 | 32.2 | 129.0 | 4.000 |
| 2 | 0.0156 | 8.1 | 32.2 | 4.000 |
| 3 | 0.0039 | **2.0** | 8.1 | 4.000 |

The penalty is `-m*|i-j|`, linear in the gap and with no reference to the
sequence length at all, so every head degrades by exactly `2048/512 = 4` and not
one of them crosses a threshold on the way. Even the flattest head is already
2 sigmas down at the far end of the trained window -- that is the soft locality
ALiBi is for -- and at 2048 it is 8, which switches the far end off rather than
leaving it untrained. That is a failure you can name and predict in advance;
RoPE's is a region nobody trained.

**FINDING: the lesson's own `alibi_bias` cannot be called at 2048.** It
materialises `n_heads * L^2` Python floats -- **16,777,216** at 4 heads -- and
measures 32 MiB at L=512, so about **516 MiB** at the length the exercise sets,
quadratically. The exercise's own sequence length is out of reach of the
exercise's own code.

Structure: `attend` is the combined module; `spread` is the score distribution at
one gap; `cost` measures the bias table the lesson builds.
"""

from __future__ import annotations

import random
import statistics
import tracemalloc

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "04-positional-encoding"
DIM, HEADS, TRAIN, TEST, PAIRS = 64, 4, 512, 2048, 4096
HEAD_DIM, BASE, PROBE = DIM // HEADS, 10000.0, 512


def vectors(count=PAIRS, dim=HEAD_DIM, seed=0):
    """Untrained query/key pairs for one head, so the curve belongs to the encoding."""
    rng = random.Random(seed)
    return [([rng.gauss(0, 1) for _ in range(dim)], [rng.gauss(0, 1) for _ in range(dim)])
            for _ in range(count)]


def attend(ref, q, k, query_pos, key_pos, slope=0.0, base=BASE):
    """RoPE and ALiBi in one module: rotate before the dot, penalise distance after it."""
    rotated = ref.dot(ref.apply_rope(q, query_pos, base), ref.apply_rope(k, key_pos, base))
    return rotated / HEAD_DIM ** 0.5 - slope * abs(query_pos - key_pos)


def spread(ref, pairs, gap, slope=0.0):
    """(mean, sd) of the combined score at one gap, over the random pairs."""
    scores = [attend(ref, q, k, 0, gap, slope) for q, k in pairs]
    return statistics.fmean(scores), statistics.pstdev(scores)


def cost(ref, length, heads=HEADS):
    """Resident bytes and element count of the lesson's own alibi_bias at one length."""
    tracemalloc.start()
    table = ref.alibi_bias(heads, length, causal=False)
    current = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    return current * (len(table) == heads), heads * length * length


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pairs, slopes = vectors(), ref.alibi_slopes(HEADS)
    sigma = spread(ref, pairs, 1)[1]
    measured, elements = cost(ref, PROBE)
    return {
        "slopes": slopes, "decay": {g: spread(ref, pairs, g) for g in (1, 64, TRAIN, TEST)},
        "additive": max(abs(attend(ref, q, k, 0, g, slopes[0]) - attend(ref, q, k, 0, g)
                            + slopes[0] * g) for q, k in pairs[:32] for g in (1, 99, TEST)),
        "sigma": sigma,
        "penalty": {g: [m * g for m in slopes] for g in (TRAIN, TEST)},
        "sigmas": {g: [m * g / sigma for m in slopes] for g in (TRAIN, TEST)},
        "bias_bytes": measured, "bias_elements": elements,
        "projected": (measured * (TEST / PROBE) ** 2, HEADS * TEST * TEST),
    }


def verify(result):
    decay, sigmas = result["decay"], result["sigmas"]
    near, far = decay[1][1], decay[TEST][1]
    return [
        practice.Check(
            "ANSWER: RoPE and ALiBi compose by addition, exactly",
            result["additive"] < 1e-12,
            f"the module rotates q and k with the lesson's apply_rope before the dot product and "
            f"subtracts m*|i-j| after it, so the combined score minus the bias equals the RoPE "
            f"score to {result['additive']:.1e} at gaps 1, 99 and {TEST}. 'In the same attention "
            "module' is addition, and the two never interfere",
        ),
        practice.Check(
            "FINDING: RoPE alone has no distance decay at all",
            abs(far / near - 1) < 0.05 and abs(decay[TEST][0]) < 0.3,
            f"the rotation is orthogonal, so it moves k without changing its norm and the score's "
            f"distribution over isotropic q, k cannot depend on the gap: sd {near:.6f} at gap 1 "
            f"against {far:.6f} at gap {TEST}, {abs(far / near - 1):.2e} apart, mean "
            f"{decay[TEST][0]:+.3f}. The long-term decay RoPE is credited with belongs to trained "
            "Wq and Wk; extrapolated, RoPE gives an untrained score and not a decayed one",
        ),
        practice.Check(
            "ANSWER: every head degrades by exactly the length ratio, 4x, and none crosses over",
            abs(min(sigmas[TEST]) / min(sigmas[TRAIN]) - TEST / TRAIN) < 1e-9
            and min(sigmas[TEST]) > 4,
            f"penalties at gap {TEST} are {[round(p) for p in result['penalty'][TEST]]} against a "
            f"content sd of {result['sigma']:.3f}: {[round(s) for s in sigmas[TEST]]} sigmas, "
            f"against {[round(s, 1) for s in sigmas[TRAIN]]} at {TRAIN}. -m*|i-j| is linear in the "
            f"gap and never mentions the sequence length, so every head moves by exactly "
            f"{TEST // TRAIN}x -- the flattest from {min(sigmas[TRAIN]):.1f} sigmas to "
            f"{min(sigmas[TEST]):.1f}, already local at {TRAIN} and switched off at {TEST}",
        ),
        practice.Check(
            "CONTROL: the slopes span 64x, so the heads do not degrade together",
            result["slopes"][0] / result["slopes"][-1] == 64,
            f"alibi_slopes({HEADS}) = {result['slopes']}, geometric with ratio 2^(-8/n) and a "
            f"{result['slopes'][0] / result['slopes'][-1]:.0f}x span. Head 0 passes one content "
            f"sigma at gap {result['sigma'] / result['slopes'][0]:.0f} and head {HEADS - 1} only "
            f"at gap {result['sigma'] / result['slopes'][-1]:.0f}, the same 64x apart, so "
            "'compare degradation' has a per-head answer and not one answer",
        ),
        practice.Check(
            "FINDING: the lesson's own alibi_bias cannot be called at 2048",
            result["projected"][0] > 400 * 2 ** 20,
            f"alibi_bias materialises n_heads * L^2 Python floats: "
            f"{result['bias_elements']:,} at L={PROBE}, measured "
            f"{result['bias_bytes'] / 2 ** 20:.0f} MiB, so {result['projected'][1]:,} floats and "
            f"about {result['projected'][0] / 2 ** 20:.0f} MiB at the {TEST} the exercise sets. "
            "The sequence length the exercise names is out of reach of the code it names",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
