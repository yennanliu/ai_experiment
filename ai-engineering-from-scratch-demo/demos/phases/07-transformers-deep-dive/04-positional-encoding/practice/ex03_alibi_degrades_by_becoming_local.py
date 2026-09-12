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
256 random query/key pairs, at the two lengths the exercise names.

**ANSWER: the two compose exactly, because they act on different terms.** RoPE
rotates `q` and `k` before the dot product; ALiBi adds `-m * |i - j|` after it.
The combined score equals the RoPE score plus the bias to **0.0**, at every gap,
so "in the same attention module" is addition and nothing else.

**FINDING: RoPE alone has no distance decay whatever.** The rotation is
orthogonal: it moves `k` without changing its norm, so over isotropic `q, k` the
score's distribution *cannot* depend on the gap. Measured, the standard deviation
is **2.137 at gap 1 and 2.137 at gap 2048**, 1e-15 apart, mean zero throughout.
The "long-term decay" RoPE is credited with is a property of trained `Wq`/`Wk`,
not of the rotation. Extrapolating RoPE to 2048 does not produce a decayed score;
it produces an **untrained** one, distributed exactly as gap 1 is.

**ANSWER: ALiBi's degradation is bounded in kind, not in size.**

| head | slope | penalty at 512 | at 2048 | at 2048, in content sigmas |
|---|---:|---:|---:|---:|
| 0 | 0.2500 | -128 | **-512** | 240 |
| 1 | 0.0625 | -32 | -128 | 60 |
| 2 | 0.0156 | -8 | -32 | 15 |
| 3 | 0.0039 | **-2** | -8 | **4** |

At 512 the flattest head's penalty is 0.9 sigma, so content still wins there. At
2048 it is four, so every head is distance-dominated. Going 512 -> 2048 turns
ALiBi from "content competes with distance" into "distance decides", which is a
failure you can name; RoPE's is a region nobody trained.

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
DIM, HEADS, TRAIN, TEST, PAIRS = 64, 4, 512, 2048, 256
BASE, PROBE = 10000.0, 512


def vectors(count=PAIRS, dim=DIM, seed=0):
    """Untrained query/key pairs, so the curve belongs to the encoding and not a model."""
    rng = random.Random(seed)
    return [([rng.gauss(0, 1) for _ in range(dim)], [rng.gauss(0, 1) for _ in range(dim)])
            for _ in range(count)]


def attend(ref, q, k, query_pos, key_pos, slope=0.0, base=BASE):
    """RoPE and ALiBi in one module: rotate before the dot, penalise distance after it."""
    rotated = ref.dot(ref.apply_rope(q, query_pos, base), ref.apply_rope(k, key_pos, base))
    return rotated / (DIM // HEADS) ** 0.5 - slope * abs(query_pos - key_pos)


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
            "ANSWER: at 2048 every head is distance-dominated; at 512 the flattest is not",
            min(sigmas[TEST]) > 3 and min(sigmas[TRAIN]) < 1.5,
            f"penalties at gap {TEST} are {[round(p) for p in result['penalty'][TEST]]} against a "
            f"content sd of {result['sigma']:.2f}: "
            f"{[round(s) for s in sigmas[TEST]]} sigmas. At {TRAIN} the flattest head is only "
            f"{min(sigmas[TRAIN]):.1f} sigmas, so content still competes. Going {TRAIN} -> {TEST} "
            "turns ALiBi from 'content competes with distance' into 'distance decides'",
        ),
        practice.Check(
            "CONTROL: the slopes span 64x, so the heads do not degrade together",
            result["slopes"][0] / result["slopes"][-1] == 64,
            f"alibi_slopes({HEADS}) = {result['slopes']}, geometric with ratio 2^(-8/n) and a "
            f"{result['slopes'][0] / result['slopes'][-1]:.0f}x span. Head 0 is purely local past "
            f"gap {result['sigma'] / result['slopes'][0]:.0f} while head {HEADS - 1} is still "
            "global past 500, so 'compare degradation' has a per-head answer, not one answer",
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
