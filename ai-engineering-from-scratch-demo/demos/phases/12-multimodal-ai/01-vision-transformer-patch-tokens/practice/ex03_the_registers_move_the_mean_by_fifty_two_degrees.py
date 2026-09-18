"""Exercise 3 — the registers move the mean by fifty-two degrees.

    Implement mean pooling over patch tokens in pure Python. Verify that
    mean-pool over 196 tokens of a DINOv2 output matches what the model's
    `forward` returns when you ask for a pooled embedding.

Reading of the exercise: there is no DINOv2 to call here -- the lesson is
stdlib and ships no weights -- so "matches what forward returns" is read as the
question forward actually has to answer, which is *which slice of the sequence
it pools over*. The lesson's own `ZOO` entry fixes the geometry, and a labelled
synthetic output with the high-norm registers the lesson describes shows what
the slice choice costs.

**ANSWER: 196 is not a DINOv2 count.** `grid_shape(224, 14)` is 16x16 = **256**
patches, and `seq_length` is **261** with the CLS token and 4 registers. 196 is
ViT-B/16's grid -- `grid_shape(224, 16)` -- so the exercise names one model's
token count and another model's patch size.

**MEASUREMENT: the pure-Python mean is exact to 1e-15.** Over 256 tokens of
1,536 dimensions, a naive per-dimension `sum(...) / n` agrees with an
`math.fsum` reference within 1e-15 on every dimension, and pooling 196 copies
of one vector returns that vector unchanged.

**FINDING: pooling the wrong 1.5% of the sequence moves the answer by 52°.**
The 4 registers are 1.5% of the 261-token sequence, but at the ~10x norm
Darcet et al. report for the artifacts they replace, including them drops the
cosine against the patch-only mean to **0.619** -- 51.8° -- and inflates the
norm by 58%. "Discarded before handoff" is not bookkeeping; it is the
difference between two different embeddings.

**FINDING: pooling is linear, which is why forward can hand it back for free.**
Pool-then-project and project-then-pool agree to 4.4e-16 across a 1,536 -> 8
projection, so a pooled embedding costs one extra mean over the tokens the
model already computed, not a second pass.

Structure: `make_fixture` builds the labelled synthetic DINOv2 output (256
Gaussian patch tokens plus 4 registers scaled to 10x the median patch norm),
`mean_pool` is the exercise's deliverable, `exact_pool` is the fsum reference
it is checked against, and `cosine` compares the two candidate slices.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "01-vision-transformer-patch-tokens"
SEED, NAMED, ARTIFACT_NORM = 20260918, 196, 10.0
PROJECT_TO = 8


def mean_pool(tokens):
    """The exercise's deliverable: mean over a list of equal-length vectors."""
    count = len(tokens)
    return [sum(token[i] for token in tokens) / count for i in range(len(tokens[0]))]


def exact_pool(tokens):
    """The same mean under exact summation, as the reference to check against."""
    count = len(tokens)
    return [math.fsum(token[i] for token in tokens) / count for i in range(len(tokens[0]))]


def norm(vector):
    return math.sqrt(math.fsum(x * x for x in vector))


def cosine(left, right):
    return math.fsum(x * y for x, y in zip(left, right)) / (norm(left) * norm(right))


def make_fixture(patches, registers, hidden):
    """Labelled synthetic encoder output: unit-scale patches, high-norm registers.

    The register tokens are scaled to ARTIFACT_NORM times the median patch norm,
    which is the artifact magnitude the lesson's register section describes.
    """
    rng = random.Random(SEED)

    def draw():
        return [rng.gauss(0, 1) for _ in range(hidden)]

    patch_tokens = [draw() for _ in range(patches)]
    target = ARTIFACT_NORM * statistics.median(norm(token) for token in patch_tokens)
    register_tokens = [[x * target / norm(token) for x in token]
                       for token in (draw() for _ in range(registers))]
    matrix = [draw() for _ in range(PROJECT_TO)]
    return patch_tokens, register_tokens, matrix


def project(matrix, vector):
    return [math.fsum(row[i] * vector[i] for i in range(len(vector))) for row in matrix]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cfg = next(c for c in ref.ZOO if c.name.startswith("DINOv2"))
    grid = ref.grid_shape(cfg.image_size, cfg.patch_size)
    patches, registers, matrix = make_fixture(grid[0] * grid[1], cfg.registers, cfg.hidden)
    patch_mean = mean_pool(patches)
    with_registers = mean_pool(patches + registers)
    return {
        "grid": list(grid), "patch_tokens": grid[0] * grid[1],
        "seq": ref.seq_length(cfg), "registers": cfg.registers,
        "named_grid": list(ref.grid_shape(cfg.image_size, ref.ZOO[0].patch_size)),
        "named": NAMED,
        "deviation": max(abs(a - b) for a, b in zip(patch_mean, exact_pool(patches))),
        "identity": mean_pool([[0.25] * cfg.hidden] * NAMED) == [0.25] * cfg.hidden,
        "cosine": round(cosine(patch_mean, with_registers), 4),
        "degrees": round(math.degrees(math.acos(cosine(patch_mean, with_registers))), 1),
        "norm_ratio": round(norm(with_registers) / norm(patch_mean), 2),
        "register_share": round(cfg.registers / ref.seq_length(cfg) * 100, 1),
        "linearity": max(abs(a - b) for a, b in zip(
            project(matrix, patch_mean), mean_pool([project(matrix, p) for p in patches]))),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 196 is not a DINOv2 count -- the grid is 16x16 = 256, the sequence 261",
            all([result["grid"] == [16, 16], result["patch_tokens"] == 256,
                 result["seq"] == 261, result["named_grid"] == [14, 14]]),
            f"grid_shape(224, 14) is {result['grid'][0]}x{result['grid'][1]} = "
            f"{result['patch_tokens']} patches and seq_length is {result['seq']} with CLS and "
            f"{result['registers']} registers. {result['named']} is grid_shape(224, 16) -- "
            f"{result['named_grid'][0]}x{result['named_grid'][1]}, ViT-B/16 -- so the exercise "
            "names one model's token count and another model's patch size",
        ),
        practice.Check(
            "MEASUREMENT: the pure-Python mean is exact to 1e-15",
            result["deviation"] <= 1e-15 and result["identity"],
            f"over {result['patch_tokens']} tokens of 1,536 dimensions the naive "
            f"sum(...)/n agrees with an fsum reference to {result['deviation']:.2e} on every "
            f"dimension, and pooling {result['named']} copies of one vector returns it "
            "unchanged",
        ),
        practice.Check(
            "FINDING: pooling the wrong 1.5% of the sequence moves the answer by 52 degrees",
            all([result["cosine"] == 0.6191, result["degrees"] == 51.8,
                 result["norm_ratio"] == 1.58, result["register_share"] == 1.5]),
            f"the {result['registers']} registers are {result['register_share']}% of the "
            f"{result['seq']}-token sequence, but at {ARTIFACT_NORM:g}x the median patch norm "
            f"including them drops the cosine to {result['cosine']} -- {result['degrees']}° -- "
            f"and inflates the pooled norm by "
            f"{(result['norm_ratio'] - 1) * 100:.0f}%. Discarding them is not bookkeeping",
        ),
        practice.Check(
            "FINDING: pooling is linear, which is why forward can hand it back for free",
            result["linearity"] <= 1e-12,
            f"pool-then-project and project-then-pool agree to {result['linearity']:.1e} "
            f"across a 1,536 -> {PROJECT_TO} projection, so a pooled embedding is one extra "
            "mean over tokens the model already has, not a second pass",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
