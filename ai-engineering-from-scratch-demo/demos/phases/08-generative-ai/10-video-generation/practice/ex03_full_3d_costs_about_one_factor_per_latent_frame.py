"""Exercise 3 — full 3D attention costs about one factor per latent frame.

    **Hard.** Use HuggingFace diffusers to run CogVideoX-2B on a local GPU. Time
    20 inference steps at 720p for a 6-second clip. Profile the spatiotemporal
    attention to identify the bottleneck.

Reading of the exercise: `diffusers` and `torch` are absent and there is no GPU,
so wall-clock timing is impossible here. What runs instead (`DESIGN D11`) is the
profile itself, done by counting: CogVideoX-2B's published shape -- 30 layers,
width 1920, 8x spatial / 4x temporal VAE, 2x2 patches, 226 text tokens, 49
frames at 8 fps = 6 s -- gives the token count, and per layer attention costs
`4 N^2 d` FLOPs against `24 N d^2` for the projections and 4x MLP. The full-vs-
factorized formula is checked by actually executing both on a tiny grid with the
lesson's own `matmul` and counting the scores it computes.

**ANSWER: at 720p the attention core is the bottleneck -- 80% of the FLOPs.**
A 720p 6 s clip is 13 x 3600 + 226 = **47,026** tokens; attention's share is
`N / (N + 6d)` = **0.80**, against 0.61 at the model's native 480x720 (17,776
tokens). 20 steps with CFG are **2.5e16** FLOPs, 81 s at an A100's 312 TFLOP/s
dense peak before any inefficiency. And eager attention cannot run at all: one
head's score matrix is **4.4 GB** in bf16, 133 GB for a layer's 30 heads, which is
why the pipeline depends on fused (flash / SDPA) attention that never builds it.

**FINDING: the lesson's own "full 3D is 16-100x factorized" does not hold for the
model the exercise names.** Full over factorized is `Nt*Ns / (Nt + Ns)`, which is
~Nt whenever a frame has more patches than there are frames: **12.95x** at 720p
and 12.88x at native -- the number of latent frames, independent of resolution. A
6 s CogVideoX clip has 13, so it sits below the doc's lower bound; 16x needs 17 (65
video frames, ~8 s). (CogVideoX itself uses full 3D attention.)

**FINDING: the lesson's "spatiotemporal DiT" contains no attention at all.** Its
denoiser is three dense layers -- 1824, 2304 and 288 multiply-adds -- over the
flattened clip, and the module defines no softmax or attention function: its
attention share is 0%, so nothing in it can be profiled the way the exercise asks.

**CONTROL:** on a 4-frame x 6-patch grid the lesson's `matmul` computes 576
scores for full attention and 240 factorized, ratio 2.40 = 24/10 as the formula
says; `diffusers`, `torch` and `transformers` all return None from `find_spec`.

Structure: `tokens` and `layer_flops` are the accounting; `executed_scores` runs
both attentions on a toy grid; `solve` assembles the profile.
"""

from __future__ import annotations

import importlib.util
import random

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "10-video-generation"
LAYERS, WIDTH, HEADS, TEXT, FRAMES, STEPS = 30, 1920, 30, 226, 49, 20
A100_BF16, DOC_RANGE, NEEDED = 312e12, (16, 100), ("diffusers", "torch", "transformers")


def tokens(height, width):
    """(latent frames, patches per frame) after the 8x/4x VAE and 2x2 patchify."""
    return 1 + (FRAMES - 1) // 4, (height // 16) * (width // 16)


def layer_flops(n):
    """(attention-core FLOPs, linear FLOPs) for one transformer layer over n tokens."""
    return 4 * n * n * WIDTH, 24 * n * WIDTH * WIDTH


def profile(height, width):
    nt, ns = tokens(height, width)
    n = nt * ns + TEXT
    attn, linear = layer_flops(n)
    return {
        "tokens": n,
        "share": attn / (attn + linear),
        "full_over_factor": nt * ns / (nt + ns),
        "total": 2 * STEPS * LAYERS * (attn + linear),
        "head_gb": n * n * 2 / 1e9,
    }


def attend(ref, grid, members):
    """Score count for attention restricted to each group of token positions."""
    count = 0
    for group in members:
        keys = [grid[p] for p in group]
        count += sum(len(ref.matmul(keys, grid[p])) for p in group)
    return count


def executed_scores(ref, nt=4, ns=6, dim=3):
    """(full, factorized) score counts, by running both on an nt x ns grid."""
    rng = random.Random(0)
    grid = {(t, s): [rng.gauss(0, 1) for _ in range(dim)] for t in range(nt) for s in range(ns)}
    full = attend(ref, grid, [list(grid)])
    spatial = attend(ref, grid, [[(t, s) for s in range(ns)] for t in range(nt)])
    temporal = attend(ref, grid, [[(t, s) for t in range(nt)] for s in range(ns)])
    return full, spatial + temporal


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    net = ref.init_net(ref.T_FRAMES * (1 + ref.POS_DIM) + 8, 48, ref.T_FRAMES, random.Random(0))
    return {
        "hd": profile(720, 1280),
        "native": profile(480, 720),
        "lesson_macs": [len(net[k]) * len(net[k][0]) for k in ("W1", "W2", "W3")],
        "lesson_attn": [
            n
            for n in dir(ref)
            if "attn" in n.lower() or "softmax" in n.lower() or "attention" in n.lower()
        ],
        "executed": executed_scores(ref),
        "absent": [m for m in NEEDED if importlib.util.find_spec(m) is None],
    }


def verify(result):
    hd, native = result["hd"], result["native"]
    full, factor = result["executed"]
    return [
        practice.Check(
            "ANSWER: at 720p the attention core is the bottleneck",
            hd["share"] > 0.75 and hd["share"] > native["share"],
            f"720p x 6 s = {hd['tokens']:,} tokens; attention is {hd['share']:.0%} of a layer's "
            f"FLOPs (native 480x720: {native['tokens']:,} tokens, {native['share']:.0%}). "
            f"{STEPS} steps with CFG = {hd['total']:.1e} FLOPs, {hd['total'] / A100_BF16:.0f} s "
            f"at an A100's dense bf16 peak; one head's score matrix is {hd['head_gb']:.1f} GB, "
            f"{hd['head_gb'] * HEADS:.0f} GB per layer, so only fused attention can run it",
        ),
        practice.Check(
            "FINDING: full 3D over factorized is ~the latent frame count, below the doc's 16x",
            hd["full_over_factor"] < DOC_RANGE[0] and native["full_over_factor"] < DOC_RANGE[0],
            f"Nt*Ns/(Nt+Ns) = {hd['full_over_factor']:.2f}x at 720p and "
            f"{native['full_over_factor']:.2f}x at native -- about Nt = {tokens(720, 1280)[0]} "
            f"latent frames whatever the resolution, under the lesson's claimed {DOC_RANGE[0]}-"
            f"{DOC_RANGE[1]}x",
        ),
        practice.Check(
            "FINDING: the lesson's 'spatiotemporal DiT' contains no attention at all",
            not result["lesson_attn"],
            f"its denoiser is three dense layers of {result['lesson_macs']} multiply-adds over "
            f"the flattened clip, and the module names no attention or softmax function "
            f"({result['lesson_attn']}): attention's share of it is 0%",
        ),
        practice.Check(
            "CONTROL: the ratio formula holds when executed, and the GPU stack is absent",
            (full, factor) == (576, 240) and result["absent"] == list(NEEDED),
            f"on a 4 x 6 grid the lesson's matmul computes {full} scores full and {factor} "
            f"factorized, {full / factor:.2f} = 24/10 as Nt*Ns/(Nt+Ns) predicts; "
            f"{result['absent']} return None from find_spec",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
