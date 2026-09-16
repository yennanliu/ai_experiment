"""Exercise 4 — the MoE's extra copy is 41.9 GB against the dense model's 8.8, until EP divides it.

    Compute the 2x parameter overhead of DualPipe for a 70B dense model with P=8
    pipeline stages and a 671B MoE model with P=16 pipeline stages. Show why the
    MoE case's overhead is proportionally smaller (most parameters are experts,
    sharded across a large EP group).

Reading of the exercise: the overhead is what a second copy of one pipeline
*stage* costs on one GPU, so it is `params / P` in both cases, and the exercise's
explanation is then tested by adding the expert-parallel group it names -- which
appears nowhere in the lesson, whose only knobs are `P` and `M`.

**ANSWER: with pipeline parallelism alone the MoE's overhead is 4.8x the dense
model's.**

    model        params    P     one stage   DualPipe's extra copy
    70B dense      70B     8       8.75B            8.75B
    671B MoE      671B    16      41.94B           41.94B

`671 / 16` is bigger than `70 / 8`, so on the two configurations the exercise
names, the MoE case's overhead is **larger** in absolute terms and larger per
GPU. The exercise's claim needs a term it does not supply.

**FINDING: expert parallelism is what makes the claim true, and it is not in the
model.** The experts are **656B of 671B, 97.8%** of the parameters. Sharding them
across an EP group divides that 97.8% and leaves the rest:

    EP = 1    stage 41.94B    extra copy 479% of the dense stage
    EP = 8    stage  6.04B    extra copy  69%
    EP = 64   stage  1.55B    extra copy  18%

At EP=64 -- DeepSeek-V3's own configuration -- the MoE stage is a *fifth* of the
dense stage, and that is the "proportionally smaller" the exercise describes.

**MECHANISM: DualPipe replicates the stage, and EP shrinks the stage.** The two
knobs act on the same quantity from opposite sides: `P` and `EP` divide the
parameters a GPU holds, and DualPipe's second copy doubles whatever is left.
Nothing about DualPipe is cheaper on an MoE -- what is cheaper is the MoE stage.

**FINDING: neither `P` nor `EP` appears in the bubble formulas as a memory
term.** `param_copies` is recorded in `summarize`'s tuples as `1` or `2` and read
by nothing; no function in the lesson takes a parameter count, an expert count or
an EP group size. The whole of Exercise 4 has to be computed outside the module.

Structure: `stage` prices one pipeline stage under a given `(P, EP)`;
`split` separates the MoE's expert parameters from the rest.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "19-dualpipe-parallelism"
DENSE_PARAMS, DENSE_STAGES = 70e9, 8
MOE_PARAMS, MOE_STAGES = 671e9, 16
HIDDEN, MOE_FF, EXPERTS, SHARED, LAYERS, DENSE_LAYERS = 7168, 2048, 256, 1, 61, 3
EP_GROUPS = (1, 8, 64)


def split():
    """DeepSeek-V3's expert parameters against everything else."""
    expert = 3 * HIDDEN * MOE_FF
    routed = expert * (EXPERTS + SHARED) * (LAYERS - DENSE_LAYERS)
    return {"expert": expert, "experts_total": routed, "rest": MOE_PARAMS - routed,
            "share": routed / MOE_PARAMS}


def stage(params, stages, expert_params=0.0, ep=1):
    """What one pipeline stage holds: dense params over P, experts over P and EP."""
    return (params - expert_params) / stages + expert_params / (stages * ep)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    parts = split()
    dense_stage = stage(DENSE_PARAMS, DENSE_STAGES)
    moe = {ep: stage(MOE_PARAMS, MOE_STAGES, parts["experts_total"], ep) for ep in EP_GROUPS}
    rows = ref.summarize(MOE_STAGES, 128)
    return {
        "dense_stage": dense_stage,
        "moe": moe,
        "ratio": {ep: value / dense_stage for ep, value in moe.items()},
        "parts": parts,
        "copies": {name: copies for name, _, copies, _ in rows},
        "memory_aware": [name for name in dir(ref)
                         if any(word in name.lower() for word in ("param_", "expert", "memory"))],
    }


def verify(result):
    moe, ratio, parts = result["moe"], result["ratio"], result["parts"]
    return [
        practice.Check(
            "ANSWER: with pipeline parallelism alone the MoE's overhead is 4.8x the dense model's",
            moe[1] > 4 * result["dense_stage"],
            f"DualPipe's overhead is a second copy of one pipeline stage, so it is params / P: "
            f"{DENSE_PARAMS / 1e9:.0f}B over {DENSE_STAGES} is "
            f"{result['dense_stage'] / 1e9:.2f}B for the dense model and "
            f"{MOE_PARAMS / 1e9:.0f}B over {MOE_STAGES} is {moe[1] / 1e9:.2f}B for the MoE -- "
            f"{ratio[1]:.1f}x larger per GPU. On the two configurations the exercise names, its "
            "claim is the wrong way round",
        ),
        practice.Check(
            "FINDING: expert parallelism is what makes the claim true, and it is not in the model",
            parts["share"] > 0.95 and ratio[64] < 0.25,
            f"the experts are {parts['experts_total'] / 1e9:.0f}B of "
            f"{MOE_PARAMS / 1e9:.0f}B, {parts['share']:.1%} of the parameters. Sharding them "
            f"across an EP group divides that share and leaves the rest: "
            + ", ".join(f"EP={ep} stage {moe[ep] / 1e9:.2f}B ({ratio[ep]:.0%} of the dense stage)"
                        for ep in EP_GROUPS)
            + ". At EP=64, DeepSeek-V3's own configuration, the MoE stage is a fifth of the dense "
            "one, and that is the 'proportionally smaller' the exercise describes",
        ),
        practice.Check(
            "MECHANISM: DualPipe replicates the stage and EP shrinks the stage",
            moe[64] * 64 * MOE_STAGES > parts["experts_total"],
            f"P and EP divide the parameters a GPU holds and DualPipe's second copy doubles "
            f"whatever is left, so the two act on the same quantity from opposite sides. Nothing "
            f"about DualPipe is cheaper on an MoE: what is cheaper is the MoE stage, "
            f"{moe[64] / 1e9:.2f}B at EP=64 against the dense model's "
            f"{result['dense_stage'] / 1e9:.2f}B",
        ),
        practice.Check(
            "FINDING: no function in the lesson takes a parameter count",
            not result["memory_aware"] and set(result["copies"].values()) == {1, 2},
            f"param_copies is recorded in summarize's tuples as {result['copies']} and read by "
            f"nothing, and the module exposes no name mentioning a parameter total, an expert "
            f"count or memory -- {result['memory_aware']}. Every number in this exercise has to "
            "be computed outside the module the exercise is about",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
