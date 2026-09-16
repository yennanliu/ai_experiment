"""Exercise 5 — 560 GB of optimizer state into 512 GB of RAM, on 5 GPUs rather than 4.

    Add ZeRO-Offload to the memory calculator. Assume CPU RAM is 512GB
    per node and NVMe is 2TB. Show how offloading optimizer states to CPU allows
    a 70B model to train on 4 GPUs instead of 16, at the cost of 30-50% slower
    optimizer steps.

Reading of the exercise: offloading is applied to the term the exercise names --
the Adam states -- leaving weights, gradients and activations on the GPU, and
the GPU count is *derived* from the lesson's own calculator rather than taken
from the exercise, at the same 70B shape used throughout (`hidden_dim=8192`,
`num_layers=80`, `sequence_length=2048`, batch 1) so the activation term is the
real one. The slowdown is priced against a stated PCIe 4.0 x16 budget of 25 GB/s
effective, because "30-50% slower" is a claim with a bandwidth behind it.

**FINDING: the optimizer state does not fit the node the exercise specifies.**
Adam at fp32 is 8 bytes per parameter, so 70B is **560 GB** -- 48 GB more than
the 512 GB of CPU RAM the exercise assumes. The premise fails its own
constraint. The 2 TB of NVMe fits it, which is ZeRO-Infinity rather than
ZeRO-Offload and an order of magnitude slower again.

**ANSWER: 5 GPUs, not 4 -- and the baseline is 13, not 16.** ZeRO-3 across N
cards needs `840/N + 10.7 <= 80`, which first holds at **13** GPUs (75.4 GB
each). Offload the 560 GB of Adam state and the requirement becomes
`280/N + 10.7 <= 80`, first satisfied at **5** (66.7 GB each). Both numbers in
the exercise are round numbers next to the ones the calculator gives.

**FINDING: 30-50% is not a slowdown, it is an unstated overlap assumption.** At
5 GPUs, offload moves gradients down and updated parameters back every step:
`2 x 140 GB / 5` = **56 GB per GPU**, which is **2.24 s** at 25 GB/s. The step's
own compute, by 6ND at 40% MFU on 5 H100s for 5 x 2048 tokens, is **2.17 s**.
Serialised, the step is **2.03x** -- an upper bound, not a prediction. Hidden
perfectly behind compute it is **1.03x**, because the transfer is the longer of
the two. The exercise's 1.30-1.50x lies inside that window and pins the overlap
to **52-71%**, which is the quantity the exercise never names. On top of that,
30-50% is quoted for the optimizer *step*, a small part of the step a user waits
on.

**FINDING: what is bought is fewer cards, at 1.03x to 2.03x the card-seconds per
token.** Each card processes the same 2048 tokens a step either way, so 13 cards
clear 26,624 tokens in one step time and 5 clear 10,240 in 1.03 to 2.03 of one
-- 1,009 to 1,986 tokens per card-second against 2,048, depending entirely on
overlap. Offload buys the ability to run at all on the hardware you have; the
exercise's "allows a 70B model to train on 4 GPUs instead of 16" reads as though
the only change were the card count.

Structure: `smallest_fit` searches the reference calculator for the first GPU
count that fits 80 GB, with or without the optimizer term.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "05-scaling-distributed"
PARAMS, CARD_GB, CPU_GB, NVME_GB = 70, 80, 512, 2000
SHAPE = dict(hidden_dim=8192, num_layers=80, sequence_length=2048)
PCIE_GBS, TFLOPS, MFU = 25.0, 990e12, 0.40
CLAIMED = (1.30, 1.50)


def per_gpu(ref, gpus, offload):
    """Per-GPU memory under ZeRO-3, with the Adam states left on the CPU or not."""
    report = ref.memory_calculator(PARAMS, num_gpus=gpus, sharding="zero3", **SHAPE)
    return report["per_gpu_total_gb"] - (report["optimizer_gb"] if offload else 0)


def smallest_fit(ref, offload):
    """The first GPU count whose per-card footprint fits an 80 GB card."""
    return next(n for n in range(1, 512) if per_gpu(ref, n, offload) <= CARD_GB)


def step_seconds(gpus, tokens_per_gpu=2048):
    """One step's compute by 6ND, at the lesson's own 40% utilisation."""
    return 6 * PARAMS * 1e9 * gpus * tokens_per_gpu / (TFLOPS * MFU * gpus)


def overlap_for(target, compute, moved):
    """The fraction of the transfer that has to hide behind compute to reach `target`."""
    return 1 - (target - 1) * compute / moved


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sharded, offloaded = smallest_fit(ref, False), smallest_fit(ref, True)
    report = ref.memory_calculator(PARAMS, num_gpus=1, sharding="none", **SHAPE)
    transfer = 2 * report["gradients_gb"] / offloaded
    compute = step_seconds(offloaded)
    return {
        "gpus": (sharded, offloaded),
        "sharded_gb": report["weights_gb"] + report["optimizer_gb"] + report["gradients_gb"],
        "footprint": (per_gpu(ref, sharded, False), per_gpu(ref, offloaded, True)),
        "optimizer_gb": report["optimizer_gb"],
        "activations_gb": report["activations_gb"],
        "fits_cpu": report["optimizer_gb"] <= CPU_GB,
        "fits_nvme": report["optimizer_gb"] <= NVME_GB,
        "transfer_gb": transfer,
        "seconds": (compute, transfer / PCIE_GBS),
        "slowdown": (compute + transfer / PCIE_GBS) / compute,
        "best": max(compute, transfer / PCIE_GBS) / compute,
        "needed": [overlap_for(t, compute, transfer / PCIE_GBS) for t in CLAIMED],
    }


def verify(result):
    sharded, offloaded = result["gpus"]
    compute, moved = result["seconds"]
    optimizer, needed = result["optimizer_gb"], result["needed"]
    return [
        practice.Check(
            f"FINDING: {result['optimizer_gb']:.0f} GB of Adam state into {CPU_GB} GB of RAM",
            not result["fits_cpu"] and result["fits_nvme"],
            f"Adam keeps two fp32 moments, 8 bytes a parameter, so {PARAMS}B is "
            f"{optimizer:.0f} GB -- {optimizer - CPU_GB:.0f} GB more than the {CPU_GB} GB of CPU "
            f"RAM the exercise assumes per node. The premise fails its own constraint. The "
            f"{NVME_GB} GB of NVMe does fit it, but paging optimizer state to NVMe is "
            "ZeRO-Infinity rather than ZeRO-Offload, and an order of magnitude slower again",
        ),
        practice.Check(
            "ANSWER: 5 GPUs, not 4 -- and the baseline is 13, not 16",
            (sharded, offloaded) == (13, 5),
            f"ZeRO-3 across N cards needs {result['sharded_gb']:.0f}/N + "
            f"{result['activations_gb']:.1f} <= {CARD_GB}, first true at {sharded} GPUs "
            f"({result['footprint'][0]:.1f} GB each). Leave the {optimizer:.0f} GB of Adam state "
            f"on the CPU and it becomes {offloaded} GPUs ({result['footprint'][1]:.1f} GB each). "
            "Both numbers the exercise gives are round numbers one step away from the ones its "
            "own calculator produces",
        ),
        practice.Check(
            "FINDING: 30-50% is not a slowdown, it is an overlap assumption of 52-71%",
            result["slowdown"] > 2.0 > result["best"] and 0.4 < needed[1] < needed[0] < 0.8,
            f"at {offloaded} GPUs, offload moves gradients down and updated parameters back every "
            f"step: {result['transfer_gb']:.0f} GB per GPU, {moved:.2f} s at {PCIE_GBS:.0f} GB/s "
            f"over PCIe 4.0 x16. The step's own compute, by 6ND at {100 * MFU:.0f}% MFU on "
            f"{offloaded} H100s for {offloaded} x 2048 tokens, is {compute:.2f} s. Serialised, "
            f"the step is {result['slowdown']:.2f}x -- an upper bound, not a prediction; hidden "
            f"perfectly behind compute it is {result['best']:.2f}x. The exercise's "
            f"{CLAIMED[0]:.2f}-{CLAIMED[1]:.2f}x sits inside that window and pins the overlap to "
            f"{100 * needed[1]:.0f}-{100 * needed[0]:.0f}%, which is the number it never states "
            "-- and it quotes the figure for the optimizer step, not the step a user waits on",
        ),
        practice.Check(
            "FINDING: what is bought is fewer cards, at 1.03x to 2.03x the card-seconds per token",
            result["best"] > 1.0 and result["slowdown"] > 1.9,
            f"each card processes 2048 tokens a step either way, so {sharded} cards clear "
            f"{sharded * 2048:,} tokens in one step time and {offloaded} clear "
            f"{offloaded * 2048:,} in {result['best']:.2f} to {result['slowdown']:.2f} of one -- "
            f"{2048 / result['slowdown']:,.0f} to {2048 / result['best']:,.0f} tokens per "
            f"card-second against 2,048, so the cost per token is between {result['best']:.2f}x "
            f"and {result['slowdown']:.2f}x depending entirely on overlap. Offload buys the "
            "ability to run at all on the hardware you have; the exercise's 'allows a 70B model "
            "to train on 4 GPUs instead of 16' reads as though the only change were the card count",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
