"""Exercise 4 — the wire cost is identical to the byte.

    Compute the communication cost of softmax InfoNCE vs sigmoid pairwise for a
    512-GPU run at batch 32k. Which scales as O(N), which as O(N^2)? Cite
    SigLIP Section 4.

Reading of the exercise: the two candidate answers are computed rather than
recalled, and the scaling exponents are *measured* -- by evaluating each cost at
two batch sizes with the local batch held fixed, which is the regime a real run
scales in. The stated constants are SigLIP SO400m's width (1,152) and bf16, and
the pair and normaliser counts are taken from the lesson's own loss functions by
instrumenting them rather than by reading them.

**ANSWER: the communication is the same number for both -- 75,350,016 bytes,
71.86 MiB, per device per tower per step.** InfoNCE all-gathers every embedding,
(P-1)/P * N * D * 2 bytes; SigLIP §4's chunked form passes its text chunk around
the ring, P-1 steps of (N/P) * D * 2. Those are the same expression.

**ANSWER: what differs is memory, by exactly P.** A device holding InfoNCE's
64 rows needs 64 x 32,768 logits -- **8 MiB** in fp32 -- because the softmax
normaliser spans a whole row. The sigmoid chunk is 64 x 64: **16 KiB**. The
ratio is **512**, which is the device count, not a property of either loss.

**FINDING: neither cost is O(N) against O(N^2) at fixed hardware.** Measured
exponents with the local batch held at 64: wire **1.00**, InfoNCE logit memory
**1.00**, sigmoid chunk memory **0.00**, pair count **2.00**. The only O(N^2)
quantity is the pair count -- and both losses have it.

**FINDING: the lesson's own functions locate the coupling.** Instrumented,
`sigmoid_loss` calls `sigmoid` exactly N^2 times and `infonce_loss` calls
`log_sum_exp` exactly 2N times, one per row and one per column. The N^2
arithmetic is shared; the 2N normalisers are what has to be global, and half of
them read columns that live on other devices.

Structure: `costs` is the four-quantity model at one (N, P), `exponent` fits
each quantity's scaling between two batch sizes at fixed local batch, and
`counted` instruments the lesson's loss functions to count their primitives.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "02-clip-contrastive-pretraining"
BATCH, DEVICES, WIDTH = 32768, 512, 1152
WIRE_BYTES, LOGIT_BYTES = 2, 4          # bf16 embeddings, fp32 logits
SMALL, LARGE = 8192, 131072
SIZES = (4, 8, 16)


def costs(batch, devices, width=WIDTH):
    """Wire bytes per device, the two per-device logit footprints, and the pair count."""
    local = batch // devices
    return {"wire": (devices - 1) * local * width * WIRE_BYTES,
            "infonce_logits": local * batch * LOGIT_BYTES,
            "sigmoid_chunk": local * local * LOGIT_BYTES,
            "pairs": batch * batch}


def exponent(small, large, key, local=64):
    """The scaling exponent of one quantity, measured at fixed local batch."""
    low = costs(small, small // local)[key]
    high = costs(large, large // local)[key]
    return round(math.log(high / low) / math.log(large / small), 2)


def counted(ref, size):
    """The lesson's own primitive counts for one batch, by instrumenting its module."""
    tally = {"sigmoid": 0, "log_sum_exp": 0}
    originals = {name: getattr(ref, name) for name in tally}

    def wrap(name):
        def counting(value):
            tally[name] += 1
            return originals[name](value)
        return counting

    try:
        for name in tally:
            setattr(ref, name, wrap(name))
        matrix = [[0.1] * size for _ in range(size)]
        ref.sigmoid_loss(matrix)
        ref.infonce_loss(matrix)
    finally:
        for name, original in originals.items():
            setattr(ref, name, original)
    return tally


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    model = costs(BATCH, DEVICES)
    allgather = (DEVICES - 1) / DEVICES * BATCH * WIDTH * WIRE_BYTES
    tallies = {size: counted(ref, size) for size in SIZES}
    return {
        "local": BATCH // DEVICES, **model,
        "wire_mib": round(model["wire"] / 2**20, 2),
        "allgather_matches": allgather == model["wire"],
        "infonce_mib": model["infonce_logits"] // 2**20,
        "sigmoid_kib": model["sigmoid_chunk"] // 1024,
        "memory_ratio": model["infonce_logits"] // model["sigmoid_chunk"],
        "exponents": {key: exponent(SMALL, LARGE, key) for key in model},
        "tallies": tallies,
        "pairs_match": all(t["sigmoid"] == n * n for n, t in tallies.items()),
        "normalisers_match": all(t["log_sum_exp"] == 2 * n for n, t in tallies.items()),
    }


def verify(result):
    exponents, tallies = result["exponents"], result["tallies"]
    return [
        practice.Check(
            "ANSWER: the communication is the same number for both -- 75,350,016 bytes",
            all([result["wire"] == 75_350_016, result["wire_mib"] == 71.86,
                 result["allgather_matches"], result["local"] == 64]),
            f"InfoNCE all-gathers every embedding, (P-1)/P * N * D * 2 bytes; SigLIP's "
            f"chunked form passes its {result['local']}-row text chunk around the ring for "
            f"{DEVICES - 1} steps. Both are {result['wire']:,} bytes "
            f"({result['wire_mib']} MiB) per device per tower per step -- the same expression",
        ),
        practice.Check(
            "ANSWER: what differs is memory, by exactly P",
            all([result["infonce_mib"] == 8, result["sigmoid_kib"] == 16,
                 result["memory_ratio"] == DEVICES]),
            f"the softmax normaliser spans a whole row, so a device holding "
            f"{result['local']} rows needs {result['local']} x {BATCH:,} logits = "
            f"{result['infonce_mib']} MiB; the sigmoid chunk is {result['local']} x "
            f"{result['local']} = {result['sigmoid_kib']} KiB. The ratio is "
            f"{result['memory_ratio']}, which is the device count, not a property of "
            "either loss",
        ),
        practice.Check(
            "FINDING: neither cost is O(N) against O(N^2) at fixed hardware",
            all([exponents["wire"] == 1.0, exponents["infonce_logits"] == 1.0,
                 exponents["sigmoid_chunk"] == 0.0, exponents["pairs"] == 2.0]),
            f"measured between N={SMALL:,} and N={LARGE:,} at a fixed local batch of 64: "
            f"{exponents}. The only O(N^2) quantity is the pair count, and both losses "
            "have it; the sigmoid's advantage is an exponent of 0, not of 1",
        ),
        practice.Check(
            "FINDING: the lesson's own functions locate the coupling",
            all([result["pairs_match"], result["normalisers_match"],
                 result["pairs"] == 1_073_741_824]),
            f"instrumented, sigmoid_loss calls sigmoid exactly N^2 times and infonce_loss "
            f"calls log_sum_exp exactly 2N times {tallies}. At N={BATCH:,} that is "
            f"{result['pairs']:,} pairs for both and {2 * BATCH:,} normalisers -- half of "
            "them over columns that live on other devices",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
