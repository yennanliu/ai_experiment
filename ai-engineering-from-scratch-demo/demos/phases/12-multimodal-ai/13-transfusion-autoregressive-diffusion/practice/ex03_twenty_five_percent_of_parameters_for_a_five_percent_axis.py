"""Exercise 3 — 25% of parameters for a 5% axis.

    MMDiT has modality-specific QKV weights. What parameter count overhead does
    this add vs Transfusion's fully-shared transformer? At 7B params, is it
    worth it?

Reading of the exercise: the overhead is computed from the standard block
budget -- 4d^2 of attention projections and 8d^2 of MLP per layer -- and "is it
worth it" is priced against two numbers this phase has already measured rather
than argued: Lesson 12.07 rates connector and block architecture at **5%** of
benchmark variance, the smallest of its six axes, and its own within-family
rows put a decade of LLM at about **+7 MMMU**.

**ANSWER: +25% of block parameters, so 7B becomes 8.75B.** Duplicating Q, K and
V costs 3d^2 per layer against a 12d^2 block, and the attention *operation* is
still shared -- the two streams are concatenated before the softmax, which is
what makes MMDiT one model rather than two.

**FINDING: duplicating the whole block instead would be +100%.** SD3's MMDiT
duplicates the MLP and the output projection as well, which is 12d^2 against
12d^2. The exercise names the cheaper half of the design, and the two readings
differ by a factor of **4** in overhead.

**ANSWER: by this phase's own numbers it is not obviously worth it.** A 1.25x
parameter increase is 0.097 decades, which at Lesson 12.07's measured **+7 MMMU
per decade** is about **+0.68 MMMU** -- a stated extrapolation, and one that
assumes the extra parameters behave like scale. The same lesson rates the whole
architecture axis at 5% of variance against **60%** for visual-token count.
Spending 25% of the parameter budget on the smallest axis is a defensible choice
only if the token count is already right.

**FINDING: and the overhead is paid on every text token too.** The duplicated
weights are per-modality, so a text-only request carries the image stream's
parameters in memory and skips them in compute. At 8.75B in bf16 that is
**3.3 GiB** of weights resident for requests that never touch them.

Structure: `block_params` is the standard budget, `overhead` prices one
duplication policy, and `scaling_gain` converts a parameter ratio into the
benchmark points Lesson 12.07 measured.
"""

from __future__ import annotations

import math

from harness import practice

BASE_PARAMS = 7e9
ATTENTION_D2, MLP_D2 = 4, 8            # 4d^2 of QKVO, 8d^2 of MLP, per block
QKV_D2 = 3                             # what MMDiT duplicates, per the exercise
FULL_BLOCK_D2 = ATTENTION_D2 + MLP_D2
MMMU_PER_DECADE = 7.0                  # Lesson 12.07, within-family
ARCHITECTURE_AXIS, TOKEN_AXIS = 5, 60  # Lesson 12.07's variance weights
DTYPE_BYTES = 2


def overhead(duplicated_d2, block_d2=FULL_BLOCK_D2):
    return round(duplicated_d2 / block_d2 * 100, 1)


def scaled(params, duplicated_d2, block_d2=FULL_BLOCK_D2):
    return params * (1 + duplicated_d2 / block_d2)


def scaling_gain(ratio, per_decade=MMMU_PER_DECADE):
    return round(math.log10(ratio) * per_decade, 2)


def solve():
    qkv_only = overhead(QKV_D2)
    whole_block = overhead(FULL_BLOCK_D2)
    grown = scaled(BASE_PARAMS, QKV_D2)
    return {
        "qkv_overhead_pct": qkv_only, "whole_block_pct": whole_block,
        "reading_ratio": round(whole_block / qkv_only, 1),
        "grown_params": grown, "ratio": round(grown / BASE_PARAMS, 3),
        "decades": round(math.log10(grown / BASE_PARAMS), 3),
        "expected_mmmu": scaling_gain(grown / BASE_PARAMS),
        "architecture_axis": ARCHITECTURE_AXIS, "token_axis": TOKEN_AXIS,
        "axis_ratio": TOKEN_AXIS // ARCHITECTURE_AXIS,
        "resident_gib": round(grown * DTYPE_BYTES / 2 ** 30, 1),
        "idle_gib": round((grown - BASE_PARAMS) * DTYPE_BYTES / 2 ** 30, 1),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: +25% of block parameters, so 7B becomes 8.75B",
            all([result["qkv_overhead_pct"] == 25.0, result["grown_params"] == 8.75e9,
                 result["ratio"] == 1.25]),
            f"duplicating Q, K and V costs {QKV_D2}d^2 per layer against a "
            f"{FULL_BLOCK_D2}d^2 block -- {result['qkv_overhead_pct']}% -- so "
            f"{BASE_PARAMS / 1e9:.0f}B becomes {result['grown_params'] / 1e9:.2f}B. The "
            "attention operation itself is still shared: the two streams concatenate before "
            "the softmax, which is what makes MMDiT one model rather than two",
        ),
        practice.Check(
            "FINDING: duplicating the whole block instead would be +100%",
            all([result["whole_block_pct"] == 100.0, result["reading_ratio"] == 4.0]),
            f"SD3's MMDiT duplicates the MLP and output projection as well, which is "
            f"{FULL_BLOCK_D2}d^2 against {FULL_BLOCK_D2}d^2 -- "
            f"{result['whole_block_pct']}%. The exercise names the cheaper half of the "
            f"design, and the two readings differ by {result['reading_ratio']}x in overhead",
        ),
        practice.Check(
            "ANSWER: by this phase's own numbers it is not obviously worth it",
            all([result["decades"] == 0.097, result["expected_mmmu"] == 0.68,
                 result["axis_ratio"] == 12]),
            f"a {result['ratio']}x parameter increase is {result['decades']} decades, which "
            f"at Lesson 12.07's measured +{MMMU_PER_DECADE:.0f} MMMU per decade is about "
            f"+{result['expected_mmmu']} -- a stated extrapolation that assumes the extra "
            f"parameters behave like scale. The same lesson rates architecture at "
            f"{ARCHITECTURE_AXIS}% of variance against {TOKEN_AXIS}% for visual-token count, "
            f"{result['axis_ratio']}x more. Spending a quarter of the budget on the smallest "
            "axis is defensible only once the token count is right",
        ),
        practice.Check(
            "FINDING: and the overhead is paid on every text token too",
            all([result["resident_gib"] == 16.3, result["idle_gib"] == 3.3]),
            f"the duplicated weights are per-modality, so a text-only request carries the "
            f"image stream's parameters in memory and skips them in compute. At "
            f"{result['grown_params'] / 1e9:.2f}B in bf16 that is "
            f"{result['resident_gib']} GiB resident, of which {result['idle_gib']} GiB is "
            "for a modality the request never touches",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
