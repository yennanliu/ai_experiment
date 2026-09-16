"""Exercise 1 — 30.4% against 5.5% is a subtraction between fractions of 46 chunks and 55 sub-chunks.

    Run `code/main.py` on `(P=8, micro_batches=16, schedule=dualpipe)` and
    `(P=8, micro_batches=16, schedule=1f1b)`. Compute the GPU utilization
    difference and express it as recovered GPU-hours per million tokens of
    training.

Reading of the exercise: the two numbers come from the lesson's own `summarize`
and the recovery from its own `gpu_hours_recovered`, and then the subtraction is
audited, because the two bubble fractions are built from different totals and
`gpu_hours_recovered` subtracts them as though they were the same quantity. The
"per million tokens" conversion is attempted and reported as unavailable, since
no function in the lesson takes a token count.

**ANSWER: 30.43% against 5.45%, a 24.98-point difference, or 249,802 GPU-hours
per million.**

    schedule       bubble     numerator / denominator
    1F1B           30.43%      14 / 46   chunks
    Zero Bubble    11.29%       7 / 62   sub-chunks
    DualPipe        5.45%       3 / 55   sub-chunks

**FINDING: the denominators are different units.** `bubble_1f1b`'s total is
`2M + 2(P-1)` forward-or-backward *chunks*; `bubble_zero_bubble`'s is
`3M + 2(P-1)` B/W *sub-chunks*, its own docstring says so; `bubble_dualpipe`'s
is `3M + (P-1)`. Subtracting 5.45% of one whole from 30.43% of another is not a
utilisation difference, and `gpu_hours_recovered` does exactly that before
multiplying by wall-clock hours.

**FINDING: "per million tokens" cannot be computed from anything the lesson
provides.** `gpu_hours_recovered(P, M, total_gpu_hours)` is the only function
that touches time, and no function anywhere takes a token count, a batch size or
a sequence length. The conversion the exercise asks for needs a tokens-per-
GPU-hour rate that has to come from outside.

**FINDING: every schedule's bubble shrinks to zero as M grows.** At P=8 the four
bubbles run 63.6/26.9/15.8/18.9% at M=4 and 0.17/0.06/0.02/0.03% at M=4096. The
lesson's takeaway that "bubbles do not grow with M for DualPipe" is true of all
four, because every numerator is a constant in `P` and every denominator is
linear in `M`.

**MECHANISM: the gap is the numerator, and the numerator is a bubble count.**
1F1B idles `2(P-1)` chunks -- a warmup and a cooldown of depth `P-1`; DualPipe
idles `(P-1)//2`. At P=8 that is 14 against 3, and the ratio of the *counts*,
4.7x, is the claim the fractions are trying to express.

Structure: `rows` reads the lesson's own `summarize`; `parts` recomputes each
formula's numerator and denominator from its own definition.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "19-dualpipe-parallelism"
STAGES, MICRO, HOURS = 8, 16, 1_000_000
LADDER = (4, 16, 64, 256, 1024, 4096)


def parts(stages, micro):
    """Each formula's own numerator and denominator, as its source writes them."""
    return {"1F1B": (2 * (stages - 1), 2 * micro + 2 * (stages - 1), "chunks"),
            "Zero Bubble": (stages - 1, 3 * micro + 2 * (stages - 1), "sub-chunks"),
            "DualPipe": ((stages - 1) // 2, 3 * micro + (stages - 1), "sub-chunks")}


def shrinking(ladder):
    """Does every schedule's bubble fall at every step of the micro-batch ladder?"""
    columns = zip(*[ladder[micro] for micro in LADDER])
    return all(all(a > b for a, b in zip(column, column[1:])) for column in columns)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {name: bubble for name, bubble, _, _ in ref.summarize(STAGES, MICRO)}
    recovered = ref.gpu_hours_recovered(STAGES, MICRO, HOURS)
    ladder = {micro: [bubble for _, bubble, _, _ in ref.summarize(STAGES, micro)]
              for micro in LADDER}
    return {
        "rows": rows,
        "parts": parts(STAGES, MICRO),
        "recovered": recovered,
        "difference": recovered["1F1B_bubble_frac"] - recovered["DualPipe_bubble_frac"],
        "ladder": ladder,
        "shrinks": shrinking(ladder),
        "token_aware": [name for name in dir(ref)
                        if "token" in name.lower() or "batch_size" in name.lower()],
        "count_ratio": parts(STAGES, MICRO)["1F1B"][0] / parts(STAGES, MICRO)["DualPipe"][0],
    }


def verify(result):
    rows, split = result["rows"], result["parts"]
    return [
        practice.Check(
            "ANSWER: 30.43% against 5.45%, 24.98 points, 249,802 GPU-hours per million",
            abs(result["difference"] - 0.2498) < 0.001,
            ", ".join(f"{name} {bubble:.2%}" for name, bubble in rows.items())
            + f". The difference between the two schedules the exercise names is "
            f"{result['difference']:.2%}, which gpu_hours_recovered turns into "
            f"{result['recovered']['recovered_gpu_hours']:,.0f} hours on a "
            f"{HOURS:,}-GPU-hour run",
        ),
        practice.Check(
            "FINDING: the denominators are different units",
            split["1F1B"][2] != split["DualPipe"][2]
            and split["1F1B"][1] != split["DualPipe"][1],
            "the formulas' own totals are "
            + ", ".join(f"{name} {num}/{den} {unit}"
                        for name, (num, den, unit) in split.items())
            + ". bubble_1f1b counts forward-or-backward chunks; bubble_zero_bubble counts B/W "
            "sub-chunks, as its docstring says; bubble_dualpipe counts sub-chunks with a "
            "different warmup term. Subtracting one whole's fraction from another's is not a "
            "utilisation difference, and gpu_hours_recovered does it before multiplying by hours",
        ),
        practice.Check(
            "FINDING: 'per million tokens' cannot be computed from anything the lesson provides",
            not result["token_aware"],
            "gpu_hours_recovered(P, M, total_gpu_hours) is the only function that touches time, "
            f"and the module exposes no name containing 'token' or 'batch_size' -- "
            f"{result['token_aware']}. Micro-batches are counted, their size is not, so the "
            "conversion the exercise asks for needs a tokens-per-GPU-hour rate from outside the "
            "lesson entirely",
        ),
        practice.Check(
            "FINDING: every schedule's bubble shrinks to zero as M grows",
            result["shrinks"],
            "at P=8 the four bubbles run "
            + " | ".join(f"M={m} " + "/".join(f"{b:.1%}" for b in result["ladder"][m])
                         for m in (LADDER[0], LADDER[-1]))
            + ". The takeaway that bubbles do not grow with M for DualPipe is true of all four, "
            f"because every numerator is a constant in P -- {split['1F1B'][0]}, "
            f"{split['Zero Bubble'][0]}, {split['DualPipe'][0]} at P={STAGES} -- and every "
            f"denominator is linear in M. The ratio of those counts, {result['count_ratio']:.1f}x, "
            "is the claim the fractions are trying to express",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
