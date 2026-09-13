"""Exercise 2 — a window of 1,024 on a 64-token block is full attention, exactly.

    **Medium.** Implement causal SWA with `window=1024` on top of the Lesson 07
    capstone. Train for 1,000 steps on tinyshakespeare. How much does val loss
    regress vs full attention? How much does peak memory drop?

Reading of the exercise: before training anything, ask what the mask does. The
Lesson 07 capstone sets `block_size = 64`. `swa_mask(64, 1024)` is compared with
`causal_mask(64)` cell by cell.

**ANSWER: it regresses by nothing and saves nothing, and no training is needed to
know it.** `swa_mask(64, 1024) == causal_mask(64)`, identically -- every row's
lower bound is `max(0, i - 1023)`, which is 0 for every `i < 64`. The two masks
are the same object's worth of `0.0` and `-inf`, so the two models are the same
function, the val losses are equal for every seed and every step, and the
attention matrix is 64x64 either way.

| window | equals causal at n=64 | cells attended | rows changed |
|---:|---|---:|---:|
| 1024 | **yes** | 2,080 | **0** |
| 64 | **yes** | 2,080 | **0** |
| 32 | no | 1,552 | 32 |
| 8 | no | 484 | 56 |

**FINDING: the window has to be under 64 before anything happens at all**, and
the first `window` rows stay causal even then. At `window=32`, half the rows are
untouched and the saving is 25% of the cells; at `window=8` it is 77%.

**FINDING: the exercise's own numbers are 16x apart.** A 1,024-token window on a
64-token context is not a small mismatch -- the window is sixteen times the
sequence. The memory question has the same answer for the same reason: the KV
cache is `min(window, n)` positions deep, and `min(1024, 64)` is 64.

**CONTROL: the saving is a mask property, so it is exact.** No query, key or
value enters any of these numbers; `count_nonmasked` is arithmetic on the shape.
That is why "how much does val loss regress" can be answered without a training
run, and why the answer is 0.000 rather than 'about zero'.

Structure: `profile` counts a mask's cells and the rows it moves relative to
causal at one window size.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "15-attention-variants"
BLOCK, ASKED, WINDOWS = 64, 1_024, (1_024, 64, 32, 8)


def profile(ref, window, n=BLOCK):
    """(equals causal, cells attended, rows that differ from causal) at this window."""
    causal, windowed = ref.causal_mask(n), ref.swa_mask(n, window)
    return (windowed == causal, ref.count_nonmasked(windowed),
            sum(1 for a, b in zip(causal, windowed) if a != b))


def depth(window, n=BLOCK):
    """KV-cache depth in positions: a window cannot remember more than there is."""
    return min(window, n)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shapes = {w: profile(ref, w) for w in WINDOWS}
    return {
        "shapes": shapes, "block": BLOCK, "asked": ASKED,
        "cache": {w: depth(w) for w in WINDOWS},
        "ratio": ASKED / BLOCK, "full": ref.count_nonmasked(ref.causal_mask(BLOCK)),
    }


def verify(result):
    shapes, cache = result["shapes"], result["cache"]
    return [
        practice.Check(
            "ANSWER: swa_mask(64, 1024) is causal_mask(64), cell for cell",
            shapes[ASKED][0] and shapes[ASKED][2] == 0,
            f"every row's lower bound is max(0, i - {ASKED - 1}), which is 0 for every i < "
            f"{BLOCK}, so the two masks are the same arrangement of 0.0 and -inf and "
            f"{shapes[ASKED][2]} rows differ. The two models are the same function: the val "
            "losses are equal at every seed and every step, and no training run is needed",
        ),
        practice.Check(
            "ANSWER: peak memory drops by nothing either",
            cache[ASKED] == BLOCK,
            f"the KV cache is min(window, n) positions deep, and min({ASKED}, {BLOCK}) is "
            f"{cache[ASKED]}. The attention matrix stays {BLOCK}x{BLOCK} and the cache stays "
            f"{BLOCK} deep -- {result['full']:,} attended cells in both arms",
        ),
        practice.Check(
            "FINDING: the window has to fall under 64 before anything happens",
            not shapes[32][0] and shapes[32][2] == 32 and shapes[8][2] == BLOCK - 8,
            "equals-causal / cells / rows moved by window: " + ", ".join(
                f"{w} {'yes' if eq else 'no'}/{cells:,}/{moved}"
                for w, (eq, cells, moved) in shapes.items())
            + f". Even at window=32 half the rows are untouched, because the first `window` rows "
              "are causal whatever the window is",
        ),
        practice.Check(
            "FINDING: the exercise's own numbers are 16x apart",
            result["ratio"] == 16,
            f"a {ASKED}-token window on a {BLOCK}-token context is not a small mismatch: the "
            f"window is {result['ratio']:.0f} times the sequence. Both halves of the question -- "
            "the loss regression and the memory drop -- have the same answer for the same reason",
        ),
        practice.Check(
            "CONTROL: the saving is a mask property, so the answer is 0.000 and not 'about zero'",
            shapes[ASKED][1] == shapes[BLOCK][1] == result["full"],
            f"no query, key or value enters any of these numbers -- count_nonmasked is arithmetic "
            f"on the shape, giving {result['full']:,} cells at window {ASKED}, window {BLOCK} and "
            "full causal alike. That is what lets the training question be answered without "
            "training",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
