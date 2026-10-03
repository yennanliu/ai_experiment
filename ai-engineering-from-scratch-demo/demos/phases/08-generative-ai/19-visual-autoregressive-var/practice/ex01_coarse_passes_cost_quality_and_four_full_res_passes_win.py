"""Exercise 1 — more passes help only if they are full-resolution passes.

    **Scale count ablation.** Train VAR with 4, 6, 8, 10 scales. Measure
    reconstruction quality vs number of autoregressive passes. More scales =
    finer residuals = better quality but more passes.

Reading of the exercise: "train VAR with N scales" is read as fitting the
lesson's own residual-VQ tokenizer (`train_codebooks`) with an N-entry `SCALES`
schedule -- one autoregressive pass per scale -- and measuring held-out
reconstruction MSE with the lesson's own `reconstruction_mse`, on the same 64
train / 16 val split `main()` draws. The lesson's grid is 8x8 and its
`downsample` only accepts divisors of 8, so there are exactly four distinct
sizes; 6, 8 and 10 scales must repeat one. Both placements are run: repeats at
the finest size, and repeats at the coarse sizes.

**ANSWER: it depends entirely on where the extra passes go.** Repeating the
8x8 scale drops val MSE from **1.5e-04** at 4 passes to **5.9e-07**, **2.0e-09**
and **2.3e-12** at 6, 8, 10. Repeating the coarse sizes instead gives
**1.2e-04**, **4.0e-04**, **4.2e-04** -- at 10 passes it is *worse* than at 4.
"More scales = better quality" holds for one placement and is false for the other.

**FINDING: the pyramid is the expensive part, not the cheap one.** Four passes
that are all 8x8 -- `SCALES=(8,8,8,8)` -- reach **3.8e-10**, about 400,000x below
the lesson's `(1,2,4,8)` at the same pass count. A single 8x8 pass alone gets
**5.2e-04**, within 3.5x of the four-pass pyramid. The coarse scales spend 16
codes on a handful of pooled means, and those codes are what the residual pays
for later.

**FINDING: the paper's 10-scale schedule cannot be expressed here.**
`(1,2,3,4,5,6,8,10,13,16)` fails at the first non-divisor: `downsample(img, 3)`
raises `ValueError` (it reshapes 64 values into 3x2x3x2).

**CONTROL: what the pyramid buys is tokens, not passes.** `(1,2,4,8)` emits 85
tokens; `(8,8,8,8)` emits 256. The coarse-only `(1,2,4)` sits at **0.071**
against **0.121** for predicting the global mean: the three coarse scales remove
only about 40% of the variance. The scale-count knob, as the lesson exposes it,
trades tokens for quality -- it does not trade passes.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "19-visual-autoregressive-var"
COUNTS = (4, 6, 8, 10)
PAPER = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)


def schedules(count):
    """Three ways to spend `count` passes on an 8x8 grid."""
    coarse = tuple(sorted((1, 2, 4, 8) + ((1, 2, 4) * 3)[: count - 4]))
    return {
        "fine": (1, 2, 4, 8) + (8,) * (count - 4),
        "coarse": coarse,
        "full": (8,) * count,
    }


def val_mse(ref, scales, train, val):
    """The lesson's own tokenizer, fitted with `scales`, scored on held-out images."""
    ref.SCALES = scales
    return ref.reconstruction_mse(val, ref.train_codebooks(train))


def paper_schedule_error(ref, image):
    try:
        ref.downsample(image, PAPER[2])
    except ValueError as err:
        return str(err)
    return ""


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = np.random.default_rng(0)
    train, val = ref.make_patterns(rng, 64), ref.make_patterns(rng, 16)
    sweep = {
        n: {k: val_mse(ref, s, train, val) for k, s in schedules(n).items()}
        for n in COUNTS
    }
    single = val_mse(ref, (8,), train, val)
    coarse_only = val_mse(ref, (1, 2, 4), train, val)
    ref.SCALES = (1, 2, 4, 8)
    return {
        "sweep": sweep,
        "single": single,
        "coarse_only": coarse_only,
        "mean_baseline": float(np.mean((val - val.mean()) ** 2)),
        "paper_error": paper_schedule_error(ref, train[0]),
        "tokens": {k: sum(s * s for s in v) for k, v in schedules(4).items()},
    }


def verify(result):
    sw = result["sweep"]
    fine = [sw[n]["fine"] for n in COUNTS]
    coarse = [sw[n]["coarse"] for n in COUNTS]
    return [
        practice.Check(
            "ANSWER: full-resolution repeats improve monotonically with passes",
            bool(np.all(np.diff(fine) < 0)) and fine[-1] < 1e-9,
            "val MSE at 4/6/8/10 passes, repeats at 8x8: "
            + ", ".join(f"{v:.1e}" for v in fine),
        ),
        practice.Check(
            "FINDING: coarse repeats make 10 passes worse than 4",
            coarse[-1] > coarse[0],
            "val MSE at 4/6/8/10 passes, repeats at 1/2/4: "
            + ", ".join(f"{v:.1e}" for v in coarse),
        ),
        practice.Check(
            "FINDING: four 8x8 passes beat the lesson's pyramid at the same pass count",
            sw[4]["full"] * 1e4 < sw[4]["fine"],
            f"(8,8,8,8) {sw[4]['full']:.1e} vs (1,2,4,8) {sw[4]['fine']:.1e}, ratio "
            f"{sw[4]['fine'] / sw[4]['full']:,.0f}x; one 8x8 pass alone {result['single']:.1e}",
        ),
        practice.Check(
            "FINDING: the paper's 10-scale schedule cannot run on the lesson's grid",
            "reshape" in result["paper_error"],
            f"downsample(img, 3) raises ValueError: {result['paper_error']}",
        ),
        practice.Check(
            "CONTROL: the pyramid saves tokens, and its coarse scales carry little",
            result["tokens"]["fine"] == 85
            and result["coarse_only"] > 0.5 * result["mean_baseline"],
            f"tokens {result['tokens']['fine']} vs {result['tokens']['full']}; coarse-only "
            f"(1,2,4) MSE {result['coarse_only']:.3f} vs predict-the-mean "
            f"{result['mean_baseline']:.3f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
