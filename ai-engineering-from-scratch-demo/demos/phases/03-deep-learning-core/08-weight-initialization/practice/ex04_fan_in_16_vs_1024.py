"""Exercise 4 — the same experiment at fan_in 16 and 1024.

    Run the experiment with fan_in = 16 vs fan_in = 1024. Xavier and Kaiming adapt
    to fan_in, but random init doesn't. Show how the gap between "works" and
    "breaks" widens with larger layers.

Reading of the exercise: `forward_deep`'s `width` is the fan_in, so this runs the lesson's own
function at four widths — 6 layers and 2 samples at 1024, where 50 x 100 would be 5 billion
pure-Python multiply-adds; the per-layer gain the checks quote is what 50 layers compounds.
"The gap between works and breaks" is made precise in check 3 as the band of scales a 50-layer
net survives, which is the thing that narrows. Check 5 is what the four strategies really are.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "08-weight-initialization"
WIDTHS, LAYERS, SAMPLES, DEEP = (16, 64, 256, 1024), 6, 2, 50
FLOOR, CEILING = 1e-6, 1e6            # `magnitude_report`'s own VANISHED / EXPLODED thresholds


def arms(ref) -> dict:
    return {"random1": lambda fi, fo: ref.random_init(fi, fo, 1.0),
            "random001": lambda fi, fo: ref.random_init(fi, fo, 0.01),
            "xavier": ref.xavier_init, "kaiming": ref.kaiming_init}


def sweep(ref, width) -> dict:
    """The lesson's own forward_deep at one fan_in, per-layer gain read off its magnitudes."""
    out = {}
    for name, init in arms(ref).items():
        mags = ref.forward_deep(init, ref.relu, n_layers=LAYERS, width=width, n_samples=SAMPLES)
        out[name] = {"end": mags[-1], "gain": (mags[-1] / mags[0]) ** (1 / (LAYERS - 1))}
    return out


def band(width) -> tuple:
    """The scales whose 50-layer net lands inside [1e-6, 1e6], from gain = sigma*sqrt(n/2)."""
    unit = math.sqrt(2.0 / width)
    return CEILING ** (-1 / DEEP) * unit, CEILING ** (1 / DEEP) * unit


def proportional(ref, width=8) -> float:
    """Worst deviation from 'every strategy is one random matrix times a constant'."""
    draws, factors = {}, {"random1": 1.0, "random001": 0.01, "xavier": math.sqrt(1.0 / width),
                          "kaiming": math.sqrt(2.0 / width)}
    for name, init in arms(ref).items():
        random.seed(42)
        draws[name] = init(width, width)
    return max(abs(draws[n][r][c] - f * draws["random1"][r][c])
               for n, f in factors.items() for r in range(width) for c in range(width))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"by_width": {w: sweep(ref, w) for w in WIDTHS},
            "bands": {w: band(w) for w in WIDTHS}, "same": proportional(ref)}


def digest(result) -> dict:
    """Every listing `verify` prints, so that stays a list of comparisons."""
    by_width, bands = result["by_width"], result["bands"]
    row = lambda k, f: ", ".join(f"n={w}: {by_width[w][k][f]:.3g}" for w in WIDTHS)  # noqa: E731
    return {"kaiming": row("kaiming", "end"), "k_gain": row("kaiming", "gain"),
            "r1": row("random1", "end"), "r1_gain": row("random1", "gain"),
            "x_gain": row("xavier", "gain"),
            "law": ", ".join(f"n={w}: {by_width[w]['random1']['gain'] / math.sqrt(w / 2):.3f}"
                             for w in WIDTHS),
            "band": ", ".join(f"n={w}: [{lo:.5f}, {hi:.5f}] wide {hi - lo:.5f}"
                              for w, (lo, hi) in bands.items())}


def verify(result):
    d, by_width, bands = digest(result), result["by_width"], result["bands"]
    off = max(abs(by_width[w][k]["gain"] / (s * math.sqrt(w / 2)) - 1)
              for w in WIDTHS for k, s in (("random1", 1.0), ("random001", 0.01)))
    narrow = (bands[WIDTHS[0]][1] - bands[WIDTHS[0]][0]) / (bands[WIDTHS[-1]][1]
                                                            - bands[WIDTHS[-1]][0])
    return [
        practice.Check("ANSWER: Kaiming holds across a 64x change in fan_in and a fixed scale "
                       "does not",
                       all(0.9 < by_width[w]["kaiming"]["gain"] < 1.15 for w in WIDTHS)
                       and by_width[WIDTHS[-1]]["random1"]["gain"] > 20,
                       f"magnitude after {LAYERS} layers — Kaiming {d['kaiming']} at a per-layer "
                       f"gain of {d['k_gain']}; random N(0,1) {d['r1']} at {d['r1_gain']}. Kaiming "
                       f"moves 0.98 -> 1.02 over the range; the fixed scale moves 2.8 -> 23"),
        practice.Check("MECHANISM: the per-layer gain is sigma * sqrt(fan_in / 2)",
                       off < 0.08,
                       f"measured gain over the closed form, at all four widths: {d['law']} — "
                       f"worst deviation {100 * off:.1f}%, and sigma = 1.0 and 0.01 give the same "
                       f"ratio to the last digit because the two matrices are proportional. A "
                       f"ReLU layer "
                       f"halves the variance of a symmetric pre-activation, so fan_in * sigma^2 / "
                       f"2 is the variance gain and its square root is the magnitude gain"),
        practice.Check("ANSWER: the band of scales a 50-layer net survives narrows as "
                       "1/sqrt(fan_in)",
                       narrow > 7.0,
                       f"a {DEEP}-layer net stays inside [{FLOOR:g}, {CEILING:g}] only while the "
                       f"gain is within [{CEILING ** (-1 / DEEP):.4f}, {CEILING ** (1 / DEEP):.4f}], "
                       f"so sigma must lie in {d['band']}. That is {narrow:.1f}x narrower at "
                       f"fan_in {WIDTHS[-1]} than at {WIDTHS[0]} — and sigma = 1.0 is outside "
                       f"every one of them"),
        practice.Check("FINDING: Xavier is not neutral under ReLU — it loses 1/sqrt(2) a layer, "
                       "at every width",
                       all(abs(by_width[w]["xavier"]["gain"] - 0.707) < 0.06 for w in WIDTHS),
                       f"Xavier's per-layer gain: {d['x_gain']} — sqrt(1/n) * sqrt(n/2) = "
                       f"{1 / math.sqrt(2):.4f}, with the fan_in cancelling. Over {DEEP} layers "
                       f"that is 2^-{DEEP // 2} = {2.0 ** (-DEEP / 2):.1e}, which is why Kaiming's "
                       f"extra factor of 2 exists"),
        practice.Check("CONTROL: all four strategies are one random matrix times a constant",
                       result["same"] == 0.0,
                       f"`forward_deep` re-seeds to 42 on every call and every init draws the same "
                       f"number of `random.gauss` values in the same order, so random N(0,1), "
                       f"random N(0,0.01), Xavier and Kaiming return matrices that agree to "
                       f"{result['same']:.1f} after dividing by 1, 0.01, sqrt(1/n) and sqrt(2/n). "
                       f"The whole experiment varies one scalar"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
