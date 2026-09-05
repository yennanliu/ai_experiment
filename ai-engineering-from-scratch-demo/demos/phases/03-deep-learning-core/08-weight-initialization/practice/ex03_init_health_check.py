"""Exercise 3 — an init health check, and the health metric the lesson gets wrong.

    Create an "init health check" function that takes a network's layer dimensions
    and activation type, then recommends the correct initialization and warns if the
    current init will cause problems.

Reading of the exercise: "warns if the current init will cause problems" needs a
definition of "problems", and the lesson supplies one — its flowchart says to verify
activation magnitudes stay in [0.5, 2.0]. Check 1 builds the check out of variance
propagation and scores it against the lesson's own six configs; check 2 shows the
flowchart's metric ranking the two dead configs first; checks 3-5 say why, and what
the check should recommend instead.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "08-weight-initialization"
SEED, WIDTH, BAND = 42, 64, (0.5, 2.0)      # SEED and BAND are the lesson's own choices
DIMS = (WIDTH,) * 51                        # the lesson's 50-layer square stack
GAIN = {"sigmoid": 4.0, "tanh": 5 / 3, "relu": math.sqrt(2.0)}
SLOPE = {"sigmoid": 0.25, "tanh": 1.0, "relu": 2 ** -0.5}    # rms of act'(z) near z = 0
CONFIGS = (("zero", "sigmoid"), ("random(1.0)", "relu"), ("random(0.01)", "relu"),
           ("xavier", "sigmoid"), ("xavier", "tanh"), ("kaiming", "relu"))
ZS, R1, R01, XS, XT, KR = CONFIGS


def run_stack(init, act, layers=50, width=WIDTH, n=24):
    """The lesson's forward_deep, instrumented to keep the activations it discards."""
    random.seed(SEED)
    rows = [[random.gauss(0, 1) for _ in range(width)] for _ in range(n)]
    for _ in range(layers):
        w = init(width, width)
        rows = [[act(sum(a * b for a, b in zip(r, s))) for r in w] for s in rows]
    return rows


def spread(rows):
    """Across-sample std per unit, rms'd: the input sensitivity mean |a| cannot see."""
    cols = list(zip(*rows))
    var = [sum((v - sum(c) / len(c)) ** 2 for v in c) / len(c) for c in cols]
    return (sum(var) / len(var)) ** 0.5


def mean_abs(rows):
    return sum(abs(v) for s in rows for v in s) / (len(rows) * len(rows[0]))


def scaled(init, g):
    return lambda fi, fo: [[g * v for v in row] for row in init(fi, fo)]


def std_of(init, fan_in, fan_out):
    """Measured from the init's own output, so the check reads any init, not a formula."""
    flat = [v for row in init(fan_in, fan_out) for v in row]
    return (sum(v * v for v in flat) / len(flat)) ** 0.5


def verdict(symmetric, big, small):
    """One word, from the same three tests whether predicted or measured."""
    return next((n for n, hit in zip(("symmetric", "explodes", "dies"),
                                     (symmetric, big, small)) if hit), "ok")


def health_check(dims, act, init):
    """Recommend a std for `act`, and predict what `init` does over these dims."""
    stds = [std_of(init, fi, fo) for fi, fo in zip(dims, dims[1:])]
    gains = [fi * s * s * SLOPE[act] ** 2 for fi, s in zip(dims, stds)]
    factor = math.prod(gains) ** 0.5
    return {"recommend": GAIN[act] / math.sqrt(dims[0]), "std": stds[0], "gain": gains[0],
            "factor": factor,
            "verdict": verdict(max(stds) == 0.0, factor > 1e3, factor < 1e-3)}


def summarize(rows, mags):
    """Per-config measurements, kept out of solve() so its complexity stays in budget."""
    return {"spread": {c: spread(r) for c, r in rows.items()},
            "mag": {c: mean_abs(r) for c, r in rows.items()},
            "seen": {c: verdict(spread(r) == 0.0, mean_abs(r) > 1e3, spread(r) < 1e-6)
                     for c, r in rows.items()},
            "band": {c: sum(1 for v in m if BAND[0] <= v <= BAND[1]) for c, m in mags.items()}}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    act = {"sigmoid": ref.sigmoid, "tanh": ref.tanh_act, "relu": ref.relu}
    init = {"zero": ref.zero_init, "xavier": ref.xavier_init, "kaiming": ref.kaiming_init,
            "random(1.0)": lambda a, b: ref.random_init(a, b, 1.0),
            "random(0.01)": lambda a, b: ref.random_init(a, b, 0.01)}
    with parity.quiet():
        rows = {c: run_stack(init[c[0]], act[c[1]]) for c in CONFIGS}
        mags = {c: ref.forward_deep(init[c[0]], act[c[1]]) for c in CONFIGS}
        pred = {c: health_check(DIMS, c[1], init[c[0]]) for c in CONFIGS}
        sweep = {g: spread(run_stack(scaled(ref.xavier_init, g), ref.sigmoid))
                 for g in (1, 2, 4, 6)}
        ladder = {d: spread(run_stack(ref.xavier_init, ref.sigmoid, layers=d)) for d in (2, 8)}
        fix = {g: run_stack(scaled(ref.xavier_init, g), ref.tanh_act) for g in (1.0, 5 / 3)}
    return {"pred": pred, "sweep": sweep, "ladder": ladder, **summarize(rows, mags),
            "fix": {g: (spread(r), mean_abs(r)) for g, r in fix.items()}}


def listings(result) -> tuple:
    """The two strings `verify` prints, kept out of it."""
    pred = result["pred"]
    return (", ".join(f"{a}+{b} {pred[(a, b)]['verdict']}" for a, b in CONFIGS),
            ", ".join(f"x{g} {v:.1e}" for g, v in result["sweep"].items()))


def verify(result):
    pred, seen, band = result["pred"], result["seen"], result["band"]
    sp, mag, fix = result["spread"], result["mag"], result["fix"]
    rate = (result["ladder"][8] / result["ladder"][2]) ** (1 / 6)
    off = [abs(math.log10(pred[c]["factor"]) - math.log10(mag[c])) for c in (R1, R01)]
    calls, swept = listings(result)
    return [
        practice.Check("ANSWER: the check calls all six of the lesson's configs correctly",
                       all(pred[c]["verdict"] == seen[c] for c in CONFIGS) and max(off) < 2.0,
                       f"from fan_in * std^2 * slope^2 and nothing else — {calls} — each "
                       f"matching the measured 50-layer outcome. It predicts the magnitude to "
                       f"{max(off):.1f} decades over the {math.log10(mag[R1]):.0f} and "
                       f"{math.log10(mag[R01]):.0f} the two random inits actually reach"),
        practice.Check("FINDING: the lesson's own health metric ranks the two DEAD configs top",
                       band[ZS] == 50 and band[XS] > band[KR] and max(sp[ZS], sp[XS]) < 1e-16,
                       f"layers with mean |activation| inside the flowchart's [{BAND[0]}, "
                       f"{BAND[1]}]: zero+sigmoid {band[ZS]}/50, xavier+sigmoid {band[XS]}/50, "
                       f"kaiming+relu {band[KR]}/50, xavier+tanh {band[XT]}/50 — yet the "
                       f"across-sample spread at layer 50 is {sp[ZS]:.1e} and {sp[XS]:.1e} for "
                       f"the first two against {sp[KR]:.3f} and {sp[XT]:.3f} for the last two"),
        practice.Check("MECHANISM: sigmoid's offset pins mean |a| at 0.5 while the signal dies",
                       abs(rate - SLOPE["sigmoid"]) < 0.03 and abs(mag[XS] - 0.5) < 0.01,
                       f"xavier+sigmoid holds mean |a| at {mag[XS]:.4f} for all 50 layers "
                       f"because sigmoid is centred on 0.5, but deviations contract by "
                       f"{rate:.3f} per layer — measured against sigmoid'(0) = "
                       f"{SLOPE['sigmoid']}, which Xavier's 2/(fan_in + fan_out) ignores"),
        practice.Check("FINDING: at this depth no init scale rescues sigmoid",
                       max(result["sweep"].values()) < 1e-6,
                       f"across-sample spread at layer 50 with Xavier scaled by — {swept} — "
                       f"all still dead. Raising the gain trades vanishing deviations for a "
                       f"saturated sigmoid, so the check should recommend changing activation"),
        practice.Check("CONTROL: the same rule does fix tanh, which has no offset",
                       fix[5 / 3][0] > 5 * fix[1.0][0] and BAND[0] <= fix[5 / 3][1] <= BAND[1],
                       f"the check recommends std = {pred[XT]['recommend']:.4f} for tanh "
                       f"(gain 5/3) against Xavier's {pred[XT]['std']:.4f}: spread at layer 50 "
                       f"rises {fix[1.0][0]:.4f} -> {fix[5 / 3][0]:.4f} and mean |a| "
                       f"{fix[1.0][1]:.4f} -> {fix[5 / 3][1]:.4f}, into the band"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
