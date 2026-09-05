"""Exercise 1 — LeCun init, and why comparing it to Xavier here proves nothing.

    Add LeCun initialization (Var = 1/fan_in, designed for SELU activation). Run
    the 50-layer experiment with LeCun init + tanh and compare to Xavier + tanh.

Reading of the exercise: the comparison is the task, so check 1 runs both and gives
the algebra that collapses them into one run on the lesson's square stack. Checks 2
and 3 ask whether that shared trajectory is any good (it is not; tanh is why), check
4 runs LeCun with SELU, and check 5 builds the stack where the two formulas differ.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "08-weight-initialization"
SEED, XSEED = 20250904, 8675309          # weights vs inputs: never the same stream
BAND = (0.5, 2.0)                        # the lesson's own flowchart criterion
SELU_L, SELU_A = 1.0507009873554805, 1.6732632423543772
WIDTHS = (16, 64, 100, 256, 1024)
WIDTH_LIST = ", ".join(str(n) for n in WIDTHS)
FUNNEL = [16 if i % 2 == 0 else 64 for i in range(51)]   # alternating fan_in/fan_out


def lecun_init(fan_in, fan_out):
    """Var = 1/fan_in, written in the lesson's own style so it shares its RNG."""
    std = math.sqrt(1.0 / fan_in)
    return [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]


def gain_init(gain):
    return lambda fan_in, _fo: [[random.gauss(0, gain / math.sqrt(fan_in))
                                 for _ in range(fan_in)] for _ in range(fan_in)]


def selu(x):
    return SELU_L * x if x > 0 else SELU_L * SELU_A * (math.exp(min(x, 0.0)) - 1.0)


def stack_rms(init, widths):
    rng = random.Random(XSEED)
    vec = [rng.gauss(0, 1) for _ in range(widths[0])]
    random.seed(SEED)
    for fan_in, fan_out in zip(widths, widths[1:]):
        vec = [sum(w * v for w, v in zip(row, vec)) for row in init(fan_in, fan_out)]
    return (sum(v * v for v in vec) / len(vec)) ** 0.5


def tanh_contraction(scale, n=20000):
    rng = random.Random(SEED)
    draws = [rng.gauss(0, scale) for _ in range(n)]
    return sum(math.tanh(z) ** 2 for z in draws) / sum(z * z for z in draws)


def phrases(result):
    xav = result["xav"]
    return {"trail": ", ".join(f"L{i + 1} {xav[i]:.4f}" for i in (0, 4, 9, 24, 49)),
            "con": ", ".join(f"rms {s}: {v:.3f}" for s, v in result["contract"].items()),
            "g": ", ".join(f"g={g:.3f} -> {v:.4f}" for g, v in result["gains"].items())}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with parity.quiet():
        xav = ref.forward_deep(ref.xavier_init, ref.tanh_act)
        lec = ref.forward_deep(lecun_init, ref.tanh_act)
        sel = ref.forward_deep(lecun_init, selu)
        gains = {g: ref.forward_deep(gain_init(g), ref.tanh_act, n_samples=30)[49]
                 for g in (1.0, math.sqrt(2.0), 5 / 3)}
        funnel = {n: stack_rms(i, FUNNEL)
                  for n, i in (("lecun", lecun_init), ("xavier", ref.xavier_init))}
    return {"xav": xav, "selu": sel, "gains": gains, "funnel": funnel,
            "square_ok": all(math.sqrt(2.0 / (n + n)) == math.sqrt(1.0 / n) for n in WIDTHS),
            "contract": {s: tanh_contraction(s) for s in (1.0, 0.5, 0.3)},
            "worst": max(abs(a - b) for a, b in zip(xav, lec)),
            "out_of_band": [i + 1 for i, m in enumerate(xav)
                            if not BAND[0] <= m <= BAND[1]]}


def verify(result):
    xav, sel, ph = result["xav"], result["selu"], phrases(result)
    gains, fun, band = result["gains"], result["funnel"], result["out_of_band"]
    return [
        practice.Check("ANSWER: on a square stack LeCun + tanh IS Xavier + tanh, bit for bit",
                       result["worst"] == 0.0 and result["square_ok"],
                       f"50 layers, worst difference {result['worst']:.1e} — {ph['trail']}. "
                       f"2/(fan_in + fan_out) == 1/fan_in exactly when fan_out == fan_in (width "
                       f"{WIDTH_LIST}), and forward_deep only ever builds width x width layers"),
        practice.Check("FINDING: the trajectory they share fails the lesson's own criterion",
                       xav[49] / xav[0] < 0.15 and len(band) == 49,
                       f"mean |activation| {xav[0]:.4f} at L1 -> {xav[49]:.4f} at L50, a "
                       f"{xav[0] / xav[49]:.1f}x fall; the flowchart asks for [{BAND[0]}, "
                       f"{BAND[1]}] and {len(band)} of 50 layers fail, from L{band[0]} on"),
        practice.Check("MECHANISM: tanh contracts, so gain 1 has no non-zero fixed point",
                       max(result["contract"].values()) < 1.0 and gains[5 / 3] > 5 * gains[1.0],
                       f"E[tanh(z)^2]/E[z^2] at {ph['con']} — under 1 everywhere, nearing 1 only "
                       f"as the signal dies, and Var(w) = 1/fan_in preserves variance across the "
                       f"*linear* map alone. Layer-50 magnitude by forward gain g: {ph['g']}"),
        practice.Check("ANSWER: LeCun is stable with the activation the exercise names",
                       0.75 < sel[49] / sel[0] < 1.3,
                       f"LeCun + SELU holds mean |activation| {sel[0]:.4f} -> {sel[49]:.4f}, a "
                       f"{abs(1 - sel[49] / sel[0]) * 100:.0f}% change over 50 layers against "
                       f"tanh's {(1 - xav[49] / xav[0]) * 100:.0f}% — SELU's negative branch "
                       f"amplifies, which is what repairs tanh's contraction"),
        practice.Check("CONTROL: the two formulas do differ — on layers that are not square",
                       0.4 < fun["lecun"] < 2.5 and fun["lecun"] > 60 * fun["xavier"],
                       f"50 alternating 16/64 linear layers: LeCun ends at rms "
                       f"{fun['lecun']:.4f}, Xavier at {fun['xavier']:.2e} — a factor of "
                       f"{fun['lecun'] / fun['xavier']:.0f}. Xavier averages the forward and "
                       f"backward requirements; a non-square layer then meets neither"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
