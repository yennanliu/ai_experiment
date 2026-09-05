"""Exercise 2 — GPT-2's 1/sqrt(2N) residual scaling, and what it is actually for.

    Implement the GPT-2 residual scaling: multiply the output of each layer by
    1/sqrt(2*N) before adding to the residual stream. Run 50 layers with and
    without scaling, measure how fast the residual magnitude grows.

Reading of the exercise: "measure how fast" is the load-bearing phrase — the lesson
says variance "grows proportionally to N", which is a testable rate, not a direction.
Checks 1-3 measure the unscaled rate and find it exponential, check 4 adds the pre-norm
the lesson never mentions and recovers the rate it claims, checks 5-6 measure what the
1/sqrt(2N) factor converges to. Tables and full traces are in the README.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "08-weight-initialization"
SEED, XSEED, WIDTH, DEPTHS = 20250904, 8675309, 48, (10, 25, 50, 100)   # two RNG streams


def rms(rows) -> float:
    flat = [v for row in rows for v in row]
    return (sum(v * v for v in flat) / len(flat)) ** 0.5


def pre(sample, norm):   # pre-LayerNorm reduced to what matters here: fix input scale
    return [v / (rms([sample]) or 1.0) for v in sample] if norm else sample


def sublayer(ref, sample, w1, w2):   # Kaiming -> ReLU -> Kaiming, the residual block
    hidden = [ref.relu(sum(a * b for a, b in zip(row, sample))) for row in w1]
    return [sum(a * b for a, b in zip(row, hidden)) for row in w2]


def start(width, n=24):
    rng = random.Random(XSEED)
    rows = [[rng.gauss(0, 1) for _ in range(width)] for _ in range(n)]
    random.seed(SEED)
    return rows


def stream(ref, scale, layers, width=WIDTH, norm=False):
    rows = start(width)
    trace = [rms(rows)]
    for _ in range(layers):
        w1, w2 = ref.kaiming_init(width, width), ref.kaiming_init(width, width)
        rows = [[a + scale * b for a, b in zip(s, sublayer(ref, pre(s, norm), w1, w2))]
                for s in rows]
        trace.append(rms(rows))
    return trace


def sublayer_gain(ref, width=WIDTH):
    rows = start(width)
    w1, w2 = ref.kaiming_init(width, width), ref.kaiming_init(width, width)
    return (rms([sublayer(ref, s, w1, w2) for s in rows]) / rms(rows)) ** 2


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with parity.quiet():
        plain = {n: stream(ref, 1.0, n) for n in DEPTHS}
        scaled = {n: stream(ref, 1.0 / math.sqrt(2 * n), n) for n in DEPTHS}
        ln_plain, ln_scaled = (stream(ref, 1.0, 50, norm=True),
                               stream(ref, 1.0 / math.sqrt(100), 50, norm=True))
        gain = sublayer_gain(ref)
    grow = {n: (t[-1] / t[0]) ** 2 for n, t in plain.items()}
    return {"plain": plain[50], "scaled": scaled[50], "gain": gain, "grow": grow,
            "hold": {n: (t[-1] / t[0]) ** 2 for n, t in scaled.items()},
            "ln_plain": ln_plain[-1] / ln_plain[0], "rate": grow[50] ** (1 / 50),
            "ln_scaled": (ln_scaled[-1] / ln_scaled[0]) ** 2}


def verify(result):
    plain, scaled, hold = result["plain"], result["scaled"], result["hold"]
    trail = ", ".join(f"L{i} {plain[i]:.2e}" for i in (0, 1, 10, 25, 50))
    blew, held = plain[-1] / plain[0], ", ".join(f"N={n} {v:.3f}" for n, v in hold.items())
    grew = ", ".join(f"N={n} {v ** 0.5:.2e}x" for n, v in result["grow"].items())
    return [
        practice.Check("ANSWER: 50 layers with and without the 1/sqrt(2N) factor",
                       blew > 1e10 and 1.4 < scaled[-1] / scaled[0] < 2.0,
                       f"unscaled residual rms {trail}; scaled by 1/sqrt(2*50) the same stream "
                       f"goes {scaled[0]:.4f} -> {scaled[-1]:.4f}, a factor of "
                       f"{scaled[-1] / scaled[0]:.2f} against {blew:.2e}"),
        practice.Check("FINDING: unscaled growth is exponential, not proportional to N",
                       result["rate"] > 2.5 and blew > 1e6 * 51 ** 0.5,
                       f"variance factor {result['rate']:.3f} per layer: L50 sits {blew:.2e}x "
                       f"above L0, {math.log10(blew / 51 ** 0.5):.1f} decades past the sqrt(51) "
                       f"= {51 ** 0.5:.2f}x that 'proportional to N' predicts"),
        practice.Check("MECHANISM: the sublayer output scales with its input, so adds multiply",
                       1.6 < result["gain"] < 2.4,
                       f"E[sublayer(x)^2]/E[x^2] = {result['gain']:.3f} for one fresh Kaiming-"
                       f"ReLU-Kaiming block: two Kaiming layers double the second moment, ReLU "
                       f"halves it, so x + f(x) carries {1 + result['gain']:.2f}x the variance"),
        practice.Check("CONTROL: pre-normalise the sublayer input and the linear rate appears",
                       abs(result["ln_plain"] / math.sqrt(101) - 1) < 0.10,
                       f"an RMS-norm in front of the sublayer — what a pre-LN transformer "
                       f"actually has — holds the unscaled stream to {result['ln_plain']:.3f}x "
                       f"against the claimed sqrt(1 + 2*50) = {math.sqrt(101):.3f}"),
        practice.Check("ANSWER: 1/sqrt(2N) bounds the stream at a depth-independent variance",
                       max(hold.values()) < 3.3 and min(hold.values()) > 2.3,
                       f"scaled variance ratio by depth — {held} — against the closed form "
                       f"(1 + 1/N)^N rising to e = {math.e:.3f}; unscaled, {grew}"),
        practice.Check("FINDING: the 2 in 2N counts two residual adds per block, not one",
                       1.8 < result["ln_scaled"] < 2.5,
                       f"normalised sublayer, scaled by 1/sqrt(2N), 50 single-add layers: ratio "
                       f"{result['ln_scaled']:.3f} against the exact 1 + 2N/(2N) = 2. GPT-2 "
                       f"scales blocks that add twice; one add per layer wants 1/sqrt(N)"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
