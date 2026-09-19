"""Exercise 5 — the rate is 50 for the transformer and 400 for the head.

    Compute the throughput budget: how fast must a Talker emit tokens to keep up
    with 16kHz speech at 50 base-layer tokens/sec?

Reading of the exercise: "tokens" is ambiguous here and the two readings differ
by a factor of eight, so both are computed. The lesson's own prose answers one of
them and then compares it against a throughput range whose bottom end fails the
requirement it just stated.

**ANSWER: 50 a second for the base layer, and 400 a second for all eight.**
A 50 Hz base stream with 8 residual codebooks is **400** tokens per second of
audio -- one every **2.5 ms** -- against the base layer's one every 20 ms. The
lesson states the 50 and not the 400.

**FINDING: the 400 is not a transformer rate, which is why the split exists.**
Lesson 12.16 measures the residual levels at 4.3 ms each against a 40 ms
first-token cost -- a small depth head, not the main pass. So the Talker's
transformer runs at **50** and its head at **350** more, and the two numbers
belong to different modules.

**FINDING: the lesson's own throughput range fails its own requirement at the
bottom.** It cites "30-80 tok/s on an H100" and calls a small Talker "fast
enough". At **30** tok/s against a 50/s requirement the real-time factor is
**1.67** -- the stream falls **0.67 seconds behind for every second spoken** --
and only **67%** of the cited points -- 50 and 80 of {30, 50, 80} -- keep up
at all.

**FINDING: and the margin at the top is thin.** At 80 tok/s the real-time factor
is **0.62**, so the whole cited range spans RTF **0.62 to 1.67** and straddles 1.
A Talker sized by that range is a coin flip on whether it streams, which is the
actual argument for a dedicated small model rather than "small is fast enough".

Structure: `per_second` converts a base rate and a codebook count into a token
rate, `rtf` is the real-time factor at a given throughput, and `THROUGHPUTS` is
the lesson's own cited range.
"""

from __future__ import annotations

from harness import practice

BASE_RATE, CODEBOOKS = 50, 8
THROUGHPUTS = (30, 50, 80)
DEPTH_HEAD_MS, FIRST_TOKEN_MS = 4.3, 40.0     # Lesson 12.16's measured split


def per_second(base=BASE_RATE, codebooks=CODEBOOKS):
    return base * codebooks


def period_ms(rate):
    return round(1000 / rate, 1)


def rtf(throughput, required=BASE_RATE):
    return round(required / throughput, 2)


def solve():
    full = per_second()
    curve = {rate: rtf(rate) for rate in THROUGHPUTS}
    keeping_up = [rate for rate, value in curve.items() if value <= 1.0]
    return {
        "base_rate": BASE_RATE, "codebooks": CODEBOOKS, "full_rate": full,
        "base_period": period_ms(BASE_RATE), "full_period": period_ms(full),
        "factor": full // BASE_RATE,
        "head_rate": full - BASE_RATE,
        "head_ms": DEPTH_HEAD_MS, "pass_ms": FIRST_TOKEN_MS,
        "head_ratio": round(FIRST_TOKEN_MS / DEPTH_HEAD_MS, 1),
        "rtf": curve,
        "keeping_up": keeping_up,
        "share_of_range": round(len(keeping_up) / len(THROUGHPUTS) * 100),
        "behind_per_second": round(curve[THROUGHPUTS[0]] - 1, 2),
        "straddles_one": min(curve.values()) < 1.0 < max(curve.values()),
    }


def verify(result):
    curve = result["rtf"]
    return [
        practice.Check(
            "ANSWER: 50 a second for the base layer, 400 for all eight",
            all([result["full_rate"] == 400, result["factor"] == CODEBOOKS,
                 result["base_period"] == 20.0, result["full_period"] == 2.5]),
            f"a {BASE_RATE} Hz base stream with {CODEBOOKS} residual codebooks is "
            f"{result['full_rate']} tokens per second of audio -- one every "
            f"{result['full_period']} ms against the base layer's every "
            f"{result['base_period']} ms. The lesson states the {BASE_RATE} and not the "
            f"{result['full_rate']}",
        ),
        practice.Check(
            "FINDING: the 400 is not a transformer rate, which is why the split exists",
            all([result["head_rate"] == 350, result["head_ratio"] == 9.3]),
            f"Lesson 12.16 measures the residual levels at {result['head_ms']} ms each "
            f"against a {result['pass_ms']:.0f} ms first-token cost -- "
            f"{result['head_ratio']}x cheaper, so a small depth head rather than the main "
            f"pass. The Talker's transformer runs at {BASE_RATE} and its head at "
            f"{result['head_rate']} more, and the two belong to different modules",
        ),
        practice.Check(
            "FINDING: the lesson's own throughput range fails its own requirement",
            all([curve == {30: 1.67, 50: 1.0, 80: 0.62},
                 result["keeping_up"] == [50, 80],
                 result["share_of_range"] == 67,
                 result["behind_per_second"] == 0.67]),
            f"it cites 30-80 tok/s and calls a small Talker fast enough. Against a "
            f"{BASE_RATE}/s requirement the real-time factors are {curve}: at 30 the stream "
            f"falls {result['behind_per_second']} seconds behind for every second spoken, "
            f"and only {result['share_of_range']}% of the cited points keep up",
        ),
        practice.Check(
            "FINDING: and the margin at the top is thin",
            all([curve[80] == 0.62, result["straddles_one"]]),
            f"at the top of the cited range the real-time factor is {curve[80]}, so the whole "
            f"range spans {min(curve.values())} to {max(curve.values())} and straddles 1. A "
            "Talker sized by that range is a coin flip on whether it streams -- which is the "
            "real argument for a dedicated small model, rather than 'small is fast enough'",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
