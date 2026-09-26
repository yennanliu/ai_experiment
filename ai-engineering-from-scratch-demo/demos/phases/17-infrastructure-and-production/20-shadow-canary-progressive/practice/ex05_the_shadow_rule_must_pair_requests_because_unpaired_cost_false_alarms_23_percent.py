"""Exercise 5 — the shadow rule must pair requests, because unpaired cost false-alarms 23%.

    Shadow mode catches a 40% cost spike before canary. Write the alert rule
    that fires in shadow.

Reading of the exercise: in shadow every production request is duplicated to
the candidate and neither answer's cost depends on the user, so the rule can
compare the two models on the *same* requests. The rule is written as data,
then run twice: at window level through the reference's `measure_stage`, and
at request level, where per-request cost is lognormal (sigma = 1, an assumed
heavy tail for token counts) and the candidate adds its multiplier times the
reference's +-8% noise.

**ANSWER: fire when sum(candidate cost) / sum(production cost) over the same
request ids exceeds 1.2, on a window of at least 50 paired requests.** The
threshold is the lesson's own cost gate. Through the reference's window-level
noise a 40% spike reads at least 1.288 and fires on
all 10,000 seeded windows, and a clean candidate reads at most 1.08 and fires
on none. It fires with zero users exposed; the canary would halt on the same
spike at 1%, after serving it.

**FINDING: pairing is what makes shadow fast.** At 50 requests the paired
ratio fires on every 40% window and no clean one. The same ratio over
different requests -- what a canary compares -- false-alarms 23.0% of clean
windows and catches 73.4% of spikes. At 200 requests unpaired is still 8.9%
and 88.6%. Requiring two independent windows would cut unpaired false alarms to
0.8% (0.089 squared); the paired rule needs neither the larger window nor the
second one.

**FINDING: shadow cannot see the lesson's quality-silent case.** Users never
see shadow output, so thumbs-down -- one of the five gates -- does not exist
there. The demo's "Quality silent + cost creep" (cost 1.15, thumbs-down 1.45)
reads 1.15 in shadow and fires on 0 paired windows; the canary halts it at
25%. `measure_stage` reports a thumbs-down rate for any traffic, and the
reference has no shadow mode at all.

Structure: `SHADOW_RULE` is the rule; `fires()` evaluates it on one window;
`window_rate()` samples seeded windows of paired or unpaired requests.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "20-shadow-canary-progressive"
SHADOW_RULE = {
    "metric": "sum(candidate_cost) / sum(production_cost) over the same request ids",
    "fire_if_above": 1.2,
    "min_requests": 50,
    "for_consecutive_windows": 1,
    "observable": ("latency_p99_ms", "cost_per_req", "error_rate", "output_len_p99"),
}
WINDOWS, TRIALS = 10_000, 1_000
GRID = ((50, True), (50, False), (200, False))  # (requests per window, paired)


def fires(candidate, production):
    enough = len(candidate) >= SHADOW_RULE["min_requests"]
    return enough and sum(candidate) / sum(production) > SHADOW_RULE["fire_if_above"]


def window_rate(n, mult, paired, seed=0):
    rng, hits = random.Random(seed), 0
    for _ in range(TRIALS):
        prod = [math.exp(rng.gauss(0, 1)) for _ in range(n)]
        other = prod if paired else [math.exp(rng.gauss(0, 1)) for _ in range(n)]
        cand = [c * mult * rng.uniform(0.92, 1.08) for c in prod]
        hits += fires(cand, other)
    return round(hits / TRIALS, 3)


def reference_windows(ref, mult):
    """Window-level shadow ratios through the reference's measure_stage."""
    base = ref.BASELINE["cost_per_req"]
    reg = ref.Regression(cost_mult=mult)
    ratios = [
        ref.measure_stage(1.0, reg, seed)["cost_per_req"] / base
        for seed in range(WINDOWS)
    ]
    return round(min(ratios), 3), round(max(ratios), 3), sum(r > 1.2 for r in ratios)


def canary_halt(ref, reg):
    for i, stage in enumerate(ref.STAGES):
        if ref.check_gates(ref.measure_stage(stage, reg, seed=ref.stage_seed(i))):
            return stage
    return None


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    source = parity.lesson_dir(PHASE, LESSON) / "code" / "main.py"
    rates = {(n, p): (window_rate(n, 1.0, p), window_rate(n, 1.4, p)) for n, p in GRID}
    silent = ref.Regression(cost_mult=1.15, thumbs_down_mult=1.45)
    return {
        "spike": reference_windows(ref, 1.4),
        "clean": reference_windows(ref, 1.0),
        "canary_spike": canary_halt(ref, ref.Regression(cost_mult=1.4)),
        "rates": rates,
        "two_windows": round(rates[(200, False)][0] ** 2, 3),
        "silent": (window_rate(50, 1.15, True), canary_halt(ref, silent)),
        "thumbs_hidden": "thumbs_down_rate" not in SHADOW_RULE["observable"],
        "no_shadow_mode": "shadow" not in source.read_text("utf-8").lower(),
    }


def verify(result):
    r, spike, clean = result, result["spike"], result["clean"]
    return [
        practice.Check(
            "ANSWER: fire when paired cost exceeds 1.2 over 50+ same requests",
            (spike[0], spike[2], clean[2], r["canary_spike"])
            == (1.288, WINDOWS, 0, 0.01)
            and clean[1] <= 1.08,
            f"a 40% spike reads {spike[0]}..{spike[1]} and fires {spike[2]}/{WINDOWS}; "
            f"clean reads up to {clean[1]} and fires {clean[2]}; the canary halts it "
            f"at {r['canary_spike']:.0%} after serving it",
        ),
        practice.Check(
            "FINDING: pairing is what makes shadow fast",
            r["rates"]
            == {
                (50, True): (0.0, 1.0),
                (50, False): (0.23, 0.734),
                (200, False): (0.089, 0.886),
            }
            and r["two_windows"] == 0.008,
            f"(requests, paired) -> (clean, spike) fire rate {r['rates']}; two unpaired "
            f"windows false-alarm {r['two_windows']}",
        ),
        practice.Check(
            "FINDING: shadow cannot see the lesson's quality-silent case",
            r["silent"] == (0.0, 0.25) and r["thumbs_hidden"] and r["no_shadow_mode"],
            f"cost 1.15 + thumbs-down 1.45: (paired shadow fire rate, canary halt stage) "
            f"= {r['silent']}; the reference has no shadow mode",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
