"""Exercise 4 — a 1.16x gate clears 7% noise, but rate gates need 3x at 1% traffic.

    Non-determinism shows ±7% on your eval. Set canary gates so you don't
    false-alarm. What multipliers do you use?

Reading of the exercise: +-7% is taken in the reference's own noise model --
a uniform multiplicative draw -- applied to both the canary and the baseline
it is compared with, because a live baseline is measured too. Clean
candidates are then run through the reference's `check_gates` with the
baseline patched to the measured one. Rate metrics get a second, independent
noise source the reference leaves out: counting a few events in a window.

**ANSWER: 1.16x for latency, cost and output length; count-based multipliers
for the rate gates.** Two +-7% draws put the clean ratio in [0.93/1.07,
1.07/0.93] = [0.8692, 1.1505], so any gate above 1.1505 never false-alarms:
over 10,000 seeded clean pairs a uniform 1.16 gate fires 0 times, 1.14
fires 100 times, and 1.10 -- safe against a fixed baseline, whose clean ratio
tops out at 1.07 -- fires 2,304 times. The lesson's gates (1.2 to 2.0) also
fire 0 times. For error
and thumbs-down rates the multiplier that keeps a window's false-alarm rate
under 1% is set by the sample count, not by the 7%: thumbs-down (3%) needs
3.33x at 50 requests, 1.67x at 500, 1.39x at 1,250 and 1.19x at 5,000;
errors (2%) need 4.0x, 1.8x, 1.48x and 1.24x.

**FINDING: the lesson's rate gates false-alarm at 1% traffic.** At the 50
requests per window the lesson gives for 1%, the 1.5x thumbs-down gate fires
on a clean model 18.9% of the time and the 2x error gate 7.8%. Over the six
stages, with a 5-minute window at 1,000 req/min, the thumbs-down gate alone
halts 21.5% of clean rollouts. The reference cannot show this, because
`measure_stage` ignores the traffic share.

**FINDING: the noise floor costs detection.** A gate g is guaranteed to
catch only regressions above g / 0.8692: 1.335 for 1.16, 1.381 for the
lesson's 1.2 cost gate. A 25% cost regression is caught per stage in 74.9%
of seeded pairs against a measured baseline and 77.8% against a fixed one.

Structure: `fired()` samples seeded canary/baseline pairs through the
reference's `check_gates`; `rate_gate()` is an exact binomial tail.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "20-shadow-canary-progressive"
NOISE, PAIRS, WINDOW = 0.07, 10_000, 5_000  # 5-minute window at 1,000 req/min
SIZES = (50, 500, 1_250, 5_000)


def gated(ref, candidate, baseline, gates):
    saved_base, saved_gates = dict(ref.BASELINE), dict(ref.GATES)
    ref.BASELINE.update(baseline)
    ref.GATES.update(gates)
    try:
        return ref.check_gates(candidate)
    finally:
        ref.BASELINE.update(saved_base)
        ref.GATES.update(saved_gates)


def draw(rng, values, mult=None):
    noisy = lambda v: v * rng.uniform(1 - NOISE, 1 + NOISE)  # noqa: E731
    return {k: noisy(v * (mult or {}).get(k, 1.0)) for k, v in values.items()}


def fired(ref, gates, cost_mult=1.0, measured=True, seed=0):
    """Seeded canary/baseline pairs on which a gate fires (only the cost gate if regressed)."""
    rng, hits = random.Random(seed), 0
    for _ in range(PAIRS):
        cand = draw(rng, ref.BASELINE, {"cost_per_req": cost_mult})
        breaches = gated(ref, cand, draw(rng, ref.BASELINE) if measured else {}, gates)
        hits += ("cost_per_req" in breaches) if cost_mult != 1.0 else bool(breaches)
    return hits


def pmf(n, p, i):
    """Binomial pmf in log space, so n = 5,000 does not overflow a float."""
    logc = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
    return math.exp(logc + i * math.log(p) + (n - i) * math.log(1 - p))


def tail(n, p, k):
    return max(0.0, 1.0 - sum(pmf(n, p, i) for i in range(k)))


def rate_gate(n, p):
    """Smallest multiplier g for which 'count > g * n * p' false-alarms <= 1%."""
    k = next(k for k in range(n + 1) if tail(n, p, k) <= 0.01)
    return round((k - 1) / (n * p), 3)


def false_alarm(n, p, gate):
    return tail(n, p, math.floor(n * p * gate) + 1)


def rollout_false_alarm(stages, p, gate):
    ns = [round(WINDOW * s) for s in stages]
    return round(1 - math.prod(1 - false_alarm(n, p, gate) for n in ns), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    thumbs, errors = ref.BASELINE["thumbs_down_rate"], ref.BASELINE["error_rate"]
    return {
        "bound": round((1 + NOISE) / (1 - NOISE), 4),
        "alarms": {
            g: fired(ref, dict.fromkeys(ref.GATES, g)) for g in (1.1, 1.14, 1.16)
        },
        "lesson_alarms": fired(ref, {}),
        "rates": tuple([rate_gate(n, p) for n in SIZES] for p in (thumbs, errors)),
        "fa_1pct": tuple(
            round(false_alarm(50, *a), 3) for a in ((thumbs, 1.5), (errors, 2))
        ),
        "rollout_fa": rollout_false_alarm(ref.STAGES, thumbs, 1.5),
        "floor": tuple(round(g * (1 + NOISE) / (1 - NOISE), 3) for g in (1.16, 1.2)),
        "catch_25": tuple(
            fired(ref, {}, 1.25, m, seed=1) / PAIRS for m in (True, False)
        ),
        "restored": ref.GATES["cost_per_req"] == 1.2,
    }


def verify(result):
    r, (t, e) = result, result["rates"]
    alarms = {1.10: 2304, 1.14: 100, 1.16: 0}
    return [
        practice.Check(
            "ANSWER: 1.16x for latency, cost and length; count-based rate gates",
            all([r["bound"] == 1.1505, r["alarms"] == alarms, r["lesson_alarms"] == 0])
            and (t, e) == ([3.333, 1.667, 1.387, 1.193], [4.0, 1.8, 1.48, 1.24]),
            f"clean ratio bound {r['bound']}; false alarms per {PAIRS} clean pairs "
            f"{r['alarms']}, lesson gates {r['lesson_alarms']}; <=1% rate gates at "
            f"n={list(SIZES)}: thumbs-down {t}, errors {e}",
        ),
        practice.Check(
            "FINDING: the lesson's rate gates false-alarm at 1% traffic",
            r["fa_1pct"] == (0.189, 0.078) and r["rollout_fa"] == 0.215,
            f"at 50 requests the 1.5x thumbs-down and 2x error gates fire on a clean "
            f"model {r['fa_1pct']}; thumbs-down halts {r['rollout_fa']:.1%} of clean "
            "six-stage rollouts",
        ),
        practice.Check(
            "FINDING: the noise floor costs detection",
            (r["floor"], r["catch_25"]) == ((1.335, 1.381), (0.7494, 0.7779))
            and r["restored"],
            f"guaranteed catch above {r['floor']} for gates 1.16 / 1.2; a 25% cost "
            f"regression is caught {r['catch_25']} per stage (measured / fixed baseline)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
