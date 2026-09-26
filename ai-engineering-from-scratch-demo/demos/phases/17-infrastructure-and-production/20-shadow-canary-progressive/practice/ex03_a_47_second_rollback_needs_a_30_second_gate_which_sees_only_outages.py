"""Exercise 3 — a 47-second rollback needs a 30-second gate, which sees only outages.

    Design a rollback that takes under 60 seconds end-to-end. List the required
    infrastructure.

Reading of the exercise: "end-to-end" is read as from the moment a bad
candidate starts serving users to the moment the last candidate request has
finished and every new request is on the pinned old model -- detection
included, since a fast flip after a slow alarm is not a fast rollback. The
design is a step budget, then replayed request by request on a virtual clock
at the lesson's traffic (1% = 10 req/min, so 1,000 req/min in all).

**ANSWER: 47 seconds, from six pieces of infrastructure.** (1) a gate
evaluator on a 30 s sliding window with (2) automatic rollback -- no human in
the loop; (3) a flag service whose write takes 1 s and (4) pushes to routers
within 5 s (streaming, not polling); (5) routers that drain in-flight
candidate requests under a 10 s timeout; (6) a model registry with pinned
digests and the baseline kept warm at full capacity, so reverting the pin is
a 1 s metadata write and nothing reloads. 30 + 0 + 1 + 5 + 10 + 1 = 47 s.
Replayed at 1,000 req/min with every 100th request on the candidate at the
reference's 1.8x latency (P99 1,608 ms on the shipped seed), the last
candidate request ends at 31.6 s, inside the 46 s the budget allows for
draining, and the pin is reverted by 47 s.

**FINDING: the lesson's own cadence misses the budget by 5x before anything
flips.** Gates checked every 5-15 minutes mean the first check is at 300 s;
the same pipeline on a 5-minute window is 317 s, and the lesson's
redeploy-to-rollback is 3 hours.

**FINDING: a 30-second gate sees only outages.** At 1% a 30 s window holds 5
candidate requests. To keep false alarms under 1% at the 2% baseline error
rate the error gate must want 2 errors of 5 -- a 40% error rate, not the
lesson's 2x -- and it catches a 50% outage 81% of the time. Cost, output
length and thumbs-down (0.15 expected events) cannot be judged in that window,
so the sub-minute path covers hard failures; the five-metric gates stay on the
slow window.

**FINDING: the reference has no rollback to time.** Its ROLLBACK is a
`print`; the module defines no flag, registry, digest or pin.

Structure: `DESIGN` is the step budget; `replay()` routes requests on the
virtual clock and records when the last candidate request ends.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "20-shadow-canary-progressive"
DESIGN = (  # (infrastructure, step, seconds)
    ("gate evaluator, 30 s sliding window", "detect", 30),
    ("auto-rollback policy, no human page", "decide", 0),
    ("flag service write", "flip", 1),
    ("streaming flag push to routers", "propagate", 5),
    ("router drain with 10 s timeout", "drain", 10),
    ("model registry pinned digest, baseline warm", "pin", 1),
)
REQ_PER_S, SHARE, BASE_ERR = 1_000 / 60, 0.01, 0.02


def budget(detect=None):
    return sum(detect if step == "detect" and detect else s for _, step, s in DESIGN)


def replay(latency_s, horizon_s=60.0):
    """End of the last request the router sent to the candidate."""
    steps = {step: s for _, step, s in DESIGN}
    routed_off = steps["detect"] + steps["decide"] + steps["flip"] + steps["propagate"]
    last_end, n = 0.0, 0
    while n / REQ_PER_S < horizon_s:
        start = n / REQ_PER_S
        if n % round(1 / SHARE) == 0 and start < routed_off:
            last_end = max(last_end, start + min(latency_s, steps["drain"]))
        n += 1
    return round(last_end, 1)


def tail(n, p, k):
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def error_gate(window_s):
    n = round(window_s * REQ_PER_S * SHARE)
    k = next(k for k in range(n + 1) if tail(n, BASE_ERR, k) <= 0.01)
    return n, k, round(tail(n, 0.5, k), 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    slow = ref.measure_stage(0.01, ref.Regression(latency_mult=1.8), ref.stage_seed(0))
    latency = slow["latency_p99_ms"] / 1_000
    names = " ".join(dir(ref)).lower()
    source = parity.lesson_dir(PHASE, LESSON) / "code" / "main.py"
    thumbs = 30 * REQ_PER_S * SHARE * ref.BASELINE["thumbs_down_rate"]
    return {
        "fast": (budget(), round(latency, 3), replay(latency)),
        "slow_total": budget(detect=300),
        "gate": (*error_gate(30), round(thumbs, 2)),
        "no_infra": not any(w in names for w in ("flag", "registry", "digest", "pin")),
        "print_only": 'print(f"  → ROLLBACK' in source.read_text(encoding="utf-8"),
    }


def verify(result):
    total, latency, last_end = result["fast"]
    n, k, power, thumbs = result["gate"]
    return [
        practice.Check(
            "ANSWER: 47 seconds, from six pieces of infrastructure",
            result["fast"] == (47, 1.608, 31.6),
            f"budget {[(step, s) for _, step, s in DESIGN]} = {total} s; at {latency} s "
            f"candidate latency the last candidate request ends at {last_end} s",
        ),
        practice.Check(
            "FINDING: the lesson's own cadence misses the budget by 5x",
            result["slow_total"] == 317,
            f"the same pipeline on a 5-minute gate window takes {result['slow_total']} s",
        ),
        practice.Check(
            "FINDING: a 30-second gate sees only outages",
            result["gate"] == (5, 2, 0.8125, 0.15),
            f"{n} candidate requests per 30 s at 1%; a <=1% false-alarm error gate "
            f"needs {k} of {n} errors and catches a 50% outage with p={power}; "
            f"{thumbs} expected thumbs-down events",
        ),
        practice.Check(
            "FINDING: the reference has no rollback to time",
            result["no_infra"] and result["print_only"],
            "ROLLBACK is a print; no flag, registry, digest or pin is in the module",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
