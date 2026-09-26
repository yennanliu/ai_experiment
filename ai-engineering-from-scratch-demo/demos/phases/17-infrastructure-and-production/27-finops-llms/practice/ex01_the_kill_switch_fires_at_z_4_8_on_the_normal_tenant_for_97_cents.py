"""Exercise 1 — the kill switch fires at z = 4.80 on the normal tenant, for 97 cents.

    Run `code/main.py`. At what z-score does the kill switch fire? How do you
    pick the threshold?

Reading of the exercise: "at what z-score" is read off the shipped run (seed
7, 10 days, threshold 4); "how do you pick" is answered by measuring what each
threshold costs -- steady tenants paused for nothing -- against what it buys --
real spend spikes caught on the day they happen -- over 1000 seeds of the
reference's own traffic model.

**ANSWER: it fires once, at z = 4.80, on tenant_A_normal.** Day 8, on $0.97
of spend against a baseline of $0.60 +- $0.08 -- 1% of that tenant's $100
contract. The tenant the demo calls abusive peaks at z = 1.89 and is never
touched, and the daily spend cap never prints a breach.

**FINDING: the "abusive" tenant is its own baseline, and inside its
contract.** tenant_C_abusive has 25x traffic from day 1, so its history
already contains the abuse and z measures nothing unusual. It averages $15.38
a day against a $20 contract and a $40 cap: by the ladder's own dollar rules
it is not abusive at all. A z-score against a tenant's own history can only
catch a *change*.

**FINDING: pick the threshold from a false-pause budget, and it cannot be
4 alone.** Over 1000 seeds x 3 steady tenants (10 days, armed from day 6):

| z | steady tenants paused | 2x day caught | 3x day caught |
|---:|---:|---:|---:|
| 2 | 30.2% | 64.9% | 81.5% |
| 3 | 11.1% | 55.1% | 85.2% |
| 4 | 5.2% | 42.2% | 80.5% |
| 5 | 1.8% | 30.1% | 72.0% |
| 6 | 0.8% | 19.8% | 62.4% |
| 8 | 0.3% | 8.8% | 42.5% |

With a 5-to-9-day baseline and traffic whose tokens per request is one
Gaussian draw a day, z = 4 pauses one steady tenant in 19 and misses most
doublings. Requiring spend over the contract as well cuts false pauses to
1.5%, all of them tenant C -- the only tenant that ever spends near its
contract. Surprise and dollars are separate tests; a pause needs both.

**FINDING: the ladder's first rung is not implemented.**
`simulate_day` never reads `rate_limit_per_min` or `minute_count`; the
"tighten rate" in the cap-breach message changes nothing either.

Structure: `run()` replays the shipped `main()` loop on fresh tenants with a
seeded `random.Random` swapped in (restored after); `zscores()` recomputes the
reference's z offline so a disarmed run can be graded at every threshold.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import random
import re

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "27-finops-llms"
SEEDS, DAYS, SPIKE_DAY = range(1000), 10, 8
THRESHOLDS, SPIKES = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0), (2.0, 3.0)
FIRED = r"KILL SWITCH\] (\w+): z=([\d.]+) on spend \$([\d.]+)"


def run(ref, seed, kill=4.0, spike=1.0):
    """The shipped main() loop on fresh tenants: (per-tenant daily spend, printed log).

    `spike` multiplies every tenant's traffic from SPIKE_DAY on; kill=inf disarms
    the switch so the full history can be replayed offline.
    """
    saved, log = (ref.TENANTS, ref.random), io.StringIO()
    ref.TENANTS = {n: (ref.TenantPolicy(p.contracted_daily_usd, p.rate_limit_per_min,
                                        kill_z_score=kill), ref.TenantState(), m)
                   for n, (p, _, m) in saved[0].items()}
    ref.random = random.Random(seed)
    try:
        with contextlib.redirect_stdout(log):
            for day in range(1, DAYS + 1):
                factor = spike if day == SPIKE_DAY else 1.0
                ref.TENANTS = {n: (p, s, m * factor) for n, (p, s, m) in ref.TENANTS.items()}
                ref.simulate_day(day, verbose=True)
                for _, s, _ in ref.TENANTS.values():
                    s.daily_history.append(s.spend_today_usd)
                    s.spend_today_usd = s.spend_today_usd if s.paused else 0.0
        return {n: s.daily_history for n, (_, s, _) in ref.TENANTS.items()}, log.getvalue()
    finally:
        ref.TENANTS, ref.random = saved


def zscores(history):
    """The reference's z for every day it is armed (day 6 on), 1-indexed."""
    stats = [(sum(history[:i]) / i, history[:i], i) for i in range(5, len(history))]
    sds = [(m, (sum((x - m) ** 2 for x in b) / (i - 1)) ** 0.5, i) for m, b, i in stats]
    return {i + 1: (history[i] - m) / (sd or 1) for m, sd, i in sds}


def first_fire(history, k, floor=0.0):
    return next((d for d, z in zscores(history).items() if z > k and history[d - 1] > floor), None)


def histories(ref, spike=1.0):
    return [(n, h) for s in SEEDS for n, h in run(ref, s, float("inf"), spike)[0].items()]


def calibrate(steady, spiked):
    """Per threshold: share of steady tenants falsely paused, and of spike days caught."""
    table = {}
    for k in THRESHOLDS:
        false = sum(first_fire(h, k) is not None for _, h in steady) / len(steady)
        caught = [sum(first_fire(h, k) == SPIKE_DAY for _, h in r) / len(r) for r in spiked]
        table[k] = tuple(round(v, 3) for v in (false, *caught))
    return table


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    history, log = run(ref, 7)
    policy = {n: p.contracted_daily_usd for n, (p, _, _) in ref.TENANTS.items()}
    steady, source = histories(ref), inspect.getsource(ref.simulate_day)
    floor = [n for n, h in steady if first_fire(h, 4.0, policy[n]) is not None]
    return {
        "fired": [(n, float(z), float(usd)) for n, z, usd in re.findall(FIRED, log)],
        "policy": policy, "cap_lines": log.count("cap breach"),
        "maxz": {n: round(max(zscores(h).values()), 2) for n, h in history.items()},
        "table": calibrate(steady, [histories(ref, mult) for mult in SPIKES]),
        "floored": (len(floor) / len(steady), sorted(set(floor))),
        "rate_used": [f for f in ("rate_limit_per_min", "minute_count") if f in source],
        "c_mean": round(sum(history["tenant_C_abusive"]) / DAYS, 2),
    }


def verify(result):
    table, fired, maxz, policy = (result[k] for k in ("table", "fired", "maxz", "policy"))
    falses = [v[0] for v in table.values()]
    return [
        practice.Check(
            "ANSWER: it fires once, at z = 4.80, on tenant_A_normal",
            fired == [("tenant_A_normal", 4.8, 0.97)] and maxz["tenant_A_normal"] == 4.8
            and maxz["tenant_C_abusive"] < 2 and result["cap_lines"] == 0,
            f"kill switch {fired}, against a ${policy[fired[0][0]]:.0f} contract; highest z "
            f"per tenant {maxz}; cap breaches printed: {result['cap_lines']}",
        ),
        practice.Check(
            "FINDING: the 'abusive' tenant is its own baseline, and inside its contract",
            result["c_mean"] < policy["tenant_C_abusive"],
            f"tenant C averages ${result['c_mean']} a day against a ${policy['tenant_C_abusive']:.0f} contract",
        ),
        practice.Check(
            "FINDING: pick the threshold from a false-pause budget, and it cannot be 4 alone",
            table[4.0] == (0.052, 0.422, 0.805) and falses == sorted(falses, reverse=True)
            and result["floored"] == (0.015, ["tenant_C_abusive"]),
            f"(false pauses, 2x caught, 3x caught) by z: {table}; z > 4 and spend over "
            f"contract pauses {result['floored'][0]:.1%}, all {result['floored'][1]}",
        ),
        practice.Check(
            "FINDING: the ladder's first rung is not implemented",
            result["rate_used"] == [],
            "simulate_day reads neither rate_limit_per_min nor minute_count",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
