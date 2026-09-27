"""Exercise 4 — honeypots need 32 of the 100 audits to catch 30% collusion, and paid from H they ship more backdoors even at zero collusion.

    Read Greenblatt et al. 2024 Section 7 (collusion). Propose a
    collusion-resistance measure and explain its expected usefulness cost.

Reading of the exercise: the measure proposed is the one the paper leans on
for untrusted monitoring: validate the monitor on honeypots, synthetic
backdoors H has planted, and shut down if it lets too many through. Paraphrase
U's output before the monitor sees it, so the monitor cannot tell a real
backdoor from a planted one. The costs are measured on the reference's own
`untrusted_monitor` and `run_protocol`, at the lesson's backdoor rate 0.10,
H = 0.10 (100 audits per 1000 tasks) and collusion 0.30. The shutdown rule is
a one-sided binomial test that falsely shuts down an honest monitor at most
5% of the time.

**ANSWER: honeypots plus paraphrasing. Its usefulness cost is close to zero;
it is paid in audit budget and monitor recall, and in this toy the audit
cost is larger than the collusion it removes.** The reference's monitor
flags 0.80 of planted backdoors when honest and 0.56 at c = 0.3. The
smallest honeypot set that catches c = 0.3 with 90% power is 32, with a
shutdown threshold of fewer than 22 flags (false shutdown 4.1%). That is 32
of the 100 audits per 1000 tasks.

**FINDING: paid from the H budget, even a perfect deterrent ships more
backdoors.** Suppose the honeypots drive collusion all the way to 0. The
monitor then has 68 audits left, and ships 0.060 backdoors against 0.048
with no measure and 100 audits. Honeypots out of H beat doing nothing only
at 14 or fewer, where the test's power against c = 0.3 is 0.63. The measure
needs its own budget, outside the one it defends.

**FINDING: the toy's usefulness moves the wrong way.** run_protocol counts an
audited benign output as not useful, so moving audits to honeypots *raises*
usefulness (0.853 to 0.872 with 32 honeypots at c = 0). Paraphrasing touches only the copy the
monitor reads, and U's original ships, so the real usefulness cost is just
latency and compute.

**FINDING: the reference's collusion is signal-blind, which is the easy
case.** untrusted_monitor takes no input from the producer, so it colludes
on planted backdoors exactly as often as on real ones, and honeypots see it.
A monitor that colludes only when it sees the producer's signal is honest on
every honeypot. The same test then shuts it down only at its 4.1% false
rate, whatever c is. Paraphrasing is what closes that gap. It pays off while
the paraphrased monitor keeps recall above 0.56, the colluding monitor's
measured catch rate: it can lose 30% of its recall and still win.

Structure: `flag_rate()` samples the reference monitor on planted backdoors;
`shutdown_test()` is the binomial test; `line()` runs run_protocol seeded.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import math
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "10-ai-control-subversion"
N, BR, H, C, ALPHA, POWER = 20000, 0.10, 0.10, 0.30, 0.05, 0.90


def flag_rate(ref, collusion, draws=20000):
    """Share of planted backdoors the reference monitor flags, 2 decimals."""
    saved, ref.random = ref.random, random.Random(3)
    try:
        return round(sum(ref.untrusted_monitor(True, collusion) for _ in range(draws)) / draws, 2)
    finally:
        ref.random = saved


def below(k, p, t):
    """P(Binomial(k, p) < t)."""
    return sum(math.comb(k, i) * p**i * (1 - p) ** (k - i) for i in range(t))


def shutdown_test(k, honest):
    """Largest threshold t with P(honest monitor flags < t of k) <= ALPHA."""
    return max(t for t in range(k + 2) if below(k, honest, t) <= ALPHA)


def line(ref, h, collusion):
    saved, out = ref.random, io.StringIO()
    ref.random = random.Random(0)
    try:
        with contextlib.redirect_stdout(out):
            ref.run_protocol("untrusted", BR, h, N, collusion=collusion)
    finally:
        ref.random = saved
    return tuple(map(float, re.findall(r"=([\d.]+)", out.getvalue())))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    honest, colluding = flag_rate(ref, 0.0), flag_rate(ref, C)
    power = {k: below(k, colluding, shutdown_test(k, honest)) for k in range(1, 200)}
    k90 = min(k for k, p in power.items() if p >= POWER)
    t90 = shutdown_test(k90, honest)
    none = line(ref, H, C)
    paid = {k: line(ref, H - k / 1000, 0.0) for k in range(0, 41)}
    breakeven = max(k for k, r in paid.items() if r[1] <= none[1])
    return {
        "honest": honest, "colluding": colluding, "k90": k90, "t90": t90,
        "false_shutdown": round(below(k90, honest, t90), 3), "power_be": round(power[breakeven], 2),
        "none": none, "perfect": paid[k90], "breakeven": breakeven,
        "monitor_args": list(inspect.signature(ref.untrusted_monitor).parameters),
    }


def verify(r):
    return [
        practice.Check(
            "ANSWER: 32 honeypots (under 22 flags shuts down) catch c = 0.3 with 90% power",
            (r["honest"], r["colluding"], r["k90"], r["t90"], r["false_shutdown"])
            == (0.8, 0.56, 32, 22, 0.041),
            f"planted-backdoor flag rate {r['honest']} honest, {r['colluding']} at c = {C}; "
            f"{r['k90']} honeypots, shut down below {r['t90']}, false shutdown {r['false_shutdown']}",
        ),
        practice.Check(
            "FINDING: paid from the H budget, even a perfect deterrent ships more backdoors",
            r["perfect"][1] > r["none"][1] and (r["none"][1], r["perfect"][1]) == (0.048, 0.06)
            and r["breakeven"] == 14 and r["power_be"] == 0.63,
            f"no measure {r['none']}; c = 0 with {r['k90']} audits spent {r['perfect']}; "
            f"break-even at {r['breakeven']} honeypots, power {r['power_be']}",
        ),
        practice.Check(
            "FINDING: the toy's usefulness moves the wrong way",
            (r["none"][0], r["perfect"][0]) == (0.853, 0.872),
            f"useful {r['none'][0]} with no measure, {r['perfect'][0]} with honeypots at c = 0",
        ),
        practice.Check(
            "FINDING: the reference's collusion is signal-blind, which is the easy case",
            r["monitor_args"] == ["is_backdoor", "collusion"]
            and round(r["colluding"] / r["honest"], 2) == 1 - C,
            f"untrusted_monitor{tuple(r['monitor_args'])}: no producer signal; paraphrase "
            f"break-even recall {r['colluding']} = {1 - r['colluding'] / r['honest']:.0%} below "
            f"{r['honest']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
