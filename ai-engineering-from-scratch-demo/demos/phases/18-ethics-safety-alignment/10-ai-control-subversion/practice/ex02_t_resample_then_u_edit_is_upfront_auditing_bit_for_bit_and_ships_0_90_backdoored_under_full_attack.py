"""Exercise 2 — T-resample-then-U-edit is upfront auditing bit for bit, and ships 0.90 backdoored under full attack against Trusted Editing's 0.20.

    Add a new protocol: "T resamples with low temperature, then U edits T's
    output." Measure its position on the frontier. Why does this fail more
    catastrophically than Trusted Editing?

Reading of the exercise: the protocol is built from the reference's own
parts. T resampling at low temperature is a deterministic draft: it spends no
randomness and cannot be backdoored. U's edit is `u_produce`, because the
edit is where a backdoor gets in. Nothing reviews the output after U, so the
only remaining defence is H, spent the way the reference spends budget with
no flag to route it: the first H x n outputs are audited. Its position is
measured on the same seeded 20,000-task runs as the reference protocols, and
"catastrophically" by sweeping the red team's backdoor rate from 0.05 to 1.0
at H = 0.10.

**ANSWER: it lands exactly on upfront auditing: (0.899 useful, 0.090
backdoor) at backdoor rate 0.10.** Seed for seed, it prints the same line as
run_protocol("upfront") at every backdoor rate tried. T's draft adds nothing
the protocol can use: whatever T wrote, U's edit decides what ships. On the
frontier it is dominated by Trusted Editing (0.970 / 0.000) and trades
usefulness for safety against Untrusted Monitoring at c = 0.3 (0.853 /
0.048).

**FINDING: its failure scales one-for-one with the attack; Trusted Editing's
does not start until T and H are both saturated.** Shipped backdoors, H = 0.10:

| backdoor rate | T-resample, U-edit | Trusted Editing |
|---:|---:|---:|
| 0.05 | 0.047 | 0.000 |
| 0.10 | 0.090 | 0.000 |
| 0.20 | 0.179 | 0.001 |
| 0.50 | 0.446 | 0.065 |
| 1.00 | 0.900 | 0.197 |

This is why it fails worse. Trusted Editing puts the trusted model *last*: a
backdoor has to get past T (T misses 30%) and then find the H budget spent.
Resample-then-edit puts the untrusted model last, so every backdoor U chooses
to write ships unless H happens to audit that output, and the shipped rate is
within 0.005 of 0.9 x the attack rate at every rate tried. Under full attack Trusted Editing
still ships 0.703 useful outputs; this protocol ships none.

Structure: `resample_then_edit()` is the new protocol as a run_protocol-style
loop over the reference's `u_produce`; `line()` runs any protocol under a
seeded random and returns (useful, backdoor, H used).
"""

from __future__ import annotations

import contextlib
import inspect
import io
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "10-ai-control-subversion"
N, H, RATES = 20000, 0.10, (0.05, 0.10, 0.20, 0.50, 1.00)


def resample_then_edit(ref, backdoor_rate, h_budget, n):
    """T drafts deterministically (temperature 0); U edits last; H audits while budget lasts."""
    useful = shipped = used_h = 0
    for _ in range(n):
        _, is_bd = ref.u_produce(backdoor_rate)  # U's edit: the only step that can plant one
        audited = used_h / n < h_budget
        used_h += audited
        useful += not is_bd
        shipped += is_bd and not audited
    print(f"resample_edit  useful={useful / n:.3f}  backdoor={shipped / n:.3f}  H used={used_h / n:.3f}")


def line(ref, call, seed=0):
    saved, out = ref.random, io.StringIO()
    ref.random = random.Random(seed)
    try:
        with contextlib.redirect_stdout(out):
            call()
    finally:
        ref.random = saved
    return tuple(map(float, re.findall(r"=([\d.]+)", out.getvalue())))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sweep = {}
    for br in RATES:
        sweep[br] = {
            "new": line(ref, lambda: resample_then_edit(ref, br, H, N)),
            "upfront": line(ref, lambda: ref.run_protocol("upfront", br, H, N)),
            "te": line(ref, lambda: ref.run_protocol("trusted_ed", br, H, N)),
        }
    um = line(ref, lambda: ref.run_protocol("untrusted", 0.10, H, N, collusion=0.30))
    defaults = inspect.signature(ref.run_protocol).parameters
    return {
        "sweep": sweep, "um": um, "t_skill": defaults["t_skill"].default,
        "same_as_upfront": all(row["new"] == row["upfront"] for row in sweep.values()),
        "worst_slope_error": round(max(abs(r["new"][1] - (1 - H) * br) for br, r in sweep.items()), 3),
    }


def verify(result):
    sweep, um = result["sweep"], result["um"]
    new, te = ({br: sweep[br][key][1] for br in RATES} for key in ("new", "te"))
    at10, full = sweep[0.10], sweep[1.0]
    return [
        practice.Check(
            "ANSWER: it lands exactly on upfront auditing",
            all([result["same_as_upfront"], at10["new"][:2] == (0.899, 0.09),
                 at10["te"][:2] == (0.97, 0.0), um[:2] == (0.853, 0.048)]),
            f"at backdoor rate 0.10: new {at10['new']}, upfront {at10['upfront']}, "
            f"Trusted Editing {at10['te']}, Untrusted Monitoring c = 0.3 {um}",
        ),
        practice.Check(
            "FINDING: its failure scales one-for-one with the attack; Trusted Editing's does not",
            all([new == {0.05: 0.047, 0.1: 0.09, 0.2: 0.179, 0.5: 0.446, 1.0: 0.9},
                 te == {0.05: 0.0, 0.1: 0.0, 0.2: 0.001, 0.5: 0.065, 1.0: 0.197},
                 (full["te"][0], full["new"][0]) == (0.703, 0.0),
                 result["worst_slope_error"] <= 0.005, result["t_skill"] == 0.7]),
            f"shipped backdoors by attack rate, new {new}, Trusted Editing {te}; useful "
            f"under full attack {full['new'][0]} vs {full['te'][0]}; worst gap to 0.9 x attack "
            f"{result['worst_slope_error']}; T removes a backdoor with p = {result['t_skill']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
