"""Exercise 5 — the 2024 column holds 0 numbers, and a 30-target campaign cannot tell 4 successes from 15, so falsify on gates, not rates.

    Pick one of the four domains and write a one-paragraph 2027 forecast
    based on the 2024-2025 trajectory. Identify the evidence that would
    falsify your forecast.

Reading of the exercise: the domain is cyber, the one row with a number
that is not an uplift ratio. The forecast is held to the lesson's own skill
file, which refuses numeric uplift forecasts. So it forecasts the human
role, which a lab report can observe directly, not the uplift. Falsifiers
are kept only if one report could show them. Success rate is tested for
that with exact Clopper-Pearson intervals at the November 2025 scale:
roughly 30 targets, "a handful" of validated intrusions, read as 3-6.

**ANSWER (forecast).** By the end of 2027, a frontier lab will publish an
AI-orchestrated intrusion case in which the model checks its own claimed
results. The report will no longer name fabricated or overstated findings
as an obstacle to autonomy, and the human share of effort will fall below
the 2025 report's 10% floor. The authorization gates (exploitation,
sensitive access, exfiltration) will still be human, because an operator
keeps those by choice (Exercise 1). **Falsified by:** a 2027 major-lab
incident report that still requires human validation of claimed results;
a reported human share of 10% or more; or a report in which the
authorization gates are gone while validation remains.

**FINDING: the lesson's table gives one numeric point per trajectory.**
The 2024 column of `DOMAINS` holds 0 numbers across its 4 rows. The 2025
column holds 2: bio's 2.53x and cyber's 80-90%. Each "2024-2025 trajectory"
is a word followed by one number, and for chem and nuclear it is two
words.

**FINDING: success rate cannot falsify a forecast at this scale.** For 3-6
successes out of 30, the 95% intervals run from 0.021-0.265 to
0.077-0.386. At 4 of 30 the interval is 0.038-0.307, and a later campaign
needs 15 of 30 before its interval clears that, a 3.75x jump. A forecast
of "more successful campaigns" is therefore one that no single report
could falsify. Gate counts and the human share can.

Structure: `interval()` is an exact Clopper-Pearson by bisection over the
binomial tail; `numbers()` counts non-year numbers in a `DOMAINS` column.
"""

from __future__ import annotations

import math
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "30-dual-use-risk-cyber-bio-chem-nuclear"
TARGETS, HANDFUL, BASE = 30, range(3, 7), 4


def tail(k, n, p, upper):
    """P(X >= k) if upper else P(X <= k), X ~ Binomial(n, p)."""
    ks = range(k, n + 1) if upper else range(k + 1)
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in ks)


def bound(k, n, upper, alpha=0.05):
    """Exact bound: P(X <= k | p) = alpha/2 for the upper, P(X >= k | p) = alpha/2 for the lower."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        above = tail(k, n, mid, upper=not upper) > alpha / 2
        # P(X >= k) rises with p, P(X <= k) falls with p
        lo, hi = (mid, hi) if above == upper else (lo, mid)
    return round((lo + hi) / 2, 3)


def interval(k, n=TARGETS):
    return bound(k, n, upper=False), bound(k, n, upper=True)


def numbers(text):
    return [
        x
        for x in re.findall(r"(?<!ASL-)\d+(?:\.\d+)?", text)
        if not re.fullmatch(r"20\d\d", x)
    ]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    skill = (
        parity.lesson_dir(PHASE, LESSON) / "outputs" / "skill-dual-use-triage.md"
    ).read_text()
    base_hi = interval(BASE)[1]
    clears = next(k for k in range(BASE, TARGETS + 1) if interval(k)[0] > base_hi)
    return {
        "per_column": {
            col: [n for d in ref.DOMAINS for n in numbers(d[col])]
            for col in ("2024_state", "2025_state")
        },
        "refuses_numeric": "numeric uplift forecast, refuse" in skill,
        "intervals": {k: interval(k) for k in HANDFUL},
        "clears": clears,
    }


def verify(result):
    r, iv = result, result["intervals"]
    return [
        practice.Check(
            "ANSWER: the forecast is on the human role, since the skill file refuses uplift numbers",
            r["refuses_numeric"],
            "skill-dual-use-triage.md: 'If the user asks for a numeric uplift forecast, refuse'",
        ),
        practice.Check(
            "FINDING: the 2024 column holds 0 numbers, the 2025 column 2",
            r["per_column"] == {"2024_state": [], "2025_state": ["2.53", "80", "90"]},
            f"non-year numbers per column {r['per_column']}",
        ),
        practice.Check(
            "FINDING: 4 of 30 is 0.038-0.307, and a rerun needs 15 of 30 to clear it",
            iv[3] == (0.021, 0.265)
            and iv[4] == (0.038, 0.307)
            and iv[6] == (0.077, 0.386)
            and r["clears"] == 15,
            f"95% Clopper-Pearson by successes of {TARGETS}: {iv}; first k clearing 4/30: {r['clears']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
