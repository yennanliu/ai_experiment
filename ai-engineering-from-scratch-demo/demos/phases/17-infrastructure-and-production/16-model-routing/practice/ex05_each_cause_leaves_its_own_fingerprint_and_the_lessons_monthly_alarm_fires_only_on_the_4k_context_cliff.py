"""Exercise 5 — each cause leaves its own fingerprint, and the lesson's monthly alarm fires only on the 4K context cliff.

    Over six months, escalation rate climbs from 8% to 22%. Diagnose three
    causes and the fix for each.

Reading of the exercise: the escalation log is the reference cascade's rule
(simple kept, medium escalated on a 50% coin, hard always) run over 1000
requests with the reference workload's token ranges, priced with its
`cost_of`. Month 0 is a mix of 88/8/4 simple/medium/hard, which escalates
exactly 8.0%. Three separate causes are each tuned to land near 22%, and a
fix is priced for each. A diagnosis rule is then asked to tell them apart
from the log alone.

**ANSWER: the traffic got harder, the cheap model got worse, or the prompts
got longer.**

| cause | month 6 | fix | after |
|---|---:|---|---:|
| mix drifts to 68/16/16 | 21.7%, $5.37 | pre-route hard queries straight to frontier | 8.1%, $5.06 |
| cheap model regresses (medium always, simple 11%) | 22.1%, $2.52 | pin the model version, roll back, recalibrate | 8.0%, $1.76 |
| +3090 tokens of retrieved context; cheap gives up past 4K | 21.9%, $5.42 | length pre-route: >4K prompts to frontier | 0.4%, $5.09 |

In the first case escalation is doing its job and quality holds. The waste
is the doomed cheap attempt on every hard query, $0.31 here. In the second
the bill rises 43% on unchanged traffic.

**FINDING: the log alone separates the three.** Re-weight month 0's
per-difficulty rates by the new mix. If that explains the climb, the cause is
mix. If not, the excess escalations sit on long prompts (length) or on
short ones (model). This rule names all three correctly.

**FINDING: the lesson's alarms stay silent on two causes, and the monthly
one fires on the third only at the 4K cliff.** The lesson flags a cascade
"kicking up-route >30%", and the Ship It plan alerts when escalation
"climbs >10 points in a month". Drive each cause linearly from month 0 to
month 6 and read the monthly rates. Mix drift climbs 8.0 -> 21.7% with no
month over 2.9 points; the model regression climbs 8.0 -> 22.1% with no
month over 3.4 points. The context growth sits near 8-11% for five months,
then jumps 11.1 points in month 6 when simple prompts cross 4K, so only that
step trips the monthly alert. The 30% alarm never fires, and the regression
case raises the bill 43% without either alarm. The gate has to be anchored
to a baseline (here +14 points since month 0), not to a monthly step.

Structure: `workload()` draws with fixed seeds; `cascade()` returns rate, cost
and the (difficulty, long prompt, escalated) log; `diagnose()` reads only
that log; `monthly()` drives each cause linearly over six months.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "16-model-routing"
RANGES = {
    "simple": ((200, 1000), (50, 200)),
    "medium": ((800, 3000), (100, 400)),
    "hard": ((2000, 8000), (200, 1500)),
}
SHIPPED = {"simple": 0.0, "medium": 0.5, "hard": 1.0}  # the reference cascade's rule
MONTH0, DRIFTED = (0.88, 0.08, 0.04), (0.68, 0.16, 0.16)
REGRESSED = {"simple": 0.11, "medium": 1.0, "hard": 1.0}
LONG = 4000  # the lesson's ">4K tokens"
CONTEXT = 3090  # retrieved-context tokens added to every prompt


def workload(ref, mix, context=0, n=1000, seed=7):
    rng, out = random.Random(seed), []
    for _ in range(n):
        u = rng.random()
        kind = "simple" if u < mix[0] else "medium" if u < mix[0] + mix[1] else "hard"
        (p0, p1), (o0, o1) = RANGES[kind]
        out.append(ref.Query(kind, rng.randint(p0, p1) + context, rng.randint(o0, o1)))
    return out


def cascade(ref, reqs, esc=SHIPPED, max_len=None, pre_route=lambda q: False):
    """(escalation rate, cost, log of (difficulty, long prompt, escalated))."""
    rng, cost, log = random.Random(11), 0.0, []
    for q in reqs:
        u = rng.random()
        if pre_route(q):
            cost += ref.cost_of("frontier", q)
            continue
        up = u < esc[q.difficulty] or (max_len is not None and q.prompt_tokens > max_len)
        cost += ref.cost_of("cheap", q) + (ref.cost_of("frontier", q) if up else 0)
        log.append((q.difficulty, q.prompt_tokens > LONG, up))
    return round(rate([u for *_, u in log]), 3), round(cost, 2), log


def rate(flags):
    return sum(flags) / max(len(flags), 1)


def diagnose(month0_log, log):
    """Is the climb explained by the mix at month-0 rates? If not, where is the excess?"""
    rate0 = {d: rate([u for k, _, u in month0_log if k == d]) for d in SHIPPED}
    excess = {True: 0.0, False: 0.0}
    for kind, long_, up in log:
        excess[long_] += up - rate0[kind]
    if abs(sum(excess.values())) < 0.02 * len(log):
        return "mix"
    return "length" if excess[True] > excess[False] else "model"


def monthly(ref, cause):
    """Escalation rate at months 0..6 as the cause moves linearly to its month-6 value."""
    def month(f):
        if cause == "mix":
            mix = [a + f * (b - a) for a, b in zip(MONTH0, DRIFTED)]
            return cascade(ref, workload(ref, mix))
        if cause == "model":
            esc = {d: v + f * (REGRESSED[d] - v) for d, v in SHIPPED.items()}
            return cascade(ref, workload(ref, MONTH0), esc)
        return cascade(ref, workload(ref, MONTH0, round(f * CONTEXT)), max_len=LONG)

    return [month(t / 6)[0] for t in range(7)]


def alarms(paths):
    """Largest monthly step per cause, the context path's >10-point months, the peak rate."""
    deltas = {k: [y - x for x, y in zip(p, p[1:])] for k, p in paths.items()}
    return {
        "paths": paths,
        "steps": {k: round(max(d), 3) for k, d in deltas.items()},
        "alerts": [d > 0.10 for d in deltas["length"]],
        "peak": max(max(p) for p in paths.values()),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    base = cascade(ref, workload(ref, MONTH0))
    drifted, long_ = workload(ref, DRIFTED), workload(ref, MONTH0, CONTEXT)
    runs = {
        "mix": cascade(ref, drifted),
        "model": cascade(ref, workload(ref, MONTH0), REGRESSED),
        "length": cascade(ref, long_, max_len=LONG),
    }
    fixes = {
        "mix": cascade(ref, drifted, pre_route=lambda q: q.difficulty == "hard"),
        "model": base,
        "length": cascade(ref, long_, max_len=LONG, pre_route=lambda q: q.prompt_tokens > LONG),
    }
    skill = parity.lesson_dir(PHASE, LESSON) / "outputs/skill-router-plan.md"
    return {
        "base": base[:2],
        "runs": {k: r[:2] for k, r in runs.items()},
        "diagnosis": {k: diagnose(base[2], r[2]) for k, r in runs.items()},
        "fixes": {k: f[:2] for k, f in fixes.items()},
        **alarms({k: monthly(ref, k) for k in runs}),
        "bill_rise": round(runs["model"][1] / base[1] - 1, 2),
        "gates": "up-route >30%" in parity.doc_text(PHASE, LESSON)
        and "climbs >10 points in a month" in skill.read_text(),
    }


def verify(result):
    runs, fixes, paths, steps = result["runs"], result["fixes"], result["paths"], result["steps"]
    return [
        practice.Check(
            "ANSWER: the traffic got harder, the cheap model got worse, or the prompts got longer",
            result["base"] == fixes["model"] == (0.08, 1.76)
            and runs == {"mix": (0.217, 5.37), "model": (0.221, 2.52), "length": (0.219, 5.42)}
            and (fixes["mix"], fixes["length"]) == ((0.081, 5.06), (0.004, 5.09)),
            f"month 0 {result['base']}; month 6 (rate, cost) {runs}; after each fix {fixes}; "
            f"the mix fix saves ${runs['mix'][1] - fixes['mix'][1]:.2f} of doomed cheap calls",
        ),
        practice.Check(
            "FINDING: the log alone separates the three",
            result["diagnosis"] == {k: k for k in runs},
            f"diagnose() names {result['diagnosis']}",
        ),
        practice.Check(
            "FINDING: the lesson's alarms stay silent on two causes, and the monthly one "
            "fires on the third only at the 4K cliff",
            result["gates"]
            and steps == {"mix": 0.029, "model": 0.034, "length": 0.111}
            and result["peak"] < 0.30
            and result["alerts"] == [False] * 5 + [True]
            and result["bill_rise"] == 0.43,
            f"monthly escalation paths {paths}; largest monthly step {steps} against the "
            "10-point alert, every peak under the 30% alarm; the regression raises the bill "
            f"{result['bill_rise']:.0%} on unchanged traffic",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
