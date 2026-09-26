"""Exercise 5 — each cause leaves its own fingerprint, and neither of the lesson's escalation alarms fires on the climb.

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

**FINDING: neither of the lesson's alarms fires on this climb.** The lesson
flags a cascade "kicking up-route >30%", and the Ship It plan alerts when
escalation "climbs >10 points in a month". A steady 8 -> 22% over six months
peaks at 22% and climbs 2.33 points a month, so both stay silent while the
regression case alone raises the bill 43%. The gate has to be anchored to a
baseline (here +14 points since month 0), not to a monthly step.

Structure: `workload()` draws with fixed seeds; `cascade()` returns rate, cost
and the (difficulty, long prompt, escalated) log; `diagnose()` reads only
that log.
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
        up = u < esc[q.difficulty] or (
            max_len is not None and q.prompt_tokens > max_len
        )
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
        "length": cascade(
            ref, long_, max_len=LONG, pre_route=lambda q: q.prompt_tokens > LONG
        ),
    }
    skill = parity.lesson_dir(PHASE, LESSON) / "outputs/skill-router-plan.md"
    return {
        "base": base[:2],
        "runs": {k: r[:2] for k, r in runs.items()},
        "diagnosis": {k: diagnose(base[2], r[2]) for k, r in runs.items()},
        "fixes": {k: f[:2] for k, f in fixes.items()},
        "max_step": round((0.22 - 0.08) / 6, 4),  # a steady climb over six months
        "gates": "up-route >30%" in parity.doc_text(PHASE, LESSON)
        and "climbs >10 points in a month" in skill.read_text(),
    }


def verify(result):
    runs, fixes = result["runs"], result["fixes"]
    return [
        practice.Check(
            "ANSWER: the traffic got harder, the cheap model got worse, or the prompts got longer",
            result["base"] == fixes["model"] == (0.08, 1.76)
            and all(0.215 <= r[0] <= 0.225 for r in runs.values())
            and (fixes["mix"], fixes["length"]) == ((0.081, 5.06), (0.004, 5.09)),
            f"month 0 {result['base']}; month 6 (rate, cost) {runs}; after each fix {fixes}",
        ),
        practice.Check(
            "FINDING: the log alone separates the three",
            result["diagnosis"] == {k: k for k in runs},
            f"diagnose() names {result['diagnosis']}",
        ),
        practice.Check(
            "FINDING: neither of the lesson's alarms fires on this climb",
            result["gates"]
            and result["max_step"] < 0.10
            and max(r[0] for r in runs.values()) < 0.30,
            f"a steady 8 -> 22% climb moves {result['max_step'] * 100:.2f} points a month "
            "against a 10-point monthly alert, and peaks under the 30% over-routing alarm",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
