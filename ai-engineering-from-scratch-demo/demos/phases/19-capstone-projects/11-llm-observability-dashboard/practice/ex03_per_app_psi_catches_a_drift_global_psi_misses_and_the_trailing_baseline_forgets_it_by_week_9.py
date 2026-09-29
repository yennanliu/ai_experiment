"""Exercise 3 — per-app PSI catches a drift global PSI misses, and the trailing baseline forgets it by week 9.

    Sharpen the drift detector: compute PSI per app-id rather than globally. Show per-app drift trails.

Reading of the exercise: the lesson's `synth_trace` builds the spans,
with `app_id` set on the root span and the prompt on the LLM span, as the
lesson has it. Three apps share the traffic, 60 / 25 / 15%, at 500 traces
a week for 10 weeks. From week 6 the small `support` app drifts for good:
90% of its prompts become refund or cancellation requests, against a
uniform mix over 6 intents before. Drift uses the lesson's own
`prompt_fingerprint` and `psi`, with the doc's step 6 rule: each week
against the trailing 4-week baseline, alert at PSI > 0.2. The trail runs
over weeks 5-10, globally and per app, with the app joined from each LLM
span's root span.

**ANSWER: per-app PSI catches the drift in week 6, and global PSI never
does.** Trails, weeks 5-10:

| scope | w5 | w6 | w7 | w8 | w9 | w10 |
|---|---:|---:|---:|---:|---:|---:|
| global | 0.014 | 0.165 | 0.037 | 0.034 | 0.007 | 0.003 |
| chatbot | 0.007 | 0.005 | 0.013 | 0.018 | 0.001 | 0.001 |
| search | 0.026 | 0.049 | 0.016 | 0.035 | 0.013 | 0.011 |
| support | 0.109 | 1.723 | 0.329 | 0.406 | 0.011 | 0.032 |

Global PSI peaks at 0.165 because 85% of the pool is unchanged. `support`
jumps to 1.723 in week 6. Neither stable app alerts in any week.

**FINDING: the doc's trailing baseline learns the drift and stops
alerting while it is still there.** Each drifted week enters the next
week's baseline. By week 9 the baseline is all drifted weeks, and
`support` reads 0.011, then 0.032. Against a frozen weeks 1-4 baseline the
same weeks read 0.494 and 0.885. With a 4-week trailing window, a
permanent shift alerts for three weeks and then disappears from the chart.

**FINDING: the lesson's fingerprint cannot see a shift between two of its
own four prompts.** `prompt_fingerprint` is a SHA-256 byte mod 8, not an
embedding. The 4 prompts fill 3 of 8 bins, and "give me a travel tip for
Tokyo" and "how warm is Tokyo this week" share bin 4. Moving all traffic
from one to the other scores PSI 0.000.

**FINDING: per-app is not measurable on the lesson's data, and its printed
PSI is noise.** `synth_trace` sets `app_id` = "chatbot" on every root span
and on no LLM span, so per-app PSI equals global. `main.py`'s 0.083 compares
two samples of one distribution (151 vs 49 traces). Over 1,000 seeds of
that same null, the median PSI is 0.040 and 31 runs (3.1%) cross the 0.2
alert.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "11-llm-observability-dashboard"
WEEKS, PER_WEEK, SEED = 10, 500, 19
APPS = {"chatbot": 0.60, "search": 0.25, "support": 0.15}
PROMPTS = {
    "chatbot": ["what is the weather in Tokyo today", "summarize the recent Tokyo forecast",
                "give me a travel tip for Tokyo", "how warm is Tokyo this week"],
    "search": ["find hotels near Shinjuku", "search flights to Haneda", "list museums open on Monday",
               "find ramen near Shibuya"],
    "support": ["reset my password", "cancel my booking", "refund my last order", "change my seat",
                "update my email address", "talk to a human agent"],
}
DRIFTED = [0.025, 0.45, 0.45, 0.025, 0.025, 0.025]


def traffic(ref):
    """(week, trace) pairs; support's intent mix moves from week 6."""
    rng, out = random.Random(SEED), []
    for week in range(1, WEEKS + 1):
        for i in range(PER_WEEK):
            app = rng.choices(list(APPS), weights=list(APPS.values()))[0]
            weights = DRIFTED if app == "support" and week >= 6 else None
            trace = ref.synth_trace(f"w{week}t{i:03d}", False, rng)
            trace[0].attributes["app_id"] = app
            trace[1].attributes["prompt"] = rng.choices(PROMPTS[app], weights=weights)[0]
            out.append((week, trace))
    return out


def fingerprints(ref, rows):
    """{(week, app): [bin]} with each LLM span's app taken from its root span."""
    fps = {}
    for week, trace in rows:
        roots = {s.span_id: s for s in trace if s.parent_span_id is None}
        for s in trace:
            if s.is_llm():
                app = roots[s.parent_span_id].attributes["app_id"]
                fps.setdefault((week, app), []).append(ref.prompt_fingerprint(s.attributes["prompt"]))
    return fps


def trail(ref, fps, apps, frozen=False):
    """Weeks 5-8 against the trailing 4 weeks (doc step 6), or against weeks 1-4 when frozen."""
    pick = lambda weeks: [b for w in weeks for a in apps for b in fps.get((w, a), [])]
    return [round(ref.psi(pick(range(1, 5) if frozen else range(w - 4, w)), pick([w])), 3)
            for w in range(5, WEEKS + 1)]


def null_psi(ref, seeds=1000):
    """main.py's windows (151 baseline, 49 current) drawn from one unchanged distribution."""
    vals = []
    for seed in range(seeds):
        rng = random.Random(seed)
        fps = [ref.prompt_fingerprint(rng.choice(PROMPTS["chatbot"])) for _ in range(200)]
        vals.append(ref.psi(fps[:151], fps[151:]))
    return vals


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fps = fingerprints(ref, traffic(ref))
    trails = {"global": trail(ref, fps, list(APPS)), **{a: trail(ref, fps, [a]) for a in APPS}}
    frozen = trail(ref, fps, ["support"], frozen=True)
    bins = [ref.prompt_fingerprint(p) for p in PROMPTS["chatbot"]]
    swap = ref.psi([bins[2]] * 100, [bins[3]] * 100)
    lesson_apps = {s.attributes.get("app_id") for i in range(200)
                   for s in ref.synth_trace(f"t{i}", False, random.Random(i))}
    nulls = null_psi(ref)
    return {"trails": trails, "frozen": frozen, "bins": bins, "swap": round(swap, 6), "lesson_apps": sorted(map(str, lesson_apps)),
            "null_median": round(statistics.median(nulls), 3), "null_alarms": sum(v > 0.2 for v in nulls)}


def verify(result):
    t = result["trails"]
    alerts = {k: [w for w, v in zip(range(5, WEEKS + 1), vals) if v > 0.2] for k, vals in t.items()}
    return [
        practice.Check(
            "ANSWER: per-app PSI flags support from week 6 while global PSI stays under 0.2 in every week",
            alerts == {"global": [], "chatbot": [], "search": [], "support": [6, 7, 8]}
            and t["support"][:4] == [0.109, 1.723, 0.329, 0.406] and max(t["global"]) == 0.165,
            "; ".join(f"{k} {v}" for k, v in t.items()) + f"; weeks over 0.2: {alerts}",
        ),
        practice.Check(
            "FINDING: the trailing 4-week baseline absorbs a permanent drift and goes silent by week 9",
            t["support"][4:] == [0.011, 0.032] and result["frozen"][4:] == [0.494, 0.885]
            and min(result["frozen"][1:]) > 0.2,
            f"support trailing {t['support']}; against frozen weeks 1-4 {result['frozen']}",
        ),
        practice.Check(
            "FINDING: two of the lesson's four prompts share a bin, so a full shift between them scores PSI 0",
            (result["bins"], result["swap"]) == ([6, 7, 4, 4], 0.0),
            f"bins of the 4 lesson prompts {result['bins']}; PSI of all-'travel tip' vs all-'how warm' "
            f"{result['swap']}",
        ),
        practice.Check(
            "FINDING: the lesson's app_id is one constant on root spans, and its null PSI alarms 3.1% of the time",
            (result["lesson_apps"], result["null_median"], result["null_alarms"]) == (["None", "chatbot"], 0.04, 31),
            f"app_id values across lesson spans {result['lesson_apps']}; null PSI median {result['null_median']}, "
            f"{result['null_alarms']}/1000 over 0.2",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
