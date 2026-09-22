"""Exercise 1 — trace_count counts spans and the judge runs twice.

    Export a week of OTel traces to Langfuse cloud (free tier). Which sessions
    failed? Why?

Reading of the exercise: the export is a network step and the question after
it is not, so what is built here is the week -- 140 sessions across 7 days,
with the failure mixes a production agent actually has -- and it is pushed
through the lesson's own `TraceCollector` and judge. The answer to "which
failed and why" is a table; the interesting part is what the table gets
wrong.

**ANSWER: 40 of 140 sessions fail and 40 more warn, on three reasons.**
Sorting by `eval_score_mean` puts **40** sessions at **0.20** (FAIL), **40**
at **0.50** (WARN) and **60** at **1.00** (PASS). Every failure carries an
`error.reason`: `rate_limited` **14** times, `tool_timeout` **13** and
`context_overflow` **13**.

**FINDING: `trace_count` is a span count.** The field is named for traces and
holds `len(spans)`, so a session with **4** spans under **1** `trace_id`
reports **4**. Across the week the collector sees **400** spans under **140**
trace ids, and every per-trace rate computed from this field is wrong by the
average fan-out -- here **2.86x**.

**FINDING: the judge runs twice per session.** `summarize` calls
`scripted_llm_judge` and `main` calls it again to recover the verdict
`summarize` discarded -- **2** call sites in the module for **1** decision.
With the scripted judge that is free; with a real one it is **280** judge
calls for **140** sessions, and the lesson's own pitfall list already says
tracing without evaluation is expensive logging.

**FINDING: a session is scored as a unit, so length does not weigh.** The
judge takes the whole span list and returns one number, so a **2**-span
session that failed and a **12**-span session that failed score identically
at **0.20**. Sorting by score puts them adjacent, and the **10** extra spans
of work lost in the second one are invisible in the ranking.

Structure: `week()` seeds the traces; `report()` runs the shipped collector
and judge over them.
"""

from __future__ import annotations

import ast
import inspect
from collections import Counter

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "24-agent-observability-platforms"
DAYS, PER_DAY = 7, 20
REASONS = ("rate_limited", "tool_timeout", "context_overflow")


def session_spans(ref, day, index):
    """One session: a healthy shape, a failing one, or a long-context one."""
    sid, tid = f"s{day:02d}{index:02d}", f"t{day:02d}{index:02d}"
    spans = [ref.SpanEvent(tid, sid, "invoke_agent",
                           attributes={"gen_ai.provider.name": "anthropic"})]
    bucket = (day * PER_DAY + index) % 7
    if bucket < 2:
        reason = REASONS[(day + index) % 3]
        spans.append(ref.SpanEvent(tid, sid, "chat", status="error",
                                   attributes={"error.reason": reason,
                                               "tokens": 0}))
        return spans
    if bucket < 4:
        spans.append(ref.SpanEvent(tid, sid, "chat",
                                   attributes={"tokens": 2500}))
        return spans
    spans += [
        ref.SpanEvent(tid, sid, "chat",
                      attributes={"gen_ai.output.reference_id": f"c{index}",
                                  "tokens": 800}),
        ref.SpanEvent(tid, sid, "tool_call search_tool",
                      attributes={"gen_ai.tool.name": "search_tool"}),
        ref.SpanEvent(tid, sid, "chat",
                      attributes={"gen_ai.output.reference_id": f"d{index}",
                                  "tokens": 400}),
    ]
    return spans


def week(ref):
    collector = ref.TraceCollector()
    for day in range(DAYS):
        for index in range(PER_DAY):
            for span in session_spans(ref, day, index):
                collector.ingest(span)
    return collector


def report(ref, collector):
    summaries, buckets, reasons = ref.summarize(collector), Counter(), Counter()
    for summary in summaries:
        spans = collector.by_session()[summary.session_id]
        buckets[ref.scripted_llm_judge(spans)[1]] += 1
        reasons.update(summary.failure_reasons)
    return {"summaries": summaries, "verdicts": dict(buckets),
            "reasons": dict(reasons)}


def judge_call_sites(ref):
    tree = ast.parse(inspect.getsource(ref))
    return sum(isinstance(node, ast.Call) and getattr(node.func, "id", "")
               == "scripted_llm_judge" for node in ast.walk(tree))


def long_failure(ref, collector):
    """A failing session with ten extra spans, scored against a short one."""
    sid, tid = "s_long", "t_long"
    spans = [ref.SpanEvent(tid, sid, "invoke_agent", attributes={})]
    spans += [ref.SpanEvent(tid, sid, "chat", attributes={"tokens": 100})
              for _ in range(10)]
    spans.append(ref.SpanEvent(tid, sid, "chat", status="error",
                               attributes={"error.reason": "tool_timeout"}))
    for span in spans:
        collector.ingest(span)
    return len(spans), ref.scripted_llm_judge(spans)[0]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    collector = week(ref)
    out = report(ref, collector)
    summaries = out["summaries"]
    by_session = collector.by_session()
    spans, traces = len(collector.spans), len({s.trace_id for s in collector.spans})
    short = next(s for s in summaries if round(s.eval_score_mean, 2) == 0.2)
    long_spans, long_score = long_failure(ref, collector)
    return {
        "sessions": len(summaries), "verdicts": out["verdicts"],
        "reasons": out["reasons"],
        "scores": sorted({round(s.eval_score_mean, 2) for s in summaries}),
        "spans": spans, "traces": traces, "fanout": round(spans / traces, 2),
        "first_session": (summaries[0].trace_count,
                          len({s.trace_id for s in by_session[summaries[0].session_id]})),
        "healthy_spans": max(s.trace_count for s in summaries),
        "call_sites": judge_call_sites(ref),
        "judge_calls": 2 * len(summaries),
        "short": (short.trace_count, round(short.eval_score_mean, 2)),
        "long": (long_spans, round(long_score, 2)),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 40 of 140 sessions fail and 40 warn, on three reasons",
            all([result["sessions"] == 140,
                 result["verdicts"] == {"FAIL": 40, "WARN": 40, "PASS": 60},
                 result["scores"] == [0.2, 0.5, 1.0],
                 result["reasons"] == {"rate_limited": 14, "tool_timeout": 13,
                                       "context_overflow": 13}]),
            f"over {result['sessions']} sessions the judge returns {result['verdicts']} "
            f"at scores {result['scores']}; failures split {result['reasons']}",
        ),
        practice.Check(
            "FINDING: trace_count is a span count",
            all([result["first_session"][1] == 1, result["spans"] == 400,
                 result["traces"] == 140, result["fanout"] == 2.86,
                 result["healthy_spans"] == 4]),
            f"the healthiest session reports {result['healthy_spans']} for one trace id, "
            f"and the week is {result['spans']} spans under {result['traces']} traces -- "
            f"any per-trace rate from this field is off by {result['fanout']}x",
        ),
        practice.Check(
            "FINDING: the judge runs twice per session",
            all([result["call_sites"] == 2, result["judge_calls"] == 280,
                 result["sessions"] == 140]),
            f"main calls scripted_llm_judge again to recover the verdict summarize "
            f"discarded -- {result['call_sites']} call sites for one decision, "
            f"{result['judge_calls']} calls for {result['sessions']} sessions",
        ),
        practice.Check(
            "FINDING: a session is scored as a unit, so length does not weigh",
            all([result["short"] == (2, 0.2), result["long"] == (12, 0.2),
                 result["short"][1] == result["long"][1]]),
            f"a {result['short'][0]}-span failed session and a {result['long'][0]}-span "
            f"one both score {result['long'][1]}, so the ten extra spans of wasted work "
            "are invisible in the ranking",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
