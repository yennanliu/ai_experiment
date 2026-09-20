"""Exercise 2 — truncation is not redaction, and the completion is a literal.

    Turn on content capture (env var) and confirm `gen_ai.content.prompt` and
    `gen_ai.content.completion` events appear. Note the implications for PII.

Reading of the exercise: the events appearing is one assertion, so the PII
half is answered by reading what they actually contain. One of the two
carries a slice of the caller's prompt and the other carries a string
constant, so "content capture" captures one direction of the conversation
and a placeholder for the other -- which changes both the privacy question
and what the feature is worth.

**ANSWER: the events appear, 2 per `llm.chat` span and 4 in the trace.**
With `OTEL_CAPTURE_CONTENT=1` each of the **2** `llm.chat` spans gains
`gen_ai.content.prompt` and `gen_ai.content.completion`; with capture off
the trace carries **0** events and is otherwise identical -- same **9**
spans, same attributes.

**PII IMPLICATION: truncation is not redaction.** The prompt is stored as
`prompt[:200]` -- a length cap, applied before anything inspects it. A
prompt whose first 200 characters contain an email address stores the email
address; one whose PII sits at character 400 is protected by accident. The
control is a slice, and a slice is not a policy.

**FINDING: the completion event does not carry the completion.** Both
`gen_ai.content.completion` events hold the literal `"sample completion"`,
which is `fake_llm_call`'s return value and is identical for both prompts.
Enabling capture therefore records **1** distinct model output across **2**
calls -- the privacy cost of the model's replies is paid in full by a field
that does not yet contain them.

**FINDING: the switch is read once, at import.** `CAPTURE_CONTENT` is a
module constant assigned from `os.environ` at import time, so flipping the
variable later changes nothing and the setting cannot vary per request, per
tenant or per span. Content capture is a property of the process.

**FINDING: the events ride the same span as everything else.** They are
appended to `span.events`, which `to_otlp` emits alongside the attributes --
so there is no separate sink, no separate sampling and no separate retention
for content. Turning it on is all-or-nothing for whoever receives the trace.

Structure: `trace_with` runs one loop under a given capture setting by
assigning the module constant, since the environment is only read at import.
"""

from __future__ import annotations

import contextlib
import inspect
import io

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "20-opentelemetry-genai"
LIMIT = 200


def trace_with(ref, capture):
    """One agent loop under a capture setting, over a cleared exporter."""
    previous = ref.CAPTURE_CONTENT
    ref.CAPTURE_CONTENT = capture
    ref.SPANS.clear()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ref.agent_loop()
        return list(ref.SPANS)
    finally:
        ref.CAPTURE_CONTENT = previous


def events(spans):
    return [event for span in spans for event in span.events]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    off = trace_with(ref, False)
    on = trace_with(ref, True)
    captured = events(on)
    source = inspect.getsource(ref)
    long_prompt = "x" * 500
    span = ref.Span(name="probe", kind="CLIENT", trace_id="t", span_id="s")
    ref.CAPTURE_CONTENT = True
    try:
        ref.fake_llm_call(span, long_prompt)
    finally:
        ref.CAPTURE_CONTENT = False
    stored = next(e["attrs"]["content"] for e in span.events if e["name"].endswith("prompt"))
    return {
        "off_events": len(events(off)), "on_events": len(captured),
        "off_spans": len(off), "on_spans": len(on),
        "names": sorted({event["name"] for event in captured}),
        "per_llm_span": [len(s.events) for s in on if s.name == "llm.chat"],
        "completions": sorted({e["attrs"]["content"] for e in captured
                               if e["name"].endswith("completion")}),
        "prompts": len({e["attrs"]["content"] for e in captured
                        if e["name"].endswith("prompt")}),
        "stored_len": len(stored), "sent_len": len(long_prompt),
        "module_constant": "CAPTURE_CONTENT = os.environ" in source,
        "attrs_match": ([sorted(a.attrs) for a in off] == [sorted(b.attrs) for b in on]),
        "otlp_carries_events": "events" in ref.Span(
            name="n", kind="INTERNAL", trace_id="t", span_id="s").to_otlp(),
        "limit": LIMIT,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the events appear, two per llm.chat span and four in the trace",
            all([result["off_events"] == 0, result["on_events"] == 4,
                 result["names"] == ["gen_ai.content.completion", "gen_ai.content.prompt"],
                 result["per_llm_span"] == [2, 2],
                 result["off_spans"] == result["on_spans"] == 9,
                 result["attrs_match"]]),
            f"with capture on each llm.chat span gains {result['per_llm_span'][0]} events, "
            f"{result['names']}, for {result['on_events']} in the trace; with it off the "
            f"trace carries {result['off_events']} and is otherwise identical -- same "
            f"{result['on_spans']} spans, same attributes",
        ),
        practice.Check(
            "PII: truncation is not redaction",
            all([result["stored_len"] == result["limit"],
                 result["sent_len"] > result["limit"]]),
            f"a {result['sent_len']}-character prompt is stored as its first "
            f"{result['stored_len']} -- a length cap applied before anything inspects it. "
            f"PII in the first {result['limit']} characters is captured and PII at "
            "character 400 is protected by accident. The control is a slice, and a slice is "
            "not a policy",
        ),
        practice.Check(
            "FINDING: the completion event does not carry the completion",
            all([result["completions"] == ["sample completion"], result["prompts"] == 2]),
            f"both completion events hold {result['completions']} -- fake_llm_call's return "
            f"value, identical for both prompts -- against {result['prompts']} distinct "
            "prompts. Capture records one model output across two calls: the privacy cost of "
            "the replies is paid in full by a field that does not yet contain them",
        ),
        practice.Check(
            "FINDING: the switch is read once at import, and content rides the same span",
            all([result["module_constant"], result["otlp_carries_events"]]),
            "CAPTURE_CONTENT is assigned from os.environ at import, so the setting cannot "
            "vary per request, per tenant or per span -- it is a property of the process. "
            "And the events are appended to span.events, which to_otlp emits alongside the "
            "attributes: no separate sink, sampling or retention for content",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
