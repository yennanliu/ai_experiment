"""Exercise 5 — the missing attribute is the one that says it went wrong.

    Read the OTel GenAI semconv spec. Identify one attribute listed in the
    semconv that this lesson's code does NOT emit. Add it.

Reading of the exercise: "one attribute" invites picking any absentee, so the
choice is made by asking which absence changes what a trace can answer.
Request parameters are missing and cost a reader detail; `error.type` is
missing and costs a reader the *question* -- with no error attribute and no
status field, a failed call and a successful one are the same span. That is
the one worth adding.

**ANSWER: `error.type`, and with it a failing call is distinguishable.**
The lesson emits **0** error attributes across **9** spans, so an exception
inside `fake_llm_call` would end a span that looks exactly like a successful
one. Adding `error.type` to the failing span and its ancestors makes **2** of
**9** spans carry it and leaves the trace answerable.

**FINDING: the absence is structural, not an omission in one place.** `Span`
has **9** fields -- name, kind, ids, timestamps, attrs, events -- and **0**
of them is a status. OTel gives a span a `status` alongside its attributes,
so `error.type` here has to live in `attrs` and there is nowhere for the
`UNSET`/`OK`/`ERROR` it normally accompanies. The attribute is addable; the
field it pairs with is not.

**FINDING: the request half of the semconv is missing too, and costs less.**
`gen_ai.request.temperature`, `gen_ai.request.max_tokens` and
`gen_ai.response.finish_reasons` are all absent, while
`gen_ai.request.model`, `gen_ai.response.model` and both token counts are
present. A reader can still answer "what did this cost" and cannot answer
"why did it stop" -- the emitted set is the billing half.

**FINDING: an error must propagate or the root stays green.** Marking only
the span that raised leaves `agent.invoke_agent` unmarked, so a trace-level
query for failures finds **0** of **1** failed runs. Marking the ancestors
too is what makes the root's status mean something -- which is a policy, not
an attribute.

Structure: `failing_loop` reruns the lesson's own agent loop with one call
raising, and `mark` applies `error.type` to a span and its ancestors.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "20-opentelemetry-genai"
ERROR_TYPE = "error.type"
SEMCONV_REQUEST = ("gen_ai.request.temperature", "gen_ai.request.max_tokens",
                   "gen_ai.response.finish_reasons")
SEMCONV_PRESENT = ("gen_ai.request.model", "gen_ai.response.model",
                   "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens")


def mark(spans, span, error_type):
    """Set error.type on a span and every ancestor, which is the policy half."""
    by_id = {item.span_id: item for item in spans}
    current = span
    marked = []
    while current is not None:
        current.attrs[ERROR_TYPE] = error_type
        marked.append(current.name)
        current = by_id.get(current.parent_span_id)
    return marked


def failing_loop(ref):
    """The lesson's loop with the second llm call raising."""
    ref.SPANS.clear()
    root = ref.start_span("agent.invoke_agent", "INTERNAL",
                          attrs={"gen_ai.operation.name": "invoke_agent"})
    llm = ref.start_span("llm.chat", "CLIENT", parent=root,
                         attrs={"gen_ai.operation.name": "chat",
                                "gen_ai.request.model": "gpt-4o"})
    try:
        raise TimeoutError("provider timed out")
    except TimeoutError as exc:
        failure = type(exc).__name__
    llm.finish()
    root.finish()
    return list(ref.SPANS), llm, failure


def carrying(spans):
    return sum(ERROR_TYPE in span.attrs for span in spans)


def root_carries(spans):
    return any(span.parent_span_id is None and ERROR_TYPE in span.attrs for span in spans)


def fields_of(ref, needle=None):
    names = [f.name for f in dataclasses.fields(ref.Span)]
    return names if needle is None else [n for n in names if needle in n.lower()]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.SPANS.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        ref.agent_loop()
    clean = list(ref.SPANS)
    emitted = {key for span in clean for key in span.attrs}

    spans, failing, failure = failing_loop(ref)
    before_marked, root_before = carrying(spans), root_carries(spans)
    marked = mark(spans, failing, failure)
    return {
        "clean_spans": len(clean),
        "error_attrs": sorted(key for key in emitted if "error" in key),
        "span_fields": fields_of(ref), "status_fields": fields_of(ref, "status"),
        "missing_request": sorted(set(SEMCONV_REQUEST) - emitted),
        "present": sorted(set(SEMCONV_PRESENT) & emitted),
        "before_marked": before_marked, "marked": sorted(marked), "failure": failure,
        "after_marked": carrying(spans), "root_marked": root_carries(spans),
        "root_before": root_before, "total": len(spans),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: error.type, and with it a failing call is distinguishable",
            all([result["error_attrs"] == [], result["clean_spans"] == 9,
                 result["before_marked"] == 0, result["after_marked"] == 2,
                 result["failure"] == "TimeoutError",
                 result["marked"] == ["agent.invoke_agent", "llm.chat"]]),
            f"the lesson emits {len(result['error_attrs'])} error attributes across "
            f"{result['clean_spans']} spans, so a failing call ends a span that looks like a "
            f"successful one. Adding error.type={result['failure']!r} marks "
            f"{result['after_marked']} of {result['total']} spans, {result['marked']}",
        ),
        practice.Check(
            "FINDING: the absence is structural, not an omission in one place",
            all([len(result["span_fields"]) == 9, result["status_fields"] == []]),
            f"Span's fields are {result['span_fields']} -- "
            f"{len(result['status_fields'])} of them a status -- so error.type has to live "
            "in attrs and there is nowhere for the UNSET/OK/ERROR it normally accompanies. "
            "The attribute is addable; the field it pairs with is not",
        ),
        practice.Check(
            "FINDING: the request half of the semconv is missing too, and costs less",
            all([result["missing_request"] == sorted(SEMCONV_REQUEST),
                 result["present"] == sorted(SEMCONV_PRESENT)]),
            f"absent: {result['missing_request']}; present: {result['present']}. A reader "
            "can answer what this cost and cannot answer why it stopped -- the emitted set "
            "is the billing half",
        ),
        practice.Check(
            "FINDING: an error must propagate or the root stays green",
            all([not result["root_before"], result["root_marked"],
                 result["after_marked"] == 2]),
            f"marking only the span that raised leaves agent.invoke_agent unmarked, so a "
            f"trace-level query for failures finds none; marking the ancestors gives "
            f"root_marked={result['root_marked']}. What makes a root's status mean something "
            "is a propagation policy, not an attribute",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
