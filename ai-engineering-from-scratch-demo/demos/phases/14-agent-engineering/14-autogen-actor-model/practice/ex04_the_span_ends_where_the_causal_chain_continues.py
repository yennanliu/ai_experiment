"""Exercise 4 — the span ends where the causal chain continues.

    Wire an OTel span per message (or a no-op stand-in). Emit
    `gen_ai.agent.name`, `gen_ai.operation.name` per Lesson 23.

Reading of the exercise: a span needs a parent, and `Message` has **5**
fields with no room for one -- so "a span per message" produces a flat list
unless the runtime threads causality itself. The stand-in is a recorder with
the same shape as an OTel span (name, attributes, parent) and no dependency,
which makes the parenting question the measurable part.

**ANSWER: a span per delivered message, with the two required attributes.**
The demo emits **8** spans, every one carrying `gen_ai.agent.name` and
`gen_ai.operation.name` -- **8** of **8** on both -- with operations
`handle` for delivered messages and `dead_letter` for the one that raised.

**FINDING: without threading, every span is a root.** Spanning each message
as the runtime dequeues it gives **8** spans and **8** roots, because
`Message` carries no parent id. Threading the parent through `Runtime.send`
-- the id of the message being handled when the send happened -- gives **8**
spans and **2** roots, one per externally sent message, in a tree **3**
levels deep.

**FINDING: the span ends before its consequences start.** A handler calls
`runtime.send`, which only appends to the queue, so the child message is
processed long after the parent's span closed: between the `review` span and
its `review_result` child the runtime handles **2** unrelated messages. The
span's duration measures the handler and not the work it caused.

**FINDING: the failing message gets a span and the sender gets nothing.**
The crash is recorded as **1** span with `gen_ai.operation.name` =
`dead_letter`, and the `review` whose reply never came has **0** spans
saying so. Tracing what the runtime *did* is easy; tracing what it stopped
doing needs the sender to be notified, which exercise 1 measured as missing.

Structure: `Tracer` is the no-op stand-in; `traced_run()` is
`Runtime.run_until_idle` with a span opened around each delivery.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "14-autogen-actor-model"
SNIPPETS = ("def add(a, b): return a + b",
            "def hazard(): eval('1+1')",
            "def silent():\n    try:\n        f()\n    except:\n        pass")


class Tracer:
    """A stand-in with an OTel span's shape and no dependency."""

    def __init__(self):
        self.spans = []

    def span(self, name, parent, attributes):
        record = {"name": name, "parent": parent, **attributes}
        self.spans.append(record)
        return record

    def roots(self):
        return [span for span in self.spans if span["parent"] is None]

    def depth(self, span, seen=0):
        parent = next((s for s in self.spans if s["name"] == span["parent"]), None)
        return seen if parent is None else self.depth(parent, seen + 1)


def build(ref, tracer, thread):
    """The lesson's runtime with send recording the message it was caused by."""
    runtime = ref.Runtime()
    runtime.register(ref.ReviewerAgent("reviewer"))
    runtime.register(ref.ChecklistAgent("checklist", partner="reviewer"))
    current = {"mid": None}
    original, parents = runtime.send, {}

    def send(sender, recipient, topic, body):
        original(sender, recipient, topic, body)
        parents[runtime.counter] = current["mid"] if thread else None

    runtime.send = send
    return runtime, current, parents


def traced_run(ref, tracer, thread=True):
    runtime, current, parents = build(ref, tracer, thread)
    runtime.send("__user__", "checklist", "start", list(SNIPPETS))
    runtime.send("__user__", "reviewer", "crash_me", {})
    order = []
    while runtime.queue:
        message = runtime.queue.popleft()
        current["mid"] = message.mid
        parent = parents.get(message.mid)
        actor = runtime.actors.get(message.recipient)
        operation = "handle"
        try:
            actor.receive(message, runtime)
        except Exception:
            operation = "dead_letter"
            runtime.dead_letters.append((message, "raised"))
        tracer.span(f"m{message.mid:03d}", f"m{parent:03d}" if parent else None,
                    {"gen_ai.agent.name": message.recipient,
                     "gen_ai.operation.name": operation})
        order.append((message.mid, message.topic))
    return tracer, order


def span_report(tracer):
    return {
        "spans": len(tracer.spans), "roots": len(tracer.roots()),
        "named": sum("gen_ai.agent.name" in span for span in tracer.spans),
        "operations": sorted({s["gen_ai.operation.name"] for s in tracer.spans}),
        "dead_letter_spans": sum(s["gen_ai.operation.name"] == "dead_letter"
                                 for s in tracer.spans),
        "depth": max(tracer.depth(span) for span in tracer.spans),
    }


def causal_gap(order):
    ids = [mid for mid, _ in order]
    review = next(mid for mid, topic in order if topic == "review")
    reply = next(mid for mid, topic in order if topic == "review_result")
    return ids.index(reply) - ids.index(review) - 1


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    threaded, order = traced_run(ref, Tracer(), thread=True)
    flat, _ = traced_run(ref, Tracer(), thread=False)
    fields = list(ref.Message.__dataclass_fields__)
    return {
        **span_report(threaded), "flat_roots": len(flat.roots()),
        "gap": causal_gap(order), "message_fields": fields,
        "parent_fields": [f for f in fields if "parent" in f or "trace" in f],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 8 spans, both attributes on every one",
            all([result["spans"] == 8, result["named"] == 8,
                 result["operations"] == ["dead_letter", "handle"],
                 result["dead_letter_spans"] == 1]),
            f"the demo emits {result['spans']} spans and {result['named']} of them carry "
            f"gen_ai.agent.name, with operations {result['operations']} -- "
            f"{result['dead_letter_spans']} of them the message that raised",
        ),
        practice.Check(
            "FINDING: without threading, every span is a root",
            all([result["flat_roots"] == 8, result["roots"] == 2,
                 result["depth"] == 2, result["parent_fields"] == []]),
            f"spanning each message as it is dequeued gives {result['flat_roots']} roots, "
            f"because Message carries {len(result['parent_fields'])} parent fields. "
            f"Threading the parent through send gives {result['roots']} roots -- one "
            f"per externally sent message -- and a tree {result['depth'] + 1} levels "
            "deep",
        ),
        practice.Check(
            "FINDING: the span ends before its consequences start",
            all([result["gap"] == 2, result["gap"] > 0]),
            f"a handler's runtime.send only appends to the queue, so the child message "
            f"is processed after {result['gap']} unrelated messages have been handled. "
            "The span's duration measures the handler, not the work it caused -- which "
            "is what a naive latency chart would report",
        ),
        practice.Check(
            "FINDING: the failing message gets a span and the sender gets nothing",
            all([result["dead_letter_spans"] == 1, result["spans"] == 8,
                 result["message_fields"] == ["sender", "recipient", "topic", "body",
                                              "mid"]]),
            f"the crash is {result['dead_letter_spans']} span with operation "
            f"dead_letter, and the request whose reply never came has none saying so. "
            f"Message carries {result['message_fields']}, so a span cannot record a "
            "reply that was never sent -- tracing absence needs the sender told",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
