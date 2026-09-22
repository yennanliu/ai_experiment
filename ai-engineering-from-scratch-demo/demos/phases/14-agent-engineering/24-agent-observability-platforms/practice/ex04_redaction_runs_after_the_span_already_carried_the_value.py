"""Exercise 4 — redaction runs after the span already carried the value.

    Read Opik's guardrail docs. Wire a PII redaction guardrail to one of your
    agent runs.

Reading of the exercise: a redaction guardrail has a placement question
before it has a regex question, and the lesson's collector makes the
placement visible. `TraceCollector.ingest` appends whatever it is handed, so
a guardrail can sit in front of `ingest` or behind it, and only one of those
two positions actually removes anything. Both are built and compared on the
same run.

**ANSWER: a guardrail in front of `ingest` redacts 4 PII kinds and leaves 0
in the store.** Emails, card numbers, phone numbers and national ids are
replaced with `[REDACTED:kind]` across **24** spans in **8** sessions; the
collector holds **0** matches afterwards where the unguarded run holds
**5**, and **5** span values were rewritten. The agent's own behaviour is
unchanged.

**FINDING: redacting after ingest is not redaction.** Mutating
`collector.spans` in place leaves **0** matches in the final list and
**every** value already having been appended, so anything that read the list,
forwarded it or persisted it in between saw the original. The guardrail has
to be a transform on the way in, which is why Opik places guardrails in the
call path rather than in the dashboard.

**FINDING: one value is missed and one is redacted halfway.** Of the **6**
PII values in the fixture, **4** are fully redacted, **1** is missed outright
(`+886 912 345 678`), and **1** comes out as `ada+[REDACTED:email]` -- a
partial redaction that reads as a complete one, which is worse than a miss
because nothing downstream will look twice at it. Reporting **0** remaining
matches is reporting zero matches *of its own regexes*, and a redactor
measured against itself always passes. That is the lesson's "self-rolled
LLM-judge without grounding" pitfall in a different register.

**FINDING: redaction changes the judge's verdict on 0 of 8 sessions.**
Replacing content with `[REDACTED:...]` does not touch `status` or
`gen_ai.output.reference_id`, so `scripted_llm_judge` scores every session
identically before and after -- **8** of **8** unchanged. That is the
argument for reference-based capture from Lesson 23: a judge that reads
structure is unaffected by redaction, and a judge that reads content is
blinded by it.

Structure: `Guarded` wraps `ingest`; `scan()` counts what survives.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "24-agent-observability-platforms"
PATTERNS = {
    "email": re.compile(r"\b[a-z0-9._-]+@[a-z0-9.-]+\.[a-z]{2,}\b"),
    "card": re.compile(r"\b(?:\d{4}[- ]){3}\d{4}\b"),
    "phone": re.compile(r"\b0\d{1,2}-\d{3,4}-\d{4}\b"),
    "national_id": re.compile(r"\b[A-Z]\d{9}\b"),
}
LEAKS = (
    "contact me at ada@example.com about the refund",
    "my card is 4111-1111-1111-1111 and it was charged twice",
    "call 02-2345-6789 between nine and five",
    "my id is A123456789, please look it up",
    "reply to ada+billing@example.com instead",
    "or try +886 912 345 678 if the landline fails",
    "the order shipped on tuesday",
    "thanks for sorting that out",
)
MISSED = re.compile(r"\+\d{3} \d{3} \d{3} \d{3}|\b[a-z0-9._-]+\+[a-z]+@[a-z.]+\b")


def redact(text):
    for kind, pattern in PATTERNS.items():
        text = pattern.sub(f"[REDACTED:{kind}]", text)
    return text


class Guarded:
    """Opik's placement: the guardrail is a transform on the way in."""

    def __init__(self, collector):
        self.collector, self.redacted = collector, 0

    def ingest(self, span):
        span.attributes = {key: redact(v) if isinstance(v, str) else v
                           for key, v in span.attributes.items()}
        self.redacted += any(isinstance(v, str) and "[REDACTED:" in v
                             for v in span.attributes.values())
        self.collector.ingest(span)


def build(ref, texts):
    collector, spans = ref.TraceCollector(), []
    for index, text in enumerate(texts):
        sid, tid = f"s{index:02d}", f"t{index:02d}"
        spans += [
            ref.SpanEvent(tid, sid, "invoke_agent",
                          attributes={"gen_ai.provider.name": "anthropic"}),
            ref.SpanEvent(tid, sid, "chat",
                          attributes={"gen_ai.input.messages": text,
                                      "gen_ai.output.reference_id": f"c{index}",
                                      "tokens": 400}),
            ref.SpanEvent(tid, sid, "tool_call lookup",
                          attributes={"gen_ai.tool.name": "lookup"})]
    return collector, spans



def scan(collector):
    pattern = re.compile("|".join(p.pattern for p in PATTERNS.values()))
    return sum(len(pattern.findall(v)) for span in collector.spans
               for v in span.attributes.values() if isinstance(v, str))


def verdicts(ref, collector):
    return {sid: ref.scripted_llm_judge(spans)[1]
            for sid, spans in collector.by_session().items()}


def run(ref, guard=False, late=False):
    """One ingest pass, with the guardrail in front of ingest, behind it, or absent."""
    collector, spans = build(ref, LEAKS)
    sink = Guarded(collector) if guard else collector
    for span in spans:
        sink.ingest(span)
    if late:
        for span in collector.spans:
            span.attributes = {k: redact(v) if isinstance(v, str) else v
                               for k, v in span.attributes.items()}
    return collector, (sink.redacted if guard else 0), len(spans)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    plain, _, span_count = run(ref)
    guarded, redacted, _ = run(ref, guard=True)
    late, _, appended = run(ref, late=True)
    before = verdicts(ref, plain)
    return {
        "spans": span_count, "sessions": len(before),
        "plain_hits": scan(plain), "guarded_hits": scan(guarded),
        "redacted": redacted, "kinds": len(PATTERNS),
        "late_hits": scan(late), "late_appended": appended,
        "missed": sum(bool(MISSED.search(redact(text))) for text in LEAKS),
        "partial": redact(LEAKS[4]), "pii_values": 6, "fully_redacted": 4,
        "verdict_changes": sum(before[sid] != after for sid, after
                               in verdicts(ref, guarded).items()),
        "verdicts": sorted(set(before.values())),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a guardrail in front of ingest leaves 0 PII in the store",
            all([result["guarded_hits"] == 0, result["plain_hits"] == 5,
                 result["spans"] == 24, result["sessions"] == 8,
                 result["redacted"] == 5, result["kinds"] == 4]),
            f"redacting on the way in covers {result['kinds']} PII kinds across "
            f"{result['spans']} spans, rewriting {result['redacted']} values and leaving "
            f"{result['guarded_hits']} matches against {result['plain_hits']} unguarded",
        ),
        practice.Check(
            "FINDING: redacting after ingest is not redaction",
            all([result["late_hits"] == 0, result["late_appended"] == 24,
                 result["late_hits"] == result["guarded_hits"]]),
            f"mutating collector.spans in place also ends at {result['late_hits']} "
            f"matches, and all {result['late_appended']} spans were appended with their "
            "original values first -- the guardrail has to be in the call path",
        ),
        practice.Check(
            "FINDING: one value is missed and one is redacted halfway",
            all([result["missed"] == 1, result["guarded_hits"] == 0,
                 result["partial"] == "reply to ada+[REDACTED:email] instead",
                 result["fully_redacted"] == 4, result["pii_values"] == 6]),
            f"of {result['pii_values']} PII values, {result['fully_redacted']} are fully "
            f"redacted, {result['missed']} missed, and one comes out as "
            f"{result['partial']!r} -- a partial redaction that reads as a complete one",
        ),
        practice.Check(
            "FINDING: redaction changes the judge's verdict on 0 of 8 sessions",
            all([result["verdict_changes"] == 0, result["sessions"] == 8,
                 result["verdicts"] == ["PASS"]]),
            f"redaction touches neither status nor gen_ai.output.reference_id, so the "
            f"judge returns {result['verdicts']} for all {result['sessions']} sessions "
            "before and after: a structural judge cannot see it, a semantic one is blinded",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
