"""Exercise 2 — the shipped judge never reads a word of the output.

    Write an LLM-judge rubric for your domain (factual correctness, tone,
    scope adherence). Test on 50 traces.

Reading of the exercise: all three criteria are properties of the *text*, and
`scripted_llm_judge` reads `status`, `name` and two attribute keys -- it
never opens the content at all. So the rubric is written against the output
text, run on 50 labelled traces, and scored against the shipped judge, which
is the comparison that says what a structural judge can and cannot stand in
for.

**ANSWER: a three-criterion rubric scores 50 traces and disagrees with the
shipped judge on 35 of them.** Factual correctness, tone and scope adherence
each contribute a third; the rubric agrees with the hand labels **40/50**
(precision **0.67**, recall **0.67**) where the shipped judge manages
**15/50** -- and the shipped judge's 15 are the 15 traces that happened to be
good, since it passes all 50. The rubric's own **10** misses are worth
naming: it reads `"the refund window is not 30 days"` as factually correct,
because substring matching cannot see negation, and marks
`"that's wrong -- the warranty is actually 2 years"` as rude. Keyword rules
are a stand-in for a judge, not a judge.

**FINDING: a structurally perfect trace can carry any text at all.**
Replacing every output in a passing session with `"I don't know"` leaves the
shipped score at **1.00**, because `has_final` tests whether the attribute
`gen_ai.output.reference_id` *exists*, not what it points at. The judge is
measuring that instrumentation ran.

**FINDING: the shipped scores cannot reach the bottom of their own range.**
The four penalties sum to **0.9**, so `max(0.0, score)` can never fire and no
session can score below **0.10**. The rubric's range is used end to end --
**0.00** to **1.00** over the same 50 traces -- and the difference matters
because a threshold at 0.1 is unreachable in one scheme and routine in the
other.

**FINDING: the three criteria disagree with each other, which is the point.**
Of the 50 traces, **10** are factually wrong while in scope and in tone, and
**5** state the right fact about the wrong subject. A single blended score
hides all of them: each scores **2/3**, so **15** of **15** specific
failures survive a 0.60 threshold on the mean. The criteria are only useful
reported separately, which is what "write a rubric" means and what a single
`eval_score_mean` field cannot hold.

Structure: `rubric()` is the three-criterion judge; `TRACES` carries the
outputs and the hand labels it is scored against.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "24-agent-observability-platforms"
FACTS = {"refund window": "30 days", "shipping": "3 to 5 business days",
         "warranty": "2 years", "returns address": "Taipei"}
IN_SCOPE = ("refund", "shipping", "warranty", "return", "order", "delivery")
RUDE = ("obviously", "as I already said", "you should have", "that's wrong")


# (answer template, hand label). The labels are judged by a human reading the
# answer, not derived from the rubric -- so the rubric can be, and is, wrong.
TEMPLATES = (
    ("the {t} is {f}", True),
    ("for {t}, it is {f}", True),
    ("the {t} is 99 years", False),
    ("obviously the {t} is {f}", False),
    ("our stock price closed up 2% today", False),
    ("the {t} is not {f}", False),
    ("our stock price is up, unrelated to your order", False),
    ("that's wrong -- the {t} is actually {f}", True),
    ("{f} applies to our quarterly stock price", False),
    ("the stock price is 99 years obviously", False),
)


def make_traces():
    """50 traces: an answer, the topic it is about, and a hand label."""
    topics = list(FACTS)
    return tuple(
        {"topic": topics[i % 4], "label": TEMPLATES[i % 10][1],
         "answer": TEMPLATES[i % 10][0].format(t=topics[i % 4],
                                               f=FACTS[topics[i % 4]])}
        for i in range(50))


TRACES = make_traces()


def rubric(row):
    """Three criteria, each worth a third: is it true, is it civil, is it on topic."""
    answer = row["answer"].lower()
    parts = {"factual": float(FACTS[row["topic"]] in row["answer"]),
             "tone": float(not any(mark in answer for mark in RUDE)),
             "scope": float(any(word in answer for word in IN_SCOPE))}
    return sum(parts.values()) / 3, parts


def healthy_spans(ref, tokens=400):
    """A structurally perfect session: tool call, reference id, no errors."""
    return [ref.SpanEvent("t", "s", "invoke_agent", attributes={}),
            ref.SpanEvent("t", "s", "tool_call lookup", attributes={}),
            ref.SpanEvent("t", "s", "chat",
                          attributes={"gen_ai.output.reference_id": "c1",
                                      "tokens": tokens})]


def shipped_score(ref, _row):
    return ref.scripted_llm_judge(healthy_spans(ref))[0]


def calls(ref):
    return [(rubric(row)[0] >= 0.99, shipped_score(ref, row) >= 0.8, row["label"])
            for row in TRACES]


def agreement(ref):
    rows = calls(ref)
    hits = sum(m and lab for m, _, lab in rows)
    return {"rubric_correct": sum(m == lab for m, _, lab in rows),
            "shipped_correct": sum(t == lab for _, t, lab in rows),
            "disagree": sum(m != t for m, t, _ in rows),
            "precision": round(hits / sum(m for m, _, _ in rows), 2),
            "recall": round(hits / sum(lab for _, _, lab in rows), 2)}


def criteria_split():
    parts = [rubric(row)[1] for row in TRACES]
    wrong = [p for p in parts if not p["factual"] and p["tone"] and p["scope"]]
    scoped = [p for p in parts if p["factual"] and not p["scope"]]
    return {"wrong_only": len(wrong), "scope_only": len(scoped),
            "specific": len(wrong + scoped),
            "survive": sum(sum(p.values()) / 3 > 0.6 for p in wrong + scoped)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    blank, penalties = healthy_spans(ref, tokens=10), (0.4, 0.3, 0.1, 0.1)
    return {
        "traces": len(TRACES), "agreement": agreement(ref),
        "blank_score": round(ref.scripted_llm_judge(blank)[0], 2),
        "penalty_sum": round(sum(penalties), 2),
        "shipped_floor": round(1.0 - sum(penalties), 2),
        "rubric_range": (min(round(rubric(row)[0], 2) for row in TRACES),
                         max(round(rubric(row)[0], 2) for row in TRACES)),
        "split": criteria_split(),
    }


def verify(result):
    agree, split = result["agreement"], result["split"]
    return [
        practice.Check(
            "ANSWER: the rubric scores 50 traces and disagrees with the judge on 35",
            all([result["traces"] == 50, agree["disagree"] == 35,
                 agree["rubric_correct"] == 40, agree["shipped_correct"] == 15,
                 agree["precision"] == 0.67, agree["recall"] == 0.67]),
            f"the rubric agrees with the labels {agree['rubric_correct']}/"
            f"{result['traces']} at precision {agree['precision']}, recall "
            f"{agree['recall']}, where the judge manages {agree['shipped_correct']} by "
            f"passing all 50 -- {agree['disagree']} disagreements",
        ),
        practice.Check(
            "FINDING: a structurally perfect trace can carry any text at all",
            all([result["blank_score"] == 1.0, agree["shipped_correct"] == 15]),
            f"a session with a tool call, a reference id and no errors scores "
            f"{result['blank_score']} whatever the text says: has_final tests that the "
            "attribute exists and never dereferences it",
        ),
        practice.Check(
            "FINDING: the shipped scores cannot reach the bottom of their own range",
            all([result["penalty_sum"] == 0.9, result["shipped_floor"] == 0.1,
                 result["rubric_range"] == (0.0, 1.0)]),
            f"the four penalties sum to {result['penalty_sum']}, so max(0.0, score) never "
            f"fires and nothing scores below {result['shipped_floor']}, where the rubric "
            f"uses {result['rubric_range'][0]} to {result['rubric_range'][1]}",
        ),
        practice.Check(
            "FINDING: the three criteria disagree with each other, which is the point",
            all([split["wrong_only"] == 10, split["scope_only"] == 5,
                 split["specific"] == 15, split["survive"] == 15]),
            f"{split['wrong_only']} traces are factually wrong while in scope and in "
            f"tone, and {split['scope_only']} state the right fact about the wrong "
            f"subject -- all {split['survive']} of {split['specific']} average above 0.60",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
