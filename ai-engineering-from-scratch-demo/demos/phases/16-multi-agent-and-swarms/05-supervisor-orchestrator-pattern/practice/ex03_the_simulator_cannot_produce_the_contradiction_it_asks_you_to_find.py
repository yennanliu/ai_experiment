"""Exercise 3 — the simulator cannot produce the contradiction it asks you to find.

    Add a conflict-detection step to the lead's synthesis: if two workers
    return contradictory answers, the lead notes the disagreement rather than
    picking one. How do you detect contradiction without calling an LLM?

Reading of the exercise: write the detector, then run it over the workers this
lesson actually ships -- because `fake_web_fetch` returns one sentence for
every query, so the three workers cannot disagree and the detector has nothing
to find.

**ANSWER: compare quantities that share a subject, and flag any subject whose
values are not all equal.** Extracting `(number, noun)` pairs from each summary
and grouping by noun gives a detector that needs no model: two workers
contradict when they report different values for the same noun. On a labelled
set of **8** pairs it scores **8/8** -- **4** contradictions found, **4**
agreements left alone. It catches exactly the disagreements that are
*commensurable*, and by construction nothing else.

**FINDING: the three shipped workers cannot disagree.** `fake_web_fetch`
returns `f"Summary for '{query}': 3 key findings from 5 sources."` -- **1**
template, with only the query interpolated. Across the lead's **3**
sub-questions the summaries reduce to **1** distinct claim set, so the
detector finds **0** conflicting pairs out of **3** possible. The exercise's
premise is unreachable from the lesson's own code.

**FINDING: the only numbers are template literals and the query's own.** The
template contributes the same **3** findings and **5** sources to every
summary, whatever the sub-question. The rest of the quantities in the corpus
come from the query text interpolated into the summary -- the demo asks about
"2023 to 2026", so **2023** enters the claim set as a reported value. Both
halves are a problem: the literals agree by construction, and the query
numerals are not claims at all. To test the detector the answers have to be
injected, which is worth saying rather than reporting 0 conflicts as a pass.

**FINDING: the cost of the extra step cannot be measured here either.**
`tokens_spent` is the literal **800** per worker and the lead adds a flat
**1200**, so the run's total is `800N + 1200` by construction. Adding a
synthesis step that reads every summary changes the real token bill and
changes this formula by **0**.

Structure: `claims()` extracts commensurable quantities; `conflicts()` groups
them by subject; `LABELLED` is the set the detector is scored on.
"""

from __future__ import annotations

import collections
import inspect
import itertools
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "05-supervisor-orchestrator-pattern"
QUANTITY = re.compile(r"(\d+(?:\.\d+)?)\s+(?:key\s+)?([a-z]+)")
LABELLED = (
    ("3 key findings from 5 sources", "4 key findings from 5 sources", True),
    ("latency fell to 120 ms", "latency fell to 340 ms", True),
    ("the study used 12 models", "the study used 30 models", True),
    ("adoption reached 60 percent", "adoption reached 25 percent", True),
    ("3 key findings from 5 sources", "3 key findings from 5 sources", False),
    ("latency fell to 120 ms", "throughput rose to 340 rps", False),
    ("the study used 12 models", "the review cites 12 models", False),
    ("adoption reached 60 percent", "coverage reached 60 percent", False),
)


def claims(text):
    """Commensurable quantities in a summary: {subject: value}."""
    return {noun: float(value) for value, noun in QUANTITY.findall(text.lower())}


def conflicts(summaries):
    """Subjects on which two summaries report different values."""
    seen = collections.defaultdict(set)
    for summary in summaries:
        for subject, value in claims(summary).items():
            seen[subject].add(value)
    return sorted(subject for subject, values in seen.items() if len(values) > 1)


def score(labelled):
    """How the detector does on pairs whose verdict is known."""
    correct = sum(bool(conflicts([left, right])) == expected
                  for left, right, expected in labelled)
    return correct, len(labelled), sum(expected for _, _, expected in labelled)


def shipped(ref):
    """The summaries the lead's own three workers produce."""
    lead = ref.Lead(ref.Trace())
    questions = lead.plan("What changed in multi-agent systems 2023 to 2026?")
    summaries = [ref.fake_web_fetch(question) for question in questions]
    pairs = list(itertools.combinations(summaries, 2))
    return summaries, pairs, sum(bool(conflicts(list(pair))) for pair in pairs)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    src = inspect.getsource(ref)
    summaries, pairs, conflicting = shipped(ref)
    correct, total, positives = score(LABELLED)
    return {
        "correct": correct, "labelled": total, "positives": positives,
        "negatives": total - positives,
        "summaries": len(summaries), "pairs": len(pairs), "conflicting": conflicting,
        "distinct_claims": len({tuple(sorted(claims(s).items())) for s in summaries}),
        "templates": len(re.findall(r'return f"Summary for', src)),
        "template": sorted(claims(ref.fake_web_fetch("a question")).values()),
        "in_corpus": sorted({value for summary in summaries
                             for value in claims(summary).values()}),
        "tokens_literal": re.findall(r"tokens_spent=(\d+)", src),
        "lead_constant": re.findall(r"\) \+ (\d{3,4})", src),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: compare quantities that share a subject",
            all([result["correct"] == result["labelled"] == 8,
                 result["positives"] == 4, result["negatives"] == 4]),
            f"grouping (number, noun) pairs by noun gives a detector needing no model: "
            f"on {result['labelled']} labelled pairs it scores {result['correct']} -- "
            f"{result['positives']} contradictions found and {result['negatives']} "
            "agreements left alone, catching exactly the commensurable disagreements",
        ),
        practice.Check(
            "FINDING: the three shipped workers cannot disagree",
            all([result["templates"] == 1, result["distinct_claims"] == 1,
                 result["conflicting"] == 0, result["pairs"] == 3]),
            f"fake_web_fetch has {result['templates']} template with only the query "
            f"interpolated, so the lead's {result['summaries']} summaries reduce to "
            f"{result['distinct_claims']} distinct claim set and the detector finds "
            f"{result['conflicting']} conflicts in {result['pairs']} possible pairs",
        ),
        practice.Check(
            "FINDING: the only numbers are template literals and the query's own",
            all([result["template"] == [3.0, 5.0],
                 set(result["in_corpus"]) > set(result["template"])]),
            f"the template contributes {result['template']} to every summary regardless "
            f"of the sub-question, and the corpus holds {result['in_corpus']} -- the "
            "extra ones are digits from the interpolated query, which the detector reads "
            "as claims; sub-questions naming different years would conflict on nothing",
        ),
        practice.Check(
            "FINDING: the cost of the extra step cannot be measured here either",
            all([result["tokens_literal"] == ["800"], result["lead_constant"] == ["1200"]]),
            f"tokens_spent is the literal {result['tokens_literal'][0]} per worker and "
            f"the lead adds a flat {result['lead_constant'][0]}, so the total is "
            "800N + 1200 by construction -- a synthesis step that reads every summary "
            "changes the real bill and changes this formula by 0",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
