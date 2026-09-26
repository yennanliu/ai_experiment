"""Exercise 3 — three hours is a normal batch and a normal sync run, so only the promise decides.

    A user complains their report took 3 hours. Was that a batch mis-triage or
    a legitimate interactive? Write the decision criterion.

Reading of the exercise: the criterion is written as a function of what is
knowable after the complaint: the lane the report ran on, how long the user
expected to wait, the report's size in documents, and the observed 3 hours.
It is applied to every combination. Sync throughput comes from the lesson's
Problem, 50,000 documents in 4 hours, i.e. 12,500 an hour.

**ANSWER: it is a batch mis-triage exactly when the report ran on batch and
the user expected it in under 24 hours.** Batch promises 24 hours, so any
shorter expectation was never going to be met on that lane. On batch with an
expectation of 24 hours or more, the complaint is about communication: say
when the report will arrive. On sync, 3 hours is never a triage error. If the
report's size alone takes longer than the user will wait, it is too big to be
interactive: precompute it on batch and serve the stored result, which is the
lesson's partial-interactivity trap. If it does not, the delay is an
incident. Over the 12 cases enumerated (2 lanes x 3 expectations x 2 sizes),
4 are mis-triage.

**FINDING: 3 hours cannot tell the lanes apart.** It is inside the lesson's
typical batch P50 of 2-6 hours. It is also what the lesson's own sync
pipeline takes for 37,500 documents. The duration is evidence of neither
lane; the lane has to be read from the job, and the verdict from the user's
expectation.

**FINDING: the lesson gives two typical latencies, and the provider a third.**
It says OpenAI batches take "2-8 hours in practice" and that typical P50 is
"2-6 hours". Anthropic's batch docs say "most batches finishing in less than
1 hour". Under Anthropic's figure a 3-hour batch is slow; under the lesson's
it is normal. So the criterion must compare against the promise, 24 hours,
rather than against a typical latency.

Structure: `verdict()` is the criterion; `solve()` enumerates it.
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "15-batch-apis"
SLA_H, OBSERVED_H = 24, 3
SYNC_DOCS_PER_H = 50_000 / 4  # the Problem: 50k documents synchronously in 4 hours
EXPECTS_H, SIZES = (1, 8, 48), (1, 37_500)
TYPICAL = ("2-8 hours in practice", "Typical P50 is 2-6 hours")


def verdict(lane, expect_h, docs, observed_h=OBSERVED_H):
    """The decision criterion: compare the user's expectation to the lane's promise."""
    if lane == "batch":
        return "mis-triage" if expect_h < SLA_H else "communicate the 24h promise"
    if docs / SYNC_DOCS_PER_H > expect_h:
        return "too big to be interactive: precompute on batch"
    return "incident" if observed_h > expect_h else "on time"


def solve():
    ref_doc = parity.doc_text(PHASE, LESSON, "en")
    cases = {
        (lane, e, n): verdict(lane, e, n)
        for lane, e, n in itertools.product(("batch", "sync"), EXPECTS_H, SIZES)
    }
    return {
        "cases": cases,
        "sync_hours": {n: n / SYNC_DOCS_PER_H for n in SIZES},
        "typical_in_doc": [t in ref_doc for t in TYPICAL],
        "p50_range": (2, 6),
    }


def verify(result):
    cases = result["cases"]
    mis = sorted(k for k, v in cases.items() if v == "mis-triage")
    sync_cases = {k: v for k, v in cases.items() if k[0] == "sync"}
    low, high = result["p50_range"]
    return [
        practice.Check(
            "ANSWER: it is a batch mis-triage exactly when it ran on batch and the user "
            "expected it in under 24 hours",
            len(cases) == 12
            and mis == sorted(k for k in cases if k[0] == "batch" and k[1] < SLA_H)
            and "mis-triage" not in sync_cases.values(),
            f"{len(mis)} of {len(cases)} cases are mis-triage: {mis}; sync verdicts "
            f"{sorted(set(sync_cases.values()))}",
        ),
        practice.Check(
            "FINDING: 3 hours cannot tell the lanes apart",
            low <= OBSERVED_H <= high and result["sync_hours"][37_500] == OBSERVED_H,
            f"{OBSERVED_H}h is inside the lesson's {low}-{high}h batch P50 and is the "
            f"lesson's own sync pipeline on 37,500 documents "
            f"({result['sync_hours'][37_500]}h at {SYNC_DOCS_PER_H:.0f} docs/h)",
        ),
        practice.Check(
            "FINDING: the lesson gives two typical latencies, and the provider a third",
            result["typical_in_doc"] == [True, True],
            f"the lesson says {TYPICAL[0]!r} and {TYPICAL[1]!r}; Anthropic's docs say most "
            "batches finish in under 1 hour, so only the 24h promise is a fixed yardstick",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
