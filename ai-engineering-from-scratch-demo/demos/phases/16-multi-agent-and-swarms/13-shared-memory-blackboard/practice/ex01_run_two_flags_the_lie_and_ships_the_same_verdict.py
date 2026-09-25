"""Exercise 1 — run two flags the lie and ships the same verdict.

    Run `code/main.py`. Confirm run 1 propagates the hallucination and run 2
    catches it.

Reading of the exercise: "catches" is read as the lesson's own expected
output -- "the pool is labeled 'flagged', the final report includes a
retraction" -- so both runs are compared entry by entry, and the honest
pipeline is run as a control.

**ANSWER: run 1 propagates it; run 2 flags it and ships it anyway.** Run 1's
analyst writes "Recommend adoption" on the 42% figure with no flag anywhere.
In run 2 the verifier finds **1** mismatch and entry 0 is flagged -- then
the analyst writes a verdict **byte-identical** to run 1's. `analyst_agent`
never reads `flags`, and no code writes a retraction: the pool holds 3
entries in both runs, and the final one does not mention the flag.

**FINDING: the flag stops one hop short.** Only entry 0 carries a
`source_uri`, and `verifier_agent` checks only entries that do. The
summarizer and analyst write `source_uri=None`, and `ProvenanceEntry` has no
field naming what an entry was derived from -- so the two entries that
repeat 42% are unflaggable by construction, 0 of 2.

**FINDING: the honest pipeline corrupts the true figure.** The summarizer
takes `latest.split('.')[0]`, and the decimal point in "4.2%" is a period:
the honest summary ends "The study reports a 4." The hallucinated 42% has no
decimal and survives intact -- the summarizer damages only the correct
number. The verifier cannot see it, for the reason above.

**FINDING: provenance cannot tell the two runs apart.** `prompt_hash` hashes
the prompt, which is the same whether or not the retriever hallucinated:
all 3 entries carry identical hashes across the honest and poisoned runs.

**FINDING: adoption is keyed to the hallucination.** The analyst's rule is
`"42%" in latest`, so it recommends adoption exactly when the lie is present
and "further review" on the true 4.2%.

Structure: `run()` replays the lesson's two runs (and the honest control)
with the reference agents and returns the pool.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "13-shared-memory-blackboard"
URI = "https://arxiv.org/paper-1"


def run(ref, hallucinate, verify_first):
    pool = ref.MessagePool()
    ref.retrieval_agent(pool, URI, hallucinate=hallucinate)
    ref.summarizer_agent(pool)
    findings = ref.verifier_agent(pool) if verify_first else []
    for eid, reason in findings:
        pool.flag(eid, reason)
    ref.analyst_agent(pool)
    return pool.read_all(), findings


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    one, _ = run(ref, True, False)
    two, findings = run(ref, True, True)
    honest, _ = run(ref, False, True)
    return {
        "verdict1": one[-1].content, "verdict2": two[-1].content,
        "sizes": (len(one), len(two)), "findings": len(findings),
        "flagged": [e.id for e in two if e.flags],
        "sourced": [e.id for e in two if e.source_uri],
        "derived_field": any(f in inspect.signature(ref.ProvenanceEntry).parameters
                             for f in ("derived_from", "parents", "cites")),
        "repeat_42": [e.id for e in two[1:] if "42%" in e.content],
        "honest_summary": honest[1].content, "honest_verdict": honest[-1].content,
        "hashes_equal": [e.prompt_hash for e in one] == [e.prompt_hash for e in honest],
        "doc_retraction": "retraction" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: run 1 propagates it; run 2 flags it and ships it anyway",
            all([result["verdict1"] == result["verdict2"], "Recommend adoption" in
                 result["verdict2"], result["findings"] == 1, result["sizes"] == (3, 3),
                 result["doc_retraction"]]),
            f"run 2 flags entry {result['flagged']} and the analyst's verdict is "
            f"byte-identical to run 1's: {result['verdict2'][:40]!r}...; both pools hold "
            f"{result['sizes'][0]} entries, so the promised retraction is never written",
        ),
        practice.Check(
            "FINDING: the flag stops one hop short",
            all([result["sourced"] == [0], result["repeat_42"] == [1, 2],
                 not result["derived_field"]]),
            f"only entry {result['sourced']} has a source_uri, the verifier checks only "
            f"those, and entries {result['repeat_42']} repeat 42% with no field naming "
            "what they came from -- 0 of 2 derived entries are flaggable",
        ),
        practice.Check(
            "FINDING: the honest pipeline corrupts the true figure",
            result["honest_summary"].endswith("The study reports a 4."),
            f"split('.')[0] cuts at the decimal point: {result['honest_summary']!r} -- the "
            "hallucinated 42% has no decimal and survives intact",
        ),
        practice.Check(
            "FINDING: provenance cannot tell the two runs apart",
            result["hashes_equal"],
            "prompt_hash hashes the prompt, not the output, so all 3 entries carry "
            "identical hashes in the honest and the poisoned run",
        ),
        practice.Check(
            "FINDING: adoption is keyed to the hallucination",
            "further review" in result["honest_verdict"],
            "the analyst tests '42%' in latest: adoption when the lie is present, "
            f"{result['honest_verdict'].split(' (')[0]!r} on the true figure",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
