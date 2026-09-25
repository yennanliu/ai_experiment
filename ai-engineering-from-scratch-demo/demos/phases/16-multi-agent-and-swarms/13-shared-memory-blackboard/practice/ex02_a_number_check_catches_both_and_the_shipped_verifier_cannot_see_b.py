"""Exercise 2 — a number check catches both, and the shipped verifier cannot see B.

    Add a second hallucination: agent B invents a dataset size. The verifier
    should catch both without being hand-tuned for either.

Reading of the exercise: the retriever fetches both fake sources (paper-1
with its decimal dropped, paper-2 honestly) and the summarizer appends "on
50,000 examples" where paper-2 says 12,500. "Not hand-tuned" is read as a
rule that names no figure, no source and no agent.

**ANSWER: check that every number in every entry occurs in a fetched source.**
The rule re-fetches each `source_uri` in the pool, collects the numbers the
sources actually contain -- {4.2, 12500} -- and flags any entry stating a
number outside that set. It flags all 3 entries carrying a hallucination
(the retriever's 42, the summarizer's 42 and 50,000, the analyst's echo of
both) and not the honest paper-2 write.

**FINDING: the shipped verifier cannot catch B at all.** It checks only
entries with a `source_uri`, and the summarizer writes `None`, so it returns
1 finding -- the retriever's -- whatever B writes. Being "not hand-tuned for
B" is not enough; it has to read entries it currently skips.

**FINDING: the shipped verifier is tuned to verbatim copying.** Its test is
`e.content != truth`. A faithful paraphrase -- "Accuracy rose 4.2% over the
baseline." -- is flagged as a mismatch, while the number check passes it.

**FINDING: on the honest pipeline the number check finds a real defect.**
The honest summary ends "reports a 4." (exercise 1), and 4 is not a number
either source contains, so the rule flags it: 1 flag, on the entry that
really is wrong.

**FINDING: without lineage a misattributed number passes.** Nothing records
which source a summary came from, so the only set to check against is the
union of all sources. With the retriever honest, a summary claiming paper-1 found a
"12,500% improvement" gets 0 flags, because 12,500 is in paper-2.

Structure: `numbers()` extracts figures; `number_verifier()` is the general
rule, read-only like the shipped one; `inventing_summarizer()` is agent B.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "13-shared-memory-blackboard"
P1, P2 = "https://arxiv.org/paper-1", "https://arxiv.org/paper-2"
FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers(text):
    return {m.replace(",", "") for m in FIGURE.findall(text)}


def number_verifier(ref, pool):
    """Flag entries stating a number that no cited source in the pool contains."""
    entries = pool.read_all()
    known = set().union(*(numbers(ref.FAKE_SOURCES[e.source_uri])
                          for e in entries if e.source_uri))
    return [e.id for e in entries if numbers(e.content) - known]


def inventing_summarizer(pool, claim=" on 50,000 examples"):
    text = pool.read_all()[0].content.rstrip(".")
    return pool.write("summarizer", f"Summary: {text}{claim}.", "Summarize retrieval", None)


def poisoned(ref, claim=" on 50,000 examples", lie=True):
    pool = ref.MessagePool()
    ref.retrieval_agent(pool, P1, hallucinate=lie)
    ref.retrieval_agent(pool, P2, hallucinate=True)
    inventing_summarizer(pool, claim)
    ref.analyst_agent(pool)
    return pool


def honest(ref):
    pool = ref.MessagePool()
    ref.retrieval_agent(pool, P1, hallucinate=False)
    ref.summarizer_agent(pool)
    return pool


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pool = poisoned(ref)
    para = ref.MessagePool()
    para.write("retriever", "Accuracy rose 4.2% over the baseline.", "Fetch", P1)
    return {
        "entries": [(e.id, e.writer) for e in pool.read_all()],
        "shipped": [eid for eid, _ in ref.verifier_agent(pool)],
        "general": number_verifier(ref, pool),
        "para_shipped": len(ref.verifier_agent(para)), "para_general": number_verifier(ref, para),
        "honest": number_verifier(ref, honest(ref)),
        "misattributed": number_verifier(ref, poisoned(ref, ", a 12,500% improvement", lie=False)),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: check that every number in every entry occurs in a fetched source",
            result["general"] == [0, 2, 3],
            f"of entries {result['entries']} the number check flags {result['general']} "
            "-- both hallucinations and the analyst's echo, and not the honest paper-2 write",
        ),
        practice.Check(
            "FINDING: the shipped verifier cannot catch B at all",
            result["shipped"] == [0],
            f"it flags {result['shipped']}: the summarizer writes source_uri=None and the "
            "verifier only reads entries that have one",
        ),
        practice.Check(
            "FINDING: the shipped verifier is tuned to verbatim copying",
            result["para_shipped"] == 1 and result["para_general"] == [],
            "a faithful paraphrase of paper-1 is flagged by content != truth and passed "
            "by the number check",
        ),
        practice.Check(
            "FINDING: on the honest pipeline the number check finds a real defect",
            result["honest"] == [1],
            f"it flags entry {result['honest']}, the summary truncated to 'reports a 4.'",
        ),
        practice.Check(
            "FINDING: without lineage a misattributed number passes",
            result["misattributed"] == [],
            f"an honest retriever plus a summary claiming a 12,500% improvement over "
            f"paper-1 gets {len(result['misattributed'])} flags -- 12,500 is in paper-2",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
