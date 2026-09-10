"""Exercise 3 — three of the four metrics cannot see the regression.

    **Hard.** Build a pytest CI gate with DeepEval. Intentionally regress the
    retriever. Verify the gate fails. Add bottom-quantile alerting via threshold
    check on the lowest 10%.

Reading of the exercise: `deepeval` is absent, so `FaithfulnessMetric` and
`ContextualRelevancyMetric` cannot be constructed and the gate is built from the
lesson's own four metrics at the thresholds `docs/en.md` Step 4 writes down --
faithfulness >= 0.85, relevancy >= 0.7. The regression is the standard one: keep
the gold chunk and inject k irrelevant chunks, k = 0 to 5. Three invented QA cases
carry it, each with a one-sentence gold chunk.

The gate does fail. It fails identically at **k = 0** and at **k = 5**, for the
same reason: answer-relevance is **0.3810** on the healthy retriever, below the
0.7 the write-up sets, and injecting five irrelevant chunks does not move it by
one digit. "Verify the gate fails" is satisfied by a gate that was already failing
before the regression and whose verdict the regression never touches.

The mechanism is in the signatures. `answer_relevance(question, answer)` is never
handed the retrieved chunks at all, so it is constant by construction.
`faithfulness` and `context_recall` join the retrieved chunks and take a set of
their tokens, so adding chunks can only grow that set: both are monotone
non-decreasing in retrieval noise and sit flat at **1.0000** for every k. Only
`context_precision` has a denominator that grows, falling **1.0000 -> 0.1667**.
Three of the four RAG metrics are blind to added noise as a matter of arithmetic,
not of tuning.

The one responsive metric is not measuring retrieval quality. `context_precision`
tests `c in relevant_chunks` -- exact string membership -- so splitting the
identical gold sentence into two halves, losing no information at all, scores
**0.0000** while faithfulness and recall stay at 1.0000. And a retriever that
returns the same gold chunk three times, which retrieves one document and calls
it top-3, scores **1.0000** on faithfulness, precision and recall alike and draws
the same verdict as the healthy retriever.

Bottom-quantile alerting cannot be written at this size. `int(0.1 * n)` is **0**
for every n below 10, so on `main()`'s three cases and on the three here the
lowest-10% check inspects an empty list and never fires. There is nothing to rank
in any case: faithfulness takes **1** distinct value across the suite.

Structure: `CASES` holds three question/gold-chunk/answer/expected rows and
`NOISE` the distractors; `score` runs all four lesson metrics over one retrieval;
`suite` averages them over the cases; `gate` applies the write-up's thresholds;
`only_precision_moved` and `scored_as_perfect` are the two verdict predicates.
"""

from __future__ import annotations

import importlib.util

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "27-llm-evaluation-frameworks"

NOISE = ("The moon landing was in 1969.", "Android launched in 2008.",
         "Nokia dominated handset sales in 2005.", "Bananas are rich in potassium.",
         "The Seine flows through Paris.")
CASES = (("When was the first iPhone released?",
          "Apple released the first iPhone on June 29, 2007.",
          "The first iPhone was released on June 29, 2007.", "June 29, 2007"),
         ("Who created Python?",
          "Guido van Rossum created the Python language in 1991.",
          "Python was created by Guido van Rossum in 1991.", "Guido van Rossum"),
         ("How tall is the Eiffel Tower?",
          "The Eiffel Tower stands 330 metres tall.",
          "The Eiffel Tower stands 330 metres tall.", "330 metres"))
FAITH_MIN, REL_MIN = 0.85, 0.7


def score(ref, case, retrieved):
    """The lesson's four RAG metrics for one case under one retrieval."""
    question, gold, answer, expected = case
    return {"faith": ref.faithfulness(answer, " ".join(retrieved)),
            "rel": ref.answer_relevance(question, answer),
            "prec": ref.context_precision(list(retrieved), [gold]),
            "rec": ref.context_recall(list(retrieved), ref.tokenize(expected))}


def suite(ref, retrieve):
    """Mean of each metric over the three cases, given a retrieval strategy."""
    rows = [score(ref, case, retrieve(case)) for case in CASES]
    return {m: round(sum(r[m] for r in rows) / len(rows), 4) for m in rows[0]}


def gate(row):
    """The write-up's Step 4 CI gate: which metrics fail, in order."""
    return tuple(name for name, ok in (("faithfulness", row["faith"] >= FAITH_MIN),
                                       ("relevancy", row["rel"] >= REL_MIN)) if not ok)


def signature_error(ref):
    """What the doc's Step 2 signature `answer_relevance(q, a, encoder, llm)` raises."""
    try:
        ref.answer_relevance("q", "a", object(), lambda prompt: prompt)
    except TypeError as exc:
        return f"TypeError: {exc}"
    return "no error"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    noisy = {k: suite(ref, lambda c, k=k: (c[1], *NOISE[:k])) for k in range(6)}
    faiths = [score(ref, case, (case[1], *NOISE))["faith"] for case in CASES]
    return {
        "deepeval": importlib.util.find_spec("deepeval") is not None,
        "noisy": noisy,
        "verdicts": {k: gate(row) for k, row in noisy.items()},
        "dropped": suite(ref, lambda c: NOISE[:2]),
        "duplicated": suite(ref, lambda c: (c[1],) * 3),
        "rechunked": suite(ref, lambda c: tuple(_halves(c[1]))),
        "decile": sorted(faiths)[:int(0.1 * len(faiths))], "distinct": sorted(set(faiths)),
        "decile_sizes": {n: int(0.1 * n) for n in (3, 5, 9, 10, 20)},
        "sig": signature_error(ref),
    }


def only_precision_moved(clean, worst):
    """True when noise injection moved context-precision and nothing else."""
    flat = clean["faith"] == worst["faith"] and clean["rec"] == worst["rec"]
    return flat and clean["rel"] == worst["rel"] and clean["prec"] > worst["prec"]


def scored_as_perfect(row, verdict):
    """True when a retrieval draws `verdict` with every retrieval-facing metric at 1.0."""
    return gate(row) == verdict and row["prec"] == 1.0 and row["rec"] == 1.0


def _halves(sentence):
    """The same sentence split into two chunks, losing no text."""
    words = sentence.split()
    return [" ".join(words[:4]), " ".join(words[4:])]


def verify(result):
    clean, worst = result["noisy"][0], result["noisy"][5]
    verdicts = result["verdicts"]
    return [
        practice.Check(
            "ANSWER: the gate returns the same verdict on the healthy and the regressed retriever",
            not result["deepeval"] and verdicts[0] == verdicts[5] == ("relevancy",),
            f"`deepeval` is absent, so the gate is the lesson's metrics at the write-up's own "
            f"Step 4 thresholds (faithfulness >= {FAITH_MIN}, relevancy >= {REL_MIN}). It reports "
            f"{list(verdicts[0])} with the clean retriever and {list(verdicts[5])} with five "
            f"irrelevant chunks injected: relevancy is {clean['rel']:.4f} either way. It fails, "
            "but it was already failing, and the regression never moved it",
        ),
        practice.Check(
            "MECHANISM: three of the four metrics are blind to added noise by construction",
            only_precision_moved(clean, worst),
            f"`answer_relevance(question, answer)` never sees the chunks; `faithfulness` and "
            f"`context_recall` take a set union over the joined chunks, which adding chunks can "
            f"only grow. Across k = 0..5 faithfulness holds {clean['faith']:.4f}, recall "
            f"{clean['rec']:.4f}, relevance {clean['rel']:.4f}; precision falls "
            f"{clean['prec']:.4f} -> {worst['prec']:.4f}",
        ),
        practice.Check(
            "CONTROL: the metrics do move when retrieval fails completely",
            result["dropped"]["faith"] < FAITH_MIN and result["dropped"]["rec"] == 0.0,
            f"retrieving only distractors gives faithfulness {result['dropped']['faith']:.4f} and "
            f"recall {result['dropped']['rec']:.4f}, so it detects total retrieval failure. "
            "It is degradation -- the thing a CI gate exists to catch -- that it cannot see",
        ),
        practice.Check(
            "FINDING: a retriever that fetches one chunk three times is scored as a perfect one",
            scored_as_perfect(result["duplicated"], verdicts[0]),
            f"returning the gold chunk duplicated gives faithfulness "
            f"{result['duplicated']['faith']:.4f}, precision {result['duplicated']['prec']:.4f} "
            f"and recall {result['duplicated']['rec']:.4f} -- every retrieval-facing metric at "
            f"ceiling and the verdict {list(gate(result['duplicated']))} identical to the "
            "healthy retriever's, for one document returned three times and called top-3",
        ),
        practice.Check(
            "FINDING: the one responsive metric measures string identity, not retrieval quality",
            result["rechunked"]["prec"] == 0.0 and result["rechunked"]["faith"] == 1.0,
            f"`context_precision` tests `c in relevant_chunks`, exact membership. Splitting "
            f"the identical gold sentence into two halves loses no text and scores "
            f"{result['rechunked']['prec']:.4f}, while faithfulness stays "
            f"{result['rechunked']['faith']:.4f} and recall {result['rechunked']['rec']:.4f}. "
            "Change the chunker and it reports a regression that did not happen",
        ),
        practice.Check(
            "CONTROL: bottom-decile alerting inspects zero cases at this size",
            result["decile"] == [] and len(result["distinct"]) == 1,
            f"`int(0.1 * n)` gives {result['decile_sizes']}, so below ten cases the lowest-10% "
            f"slice is empty -- {result['decile']} here, and the same on `main()`'s three cases. "
            f"There is nothing to rank regardless: faithfulness takes {result['distinct']} as its "
            f"only value across the suite. And Step 2's API is absent too: {result['sig']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
