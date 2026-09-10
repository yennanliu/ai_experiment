"""Exercise 3 — the context is shared, so it cannot discriminate.

    **Hard.** Implement contextual retrieval. Measure MRR improvement over
    baseline recursive. Report index cost (LLM calls) vs accuracy gain.

Reading of the exercise: no LLM is reachable, so the context each chunk is
prefixed with is written from the document itself -- the shared move a first
implementation makes. Three prefixes are measured against the recursive baseline
on exercise 2's 30 questions: a corpus-level summary, the document's name, and
the document's name plus its opening line.

All three make it worse. MRR goes **0.6746 -> 0.6624, 0.6585, 0.6557**, and
recall@1 goes 0.5333 -> 0.5000 in every arm, at a cost of one LLM call per
chunk. The ratio the exercise asks to report -- index cost against accuracy gain
-- has a negative denominator, so there is no cost per point to quote.

The mechanism is that the prefix is shared. Prepending the same tokens to every
chunk of a document adds a constant to the numerator of every cosine in that
document and a varying amount to each denominator, because the vectors are
re-normalised after the prefix is added. Nothing that appears in every chunk can
separate chunks; it can only dilute what does. The corpus-level summary is shared
by all 10 chunks and the document name by all chunks of its document, and all 30
questions are answered inside a single document -- so the routing the prefix
could help with is routing that is never needed.

Anthropic's contextual retrieval writes a *chunk-specific* sentence, generated
per chunk from the whole document, and that is the part an LLM is for. The
version measurable without one is the version that cannot work, and it fails for
a reason visible in the arithmetic rather than in the quality of the writing.

The prefix's size says how much dilution to expect: the corpus summary adds 91
characters to chunks averaging 231, so nearly 30% of every embedded string is
text shared with every other chunk.

Structure: exercise 2's corpus and scorers are loaded rather than copied;
`prefixed` builds each variant's index, `mrr` scores it against the *original*
chunk texts so the gold spans still match, and `calls` counts the LLM calls the
index would cost.
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "23-chunking-strategies-rag"

EVAL = practice.load_module(
    pathlib.Path(__file__).resolve().parent / "ex02_recall_at_five_rewards_making_five_chunks.py")
DOCUMENTS, QUERIES, CHUNK = EVAL.DOCUMENTS, EVAL.QUERIES, EVAL.CHUNK


def baseline(ref):
    """The recursive index, with the document each chunk came from."""
    chunks, owners = [], []
    for name, text in DOCUMENTS.items():
        parts = ref.chunk_recursive(text, CHUNK)
        chunks += parts
        owners += [name] * len(parts)
    return chunks, owners


def prefixes(ref, owners):
    """One prefix per chunk, for each way of writing context without an LLM."""
    opening = {name: ref.split_sentences(text)[0] for name, text in DOCUMENTS.items()}
    summary = " ".join(f"{name} agreement." for name in DOCUMENTS)
    return {
        "corpus summary": [summary for _ in owners],
        "document name": [f"From the {name} document." for name in owners],
        "name and opening line": [f"From the {name} document. {opening[name]}" for name in owners],
    }


def mrr(ref, chunks, texts):
    """Mean reciprocal rank of the first chunk whose *original* text holds the gold span."""
    total = 0.0
    for query, gold in QUERIES:
        for position, (_, i) in enumerate(EVAL.ranked(ref, chunks, query), 1):
            if gold.lower() in texts[i].lower():
                total += 1 / position
                break
    return round(total / len(QUERIES), 4)


def at_one(ref, chunks, texts):
    """Recall@1 against the original chunk texts."""
    hits = sum(1 for query, gold in QUERIES
               if gold.lower() in texts[EVAL.ranked(ref, chunks, query)[0][1]].lower())
    return round(hits / len(QUERIES), 4)


def cross_document():
    """Gold spans that appear in more than one document -- the case routing would help."""
    return sum(1 for _, gold in QUERIES
               if sum(1 for text in DOCUMENTS.values() if gold.lower() in text.lower()) > 1)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    chunks, owners = baseline(ref)
    written = prefixes(ref, owners)
    arms = {name: [f"{p} {c}" for p, c in zip(parts, chunks)] for name, parts in written.items()}
    return {
        "chunks": len(chunks),
        "mean_chars": round(sum(len(c) for c in chunks) / len(chunks)),
        "prefix_chars": {n: len(parts[0]) for n, parts in written.items()},
        "baseline": {"mrr": mrr(ref, chunks, chunks), "at1": at_one(ref, chunks, chunks)},
        "arms": {n: {"mrr": mrr(ref, index, chunks), "at1": at_one(ref, index, chunks),
                     "calls": len(chunks)} for n, index in arms.items()},
        "shared": {n: len(set(parts)) for n, parts in written.items()},
        "docs": len(DOCUMENTS), "cross_document": cross_document(),
    }


def verify(result):
    base, arms = result["baseline"], result["arms"]
    gains = {n: round(row["mrr"] - base["mrr"], 4) for n, row in arms.items()}
    return [
        practice.Check(
            "ANSWER: every context variant lowers MRR",
            all(gain < 0 for gain in gains.values()),
            f"no LLM is reachable, so the context is written from the document itself -- the "
            f"shared move a first implementation makes. Against a baseline of {base['mrr']}, the "
            f"three variants score {[row['mrr'] for row in arms.values()]}: gains {gains}",
        ),
        practice.Check(
            "MECHANISM: and recall@1 falls in every arm, so it is not an MRR artefact",
            all(row["at1"] < base["at1"] for row in arms.values()),
            f"recall@1 goes {base['at1']} to {[row['at1'] for row in arms.values()]} across the "
            f"three variants -- one question in {len(QUERIES)}, in the same direction each time",
        ),
        practice.Check(
            "FINDING: the ratio the exercise asks for has a negative denominator",
            all(row["calls"] == result["chunks"] for row in arms.values()),
            f"index cost is one LLM call per chunk, {result['chunks']} here, against an accuracy "
            f"gain of {min(gains.values())} to {max(gains.values())} MRR. There is no cost per "
            "point to quote because the points go the wrong way",
        ),
        practice.Check(
            "MECHANISM: nothing shared by every chunk can separate chunks",
            result["shared"]["corpus summary"] == 1,
            f"the corpus summary is one string across all {result['chunks']} chunks and the "
            f"document name takes {result['shared']['document name']} values across "
            f"{result['docs']} documents. A prefix that appears everywhere adds a constant to "
            "every numerator and re-normalises every denominator: it can only dilute",
        ),
        practice.Check(
            "MECHANISM: and the routing it could help with is never needed",
            result["cross_document"] == 0,
            f"{result['cross_document']} of {len(QUERIES)} gold spans appear in more than one "
            "document, so no question requires choosing between documents. The one thing a "
            "document-level prefix could contribute is the thing this eval never asks for",
        ),
        practice.Check(
            "CONTROL: the dilution is a quarter of the embedded string",
            result["prefix_chars"]["corpus summary"] > result["mean_chars"] / 4,
            f"chunks average {result['mean_chars']} characters and the prefixes add "
            f"{result['prefix_chars']}. Contextual retrieval as Anthropic describes it writes a "
            "chunk-specific sentence per chunk, which is what the LLM call is for; the shared "
            "version is the one measurable here, and it is the one that cannot work",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
