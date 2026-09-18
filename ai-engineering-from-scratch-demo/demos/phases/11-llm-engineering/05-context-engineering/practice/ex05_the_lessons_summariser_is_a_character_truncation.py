"""Exercise 5 — the lesson's summariser is a truncation, and only extraction reads the query.

    Build a multi-strategy context compressor. Implement three compression
    strategies (truncation, summarization, extraction of key sentences) and
    benchmark them on a set of 20 documents. Measure the tradeoff between
    compression ratio and information retention (does the compressed version
    still contain the answer to the query?).

Reading of the exercise: the lesson ships 10 knowledge-base sentences, so the 20
documents are built by pairing them. "Summarization" is implemented as the
lesson's own summariser -- `ConversationManager._summarize_turns` -- because that
is the only one in the codebase. Retention is whether the answer term survives,
measured over the 20 of the 100 document-query pairs whose document contains
its answer at all.

**ANSWER: extraction wins on both axes at a matched 20-token budget.**
Compression ratio 0.518 against truncation's 0.717 and the summariser's 0.891,
and retention 18 of 20 against 12 and 16. There is no trade-off to plot here:
the strategy that reads the query is cheaper *and* keeps more.

**FINDING: only extraction reads the query.** Truncation and the summariser are
pure functions of the document: 0 of the 20 documents change form when the query
changes, against extraction's 8. For two of the three strategies a "compression
ratio versus information retention" curve is a curve in one variable, and
retention is decided before the compressor runs.

**MECHANISM: the summariser is `content[:100] + "..."`.** A character prefix, 16
words long, that also prepends `Previous: doc: ` -- three words of scaffolding
that are not in the document. So the "summarization" row measures a unit change
plus invented text, not a technique.

**FINDING: truncation's retention is entirely answer position.** It keeps 10 of
the 10 pairs whose answer is in the opening fragment and 2 of the 10 where it is
later. Extraction keeps 10 and 8, which is the whole difference between them.

**CONTROL: at the strategies' natural settings the trade-off appears, and it is
the budget's.** Truncating to half the words gives the best ratio of the three,
0.479, and the worst retention, 10 of 20. Same technique, different budget,
opposite verdict -- so what the exercise's plot would show is the budgets
chosen, not the strategies.

Structure: `make_corpus` builds the 20 documents by pairing knowledge-base
sentences, and `sentences_of` is the lesson's own full-stop split, decimals and
all. `STRATEGIES` holds the three compressors: `truncate` keeps the first words
inside the budget, `summarise` calls the lesson's `_summarize_turns` with the
document as one turn, and `extract` keeps the highest-scoring fragments until
the budget is spent. `benchmark` runs the 20 x 5 grid and reports the ratio plus
retention split by which fragment held the answer; `half` is truncation at its
own natural setting rather than a matched budget.
"""


from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "05-context-engineering"
QUERIES = ["refund window", "rate limits", "vector index", "test coverage", "error logging"]
ANSWERS = ["pgvector", "100", "HNSW", "80%", "JSON"]
BUDGET_TOKENS = 20


def make_corpus(sentences, n=20):
    return [f"{sentences[i % len(sentences)]} {sentences[(i + 3) % len(sentences)]}"
            for i in range(n)]


def sentences_of(doc):   # the lesson's own split, decimals and all
    return [s.strip() + "." for s in doc.split(".") if s.strip()]


def answer_position(doc, answer):
    """1 if the answer is in the opening fragment, 2 if later, None if absent."""
    hits = [i for i, part in enumerate(sentences_of(doc), 1) if answer.lower() in part.lower()]
    return min(hits[0], 2) if hits else None


def truncate(ref, doc, query, budget=BUDGET_TOKENS):
    return " ".join(doc.split()[:int(budget / 1.3)])


def summarise(ref, doc, query, budget=BUDGET_TOKENS):
    return ref.ConversationManager()._summarize_turns([{"role": "doc", "content": doc}])


def extract(ref, doc, query, budget=BUDGET_TOKENS):
    parts = sentences_of(doc)
    ranked = sorted(parts, key=lambda p: -ref.score_relevance(query, [p])[0])
    kept, spent = [], 0
    for part in ranked:
        if spent + ref.count_tokens(part) <= budget:
            kept.append(part)
            spent += ref.count_tokens(part)
    return " ".join(kept) or ranked[0]


STRATEGIES = {"truncation": truncate, "summarization": summarise, "extraction": extract}


def positions(live):   # (kept, seen) split by which fragment held the answer
    return ({n: sum(h for h, p in live if p == n) for n in (1, 2)},
            {n: sum(p == n for _, p in live) for n in (1, 2)})


def benchmark(ref, corpus, compress):
    grid = [(compress(ref, doc, query), doc, answer, answer_position(doc, answer))
            for query, answer in zip(QUERIES, ANSWERS) for doc in corpus]
    before = sum(ref.count_tokens(doc) for _, doc, _, _ in grid)
    after = sum(ref.count_tokens(out) for out, _, _, _ in grid)
    live = [(a.lower() in out.lower(), p) for out, _, a, p in grid if p]
    kept, seen = positions(live)
    return {"ratio": round(after / before, 3), "pairs": len(live), "kept": kept,
            "seen": seen, "retained": sum(h for h, _ in live)}


def half(ref, doc, query):
    return " ".join(doc.split()[:len(doc.split()) // 2])


def shape(outputs, corpus):
    sample = outputs["summarization"][0][0]
    return {"query_sensitive": {n: sum(len(set(r)) > 1 for r in rows)
                                for n, rows in outputs.items()},
            "summary_of": sample[:30], "is_prefix": sample.endswith("..."),
            "summary_words": len(sample.split()), "scaffold":
            sample.startswith("Previous: doc: "),
            "parts": sorted({len(sentences_of(d)) for d in corpus})}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    corpus = make_corpus(ref.ContextEngine().knowledge_base)
    natural = dict(STRATEGIES, truncation=half)
    outputs = {name: [[fn(ref, doc, q) for q in QUERIES] for doc in corpus]
               for name, fn in STRATEGIES.items()}
    return {
        "documents": len(corpus), "queries": len(QUERIES), **shape(outputs, corpus),
        "matched": {name: benchmark(ref, corpus, fn) for name, fn in STRATEGIES.items()},
        "natural": {name: benchmark(ref, corpus, fn) for name, fn in natural.items()},
    }


def verify(result):
    matched, natural = result["matched"], result["natural"]
    sensitive = result["query_sensitive"]
    return [
        practice.Check(
            "ANSWER: extraction wins on both axes at a matched budget",
            all([matched["extraction"]["ratio"] < matched["truncation"]["ratio"],
                 matched["extraction"]["ratio"] < matched["summarization"]["ratio"],
                 matched["extraction"]["retained"] == 18]),
            f"ratio / retention out of {matched['extraction']['pairs']} at a matched "
            f"{BUDGET_TOKENS}-token budget: extraction {matched['extraction']['ratio']} / "
            f"{matched['extraction']['retained']}, truncation "
            f"{matched['truncation']['ratio']} / {matched['truncation']['retained']}, "
            f"summarization {matched['summarization']['ratio']} / "
            f"{matched['summarization']['retained']}",
        ),
        practice.Check(
            "FINDING: only extraction reads the query",
            all([sensitive["truncation"] == 0, sensitive["summarization"] == 0,
                 sensitive["extraction"] > 0]),
            f"documents whose compressed form changes with the query, out of "
            f"{result['documents']}: {sensitive}. For two of the three strategies 'does "
            "the compressed version still contain the answer' is settled before the call",
        ),
        practice.Check(
            "MECHANISM: the summariser is a character prefix with invented scaffolding",
            all([result["is_prefix"], result["scaffold"], result["summary_words"] == 16]),
            f"`_summarize_turns` is content[:100] + '...', {result['summary_words']} words "
            f"here -- {result['summary_of']!r}... -- and it prepends 'Previous: doc: ', "
            "three words not in the document. A unit change plus invented text",
        ),
        practice.Check(
            "FINDING: truncation's retention is entirely answer position",
            all([matched["truncation"]["kept"][1] == matched["truncation"]["seen"][1],
                 matched["truncation"]["kept"][2] <= 2,
                 matched["extraction"]["kept"][2] > matched["truncation"]["kept"][2]]),
            f"retention by where the answer sits: truncation "
            f"{matched['truncation']['kept']} of {matched['truncation']['seen']}, "
            f"extraction {matched['extraction']['kept']}. Truncation keeps every answer in "
            "the opening fragment and almost none later -- that gap is the difference",
        ),
        practice.Check(
            "CONTROL: at natural settings the trade-off appears, and it is the budget's",
            all([natural["truncation"]["ratio"] < matched["extraction"]["ratio"],
                 natural["truncation"]["retained"] < matched["truncation"]["retained"]]),
            f"truncating to half the words gives ratio {natural['truncation']['ratio']}, "
            f"best of the three, and retention {natural['truncation']['retained']} of "
            f"{natural['truncation']['pairs']}, worst. Same technique, different budget, "
            f"opposite verdict -- and the documents split into {result['parts']} fragments, "
            "because the splitter breaks on every decimal",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
