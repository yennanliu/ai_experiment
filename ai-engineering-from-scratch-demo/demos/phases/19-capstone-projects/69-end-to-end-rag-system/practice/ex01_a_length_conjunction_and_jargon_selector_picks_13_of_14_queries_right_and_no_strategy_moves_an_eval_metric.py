"""Exercise 1 — a length, conjunction and jargon selector picks right on 13 of 14 queries, and no strategy moves a single eval metric.

    Add a per-query strategy selector inside the rewriter: heuristics from lesson 67 (length, conjunctions, jargon ratio) pick HyDE, multi-query, or decomposition.

Reading of the exercise: lesson 67's "Use It" section gives the rule:
short atomic queries get multi-query, multi-clause queries get decomposition,
and jargon-heavy queries get HyDE. The selector `pick` computes three
features. Conjunctions: the query splits on and / or / versus into clauses.
Length: a split counts only if its right clause opens with a question word or
both clauses have 3 or more content tokens, so "lexical and semantic
retrieval" stays whole. Jargon ratio: the share of content tokens that are
corpus identifiers. Those are the words the corpus writes in camelCase,
ALLCAPS or with a digit, plus the lesson's own four jargon words. A ratio
of 0.25 or more picks HyDE. The selector is installed on the lesson's
`Rewriter` instance, so `Pipeline.query` calls it. It is scored on 14
hand-labelled queries: the 4 eval queries, 2 atomic queries with "and", 4
jargon queries and 4 multi-topic ones. The lesson's own `pick_strategy` is
scored on the same 14. Then the lesson's eval is re-run under each forced
strategy.

**ANSWER: the selector picks right on 13 of 14 labelled queries; the
lesson's `pick_strategy` gets 7 of 14.** The selector's one miss is "what
does p95 sizing mean for workers": 1 jargon token in 5 content tokens is a
ratio of 0.20, under the 0.25 cut. The lesson's version splits both atomic
"and" queries and never splits on "or" or "versus". Its four-word jargon
list (api, function, endpoint, config) misses 3 of the 4 jargon queries;
it catches only the one that says "config". On the 4 eval queries both
selectors pick multi-query.

**FINDING: the rewriter cannot move the eval.** Forcing HyDE, multi-query,
decomposition, or no rewrite at all gives the same five metrics: recall@5
1.000, precision@1 0.500, mrr 0.708. The candidate pool holds 8 to 10 of
the 12 chunks and always contains the gold documents (4 of 4 queries), and
the reranker re-sorts that pool the same way.

**FINDING: the HyDE table is the eval set's answers, and nothing reaches
it.** `_REWRITE_HYDE` has exactly 4 keys, the 4 eval queries. 2 of its 4
hypotheticals are the gold document word for word (d3, d7). No other query
gets one: `rewrite_hyde` returns None on all 4 jargon queries, and the
pipeline falls back to a plain search.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "69-end-to-end-rag-system"
WH = {"how", "what", "where", "which", "why", "when", "who"}
LABELLED = [
    ("how is lexical and semantic retrieval combined", "multiquery"),
    ("which roles and groups does the permission gate inherit", "multiquery"),
    ("what does AbortMultipartOnFail do", "hyde"),
    ("which config sets the TTL on the OPA cache", "hyde"),
    ("what does p95 sizing mean for workers", "hyde"),
    ("how much memory does a float32 ANN vector use", "hyde"),
    ("what is the abort threshold and how is authorization checked", "decompose"),
    ("how are workers cancelled and how is the pool sized", "decompose"),
    ("where is the permission gate or how are stale records dropped", "decompose"),
    ("compare reciprocal rank fusion versus simple union of results", "decompose"),
]


def jargon_lexicon(ref):
    raw = " ".join(t for _, t in ref.CORPUS)
    ids = re.findall(r"\b(?:[A-Z]{2,}\w*|\w*\d\w*|[A-Za-z]+[a-z][A-Z]\w*)\b", raw)
    return {w.lower() for w in ids if len(ref.tokenize(w)) == 1} | {"api", "function", "endpoint", "config"}


def splits(ref, query):
    clauses = re.split(r"\s+(?:and|or|versus|vs)\s+", query.lower(), maxsplit=1)
    if len(clauses) < 2:
        return False
    opener = ref.tokenize(clauses[1])[:1]
    return bool(opener and opener[0] in WH) or min(len(ref._content_tokens(c)) for c in clauses) >= 3


def make_pick(ref, lexicon):
    def pick(query):
        if splits(ref, query):
            return "decompose"
        content = [t for t in ref.tokenize(query) if t not in ref._STOP]
        if content and sum(t in lexicon for t in content) / len(content) >= 0.25:
            return "hyde"
        return "multiquery"

    return pick


def forced_eval(ref, p, strategy):
    p.rewriter.pick_strategy = lambda q: strategy
    return {k: round(v, 3) for k, v in ref.run_eval(p).items()}


def selector_scores(ref, pick, lesson_pick):
    labelled = [(e.query, "multiquery") for e in ref.EVAL_QUERIES] + LABELLED
    return {
        "mine": sum(pick(q) == want for q, want in labelled), "lesson": sum(lesson_pick(q) == want for q, want in labelled),
        "n": len(labelled), "eval_picks": sorted({pick(e.query) for e in ref.EVAL_QUERIES}),
        "lesson_misses": [q for q, want in labelled if lesson_pick(q) != want],
    }


def pool_audit(ref, p):
    pool = [p.index.search(e.query, k_out=p.top_n) for e in ref.EVAL_QUERIES]
    return {
        "pool": [len(c) for c in pool],
        "gold_in_pool": sum(e.gold_doc_ids <= {c.doc_id for c in cs} for e, cs in zip(ref.EVAL_QUERIES, pool)),
    }


def hyde_audit(ref, p):
    hyde = {ref._REWRITE_HYDE.get(e.query) for e in ref.EVAL_QUERIES}
    return {
        "hyde_keys_are_eval": set(ref._REWRITE_HYDE) == {e.query for e in ref.EVAL_QUERIES},
        "verbatim": sorted(d for d, t in ref.CORPUS if t in hyde),
        "hyde_elsewhere": sum(p.rewriter.rewrite_hyde(q) is not None for q, w in LABELLED if w == "hyde"),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pick = make_pick(ref, jargon_lexicon(ref))
    p = ref.build_pipeline()
    out = {**selector_scores(ref, pick, p.rewriter.pick_strategy), **hyde_audit(ref, p), **pool_audit(ref, p)}
    p.rewriter.pick_strategy = pick
    out["installed"] = {k: round(v, 3) for k, v in ref.run_eval(p).items()}
    out["forced"] = {s: forced_eval(ref, p, s) for s in ("hyde", "multiquery", "decompose", "none")}
    return out


def verify(result):
    r = result
    base = {"recall@5": 1.0, "precision@1": 0.5, "mrr": 0.708, "faithfulness": 1.0, "answer_relevance": 1.0}
    return [
        practice.Check(
            "ANSWER: the selector picks right on 13/14 labelled queries, the lesson's pick_strategy on 7/14",
            (r["mine"], r["lesson"], r["n"], r["eval_picks"], len(r["lesson_misses"])) == (13, 7, 14, ["multiquery"], 7),
            f"selector {r['mine']}/{r['n']}, lesson {r['lesson']}/{r['n']}; eval queries get {r['eval_picks']}; "
            f"lesson misses {r['lesson_misses']}",
        ),
        practice.Check(
            "FINDING: every forced strategy, and no rewrite at all, gives the same five metrics",
            all(m == base for m in r["forced"].values()) and r["installed"] == base
            and (r["gold_in_pool"], min(r["pool"]), max(r["pool"])) == (4, 8, 10),
            f"forced {r['forced']}; candidate pool sizes {r['pool']} of 12, gold in pool {r['gold_in_pool']}/4",
        ),
        practice.Check(
            "FINDING: the HyDE table is keyed by the 4 eval queries and 2 hypotheticals are gold docs verbatim",
            (r["hyde_keys_are_eval"], r["verbatim"], r["hyde_elsewhere"]) == (True, ["d3", "d7"], 0),
            f"keys == eval queries: {r['hyde_keys_are_eval']}; verbatim gold docs {r['verbatim']}; "
            f"hypotheticals for the 4 jargon queries: {r['hyde_elsewhere']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
