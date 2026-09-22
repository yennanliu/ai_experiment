"""Exercise 2 — the scorer has no term frequency, because it uses a set.

    Implement BM25 properly over the archival store (term frequency, inverse
    document frequency). Measure recall@10 on a toy fact set versus the
    token-overlap baseline.

Reading of the exercise: `ArchivalStore.search` is described as "BM25-esque"
and is Jaccard over `set(text.split())` -- no term frequency, because a set
has none; no inverse document frequency, because every shared token counts
one; and a length penalty in the denominator that works the wrong way round.
Each of the three missing pieces is measured on its own, and then both
scorers are run over a **38**-record store with **8** queries of exactly one
relevant record each.

**ANSWER: recall@10 is **8/8** for BM25 and **0/8** for token overlap.**
Every query shares four common words with **30** filler records and one rare
word with its target. The overlap scorer ranks all thirty fillers above the
target; BM25 ranks the target first every time.

**FINDING: no term frequency.** Two records that differ only in repeating
the rare term five times score **identically** under the shipped scorer,
because `set(record.text.lower().split())` has discarded the counts before
the arithmetic starts. BM25 separates them.

**FINDING: no inverse document frequency.** `agent` appears in **38** of
**38** records and the rare term in **1**, and both contribute exactly **1**
to the overlap. A word that cannot discriminate is weighted like a word that
can only discriminate.

**FINDING: the length penalty runs backwards.** The target record contains
**every** query term and scores **0.227**; a filler missing one of them
scores **0.400**, because the union in Jaccard's denominator grows with the
document. The record that said more is ranked lower for having said it.

Structure: `bm25()` is the port; `FILLERS`/`TARGETS` are built so the two
scorers disagree for a stated reason rather than by luck.
"""

from __future__ import annotations

import collections
import math

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "07-memory-virtual-context-memgpt"
RARE = ("ava", "kwame", "yusuf", "mirela", "tobias", "anjali", "ingrid", "rashid")
COMMON = "agent memory note about"
FILLERS = 30
K1, B, TOP_K = 1.5, 0.75, 10


def filler(index):
    return f"{COMMON} session {index} tier and window"


def target(name):
    return (f"{COMMON} session 99 tier and window plus the detail that {name} "
            "configured the sales retrieval bot with twelve registered tools")


def build(ref, extra=()):
    store = ref.ArchivalStore()
    for index in range(FILLERS):
        store.insert(filler(index), tags=("filler",))
    ids = {name: store.insert(target(name), tags=("target",)) for name in RARE}
    for text in extra:
        store.insert(text, tags=("extra",))
    return store, ids


def score_doc(doc, terms, df, total, avgdl, k1=K1, b=B):
    """One document's BM25 score: term frequency, inverse document frequency, length."""
    counts, score = collections.Counter(doc), 0.0
    for term in terms & set(doc):
        idf = math.log(1 + (total - df[term] + 0.5) / (df[term] + 0.5))
        norm = counts[term] + k1 * (1 - b + b * len(doc) / avgdl)
        score += idf * counts[term] * (k1 + 1) / norm
    return score


def bm25(store, query, top_k=TOP_K):
    """Okapi BM25 over the same records the shipped scorer reads."""
    records = store._records
    docs = [record.text.lower().split() for record in records]
    total, avgdl = len(docs), sum(len(doc) for doc in docs) / len(docs)
    df = collections.Counter(term for doc in docs for term in set(doc))
    terms = set(query.lower().split())
    scored = [(score_doc(doc, terms, df, total, avgdl), record)
              for record, doc in zip(records, docs)]
    scored = sorted((row for row in scored if row[0]), key=lambda pair: -pair[0])
    return [record for _, record in scored[:top_k]]


def recall(store, ids, scorer):
    hits = 0
    for name in RARE:
        found = scorer(store, f"{COMMON} {name}")
        hits += ids[name] in [record.rid for record in found]
    return hits


def jaccard(query, text):
    left, right = set(query.lower().split()), set(text.lower().split())
    return round(len(left & right) / len(left | right), 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    store, ids = build(ref)
    repeated = f"{target('kwame')} kwame kwame kwame kwame"
    twin_store, twin_ids = build(ref, extra=(repeated,))
    twin_query = f"{COMMON} kwame"
    docs = [record.text.lower().split() for record in store._records]
    df = collections.Counter(term for doc in docs for term in set(doc))
    return {
        "records": store.count(),
        "bm25": recall(store, ids, lambda s, q: bm25(s, q)),
        "overlap": recall(store, ids, lambda s, q: s.search(q, top_k=TOP_K)),
        "queries": len(RARE),
        "twin_overlap": (jaccard(twin_query, target("kwame")),
                         jaccard(twin_query, repeated)),
        "twin_bm25_order": [record.tags[0] for record in bm25(twin_store, twin_query, 2)],
        "df_common": df["agent"], "df_rare": df["ava"],
        "overlap_of": (jaccard(f"{COMMON} ava", target("ava")),
                       jaccard(f"{COMMON} ava", filler(0))),
        "terms_in_target": len(set(f"{COMMON} ava".split())
                               & set(target("ava").lower().split())),
        "terms_in_filler": len(set(f"{COMMON} ava".split())
                               & set(filler(0).lower().split())),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: recall@10 is 8/8 for BM25 and 0/8 for token overlap",
            all([result["bm25"] == 8, result["overlap"] == 0,
                 result["queries"] == 8, result["records"] == 38]),
            f"over {result['records']} records and {result['queries']} queries with one "
            f"relevant record each, BM25 reaches recall@10 of {result['bm25']}/"
            f"{result['queries']} and the shipped token overlap "
            f"{result['overlap']}/{result['queries']}. The overlap scorer puts all 30 "
            "fillers above the target on every query",
        ),
        practice.Check(
            "FINDING: no term frequency, because a set has none",
            all([result["twin_overlap"][0] == result["twin_overlap"][1],
                 result["twin_bm25_order"][0] == "extra"]),
            f"a record and the same record with the rare term repeated five times score "
            f"{result['twin_overlap']} -- identical -- because "
            f"set(record.text.lower().split()) discards the counts before the arithmetic "
            f"starts. BM25 ranks them {result['twin_bm25_order']}, the repeated one first",
        ),
        practice.Check(
            "FINDING: no inverse document frequency",
            all([result["df_common"] == 38, result["df_rare"] == 1,
                 result["terms_in_target"] == 5, result["terms_in_filler"] == 4]),
            f"'agent' appears in {result['df_common']} of {result['records']} records and "
            f"the rare name in {result['df_rare']}, and both contribute exactly 1 to the "
            f"overlap count -- {result['terms_in_target']} shared terms for the target "
            f"against {result['terms_in_filler']} for a filler. A word that cannot "
            "discriminate is weighted like the only word that can",
        ),
        practice.Check(
            "FINDING: the length penalty runs backwards",
            all([result["overlap_of"] == (0.227, 0.4),
                 result["overlap_of"][0] < result["overlap_of"][1],
                 result["terms_in_target"] > result["terms_in_filler"]]),
            f"the target contains every query term and scores "
            f"{result['overlap_of'][0]}; a filler missing one of them scores "
            f"{result['overlap_of'][1]}, because Jaccard's union grows with the document. "
            "The record that said more is ranked lower for having said it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
