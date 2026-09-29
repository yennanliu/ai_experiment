"""Exercise 4 -- BM25F lifts symbol lookups from 375 to 383 of 400, and the multiplier trick misses docs with long bodies.

    Implement BM25F properly (per-field length normalization rather than the multiplier trick) and compare on a corpus where symbol matches matter most.

Reading of the exercise: BM25F is the per-field form of Zaragoza, Craswell,
Taylor, Saria and Robertson, "Microsoft Cambridge at TREC-13", section 4.1
(https://trec.nist.gov/pubs/trec13/papers/microsoft-cambridge.web.hard.pdf,
read 2026-09-29): each field's term count is divided by
(1 - b_f + b_f * len_f / avglen_f), the fields are summed with weights W_f,
and one saturation is applied to the sum. The paper writes x / (K1 + x); the
constant (k1 + 1) factor is kept here so that a one-field BM25F equals the
lesson's BM25 exactly. The lesson's "multiplier trick" (repeat title tokens
3x, normalise by the combined length) is the earlier Robertson, Zaragoza and
Taylor CIKM 2004 scheme, which does not normalise per field. Both run with
the lesson's settings: title weight 3, body 1, b = 0.75, k1 = 1.5, and the
lesson's smoothed IDF (as in https://en.wikipedia.org/wiki/Okapi_BM25, read
2026-09-29). The corpus where symbols matter most is a code index generated
by `make_fixture()`: 10 seeds x 40 docs, each titled with a camelCase symbol,
with a filler body of 8-150 words that calls 0-8 other symbols. Each symbol is
queried, and its own doc is the one right answer.

**ANSWER: BM25F finds the defining doc first for 383 of 400 symbols, the
multiplier trick for 375.** MRR is 0.9788 against 0.9658. BM25F ranks the
defining doc higher on 18 queries and lower on 8. With only a body field,
this BM25F reproduces the lesson's `BM25Index` scores to 8.9e-16.

**FINDING: the multiplier trick charges a title match for the body's
length.** The 25 docs it fails to put first have bodies averaging 130.2
words, against 83.3 for the corpus. Their repeated title tokens are divided
by the combined title-plus-body length, so a short caller that mentions the
symbol a few times can outscore it. BM25F normalises the title by title
length only; its 17 misses average 82.1 words, the corpus norm. The lesson
says the multiplier "keeps the math identical"; it keeps BM25's formula, but
not the per-field normalisation this exercise asks for.

Structure: `bm25f()` indexes any fields, `term_score()` is the per-term
formula and `search()` ranks with it; `make_fixture()`
generates the code corpus; `ranks()` queries each symbol.
"""

from __future__ import annotations

import math
import random
from collections import Counter

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "65-hybrid-retrieval-bm25-dense"
FIELDS = {"title": (3.0, 0.75), "body": (1.0, 0.75)}  # (weight, b): the lesson's 3/1 and 0.75
FILLER = ("the a value is returned when called with config request handler cache retry timeout buffer "
          "stream parse token file path user session error state queue worker lock").split()
SEEDS, N_DOCS = range(10), 40


def bm25f(docs, tokenize, fields=FIELDS, k1=1.5):
    """Index `docs` field by field; returns search(query, k) -> [(doc, score)]."""
    tfs = [{f: Counter(tokenize(d.field_text(f))) for f in fields} for d in docs]
    ix = {"docs": docs, "tfs": tfs, "tok": tokenize, "fields": fields, "k1": k1, "n": len(docs)} | field_stats(tfs, fields)
    return lambda query, k=10: search(ix, query, k)


def field_stats(tfs, fields):
    """Average length per field, and document frequency over all fields."""
    return {"avg": {f: sum(sum(t[f].values()) for t in tfs) / len(tfs) or 1.0 for f in fields},
            "df": Counter(term for t in tfs for term in set().union(*(t[f] for f in fields)))}


def term_score(ix, t, term):
    """Zaragoza et al. 2004: per-field normalised tf, weighted sum, then one saturation."""
    x = sum(w * t[f][term] / (1 - b + b * sum(t[f].values()) / ix["avg"][f]) for f, (w, b) in ix["fields"].items())
    idf = math.log((ix["n"] - ix["df"][term] + 0.5) / (ix["df"][term] + 0.5) + 1.0)
    return idf * x * (ix["k1"] + 1) / (ix["k1"] + x)


def search(ix, query, k):
    terms = [x for x in ix["tok"](query) if ix["df"][x]]
    scored = [(d, sum(term_score(ix, t, q) for q in terms)) for d, t in zip(ix["docs"], ix["tfs"])]
    scored.sort(key=lambda p: -p[1])
    return [p for p in scored[:k] if p[1] > 0]


def make_fixture(ref, seed):
    """40 symbol docs: a camelCase name as title, a body of 8-150 filler words that
    mentions 0-8 other symbols, as a caller's docstring would."""
    rng = random.Random(seed)
    verbs, nouns = ["get", "set", "parse", "load", "flush", "build", "check", "sync"], ["Config", "Session", "Buffer", "Token", "Queue", "Cache", "Lock", "Stream"]
    syms = [f"{rng.choice(verbs)}{rng.choice(nouns)}{i}" for i in range(N_DOCS)]
    docs = []
    for i, name in enumerate(syms):
        body = [rng.choice(FILLER) for _ in range(rng.randint(8, 150))]
        for _ in range(rng.randint(0, 8)):
            body.insert(rng.randrange(len(body)), rng.choice(syms))
        docs.append(ref.Doc(f"s{i}", name, " ".join(body)))
    return docs


def ranks(search, docs):
    """Rank of each symbol's own doc when the symbol is the query."""
    return [[h.doc_id for h, _ in search(d.title, len(docs))].index(d.doc_id) + 1 for d in docs]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    mult, prop, lens = [], [], []
    for seed in SEEDS:
        docs = make_fixture(ref, seed)
        idx = ref.BM25Index()
        for d in docs:
            idx.add(d)
        mult += ranks(idx.search, docs)
        prop += ranks(bm25f(docs, ref.tokenize), docs)
        lens += [len(d.body.split()) for d in docs]
    one = ref.BM25Index(field_weights={"body": 1})
    for d in ref.CORPUS:
        one.add(d)
    f = bm25f(ref.CORPUS, ref.tokenize, {"body": (1.0, 0.75)})
    gap = max(abs(a[1] - b[1]) for q in ("retry budget upload", "the policy engine") for a, b in zip(one.search(q, 7), f(q, 7)))
    return {"mult": mult, "bm25f": prop, "lens": lens, "parity": gap}


def summary(rs):
    """(MRR, hits@1) over the symbol queries."""
    return round(sum(1 / r for r in rs) / len(rs), 4), sum(r == 1 for r in rs)


def miss_length(lens, rs):
    """Mean body length of the docs a ranker fails to put first, and how many."""
    miss = [n for n, r in zip(lens, rs) if r > 1]
    return round(sum(miss) / len(miss), 1), len(miss)


def verify(result):
    r, mult, prop = result, summary(result["mult"]), summary(result["bm25f"])
    better = sum(a < b for a, b in zip(r["bm25f"], r["mult"]))
    worse = sum(a > b for a, b in zip(r["bm25f"], r["mult"]))
    lens = (miss_length(r["lens"], r["mult"]), miss_length(r["lens"], r["bm25f"]), round(sum(r["lens"]) / len(r["lens"]), 1))
    return [
        practice.Check(
            "ANSWER: per-field BM25F finds the symbol's own doc more often than the multiplier trick",
            (mult, prop, len(r["mult"]), better, worse) == ((0.9658, 375), (0.9788, 383), 400, 18, 8) and r["parity"] < 1e-12,
            f"(MRR, hits@1) of {len(r['mult'])}: multiplier {mult}, BM25F {prop}; BM25F ranks the doc higher on "
            f"{better}, lower on {worse}; one-field BM25F vs the lesson's BM25: {r['parity']:.1e}",
        ),
        practice.Check(
            "FINDING: the multiplier trick charges a title match for the body's length",
            lens == ((130.2, 25), (82.1, 17), 83.3),
            f"mean body length of missed docs: multiplier {lens[0][0]} ({lens[0][1]} misses), "
            f"BM25F {lens[1][0]} ({lens[1][1]}), corpus {lens[2]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
