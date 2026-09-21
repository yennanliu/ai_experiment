"""Exercise 3 — only the description is indexed, and the code is the API.

    Replace token-overlap retrieval with sentence-transformers embeddings (or
    a BM25 stdlib impl). Measure retrieval@5 on a 50-skill toy library.

Reading of the exercise: BM25 is the stdlib option, so that is what is built
-- but swapping the *scorer* turns out to be the smaller half. `search`
tokenises `skill.description` and nothing else, while `Skill` also carries
`code` and `tags`, and the thing a caller usually knows is the name of the
function they want to call. The measurement therefore separates the scorer
from the field set.

**ANSWER: retrieval@5 over a 50-skill library is 20/20 for BM25 across all
three fields and 0/20 for the shipped search.** Each query names an
identifier that appears in a skill's `code` and not in its prose, so the
shipped scorer finds **0** candidates to rank -- it returns an empty list
rather than a wrong answer.

**FINDING: the scorer is not the reason.** Run over descriptions only, BM25
also scores **0/20**: the term is not in the field. Run over the same three
fields, plain token overlap scores **20/20**. The field set moves the number
by **20** and the scorer moves it by **0** on this query shape.

**FINDING: where the scorer does matter is length.** Two skills that both
contain every query term score **0.667** and **0.1** under the shipped
Jaccard -- a **6.7x** penalty purely for having a longer description -- and
**2.0x** apart under BM25, which normalises length against the corpus mean
instead of against the query's union.

**FINDING: `tag_filter` fails silently.** It is an exact `in skill.tags`
test, so `tag_filter="crafting"` against skills tagged `craft` returns **0**
results with no error -- indistinguishable from a library that genuinely has
no crafting skills.

Structure: `index_text()` chooses the field set, `bm25()` and `overlap()`
are the two scorers, and every combination is measured on the same library.
"""

from __future__ import annotations

import collections
import math

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "10-skill-libraries-voyager"
K1, B, TOP_K = 1.5, 0.75, 5
VERBS = ("gather", "smelt", "place", "craft", "scout", "trade", "repair", "store",
         "plant", "tame")
NOUNS = ("ore", "plank", "torch", "anvil", "beacon")
VERBOSE = ("smelt ore in the furnace while keeping the fuel topped up and logging "
           "every batch for the operator to review later on")


def build(ref):
    """Fifty skills whose prose never repeats the identifier in their code."""
    lib = ref.SkillLibrary()
    for verb in VERBS:
        for noun in NOUNS:
            lib.register(ref.Skill(
                name=f"{verb}_{noun}", code=f"api_{verb}{noun}(count, retries)",
                description=f"a routine that handles {noun} in the world by {verb}ing "
                            "carefully and reporting back",
                fn=lambda ctx: "ok", tags=(verb, noun)))
    return lib


def prepare(lib, fields):
    """The field set is the variable the exercise is really about."""
    names = lib.list_names()
    parts = {"description": lambda s: s.description, "code": lambda s: s.code,
             "tags": lambda s: " ".join(s.tags)}
    docs = [" ".join(parts[f](lib.get(name)) for f in fields).lower()
            .replace("(", " ").replace(",", " ").split() for name in names]
    return names, docs


def overlap(lib, query, fields, top_k=TOP_K):
    names, docs = prepare(lib, fields)
    terms = set(query.lower().split())
    scored = [(len(terms & set(d)) / len(terms | set(d)), n)
              for n, d in zip(names, docs) if terms & set(d)]
    return [name for _, name in sorted(scored, reverse=True)[:top_k]]


def score_doc(doc, terms, df, total, avgdl):
    counts, score = collections.Counter(doc), 0.0
    for term in terms & set(doc):
        idf = math.log(1 + (total - df[term] + 0.5) / (df[term] + 0.5))
        score += idf * counts[term] * (K1 + 1) / (
            counts[term] + K1 * (1 - B + B * len(doc) / avgdl))
    return score


def bm25_all(lib, query, fields):
    names, docs = prepare(lib, fields)
    avgdl = sum(len(doc) for doc in docs) / len(docs)
    df = collections.Counter(term for doc in docs for term in set(doc))
    terms = set(query.lower().split())
    return {n: score_doc(d, terms, df, len(docs), avgdl) for n, d in zip(names, docs)}


def bm25(lib, query, fields, top_k=TOP_K):
    scored = [(score, name) for name, score in bm25_all(lib, query, fields).items()
              if score]
    return [name for _, name in sorted(scored, reverse=True)[:top_k]]


def recall(lib, scorer, fields, queries):
    return sum(want in scorer(lib, q, fields) for q, want in queries)


def length_penalty(ref):
    """Two skills that both contain every query term, differing only in length."""
    lib = ref.SkillLibrary()
    for name, text in (("terse", "smelt ore quickly"), ("verbose", VERBOSE)):
        lib.register(ref.Skill(name, text, "x()", lambda ctx: "ok"))
    shipped = {skill.name: score for score, skill in lib.search("smelt ore", top_k=2)}
    raw = bm25_all(lib, "smelt ore", ("description",))
    scores = {n: round(raw[n], 3) for n in ("terse", "verbose")}
    return {"shipped": {k: round(v, 3) for k, v in shipped.items()}, "bm25": scores,
            "shipped_ratio": round(shipped["terse"] / shipped["verbose"], 1),
            "bm25_ratio": round(scores["terse"] / scores["verbose"], 1)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    lib = build(ref)
    api = [(f"api_{v}{n}", f"{v}_{n}") for v in VERBS[:4] for n in NOUNS][:20]
    everything = ("description", "code", "tags")
    return {
        "penalty": length_penalty(ref),
        "library": len(lib.list_names()), "queries": len(api),
        "shipped": sum(bool(lib.search(q, TOP_K)) for q, _ in api),
        "bm25_all": recall(lib, bm25, everything, api),
        "bm25_desc": recall(lib, bm25, ("description",), api),
        "overlap_all": recall(lib, overlap, everything, api),
        "tag_hits": len(lib.search("routine", TOP_K, tag_filter="crafting")),
        "real_tag_hits": len(lib.search("routine", TOP_K, tag_filter="craft")),
    }


def verify(result):
    penalty = result["penalty"]
    return [
        practice.Check(
            "ANSWER: 20/20 for BM25 over three fields, 0/20 for the shipped search",
            all([result["library"] == 50, result["queries"] == 20,
                 result["bm25_all"] == 20, result["shipped"] == 0]),
            f"over {result['library']} skills whose queries name an identifier from "
            f"their code, BM25 across all three fields reaches retrieval@5 of "
            f"{result['bm25_all']}/{result['queries']} and the shipped search returns a "
            f"non-empty result {result['shipped']} times",
        ),
        practice.Check(
            "FINDING: the scorer is not the reason",
            all([result["bm25_desc"] == 0, result["overlap_all"] == 20,
                 result["bm25_all"] == result["overlap_all"]]),
            f"BM25 over descriptions only scores {result['bm25_desc']}/"
            f"{result['queries']} and token overlap over all three fields scores "
            f"{result['overlap_all']}/{result['queries']} -- the field set moves the "
            "number by 20 and the scorer by 0",
        ),
        practice.Check(
            "FINDING: where the scorer does matter is length",
            all([penalty["shipped_ratio"] == 6.7, penalty["bm25_ratio"] == 2.0,
                 penalty["shipped"]["terse"] > penalty["shipped"]["verbose"]]),
            f"two skills containing every query term score {penalty['shipped']} under "
            f"Jaccard -- a {penalty['shipped_ratio']}x penalty for the longer prose -- "
            f"and {penalty['bm25']} under BM25, a {penalty['bm25_ratio']}x one. BM25 "
            "normalises length against the corpus mean rather than the query's union",
        ),
        practice.Check(
            "FINDING: tag_filter fails silently",
            all([result["tag_hits"] == 0, result["real_tag_hits"] == TOP_K]),
            f"tag_filter is an exact `in skill.tags` test, so 'crafting' against skills "
            f"tagged 'craft' returns {result['tag_hits']} while 'craft' returns "
            f"{result['real_tag_hits']} -- a typo is indistinguishable from an empty "
            "library",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
