"""Exercise 1 — the drift question is answerable without an embedding model.

    Replace the toy vector similarity with a real embedding model
    (sentence-transformers, Ollama, OpenAI embeddings). Measure recall@10 on a
    synthetic long conversation. Does the ranking drift over 1000 writes?

Reading of the exercise: a downloaded model is not available at this tier, so
the substitution is made with the property that actually differs -- a dense
vector with cosine similarity, here hashed character trigrams, which unlike
token overlap survives a word changing its ending. The drift half needs no
model at all: it is a question about the shipped fusion scorer as the store
grows, and that is measured directly.

**ANSWER: recall@10 is **19/20** for trigram cosine and **0/20** for token
overlap.** Over **1020** records, each query is a single word in the
inflection the record does not use -- `drifted` against `drifting`. The
shipped scorer shares **0** tokens with the record it is looking for and
returns nothing at all; a trigram scorer shares `dri`, `rif`, `ift` and finds
**19**. The one miss is the honest part: hashed trigrams are robust to a word
ending, not to meaning, so a stem whose trigrams collide with the filler
vocabulary is still lost.

**FINDING: it drifts once and then freezes, which is worse.** For one fixed
query the shipped top-10 at 100, 250, 500 and 1000 writes overlaps its
predecessor by **0.43**, then **1.0**, then **1.0**. Between 250 and 1000
writes -- **750** new memories -- **0** of them enter the top ten. The
scorer is stable because the best available score was already reached, so
after a few hundred writes the store stops being able to surface anything
new for this query.

**FINDING: the KV arm ignores the query.** `Mem0.search` appends every KV
record for the user at a fixed relevance of **0.4**, with no query term and
no scope check. Asking `refund invoice` as a user with **0** matching records
returns **3** results, all of them KV-floor hits about other subjects.

**FINDING: recency stops contributing within a week.** With the shipped
half-life of **86400** seconds, a record **7** days old scores **0.0078** on
recency and contributes **0.0016** of a fused score. Over a long
conversation the fusion is relevance plus importance, and importance is a
number the writer chose.

Structure: `trigram()` and `cosine()` are the substitute scorer; `drift()`
measures the shipped one at four corpus sizes.
"""

from __future__ import annotations

import math
import zlib

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "09-hybrid-memory-mem0"
STEMS = ("drift", "consolidat", "invalidat", "paginat", "throttl", "shard",
         "reconcil", "backfill", "transcod", "quarantin", "replicat", "compact",
         "rebalanc", "checkpoint", "vectoriz", "deduplicat", "summariz", "escalat",
         "annotat", "provision")
SIZES = (100, 250, 500, 1000)
DRIFT_QUERY = "note about routine agent activity"
VOCAB = ("routine", "agent", "activity", "pipeline", "note", "session", "ingestion",
         "retry", "queue", "batch", "index", "token", "window", "digest", "sync",
         "drain", "replay", "audit")
DIM, TOP_K = 256, 10


def target_text(stem, index):
    return f"session {index} note about {stem}ing the ingestion pipeline"


def filler_text(index):
    """Varied enough that scores differ, so drift is measurable rather than a tie."""
    words = [VOCAB[(index * (k + 3) + k) % len(VOCAB)] for k in range(6)]
    return f"session {index} " + " ".join(words)


def query_for(stem):
    """One word, in the inflection the record does not use."""
    return f"{stem}ed"


def trigram(text, dim=DIM):
    """A dense vector without a download: CRC-hashed character trigrams."""
    vector = [0.0] * dim
    padded = f"  {text.lower()}  "
    for index in range(len(padded) - 2):
        vector[zlib.crc32(padded[index:index + 3].encode()) % dim] += 1.0
    return vector


def cosine(left, right):
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return sum(a * b for a, b in zip(left, right)) / norm if norm else 0.0


def corpus(ref, size):
    mem = ref.Mem0()
    for index in range(size):
        mem.add(filler_text(index), user_id="ava", importance=0.5)
    ids = {stem: mem.add(target_text(stem, 99), user_id="ava", importance=0.5)
           for stem in STEMS}
    return mem, ids


def recall(mem, ids, scorer):
    return sum(ids[stem] in scorer(mem, query_for(stem)) for stem in STEMS)


def overlap_top(mem, query):
    return [record.rid for _, record in mem.vector.search(query, top_k=TOP_K)]


def cosine_top(mem, query):
    probe = trigram(query)
    scored = sorted(((cosine(probe, trigram(record.text)), record.rid)
                     for record in mem.vector._records.values()), reverse=True)
    return [rid for _, rid in scored[:TOP_K]]


def drift(ref, query):
    tops = [set(overlap_top(corpus(ref, size)[0], query)) for size in SIZES]
    return [round(len(a & b) / len(a | b), 2) for a, b in zip(tops, tops[1:])]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    mem, ids = corpus(ref, SIZES[-1])
    lonely = ref.Mem0()
    for text, key in (("ava prefers terse writing", ("writing_style", "terse")),
                      ("ava lives in Lisbon", ("city", "Lisbon")),
                      ("ava owns the curriculum", ("project", "c"))):
        lonely.add(text, user_id="ava", kv_triples=(key,))
    config, week = ref.Mem0Config(), 7 * 86400
    return {
        "records": len(mem.vector._records), "queries": len(STEMS),
        "overlap_recall": recall(mem, ids, overlap_top),
        "cosine_recall": recall(mem, ids, cosine_top),
        "drift": drift(ref, DRIFT_QUERY),
        "kv_hits": len(lonely.search("refund invoice", user_id="ava", top_k=5)),
        "kv_vector_hits": len(lonely.vector.search("refund invoice", top_k=5)),
        "kv_floor": config.w_relevance * 0.4,
        "halflife": config.recency_halflife_s,
        "week_recency": round(0.5 ** (week / config.recency_halflife_s), 4),
        "week_contribution": round(config.w_recency * 0.5 ** (week / 86400), 4),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: recall@10 is 19/20 for trigram cosine and 0/20 for token overlap",
            all([result["cosine_recall"] == 19, result["overlap_recall"] == 0,
                 result["queries"] == 20, result["records"] == 1020]),
            f"over {result['records']} records, each query one word in an inflection the "
            f"record does not use, token overlap reaches recall@10 of "
            f"{result['overlap_recall']}/{result['queries']} and trigram cosine "
            f"{result['cosine_recall']}/{result['queries']}. The single miss is where "
            "trigrams collide rather than mean",
        ),
        practice.Check(
            "FINDING: it drifts once and then freezes, which is worse",
            all([result["drift"] == [0.43, 1.0, 1.0], len(result["drift"]) == 3]),
            f"for one fixed query the shipped top-10 at {SIZES} overlaps its predecessor "
            f"by {result['drift']} on Jaccard. It churns once and then never again: "
            "between 250 and 1000 writes, 750 new memories arrive and 0 of them enter "
            "the top ten, because the best reachable score was already taken",
        ),
        practice.Check(
            "FINDING: the KV arm ignores the query",
            all([result["kv_hits"] == 3, result["kv_vector_hits"] == 0,
                 result["kv_floor"] == 0.24]),
            f"Mem0.search appends every KV record for the user at a fixed relevance of "
            f"0.4 -- a floor of {result['kv_floor']} -- with no query term and no scope "
            f"check, so a query with {result['kv_vector_hits']} vector matches still "
            f"returns {result['kv_hits']} results",
        ),
        practice.Check(
            "FINDING: recency stops contributing within a week",
            all([result["halflife"] == 86400.0, result["week_recency"] == 0.0078,
                 result["week_contribution"] == 0.0016]),
            f"a record 7 days old scores {result['week_recency']} on recency and "
            f"contributes {result['week_contribution']} of a fused score at the shipped "
            f"half-life of {result['halflife']:.0f}s. Over a long conversation the fusion "
            "is relevance plus importance, and importance is a writer's guess",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
