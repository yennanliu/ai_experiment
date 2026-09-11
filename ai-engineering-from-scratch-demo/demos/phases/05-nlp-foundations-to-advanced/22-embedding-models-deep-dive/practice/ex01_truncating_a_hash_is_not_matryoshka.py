"""Exercise 1 — truncating a hash is not Matryoshka.

    **Easy.** Encode 100 sentences with `bge-small-en-v1.5` at full dim (384),
    then at Matryoshka 128. Measure MRR drop on 10 queries.

Reading of the exercise: `sentence_transformers` and `torch` are absent and the
checkpoint is not downloadable, so the encoder is the lesson's own `hash_embed`
at 256 dimensions over 40 sentences on 10 topics, queried by their topic names.
MRR falls from **0.9500** to **0.6663** between 256 and 128 -- a drop of 0.2837,
about 30% relative. The number is real and it does not mean what the exercise
wants it to mean.

Matryoshka is a property of *training*: a nested loss makes the first `k`
coordinates a usable embedding on their own. A hash embedding has independent,
interchangeable dimensions, so truncating it does not compress the
representation -- it deletes vocabulary. Exactly the tokens whose hash index
lands at or above the cut disappear, and **108 of 236 vocabulary types survive at
128**, 56 at 64, 30 at 32. The measured "Matryoshka drop" is a bucket-loss curve.

Below 128 the deletion starts removing whole documents. 2 documents have no
surviving token at 128, 6 at 64 and 14 at 32, and `truncate_matryoshka` returns a
zero vector un-normalised, so those documents score cosine 0.0 against every
query. `rank` sorts `(score, index)` tuples in reverse, so ties break by
*descending index* -- at 32 dimensions a seventh of the ranking is corpus
insertion order, read backwards.

The 256-dimension baseline is itself lossy, which makes the drop an
underestimate. 236 vocabulary types occupy 151 of the 256 buckets: 61 buckets
hold more than one token and 38 of those hold tokens with opposite signs, which
cancel in the sum. "Full dimension" here is already a compression with collisions
in it.

Structure: `TOPICS` is one line per topic -- the query, then four sentences;
`embed` and `shrink` wrap the lesson's encoder and truncation, `mrr` and
`recall_at` score a truncation width, and `sweep` runs the widths.
"""

from __future__ import annotations

import importlib.util

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "22-embedding-models-deep-dive"

UNAVAILABLE = ("sentence_transformers", "transformers", "torch")
DIM, WIDTHS = 256, (256, 192, 128, 96, 64, 32)
TOPICS = """iphone release date|Apple released the first iPhone on June 29 2007.|Macworld 2007 featured the iPhone announcement by Steve Jobs.|The original iPhone went on sale in the United States in 2007.|Apple sold one million iPhone units within seventy four days.
android operating system|Android launched in 2008 as Google mobile operating system.|The first Android handset shipped in October 2008.|Google acquired Android Incorporated in 2005.|Android became the most widely installed mobile platform.
photosynthesis in plants|Photosynthesis converts light energy into chemical energy in plants.|Chloroplasts contain the chlorophyll that absorbs sunlight.|Plants release oxygen as a by product of photosynthesis.|The Calvin cycle fixes carbon dioxide into sugar.
insulin and diabetes|Insulin regulates the amount of glucose in the blood.|Type one diabetes results from insufficient insulin production.|The pancreas secretes insulin after a meal raises blood sugar.|Frederick Banting isolated insulin in 1921.
roman empire history|The Roman Empire reached its greatest extent under Trajan.|Augustus became the first Roman emperor in 27 BC.|The western Roman Empire fell in the year 476.|Roman legions built roads across the conquered provinces.
climate and rainfall|Rainfall in the region declined by thirty percent since 2019.|Drought conditions persisted across the southern plains.|Warmer oceans increase the moisture that storms can carry.|Monsoon patterns shifted later into the summer season.
neural network training|Backpropagation computes gradients through the network layers.|Stochastic gradient descent updates weights on small batches.|Overfitting occurs when a model memorises its training data.|Dropout randomly disables units during training.
mortgage interest rates|The central bank raised interest rates by a quarter point.|Mortgage costs rose sharply for new borrowers this year.|Fixed rate loans protect borrowers from later increases.|Lenders tightened approval standards after the rate rise.
volcanic eruptions|Magma reaches the surface where tectonic plates diverge.|The eruption sent an ash cloud across the region.|Lava flows destroyed several villages on the island slope.|Volcanic winters follow the largest explosive eruptions.
shakespeare plays|Hamlet was first performed in the early seventeenth century.|Shakespeare wrote thirty seven plays and many sonnets.|The Globe theatre staged most of his later works.|Macbeth dramatises the corrupting effect of ambition."""
ROWS = tuple(tuple(line.split("|")) for line in TOPICS.splitlines())
DOCS = tuple(text for row in ROWS for text in row[1:])
GOLD = tuple(index for index, row in enumerate(ROWS) for _ in row[1:])
QUERIES = tuple((row[0], index) for index, row in enumerate(ROWS))
PER_TOPIC = len(DOCS) // len(ROWS)


def shrink(ref, vec, width):
    """The lesson's truncation, with full width passed through unchanged."""
    return vec if width == DIM else ref.truncate_matryoshka(vec, width)


def scored(ref, width):
    """The ranked corpus for every query, at one truncation width."""
    corpus = [shrink(ref, ref.hash_embed(doc, DIM), width) for doc in DOCS]
    return {topic: ref.rank(corpus, shrink(ref, ref.hash_embed(query, DIM), width))
            for query, topic in QUERIES}


def tasks(ranked):
    """MRR and recall@10 from one width's rankings."""
    reciprocal = [1 / next(pos for pos, (_, i) in enumerate(order, 1) if GOLD[i] == topic)
                  for topic, order in ranked.items()]
    hits = [sum(1 for _, i in order[:10] if GOLD[i] == topic) / PER_TOPIC
            for topic, order in ranked.items()]
    return {"mrr": round(sum(reciprocal) / len(QUERIES), 4),
            "recall": round(sum(hits) / len(QUERIES), 4)}


def sweep(ref):
    """The two tasks plus surviving vocabulary and empty documents, per width."""
    vocab = sorted({token for doc in DOCS for token in ref.tokenize(doc)})
    rows = {}
    for width in WIDTHS:
        rows[width] = dict(
            tasks(scored(ref, width)),
            vocab=sum(1 for token in vocab if ref.hash_token(token, DIM) < width),
            empty=sum(1 for doc in DOCS if not any(shrink(ref, ref.hash_embed(doc, DIM), width))),
        )
    return rows, len(vocab)


def collisions(ref, vocab_size):
    """Buckets shared by more than one token, and those whose signs cancel."""
    buckets = {}
    for token in sorted({token for doc in DOCS for token in ref.tokenize(doc)}):
        buckets.setdefault(ref.hash_token(token, DIM), []).append(token)
    shared = [group for group in buckets.values() if len(group) > 1]
    cancelling = [group for group in shared
                  if len({ref.hash_token(token, 2, seed=1) for token in group}) > 1]
    return {"used": len(buckets), "shared": len(shared), "cancelling": len(cancelling),
            "vocab": vocab_size, "example": sorted(cancelling, key=len)[-1]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows, vocab_size = sweep(ref)
    return {
        "absent": [n for n in UNAVAILABLE if importlib.util.find_spec(n) is None],
        "docs": len(DOCS), "queries": len(QUERIES), "rows": rows,
        "drop": round(rows[DIM]["mrr"] - rows[128]["mrr"], 4),
        "relative": round((1 - rows[128]["mrr"] / rows[DIM]["mrr"]) * 100, 1),
        "hash": collisions(ref, vocab_size),
        "ties": ref.rank([[0.0] * DIM] * 4, [0.0] * DIM),
    }


def verify(result):
    rows, hash_stats = result["rows"], result["hash"]
    return [
        practice.Check(
            "ANSWER: MRR falls from 0.9500 to 0.6663 between 256 and 128 dimensions",
            rows[128]["mrr"] < rows[DIM]["mrr"],
            f"{result['absent']} are all absent, so the encoder is the lesson's own `hash_embed` "
            f"over {result['docs']} sentences on {result['queries']} topics: MRR "
            f"{rows[DIM]['mrr']} at {DIM} against {rows[128]['mrr']} at 128, a drop of "
            f"{result['drop']} or {result['relative']}% relative",
        ),
        practice.Check(
            "MECHANISM: but truncation here deletes vocabulary rather than compressing it",
            rows[128]["vocab"] < hash_stats["vocab"],
            f"Matryoshka is a training property -- a nested loss makes the first k coordinates a "
            f"usable embedding. `hash_embed` has interchangeable dimensions, so cutting at k drops "
            f"every token whose hash index is at or above k: "
            f"{[rows[w]['vocab'] for w in WIDTHS]} of {hash_stats['vocab']} types survive at "
            f"{list(WIDTHS)}. The curve measures bucket loss",
        ),
        practice.Check(
            "FINDING: below 128 the deletion starts removing whole documents",
            rows[32]["empty"] > rows[128]["empty"] > 0,
            f"documents with no surviving token become the zero vector, which "
            f"`truncate_matryoshka` returns un-normalised: {[rows[w]['empty'] for w in WIDTHS]} "
            f"of {result['docs']} documents at {list(WIDTHS)}. Each scores cosine 0.0 against "
            "every query",
        ),
        practice.Check(
            "MECHANISM: and their order is the corpus's, read backwards",
            [index for _, index in result["ties"]] == [3, 2, 1, 0],
            f"`rank` sorts `(score, index)` tuples in reverse, so equal scores break by "
            f"descending index: four identical vectors rank {result['ties']}. At 32 dimensions "
            f"{rows[32]['empty']} documents are tied at 0.0, and that part of the ranking is "
            "insertion order",
        ),
        practice.Check(
            "FINDING: the full-width baseline is already a lossy compression",
            hash_stats["used"] < hash_stats["vocab"],
            f"{hash_stats['vocab']} vocabulary types occupy {hash_stats['used']} of the {DIM} "
            f"buckets: {hash_stats['shared']} buckets hold more than one token and "
            f"{hash_stats['cancelling']} of those hold opposite signs, which cancel in the sum -- "
            f"{hash_stats['example']} share one coordinate. The drop is measured from a floor",
        ),
        practice.Check(
            "CONTROL: recall moves the same way, so the shape is not an MRR artefact",
            rows[DIM]["recall"] > rows[128]["recall"] > rows[32]["recall"],
            f"recall@10 runs {[rows[w]['recall'] for w in WIDTHS]} across {list(WIDTHS)}, "
            f"against MRR {[rows[w]['mrr'] for w in WIDTHS]}. Both track the surviving "
            "vocabulary, which is the quantity actually being varied",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
