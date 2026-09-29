"""Exercise 2 — 25% pruning already drops MaxSim below OCR, and the shipped 50% loses 11%.

    Prune embeddings aggressively (75%, 90%). Find the compression cliff: the point where ViDoRe nDCG@5 drops below the OCR baseline.

Reading of the exercise: ViDoRe is not in the lesson, so nDCG@5 is taken
over its 10-page `CORPUS` with 14 labelled queries (one gold page each; the
first four are `main()`'s). The vision side is the lesson's own pipeline:
`Page.embed_patches`, `doc_prune(keep_fraction)`, `Index.retrieve`. It is
swept over keep fractions 1.0, 0.9, 0.75, 0.5 (the shipped default), 0.25
(the exercise's 75%) and 0.1 (90%), averaged over 50 salts of the builtin
`hash()` that `hash_embed` seeds from (pinned to salted CRC32 so runs
repeat). The OCR baseline is BM25 over an OCR transcript of each page:
typed pages read exactly, chart pages lose the words that are only drawn
("line chart", "bar chart"), and the two handwritten pages come back with
misread characters (`OCR` below).

**ANSWER: the cliff is at 25% pruning (keep 0.75), long before the
exercise's 75% and 90%.** The OCR baseline scores nDCG@5 0.974. Unpruned
MaxSim scores 0.992, 10% pruning 0.975 (a tie in all but the third
decimal), 25% 0.964, the shipped 50% 0.879, 75% 0.624 and 90% 0.511.

**FINDING: the lesson's 50% default is not "without measurable accuracy
loss".** The doc and skill file promise under 0.5% loss at 50%; the lesson's
own pruner loses 0.113 nDCG@5 (11.4%) and 0.219 hit@1 (0.979 -> 0.760).

**FINDING: `doc_prune` ranks by nothing the query can use.** Every patch is
unit-length (largest L2 deviation from 1 is below 1e-12), so the "per-patch
norm" it sorts by is the L1 norm of a unit vector, not the "high-variance"
signal the doc names. Dropping the same number of patches uniformly at
random scores 0.883 at 50%, slightly above the pruner's 0.879.
"""

from __future__ import annotations

import collections
import math
import random
import zlib

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "04-multimodal-document-qa"
SALTS = 50
KEEPS = (1.0, 0.9, 0.75, 0.5, 0.25, 0.1)
QUERIES = [  # (query, gold doc, gold page); the first four are main()'s own
    ("what was the 2024 operating margin change for EMEA", "10k-2024", 88),
    ("late interaction retrieval vs OCR", "paper-vidore-v3", 3),
    ("handwritten experimental figures with error bars", "handwritten-lab", 6),
    ("bar chart comparing segment margins", "chart-report", 12),
    ("segment operating margin table", "10k-2024", 88), ("FX impact and macro headwinds in EMEA", "10k-2024", 92),
    ("consolidated revenue growth executive summary", "10k-2024", 14),
    ("nDCG results vision first vs OCR", "paper-vidore-v3", 7),
    ("M3DocVQA multi page evaluation protocol", "paper-m3docrag", 2),
    ("pH readings in the lab notes", "handwritten-lab", 5), ("circuit board experiment notes", "handwritten-lab", 5),
    ("revenue by segment line chart", "chart-report", 11), ("quarterly revenue Q1 to Q4 APAC", "chart-report", 11),
    ("bar chart of operating margin 2023 vs 2024", "chart-report", 12),
]
OCR = {  # what OCR reads where it differs from the page text; typed pages read exactly
    ("chart-report", 11): "revenue by segment EMEA americas APAC Q1 Q4",
    ("chart-report", 12): "operating margin by segment 2023 2024",
    ("handwritten-lab", 5): "exper1ment nites circuit b0ard pH read1ngs",
    ("handwritten-lab", 6): "qraph w1th annotaled err0r bars fiqure 3 capti0n",
}


def ndcg5(ranked, gold):
    i = ranked.index(gold)
    return 1 / math.log2(i + 2) if i < 5 else 0.0


def bm25_ranking(ref, query, k1=1.2, b=0.75):
    docs = [((d, p), ref.tokenize(OCR.get((d, p), text))) for d, p, text in ref.CORPUS]
    avg = sum(len(t) for _, t in docs) / len(docs)
    df = collections.Counter(w for _, t in docs for w in set(t))
    scores = {}
    for key, toks in docs:
        tf = collections.Counter(toks)
        scores[key] = sum(
            math.log(1 + (len(docs) - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * (k1 + 1)
            / (tf[w] + k1 * (1 - b + b * len(toks) / avg))
            for w in ref.tokenize(query) if tf[w])
    return sorted(scores, key=lambda k: -scores[k])


def random_prune(patches, keep, rng):
    return rng.sample(patches, max(1, int(len(patches) * keep)))


def vision_scores(ref, keep, pruner="doc_prune"):
    """Mean nDCG@5 and hit@1 of the lesson's MaxSim index over all salts."""
    ndcg = hits = 0.0
    for salt in range(SALTS):
        ref.hash = lambda tok, s=salt: zlib.crc32(f"{s}|{tok}".encode())
        rng, idx = random.Random(salt), ref.Index()
        for doc, page, text in ref.CORPUS:
            p = ref.Page(doc_id=doc, page_num=page, content_tokens=ref.tokenize(text))
            p.embed_patches()
            p.patches = ref.doc_prune(p.patches, keep) if pruner == "doc_prune" else random_prune(p.patches, keep, rng)
            idx.add(p)
        for q, d, pg in QUERIES:
            ranked = [(p.doc_id, p.page_num) for p, _ in idx.retrieve(q, k=len(idx.pages))]
            ndcg += ndcg5(ranked, (d, pg))
            hits += ranked[0] == (d, pg)
    n = SALTS * len(QUERIES)
    return round(ndcg / n, 3), round(hits / n, 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ocr = round(sum(ndcg5(bm25_ranking(ref, q), (d, p)) for q, d, p in QUERIES) / len(QUERIES), 3)
    curve = {keep: vision_scores(ref, keep) for keep in KEEPS}
    cliff = next(k for k in KEEPS if curve[k][0] < ocr)
    unit = [ref.hash_embed(t) for _, _, text in ref.CORPUS for t in ref.tokenize(text)]
    return {
        "ocr": ocr, "curve": curve, "cliff_keep": cliff,
        "loss_at_default": round(curve[1.0][0] - curve[0.5][0], 3),
        "random_half": vision_scores(ref, 0.5, pruner="random")[0],
        "max_l2_dev": max(abs(math.sqrt(sum(x * x for x in v)) - 1) for v in unit),
    }


def verify(result):
    r, c = result, result["curve"]
    return [
        practice.Check(
            "ANSWER: the cliff is at 25% pruning (keep 0.75), before the exercise's 75% and 90%",
            (r["ocr"], r["cliff_keep"], [c[k][0] for k in KEEPS]) ==
            (0.974, 0.75, [0.992, 0.975, 0.964, 0.879, 0.624, 0.511]),
            f"OCR baseline nDCG@5 {r['ocr']}; MaxSim nDCG@5 by keep fraction "
            f"{ {k: c[k][0] for k in KEEPS} }; first below baseline at keep {r['cliff_keep']}",
        ),
        practice.Check(
            "FINDING: the shipped 50% prune costs 11% of nDCG@5, not the promised < 0.5%",
            (r["loss_at_default"], c[1.0][1], c[0.5][1]) == (0.113, 0.979, 0.76),
            f"nDCG@5 loss at keep 0.5: {r['loss_at_default']}; hit@1 {c[1.0][1]} -> {c[0.5][1]}",
        ),
        practice.Check(
            "FINDING: doc_prune sorts unit vectors by L1 norm and does no better than random dropping",
            r["max_l2_dev"] < 1e-12 and r["random_half"] == 0.883,
            f"max |L2 - 1| {r['max_l2_dev']:.1e}; random 50% prune nDCG@5 {r['random_half']} vs "
            f"doc_prune {c[0.5][0]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
