"""Exercise 5 — the shipped index retrieves handwriting worse than OCR misreading 1 character in 5.

    Add handwritten-note support. Render the handwriting corpus, embed with ColQwen, measure retrieval. Compare against a handwriting OCR pipeline.

Reading of the exercise: the handwriting corpus is the two handwritten-lab
pages already in the lesson's `CORPUS`, retrieved against all 10 pages by 4
handwriting queries (the first is `main()`'s own). "Embed with ColQwen" is
the lesson's index as `build_index()` ships it (hash patches, 50%
`doc_prune`, MaxSim), averaged over 50 salts of the builtin `hash()` that
`hash_embed` seeds from (pinned to salted CRC32 so runs repeat). The
handwriting OCR pipeline is BM25 over a transcript in which every letter or
digit of the handwritten pages is misread with probability CER (typed pages
read exactly), swept from 0 to 0.5 over 20 seeded draws.

**ANSWER: the shipped index scores nDCG@5 0.837 (hit@1 0.735) on
handwriting; OCR matches or beats that up to a character error rate of 0.2
(0.850) and falls below it at 0.25 (0.637).**

| CER | 0 | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OCR-BM25 nDCG@5 | 1.000 | 1.000 | 0.975 | 0.870 | 0.850 | 0.637 | 0.600 | 0.225 |

**FINDING: there is no handwriting to render.** A lesson `Page` holds
`doc_id, page_num, content_tokens, patches` and no image; the handwritten
pages are typed word lists, so the vision side reads them with zero errors.
Unpruned it scores 1.000, the same as perfect OCR.

**FINDING: every point the shipped index loses on handwriting comes from its
own pruning.** From 1.000 unpruned to 0.837 at the 50% default: the prune
costs as much as a 20-25% character error rate costs OCR.
"""

from __future__ import annotations

import collections
import dataclasses
import math
import random
import zlib

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "04-multimodal-document-qa"
SALTS, DRAWS = 50, 20
CERS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5)
HANDWRITTEN = {("handwritten-lab", 5), ("handwritten-lab", 6)}
QUERIES = [  # handwriting queries; the first is main()'s own
    ("handwritten experimental figures with error bars", "handwritten-lab", 6),
    ("pH readings in the lab notes", "handwritten-lab", 5),
    ("circuit board experiment notes", "handwritten-lab", 5),
    ("annotated graph figure 3 caption", "handwritten-lab", 6),
]
def ndcg5(ranked, gold):
    i = ranked.index(gold)
    return 1 / math.log2(i + 2) if i < 5 else 0.0


def bm25_ranking(ref, query, pages, k1=1.2, b=0.75):
    avg = sum(len(t) for _, t in pages) / len(pages)
    df = collections.Counter(w for _, t in pages for w in set(t))
    scores = {}
    for key, toks in pages:
        tf = collections.Counter(toks)
        scores[key] = sum(
            math.log(1 + (len(pages) - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * (k1 + 1)
            / (tf[w] + k1 * (1 - b + b * len(toks) / avg))
            for w in ref.tokenize(query) if tf[w])
    return sorted(scores, key=lambda k: -scores[k])

def misread(text, cer, rng):
    """Handwriting OCR stand-in: each letter or digit misread with probability cer."""
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(rng.choice(alphabet) if c.isalnum() and rng.random() < cer else c for c in text.lower())


def ocr_scores(ref, cer):
    ndcg = hits = 0.0
    for draw in range(DRAWS):
        rng = random.Random(draw)
        pages = [((d, p), ref.tokenize(misread(t, cer, rng) if (d, p) in HANDWRITTEN else t))
                 for d, p, t in ref.CORPUS]
        for q, d, p in QUERIES:
            ranked = bm25_ranking(ref, q, pages)
            ndcg += ndcg5(ranked, (d, p))
            hits += ranked[0] == (d, p)
    n = DRAWS * len(QUERIES)
    return round(ndcg / n, 3), round(hits / n, 3)


def vision_scores(ref, prune):
    ndcg = hits = 0.0
    for salt in range(SALTS):
        ref.hash = lambda tok, s=salt: zlib.crc32(f"{s}|{tok}".encode())
        idx = ref.build_index(prune=prune)
        for q, d, p in QUERIES:
            ranked = [(pg.doc_id, pg.page_num) for pg, _ in idx.retrieve(q, k=len(idx.pages))]
            ndcg += ndcg5(ranked, (d, p))
            hits += ranked[0] == (d, p)
    n = SALTS * len(QUERIES)
    return round(ndcg / n, 3), round(hits / n, 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ocr = {cer: ocr_scores(ref, cer) for cer in CERS}
    vision = vision_scores(ref, prune=True)
    return {
        "ocr": ocr, "vision": vision, "vision_unpruned": vision_scores(ref, prune=False),
        "crossover": next(c for c in CERS if ocr[c][0] < vision[0]),
        "page_fields": [f.name for f in dataclasses.fields(ref.Page)],
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: shipped vision 0.837 nDCG@5 on handwriting; OCR drops below it between CER 0.2 and 0.25",
            r["vision"] == (0.837, 0.735) and r["crossover"] == 0.25
            and [r["ocr"][c][0] for c in CERS] == [1.0, 1.0, 0.975, 0.87, 0.85, 0.637, 0.6, 0.225],
            f"vision (pruned) {r['vision']}, unpruned {r['vision_unpruned']}; OCR by CER {r['ocr']}; "
            f"OCR falls below vision at CER {r['crossover']}",
        ),
        practice.Check(
            "FINDING: the scaffold has no image to render: a Page is a list of words",
            r["page_fields"] == ["doc_id", "page_num", "content_tokens", "patches"],
            f"Page fields {r['page_fields']}",
        ),
        practice.Check(
            "FINDING: unpruned, the vision side is perfect OCR; the 50% prune is the whole loss",
            r["vision_unpruned"] == (1.0, 1.0) == r["ocr"][0.0],
            f"unpruned {r['vision_unpruned']} vs pruned {r['vision']}; OCR at CER 0 {r['ocr'][0.0]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
