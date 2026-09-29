"""Exercise 3 — RRF lands between MaxSim and OCR, and only the reranker makes the hybrid win.

    Build a hybrid: run OCR-then-text and ColQwen in parallel, fuse with RRF, rerank with a cross-encoder. Does the hybrid beat either alone? Where does it help most?

Reading of the exercise: "ColQwen" is the lesson's own late-interaction
index as `build_index()` ships it (hash patches, 50% `doc_prune`, MaxSim);
"OCR-then-text" is BM25 over an OCR transcript of each page (typed pages
exact, chart pages without their drawn-only words, handwritten pages with
misread characters, `OCR` below). Both rank all 10 pages of the lesson's
`CORPUS` for 14 labelled queries, fused by reciprocal-rank fusion (k = 60).
The lesson has no cross-encoder and none can be loaded offline, so the
reranker is a stand-in that reads query and OCR text together and counts
query words present in the page, a word matching on its first four letters
("margins" / "margin"); it reorders the fused top 5. Scores are nDCG@5 per
content class, averaged over 50 salts of the builtin `hash()` the lesson's
`hash_embed` seeds from (pinned to salted CRC32 so runs repeat).

**ANSWER: only with the reranker.** Overall nDCG@5: MaxSim 0.879,
OCR-BM25 0.974, RRF 0.925, RRF + rerank 0.986. The reranked hybrid beats
both alone, and it helps most on tables: 0.936 against 0.877 for the better
single retriever. Everywhere else it only matches OCR (text 1.000,
handwriting 1.000, charts 0.998 vs 1.000).

| nDCG@5 | text | table | chart | handwriting | all |
|---|---:|---:|---:|---:|---:|
| MaxSim (lesson) | 0.930 | 0.865 | 0.905 | 0.791 | 0.879 |
| OCR-BM25 | 1.000 | 0.877 | 1.000 | 1.000 | 0.974 |
| RRF | 0.955 | 0.887 | 0.927 | 0.919 | 0.925 |
| RRF + rerank | 1.000 | 0.936 | 0.998 | 1.000 | 0.986 |

**FINDING: RRF alone lands between its inputs.** It beats both only on
tables (0.887); on text, charts and handwriting it drags OCR down (1.000 to
0.955, 0.927, 0.919).

**FINDING: the lesson's "vision-first" retriever loses to OCR text exactly
where the doc says vision pulls ahead.** On charts (0.905 vs 1.000) and
handwriting (0.791 vs 1.000) it trails BM25 over a transcript that dropped
the chart words and misread the handwriting. The scaffold's patches are
hashed words, so it is a noisy lexical matcher, and its 50% prune drops
words the query needs.
"""

from __future__ import annotations

import collections
import math
import zlib

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "04-multimodal-document-qa"
SALTS = 50
CLASS = {("10k-2024", 88): "table", ("paper-vidore-v3", 7): "table", ("chart-report", 11): "chart",
         ("chart-report", 12): "chart", ("handwritten-lab", 5): "handwriting", ("handwritten-lab", 6): "handwriting"}
CLASSES = ("text", "table", "chart", "handwriting")
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


def rrf(*rankings, k=60):
    score = {key: sum(1 / (k + r.index(key) + 1) for r in rankings) for key in rankings[0]}
    return sorted(score, key=lambda key: -score[key])


def cross_score(ref, query, page):
    """Reranker stand-in: query words found in the page's OCR text, 4-letter prefix match."""
    return sum(any(w == t or (len(w) >= 4 and w[:4] == t[:4]) for t in page) for w in ref.tokenize(query))


def rankings(ref, idx, query):
    ocr_text = {(d, p): ref.tokenize(OCR.get((d, p), t)) for d, p, t in ref.CORPUS}
    vision = [(p.doc_id, p.page_num) for p, _ in idx.retrieve(query, k=len(idx.pages))]
    ocr = bm25_ranking(ref, query)
    fused = rrf(vision, ocr)
    reranked = sorted(fused[:5], key=lambda key: -cross_score(ref, query, ocr_text[key])) + fused[5:]
    return {"vision": vision, "ocr": ocr, "rrf": fused, "rrf+rerank": reranked}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    total = collections.defaultdict(float)
    for salt in range(SALTS):
        ref.hash = lambda tok, s=salt: zlib.crc32(f"{s}|{tok}".encode())
        idx = ref.build_index(prune=True)
        for q, d, p in QUERIES:
            for name, ranked in rankings(ref, idx, q).items():
                for cls in (CLASS.get((d, p), "text"), "all"):
                    total[name, cls] += ndcg5(ranked, (d, p))
    count = collections.Counter([CLASS.get((d, p), "text") for _, d, p in QUERIES] + ["all"] * len(QUERIES))
    return {"table": {name: {c: round(total[name, c] / (SALTS * count[c]), 3) for c in (*CLASSES, "all")}
                      for name in ("vision", "ocr", "rrf", "rrf+rerank")}}


def verify(result):
    t = result["table"]
    col = {name: [row[c] for c in (*CLASSES, "all")] for name, row in t.items()}
    return [
        practice.Check(
            "ANSWER: fused and reranked the hybrid beats both (0.986 vs 0.974 / 0.879), most on tables",
            col == {"vision": [0.93, 0.865, 0.905, 0.791, 0.879], "ocr": [1.0, 0.877, 1.0, 1.0, 0.974],
                    "rrf": [0.955, 0.887, 0.927, 0.919, 0.925],
                    "rrf+rerank": [1.0, 0.936, 0.998, 1.0, 0.986]},
            f"nDCG@5 per class {list(CLASSES) + ['all']}: {col}",
        ),
        practice.Check(
            "FINDING: RRF alone loses to OCR-BM25 on every class but tables",
            all(t["rrf"][c] < t["ocr"][c] for c in ("text", "chart", "handwriting"))
            and t["rrf"]["table"] > max(t["ocr"]["table"], t["vision"]["table"]),
            f"rrf {t['rrf']} vs ocr {t['ocr']}",
        ),
        practice.Check(
            "FINDING: the vision-first retriever trails OCR-BM25 on charts and handwriting",
            (t["vision"]["chart"], t["vision"]["handwriting"]) == (0.905, 0.791)
            and t["ocr"]["chart"] == t["ocr"]["handwriting"] == 1.0,
            f"MaxSim chart {t['vision']['chart']}, handwriting {t['vision']['handwriting']}; OCR-BM25 1.0 on both",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
