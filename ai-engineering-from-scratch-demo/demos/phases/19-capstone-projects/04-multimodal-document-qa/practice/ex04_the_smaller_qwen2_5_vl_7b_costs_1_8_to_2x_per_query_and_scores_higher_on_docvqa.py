"""Exercise 4 — the smaller Qwen2.5-VL-7B costs 1.8-2x per query and scores higher on DocVQA.

    Swap Qwen3-VL-30B for a smaller VLM (Qwen2.5-VL-7B). Measure the accuracy-per-dollar curve.

Reading of the exercise: the lesson ships no answerer, so the curve is
built from what it does ship plus published facts. The measured part is the
lesson's retriever as `build_index()` ships it (MaxSim, 50% `doc_prune`):
recall@k for k = 1..5 pages over its `CORPUS` and 14 labelled queries,
averaged over 50 salts of the builtin `hash()` its `hash_embed` seeds from
(pinned to salted CRC32 so runs repeat). Each answerer is assumed to answer
correctly at its published DocVQA rate when the gold page is among the k
pages it is shown, so accuracy(k) = recall@k x DocVQA. Cost per query is k
page images at the doc's 1536x2048 render, tokenised by each model's own
`smart_resize` rule (patch x merge pixels per token), plus 60 prompt and 120
answer tokens, at list prices. Facts read 2026-09-29: preprocessor_config.json
of both Hugging Face repos (Qwen2.5-VL: patch 14, merge 2; Qwen3-VL: patch 16,
merge 2); DocVQA test 95.7 (Qwen2.5-VL-7B model card) and 95.0 (Qwen3-VL-30B-A3B
Instruct, arXiv:2511.21631 Table 3); prices per million input/output tokens
$0.20/$0.20 and $0.13/$0.52 (pricepertoken.com, Qwen provider page).

**ANSWER: the swap moves the whole curve down, not to the left.** Measured
recall@k is 0.760, 0.884, 0.926, 0.959, 0.974. At every k the 7B costs
1.79-1.96x as much per query and is 0.7% more accurate, so the 30B-A3B buys
1.8-1.9x the accuracy per dollar at every point. For both models k = 1 gives
the most accuracy per dollar (1,538 vs 867 correct answers per dollar, at
accuracy 0.722 vs 0.727), and k = 5 gives the most accuracy (0.925 vs 0.932,
at $2.07 vs $4.05 per 1,000 queries).

**FINDING: the "smaller" model is the bigger bill.** Qwen3-VL-30B-A3B is a
mixture of experts with 3B active parameters, so its list price per input
token ($0.13) is below the dense 7B's ($0.20), and its 32-pixel tokens turn a
1536x2048 page into 3,072 tokens where Qwen2.5-VL's 28-pixel tokens make
4,015.

**FINDING: the doc's render settings disagree.** 180 DPI on a US Letter page
is 1530x1980 and on A4 1489x2104, neither of which is the 1536x2048 the
Build It step asks for.
"""

from __future__ import annotations

import zlib

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "04-multimodal-document-qa"
SALTS = 50
PAGE = (2048, 1536)  # height, width: Build It step 1
PROMPT_TOKENS, ANSWER_TOKENS = 60, 120
MODELS = {  # pixels per token side (patch * merge), DocVQA test, $ per 1M input / output tokens
    "Qwen3-VL-30B-A3B": {"px": 16 * 2, "docvqa": 0.950, "in": 0.13, "out": 0.52},
    "Qwen2.5-VL-7B": {"px": 14 * 2, "docvqa": 0.957, "in": 0.20, "out": 0.20},
}
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


def page_tokens(px, height=PAGE[0], width=PAGE[1]):
    """Qwen-VL smart_resize: each side rounded to a multiple of px; one token per px x px cell."""
    return round(height / px) * round(width / px)


def recall_at_k(ref, ks=range(1, 6)):
    hits = {k: 0 for k in ks}
    for salt in range(SALTS):
        ref.hash = lambda tok, s=salt: zlib.crc32(f"{s}|{tok}".encode())
        idx = ref.build_index(prune=True)
        for q, d, p in QUERIES:
            top = [(pg.doc_id, pg.page_num) for pg, _ in idx.retrieve(q, k=max(ks))]
            for k in ks:
                hits[k] += (d, p) in top[:k]
    return {k: round(h / (SALTS * len(QUERIES)), 3) for k, h in hits.items()}


def curve(recall):
    rows = {}
    for name, m in MODELS.items():
        for k, r in recall.items():
            cost = ((k * page_tokens(m["px"]) + PROMPT_TOKENS) * m["in"] + ANSWER_TOKENS * m["out"]) / 1e6
            acc = r * m["docvqa"]
            rows[name, k] = {"acc": round(acc, 3), "usd": round(cost, 6), "acc_per_usd": round(acc / cost)}
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    recall = recall_at_k(ref)
    rows = curve(recall)
    big, small = MODELS
    return {
        "recall": recall, "rows": rows,
        "tokens": {name: page_tokens(m["px"]) for name, m in MODELS.items()},
        "cost_ratio": [round(rows[small, k]["usd"] / rows[big, k]["usd"], 2) for k in recall],
        "value_ratio": [round(rows[big, k]["acc_per_usd"] / rows[small, k]["acc_per_usd"], 1) for k in recall],
        "best_value_k": {n: max(recall, key=lambda k, n=n: rows[n, k]["acc_per_usd"]) for n in MODELS},
        "letter_a4": [(round(w * 180), round(h * 180)) for w, h in ((8.5, 11), (8.27, 11.69))],
    }


def verify(result):
    r, rows = result, result["rows"]
    big, small = MODELS
    return [
        practice.Check(
            "ANSWER: the 7B costs 1.8-2.0x per query at every k; the 30B-A3B gives 1.8-1.9x accuracy per dollar",
            r["recall"] == {1: 0.76, 2: 0.884, 3: 0.926, 4: 0.959, 5: 0.974}
            and r["cost_ratio"] == [1.79, 1.89, 1.93, 1.95, 1.96] and r["value_ratio"] == [1.8] + [1.9] * 4
            and (rows[big, 1]["acc_per_usd"], rows[small, 1]["acc_per_usd"]) == (1538, 867)
            and (rows[big, 5]["acc"], rows[small, 5]["acc"]) == (0.925, 0.932)
            and r["best_value_k"] == {big: 1, small: 1},
            f"recall@k {r['recall']}; curve {rows}",
        ),
        practice.Check(
            "FINDING: the smaller model needs 31% more tokens per page at a 54% higher input price",
            r["tokens"] == {big: 3072, small: 4015} and MODELS[small]["in"] / MODELS[big]["in"] > 1.5,
            f"tokens per 1536x2048 page {r['tokens']}",
        ),
        practice.Check(
            "FINDING: 180 DPI does not give the 1536x2048 render on Letter or A4",
            r["letter_a4"] == [(1530, 1980), (1489, 2104)],
            f"180 DPI Letter, A4: {r['letter_a4']} vs {PAGE[::-1]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
