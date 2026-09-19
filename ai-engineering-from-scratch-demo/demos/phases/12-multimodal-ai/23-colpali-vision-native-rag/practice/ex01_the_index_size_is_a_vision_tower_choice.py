"""Exercise 1 — the index size is a vision-tower choice.

    A 200-page annual report at 729 patches per page, 128-dim emb, 4-byte
    floats. Compute raw storage and PQ-compressed (8x) storage.

Reading of the exercise: the arithmetic is done as stated and then the 729 is
questioned, because it is not a property of the report -- it is SigLIP SO400m at
384 pixels, 27 x 27, and every other tower in this phase gives a different
number for the identical page.

**ANSWER: 71.2 MiB raw and 8.9 MiB at PQ 8x.** A page is 729 x 128 x 4 =
**364.5 KiB** raw and **45.6 KiB** compressed, which reproduces the lesson's own
table rows of "365 KB" and "46 KB".

**FINDING: even compressed, the index is 15.2x a text-RAG one.** The lesson puts
text-RAG and VisRAG at 3.0 KB a page; ColPali raw is **121.5x** that and PQ 8x is
still **15.2x**. The compression closes most of the gap and none of the
comparison.

**FINDING: the 729 is a tower setting, not a document property.** 729 is 27 x 27
-- SigLIP SO400m at 384. The same page through Qwen2.5-VL at its native 1280x720
is **4,641** patches (Lesson 12.01), **6.37x** the storage: **2.27 MiB** a page
raw. The index size is chosen when the encoder is chosen.

**FINDING: and PQ is a lossy codec applied to exactly what MaxSim maximises
over.** The lesson reports **8x** and no recall figure. Quantising the patch
vectors perturbs every cosine in the `max`, and the retrieval effect of that is
the number nobody prints -- the same shape as Lesson 12.21's FAST tokenizer,
which reports a compression ratio and ships no inverse.

Structure: `page_bytes` is the per-page arithmetic, `TOWERS` compares patch
counts across this phase's encoders, and `ratio_to_text` prices the index against
the lesson's own text-RAG row.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "23-colpali-vision-native-rag"
PATCHES, DIM, FLOAT_BYTES, PAGES = 729, 128, 4, 200
PQ_FACTOR = 8
TEXT_RAG_KIB = 3.0
TOWERS = {"SigLIP SO400m @ 384": 729, "Qwen2.5-VL @ 1280x720": 4641}


def page_bytes(patches=PATCHES, dim=DIM, size=FLOAT_BYTES):
    return patches * dim * size


def kib(value):
    return round(value / 1024, 1)


def mib(value):
    return round(value / 2 ** 20, 1)


def ratio_to_text(value, text_kib=TEXT_RAG_KIB):
    return round(value / 1024 / text_kib, 1)


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    raw = page_bytes()
    compressed = raw // PQ_FACTOR
    return {
        "page_raw": raw, "page_raw_kib": kib(raw),
        "page_pq": compressed, "page_pq_kib": kib(compressed),
        "total_raw_mib": mib(raw * PAGES), "total_pq_mib": mib(compressed * PAGES),
        "raw_vs_text": ratio_to_text(raw), "pq_vs_text": ratio_to_text(compressed),
        "text_kib": TEXT_RAG_KIB,
        "towers": {name: page_bytes(count) for name, count in TOWERS.items()},
        "tower_ratio": round(TOWERS["Qwen2.5-VL @ 1280x720"]
                             / TOWERS["SigLIP SO400m @ 384"], 2),
        "native_mib": round(page_bytes(TOWERS["Qwen2.5-VL @ 1280x720"]) / 2 ** 20, 2),
        "grid": int(PATCHES ** 0.5),
        "pq_factor": PQ_FACTOR, "recall_figures": 0,
    }


def verify(result):
    towers = result["towers"]
    return [
        practice.Check(
            "ANSWER: 71.2 MiB raw and 8.9 MiB at PQ 8x",
            all([result["page_raw"] == 373_248, result["page_raw_kib"] == 364.5,
                 result["page_pq_kib"] == 45.6,
                 result["total_raw_mib"] == 71.2, result["total_pq_mib"] == 8.9]),
            f"a page is {PATCHES} x {DIM} x {FLOAT_BYTES} = {result['page_raw']:,} bytes, "
            f"{result['page_raw_kib']} KiB, and {result['page_pq_kib']} KiB at "
            f"{PQ_FACTOR}x -- so {PAGES} pages are {result['total_raw_mib']} MiB and "
            f"{result['total_pq_mib']} MiB, reproducing the lesson's own 365 KB and 46 KB "
            "rows",
        ),
        practice.Check(
            "FINDING: even compressed, the index is 15.2x a text-RAG one",
            all([result["raw_vs_text"] == 121.5, result["pq_vs_text"] == 15.2,
                 result["text_kib"] == 3.0]),
            f"the lesson puts text-RAG and VisRAG at {result['text_kib']} KB a page; ColPali "
            f"raw is {result['raw_vs_text']}x that and PQ {PQ_FACTOR}x is still "
            f"{result['pq_vs_text']}x. The compression closes most of the gap and none of "
            "the comparison",
        ),
        practice.Check(
            "FINDING: the 729 is a tower setting, not a document property",
            all([result["grid"] == 27, towers["SigLIP SO400m @ 384"] == 373_248,
                 result["tower_ratio"] == 6.37, result["native_mib"] == 2.27]),
            f"{PATCHES} is {result['grid']} x {result['grid']} -- SigLIP SO400m at 384. The "
            f"same page through Qwen2.5-VL at 1280x720 is "
            f"{TOWERS['Qwen2.5-VL @ 1280x720']:,} patches (Lesson 12.01), "
            f"{result['tower_ratio']}x the storage at {result['native_mib']} MiB a page. The "
            "index size is chosen when the encoder is",
        ),
        practice.Check(
            "FINDING: PQ is a lossy codec applied to what MaxSim maximises over",
            all([result["pq_factor"] == 8, result["recall_figures"] == 0]),
            f"the lesson reports {result['pq_factor']}x and {result['recall_figures']} recall "
            "figures. Quantising the patch vectors perturbs every cosine inside the max, and "
            "the retrieval effect is the number nobody prints -- the same shape as Lesson "
            "12.21's FAST tokenizer, which reports a ratio and ships no inverse",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
