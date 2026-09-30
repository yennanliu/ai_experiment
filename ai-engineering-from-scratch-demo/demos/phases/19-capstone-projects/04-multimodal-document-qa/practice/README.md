<!-- generated:start -->
# 19-capstone-projects / 04-multimodal-document-qa

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/04-multimodal-document-qa/) · upstream spec
`phases/19-capstone-projects/04-multimodal-document-qa/docs/en.md`

```bash
uv run demo practice run 04-multimodal-document-qa --ex 1
uv run demo explain 04-multimodal-document-qa --ex 1
uv run pytest demos/phases/19-capstone-projects/04-multimodal-document-qa
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Measure ColQwen2.5-v0.2 vs ColQwen3-omni on the same corpus. Which pages does one get right a… | code | T0 | `ex01_two_stand_in_models_disagree_less_than_one_model_does_with_itself_across_processes.py` |
| 2 | Prune embeddings aggressively (75%, 90%). Find the compression cliff: the point where ViDoRe… | code | T0 | `ex02_pruning_25pct_already_drops_maxsim_below_ocr_and_the_shipped_50pct_loses_11pct.py` |
| 3 | Build a hybrid: run OCR-then-text and ColQwen in parallel, fuse with RRF, rerank with a cross… | code | T0 | `ex03_rrf_lands_between_maxsim_and_ocr_and_only_the_reranker_makes_the_hybrid_win.py` |
| 4 | Swap Qwen3-VL-30B for a smaller VLM (Qwen2.5-VL-7B). Measure the accuracy-per-dollar curve. | code | T0 | `ex04_the_smaller_qwen2_5_vl_7b_costs_1_8_to_2x_per_query_and_scores_higher_on_docvqa.py` |
| 5 | Add handwritten-note support. Render the handwriting corpus, embed with ColQwen, measure retr… | code | T0 | `ex05_the_shipped_index_retrieves_handwriting_worse_than_ocr_misreading_1_character_in_5.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` is a toy version of ColPali retrieval. Each page
is a short list of words, and each word becomes one "patch": a 16-dim random
unit vector seeded from Python's builtin `hash()` of the word. `doc_prune`
keeps half the patches, and `Index.retrieve` ranks pages by MaxSim. There is
no page image, no ColQwen, no VLM and no OCR. Every exercise runs this code
as shipped, over its 10-page `CORPUS`, with 14 labelled queries (the four in
`main()` plus ten more, one gold page each).

Python salts `hash()` differently in every process, so the lesson's own
output changes from run to run. The solutions pin `hash` to a salted CRC32
and average over 50 salts. External facts were read on 2026-09-29: the Qwen
preprocessor configs and model card on Hugging Face, the Qwen3-VL technical
report (arXiv:2511.21631) and pricepertoken.com.

### 1 — two stand-in models disagree less than one model does with itself across processes

**The stand-in for the other model (the same embedder at `EMB_DIM = 128`,
the doc's ColQwen patch size) beats the shipped `EMB_DIM = 16` on 12 of 14
queries.** The shipped size wins two:

| query | top-1 hits of 50, dim 16 | dim 128 |
|---|---:|---:|
| what was the 2024 operating margin change for EMEA | 22 | 15 |
| late interaction retrieval vs OCR | 33 | 29 |
| pH readings in the lab notes | 24 | 44 |

Each `Page` gets a `content_class` tag. The router reads the tag of the
shipped model's top hit and picks whichever model won that class on salts
0-24. On salts 25-49 it scores hit@1 0.783, against 0.740 (dim 16) and 0.811
(dim 128). It learns "table: use dim 16" from the EMEA query, but 27 of the
79 queries it routes as tables are not table queries.

**Which pages one model misses is mostly process noise.** Dim 16 and dim 128
disagree on 3.80 of 14 queries per run. Dim 16 disagrees with itself under
another salt on 4.36. Run under PYTHONHASHSEED 0-9, the lesson's own
`main.py` puts 10-K p.88 first for its EMEA demo query in only 5 of 10
processes. That page shares three words with the query (operating, margin,
emea); the chart page p.12 shares three too (operating, margin, 2024).

### 2 — 25% pruning already drops MaxSim below OCR, and the shipped 50% loses 11%

**The cliff is at 25% pruning, well before the exercise's 75% and 90%.**
The OCR baseline is BM25 over a transcript that reads typed pages exactly,
drops the words that are only drawn on chart pages, and misreads the
handwriting. It scores nDCG@5 0.974.

| keep fraction | 1.0 | 0.9 | 0.75 | 0.5 (shipped) | 0.25 | 0.1 |
|---|---:|---:|---:|---:|---:|---:|
| MaxSim nDCG@5 | 0.992 | 0.975 | 0.964 | 0.879 | 0.624 | 0.511 |

**The lesson promises "< 0.5% accuracy loss" at 50%; its own pruner loses
11.4%.** nDCG@5 drops 0.113, and hit@1 goes from 0.979 to 0.760. Every patch
has L2 norm 1 (the largest deviation is below 1e-12), so the "norm" that
`doc_prune` sorts by is really the L1 norm of a unit vector. It carries no
signal: dropping the same number of patches at random scores 0.883, slightly
better than the pruner's 0.879.

### 3 — RRF lands between MaxSim and OCR, and only the reranker makes the hybrid win

**The hybrid wins only with the reranker, and it helps most on tables.** The
lesson has no cross-encoder, so the reranker is a stand-in. It reads the
query and the page's OCR text together and counts query words found in the
page, letting a word match on its first four letters.

| nDCG@5 | text | table | chart | handwriting | all |
|---|---:|---:|---:|---:|---:|
| MaxSim (lesson) | 0.930 | 0.865 | 0.905 | 0.791 | 0.879 |
| OCR-BM25 | 1.000 | 0.877 | 1.000 | 1.000 | 0.974 |
| RRF | 0.955 | 0.887 | 0.927 | 0.919 | 0.925 |
| RRF + rerank | 1.000 | 0.936 | 0.998 | 1.000 | 0.986 |

RRF alone lands between its two inputs. It beats both only on tables. The
lesson's "vision-first" retriever trails OCR on charts and handwriting, the
classes where the doc says vision pulls ahead. Its patches are hashed words,
so it is a noisy word matcher, and the 50% prune throws away words the
query needs.

### 4 — the smaller Qwen2.5-VL-7B costs 1.8-2x per query and scores higher on DocVQA

**The swap moves the whole curve down, not to the left.** Accuracy at k
pages is the lesson's recall@k times the model's published DocVQA score.
Cost covers k page images plus 60 prompt tokens and 120 answer tokens, at
list prices.

| k pages | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| recall@k (lesson retriever) | 0.760 | 0.884 | 0.926 | 0.959 | 0.974 |
| 30B-A3B: accuracy / $ per 1k queries | 0.722 / 0.47 | 0.840 / 0.87 | 0.880 / 1.27 | 0.911 / 1.67 | 0.925 / 2.07 |
| 7B: accuracy / $ per 1k queries | 0.727 / 0.84 | 0.846 / 1.64 | 0.886 / 2.45 | 0.918 / 3.25 | 0.932 / 4.05 |

At every k the 7B is 0.7% more accurate (DocVQA 95.7 vs 95.0) and costs
1.79-1.96x as much. The 30B-A3B therefore gets 1.8-1.9x the accuracy per
dollar, and k = 1 is the best value for both models. The "smaller" model is
the bigger bill for two reasons. The 30B-A3B is a mixture of experts with 3B
active parameters and costs $0.13 per million input tokens, against $0.20
for the 7B. And its 32-pixel tokens turn a 1536x2048 page into 3,072 tokens,
where the 7B's 28-pixel tokens make 4,015. Separately, the doc's render
settings disagree with each other: 180 DPI gives 1530x1980 on a Letter page
and 1489x2104 on A4, not 1536x2048.

### 5 — the shipped index retrieves handwriting worse than OCR misreading 1 character in 5

**The shipped index scores nDCG@5 0.837 on the four handwriting queries.**
The OCR pipeline misreads each character of the handwritten pages with
probability CER. It matches or beats that score up to CER 0.2 and falls
below it at 0.25.

| CER | 0 | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OCR-BM25 nDCG@5 | 1.000 | 1.000 | 0.975 | 0.870 | 0.850 | 0.637 | 0.600 | 0.225 |

There is nothing to render. A lesson `Page` holds `doc_id, page_num,
content_tokens, patches` and no image, and the handwritten pages are typed
word lists, so the "vision" side reads them with zero errors. Unpruned it
scores 1.000, the same as perfect OCR. All of its handwriting loss comes
from its own 50% prune.
