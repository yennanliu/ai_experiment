<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 23-chunking-strategies-rag

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/23-chunking-strategies-rag/) · upstream spec
`phases/05-nlp-foundations-to-advanced/23-chunking-strategies-rag/docs/en.md`

```bash
uv run demo practice run 23-chunking-strategies-rag --ex 1
uv run demo explain 23-chunking-strategies-rag --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/23-chunking-strategies-rag
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Chunk one 20-page document with fixed(512, 0), recursive(512, 0), and recursive(512, 10… | code | T0 | `ex01_one_of_the_three_configurations_does_not_exist.py` |
| 2 | Medium. Build a 30-query eval set over 5 documents. Measure recall@5 for recursive, semantic,… | code | T0 | `ex02_recall_at_five_rewards_making_five_chunks.py` |
| 3 | Hard. Implement contextual retrieval. Measure MRR improvement over baseline recursive. Report… | code | T0 | `ex03_the_context_is_shared_so_it_cannot_discriminate.py` |
<!-- generated:end -->

## Answers

Three exercises about chunking, and in each one the thing being compared is not
the thing named. Exercise 1 asks for a configuration the code cannot express;
exercise 2's metric measures index size; exercise 3's technique needs the LLM
call it is priced by, and without it the arithmetic works against you.

All three run at **T0** — `code/main.py` imports only `hashlib`, `math` and `re`.

### 1 — One of the three configurations does not exist

**ANSWER: `recursive(512, 100)` is not a call this code supports.**
`chunk_recursive(text, size, seps)` has no overlap parameter, so 100 binds to
`seps` and the first loop raises `TypeError: 'int' object is not iterable`.

**MECHANISM: and `recursive(512, 0)` does not respect 512 either.** The function
is not recursive — it splits on the first separator present anywhere and never
descends into a part that is still too long.

| requested size | fixed: chunks / longest | recursive: chunks / longest |
|---:|---:|---:|
| 512 | 4 / 512 | 3 / **893** |
| 300 | 6 / 300 | 4 / **893** |
| 200 | 9 / 200 | 6 / **893** |

One 893-character paragraph comes back whole at every size. `size` is an upper
bound only when the document already satisfies it — on `main()`'s own contract,
whose paragraphs all fit, nothing exceeds the limit and the bug is invisible.

**FINDING: boundary quality separates the two arms completely.** At 512
characters, chunks that both start and end at a sentence: **fixed 0 of 4**,
**recursive 3 of 3**. The sizes say why — `512, 512, 512, 75` against
`440, 274, 893`.

**CONTROL: overlap only exists on the arm that needs it.** At 200 characters
`chunk_fixed(size, 100)` returns 17 chunks against 9, re-reading half of every
chunk to repair cuts it made itself. A separator-aligned chunker has nothing to
repair.

### 2 — Recall@5 rewards making five chunks

30 questions over 5 documents of 6 sections each:

| strategy | chunks (distinct) | k=5 covers | recall@5 | recall@1 |
|---|---:|---:|---:|---:|
| fixed(300, 50) | 10 (10) | 0.50 | 0.8333 | 0.5000 |
| **recursive(300)** | 10 (10) | 0.50 | **0.8667** | 0.5333 |
| semantic | 30 (30) | 0.17 | 0.8333 | 0.5667 |
| sentence×3 | 21 (21) | 0.24 | 0.8333 | 0.5667 |
| **parent-document** | **5 (5)** | **1.00** | **1.0000** | 0.5667 |
| parent, as `main()` builds it | 15 (**5**) | 0.33 | **0.7333** | 0.5667 |

**ANSWER: parent-document wins at 1.0000 — which is the whole index.** It builds
5 chunks for 5 documents, so at k=5 the retriever returns everything. 1.0000
holds for any ranking function, including a constant one. It *does* match the
blog posts, and it is not evidence for them.

**FINDING: at equal coverage the strategies are indistinguishable.** At k=1 the
spread is two questions in thirty, and parent-document is tied rather than best.

**FINDING: the same strategy is both best and worst.** `chunk_parent_child`
returns one row per *child*, so `main()`'s own index expression
`[m["parent"] for m in pc]` holds 15 entries and 5 distinct texts; duplicates
fill the top-5 slots. **The difference between the winner and the loser is a
`set()`.**

**CONTROL: the lesson's own demonstration cannot see any of this.** Three queries
over one contract, 3 of 3 for all five strategies — including a parent index
containing exactly one distinct document.

### 3 — The context is shared, so it cannot discriminate

No LLM is reachable, so the context is written from the document itself — the
shared move a first implementation makes.

| index | MRR | recall@1 | LLM calls |
|---|---:|---:|---:|
| baseline recursive | **0.6746** | **0.5333** | 0 |
| + corpus summary | 0.6624 | 0.5000 | 10 |
| + document name | 0.6585 | 0.5000 | 10 |
| + name and opening line | 0.6557 | 0.5000 | 10 |

**ANSWER: every variant lowers MRR**, and recall@1 falls in all three, so it is
not an MRR artefact. **The ratio the exercise asks to report has a negative
denominator** — there is no cost per point to quote.

**MECHANISM: nothing shared by every chunk can separate chunks.** The corpus
summary is one string across all 10 chunks; the document name takes 5 values
across 5 documents. A prefix that appears everywhere adds a constant to every
numerator and re-normalises every denominator, so it can only dilute — and chunks
average 231 characters while the summary adds 91, nearly 30% of the embedded
string.

**MECHANISM: and the routing it could help with is never needed.** 0 of 30 gold
spans appear in more than one document, so no question requires choosing between
documents.

**CONTROL:** Anthropic's contextual retrieval writes a *chunk-specific* sentence
per chunk, which is exactly what the LLM call buys. The version measurable
without one is the version that cannot work, and it fails in the arithmetic
rather than in the quality of the writing.
