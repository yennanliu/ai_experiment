<!-- generated:start -->
# 19-capstone-projects / 64-chunking-strategies-advanced

Solutions to all 4 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/64-chunking-strategies-advanced/) · upstream spec
`phases/19-capstone-projects/64-chunking-strategies-advanced/docs/en.md`

```bash
uv run demo practice run 64-chunking-strategies-advanced --ex 1
uv run demo explain 64-chunking-strategies-advanced --ex 1
uv run pytest demos/phases/19-capstone-projects/64-chunking-strategies-advanced
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a sixth strategy: token-window using `tiktoken` instead of character counts. Compare agai… | code | T1 | `ex01_token_windows_tie_fixed_windows_at_8_of_9_and_cut_0_words_where_fixed_cuts_9.py` |
| 2 | Inject a 30 percent fraction of code blocks into the prose fixture. Re-run the table. Explain… | code | T0 | `ex02_structural_keeps_3_of_3_only_as_one_whole_document_chunk_and_one_comment_per_block_drops_it_to_2.py` |
| 3 | Replace the deterministic embedding with the one from your project's real provider. Measure t… | code | T0 | `ex03_no_stand_in_moves_semantic_recall_by_more_than_1_of_9_and_a_wider_hash_lifts_the_rest_to_9_of_9.py` |
| 4 | Add a `summary` field per chunk: a one-sentence centroid description. Re-run the eval with th… | code | T0 | `ex04_the_centroid_sentence_summary_costs_4_of_45_top_1_hits_because_it_repeats_words_the_chunk_has.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`: five chunkers, a 96-dimension
hashed bag-of-words `mock_embed`, and `eval_recall`, which builds one index per
document and scores 9 gold-span queries over three short documents (prose,
markdown, mixed). Hits are counted out of those 9 queries, or out of the prose
document's 3.

### 1 — token windows tie fixed windows at 8 of 9, and cut 0 words where fixed cuts 9

**At a matched budget the two tie.** The fixture's 2,189 characters are 413
`cl100k_base` tokens, so the lesson's 400/80-character window becomes 75/15
tokens. Both make 8 chunks, and both score 8/9 at k=1 and 9/9 at k=3 and k=5.
The one difference is where they cut. Fixed-window puts 9 chunk boundaries
inside a word; the token window puts 0 there.

The fixture cannot tell the two apart beyond that. No document gets more than
3 chunks under either windowing, and the index is per document, so k=3 returns
the whole document. At k=1, window size matters more than the unit. Over 18
matched budgets from 40 to 125 tokens, the token window wins 2, fixed-window
wins 5, and 11 tie. Each ranges over 7-9 hits as the size changes.

### 2 — structural keeps 3 of 3 only as one whole-document chunk, and one `#` comment per block drops it to 2

**With unrelated code at 30.3% of the characters, the premise holds at k=1.
Structural survives because it never cuts.** One fenced Python block went after
each prose paragraph. Hits@1 out of 3:

| code injected | fixed | sentence | recursive | semantic | structural |
|---|---:|---:|---:|---:|---:|
| none (lesson) | 2 | 2 | 3 | 2 | 3 |
| unrelated, 30.3% | 1 | 1 | 1 | 1 | 3 |
| same + one `#` comment per block, 32.7% | 2 | 1 | 2 | 1 | 2 |
| code documenting the prose, 31.8% | 3 | 3 | 3 | 2 | 1 |

The prose fixture has no headings. `structural_markdown` therefore returns the
whole 1,271-character document as one chunk, which overlaps every gold span at
any k. The other strategies mix code into prose chunks, and all 6 top-1 misses
of fixed, sentence and recursive go to a chunk that holds a code fence. At k=3
only semantic still loses (3 to 2).

**Add ordinary Python comments and the result reverses.** `structural_markdown`
reads `# ...` at line start as a heading, and it never emits the text before the
first heading. The first 290 characters land in no chunk: the retry-budget
paragraph and the fence line after it. "When does the retry budget reset" is lost
at every k. When the code documents the prose, fixed, sentence and recursive all
reach 3/3 and structural falls to 1.

### 3 — no offline stand-in moves semantic recall by more than 1 of 9, and a wider hash lifts the rest to 9 of 9

**Semantic moves by at most one query, and the spread has no consistent
direction.** No provider is reachable offline. Five stand-ins take the place of
`mock_embed`, which is the single global that both `DenseIndex` and
`semantic_chunks` read. Hits@1 of 9, and the k=1 spread (best minus worst):

| embedding | fixed | sentence | recursive | semantic | structural | spread | semantic chunks |
|---|---:|---:|---:|---:|---:|---:|---:|
| lesson hash, 96 dims | 8 | 7 | 9 | 6 | 9 | 3 | 20 |
| same hash, 4,096 dims | 9 | 9 | 9 | 7 | 9 | 2 | 23 |
| exact bag of words | 9 | 9 | 9 | 7 | 9 | 2 | 23 |
| TF-IDF | 9 | 9 | 9 | 6 | 9 | 3 | 23 |
| LSA-8, unit rows | 8 | 7 | 8 | 7 | 9 | 2 | 17 |
| LSA-8, raw rows | 9 | 9 | 8 | 5 | 8 | 4 | 15 |

The two LSA rows differ only in whether the TF-IDF rows are normalised before
the SVD, and that choice alone flips the verdict from narrow to widen. At k=3
and k=5 the spread is 0 or 1 under every embedding.

**The lesson's table mostly ranks hash collisions.** The fixture has 195 words
and the hash has 96 dimensions. Give the same function 4,096 dimensions and
every strategy except semantic reaches 9/9, which is what the exact bag of words
scores too. **Swapping the embedding also re-chunks the corpus.** Under each
lexical stand-in, the 0.55 threshold cuts before every one of the 23 sentences.
The "semantic" chunker then just splits sentences, which is the doc's
"stale embeddings" failure mode, measured.

### 4 — the centroid-sentence summary costs 4 of 45 top-1 hits, because it repeats words the chunk already has

**The lift is negative.** Each chunk's summary is its own sentence nearest the
mean of its sentences' embeddings, stored as `chunk.summary` and appended to the
embedded text. Top-1 hits across the 5 strategies:

| summary | fixed | sentence | recursive | semantic | structural | total of 45 |
|---|---:|---:|---:|---:|---:|---:|
| none | 8 | 7 | 9 | 6 | 9 | 39 |
| centroid sentence | 8 | 6 | 8 | 5 | 8 | 35 |
| "This section covers a, b, c." | 8 | 8 | 9 | 6 | 9 | 40 |

At k=3 and k=5 every arm scores 9/9, so k=1 is the only place a lift could show.
A centroid sentence adds no new words, only weight. On a one-sentence chunk it
doubles every count, and after unit normalisation the vector is unchanged. That
covers 22 of the 50 chunks, including 17 of the 20 semantic chunks. The keyword
summary gains 1 of 45, all of it on sentence.
