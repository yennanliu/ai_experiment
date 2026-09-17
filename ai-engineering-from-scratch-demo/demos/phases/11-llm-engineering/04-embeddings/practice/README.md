<!-- generated:start -->
# 11-llm-engineering / 04-embeddings

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/11-llm-engineering/04-embeddings/) · upstream spec
`phases/11-llm-engineering/04-embeddings/docs/en.md`

```bash
uv run demo practice run 04-embeddings --ex 1
uv run demo explain 04-embeddings --ex 1
uv run pytest demos/phases/11-llm-engineering/04-embeddings
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Metric comparison: run the same 5 queries against the sample documents using cosine similarit… | code | T0 | `ex01_every_disagreement_is_inside_a_tie.py` |
| 2 | Chunk size experiment: index the sample documents with chunk sizes of 50, 100, 200, and 500 w… | code | T0 | `ex02_the_first_chunk_size_the_exercise_asks_for_cannot_terminate.py` |
| 3 | Matryoshka simulation: build a SimpleEmbedder that produces 500-d vectors. Truncate to 50, 10… | code | T0 | `ex03_truncation_keeps_the_alphabetically_first_dimensions.py` |
| 4 | Binary quantization: take the embeddings from the search engine, convert them to binary (1 if… | code | T0 | `ex04_overlap_at_ten_is_forced_by_an_index_of_eleven.py` |
| 5 | Sentence-based chunking: replace fixed-size chunking with `chunk_by_sentences`. Run the same… | code | T0 | `ex05_sentence_chunking_changes_nothing_but_the_tokens.py` |
<!-- generated:end -->

## Answers

The lesson is numpy plus stdlib, so all five exercises are **T0** on the `math`
group and run in CI. Four of the five questions turn out to be about the
fixture rather than the technique: five documents of 91–102 words, a 270-word
vocabulary that *is* the embedding dimensionality, and an index of 11 chunks.
The fifth — chunk size 50 — cannot be run at all.

### 1 — every disagreement is inside a tie

| query | cosine | dot | euclidean | |
|---|---|---|---|---|
| refund policy for enterprise customers | `[0, 4, 1]` | `[0, 4, 1]` | `[0, 4, 1]` | agree |
| **how do I reset my password** | `[0, 1, 2]` | `[0, 1, 2]` | `[4, 0, 1]` | **differ** |
| what is the SLA uptime guarantee | `[4, 1, 3]` | `[4, 1, 3]` | `[4, 1, 3]` | agree |
| data retention and deletion | `[2, 1, 3]` | `[2, 1, 3]` | `[2, 1, 3]` | agree |
| **api rate limits** | `[3, 0, 1]` | `[3, 0, 1]` | `[3, 1, 2]` | **differ** |

**ANSWER: two of five, and neither is about similarity.**
`SimpleEmbedder.embed` divides by the L2 norm, so on the stored vectors

```text
cosine(q, v) == dot(q, v)                agree to 5.6e-17
euclidean(q, v) == sqrt(2 - 2 cos)       agree to 1e-16
```

One ranking, three scales. Re-run the identical scoring with ties broken by
index and all five queries agree.

**MECHANISM: the metrics differ only in how they break ties.** `sort` is stable,
so cosine and dot keep insertion order. Euclidean scores `-||q - v||`, and the
stored norms take **two** distinct values spanning **1.1e-16** — so it orders
tied chunks by the last bit of their normalisation.

**FINDING: one query embeds to the zero vector.** `"how do I reset my password"`
has 0 of 270 dimensions non-zero — not one of its six words is in the
vocabulary. Cosine returns 0.0 for every chunk; euclidean returns `-||v||` and
ranks float noise. For that query the `sqrt(2 - 2 cos)` identity is off by
**0.41**, because the query never normalised.

**FINDING: the correspondence is exact.** Chunks scoring exactly 0.0 per query
are `[0, 5, 0, 0, 4]`. The queries with a non-zero tie count are indices 1 and 4.
The queries that disagree are indices 1 and 4. No tie, no disagreement.

### 2 — the first chunk size the exercise asks for cannot terminate

```python
def chunk_text(text, chunk_size=200, overlap=50):
    ...
    start += chunk_size - overlap        #  50 - 50  ==  0
```

**ANSWER: chunk size 50 hangs.** A bounded clone emits 1,000 identical chunks
with `start` still at 0. `SemanticSearchEngine(chunk_size=50)` never returns from
`index_documents`, and the overlap is 50 because that is what the engine
defaults to.

| chunk size | chunks | IDF sum | top-1 per query |
|---:|---:|---:|---|
| 50 | — | — | **does not terminate** |
| 100 | 11 | 660.65 | 0.4502, 0.0, 0.3614, 0.4152, 0.4070 |
| 200 | 5 | 537.35 | 0.4714, 0.0, 0.3539, 0.4389, 0.4366 |
| 500 | 5 | 537.35 | *identical to 200* |

**FINDING: two of the remaining three are the same experiment.** The longest
document is 102 words, so any size at or above it gives one chunk per document —
200 and 500 hold byte-identical text.

**FINDING: the scores are not comparable across sizes.** `index_documents` calls
`self.embedder.fit(all_chunks)`, so the IDF vector is a function of the chunk
*count*. The vocabulary stays at 270 words and the same query still moves:
cosine **0.998785** between its size-100 and size-200 embeddings.

**ANSWER: no knee, and no consistent direction.** From 100 to 200, three queries
improve, one gets worse, one is 0.0 both ways.

**CONTROL: fit one embedder on the documents and reuse it.** Per-query spread
across all three sizes: `[0.0, 0.0, 0.003, 0.0, 0.0]`. What the exercise's plot
would have shown is the refit.

### 3 — truncation keeps the alphabetically first words

| dimension order | 50 | 100 | 200 | 270 |
|---|---:|---:|---:|---:|
| **alphabetical** (as shipped) | 0.667 | 0.833 | **0.667** | 1.000 |
| by document frequency | 0.583 | 0.917 | **1.000** | 1.000 |
| by IDF | 0.500 | 0.500 | 0.750 | 1.000 |

*(top-3 recall against the untruncated ranking, over the four queries that embed
to something)*

**ANSWER: recall does not degrade; it wanders.** Adding dimensions 101 to 200
makes retrieval **worse**. That is not a degradation curve.

**MECHANISM: `self.vocab = sorted(vocab_set)`.** So "the first *d* dimensions"
means "the *d* alphabetically first words", and the first seven are:

```text
$29   $500   $99   0.1%   1.3.   10%   100
```

Matryoshka's premise is that a prefix is *sufficient*. An alphabetical prefix is
arbitrary, and 270 is the vocabulary size, not a dimensionality anyone chose.

**FINDING: at 50 dimensions, 2 of the 4 live queries truncate to the zero
vector** and fall back to insertion order, so part of the short-end recall is
scored against an ordering nothing produced.

**CONTROL: document-frequency order makes it a curve** — monotone, full recall at
200 of 270. No training needed, only an ordering that means something. And the
obvious importance proxy goes the other way: **IDF order is worse than the
alphabet** at two of four points, because the rarest words are the ones fewest
chunks share. Which prefix is sufficient is exactly what the training trick
decides.

### 4 — 90% overlap at k=10 is forced by an index of 11

| k | Hamming vs cosine | Jaccard vs cosine |
|---:|---:|---:|
| 1 | **0.000** | 0.400 |
| 3 | 0.333 | 0.867 |
| 5 | 0.400 | 0.800 |
| 10 | **0.900** | 1.000 |

**ANSWER: 90% on all five queries, and it cannot be less.** The index holds 11
chunks, so two subsets of size 10 must share at least `2·10 − 11 = 9`. The
number the exercise asks for is pinned by its own fixture, and would read 90%
for a random metric.

**FINDING: wherever the number can move, it collapses** — 0.00 at k=1 on every
query. Binary quantization gets the top hit wrong every time.

**MECHANISM: tf-idf is non-negative.** Smallest IDF **1.087**, smallest vector
entry **0.0** — so `binarize(v)` is exactly `(v != 0)`, a presence bitmap.
Every weight is discarded, and Hamming distance becomes the size of a symmetric
difference of word sets.

**FINDING: what survives is length.** Ones per chunk correlate **0.9922** with
word count, and the returned Hamming order is the chunks sorted by size:

```text
ones in Hamming order:  2  37  38  39  39  42  67  70  72  71  74
```

A query has a handful of ones; the nearest chunk in Hamming terms is the
shortest one.

**CONTROL: Jaccard over the identical bitmaps** — intersection over union, the
length-invariant reading of the same bits — scores 0.40 at k=1 and 0.867 at
k=3. The quantization was never the problem; the unnormalised distance was.

### 5 — sentence chunking changes nothing but the tokens

| | chunks | whitespace words | vocabulary |
|---|---:|---:|---:|
| `chunk_text(d, 200, 50)` | 5 | 471 | 270 |
| `chunk_by_sentences(d, 200)` | **5** | **479** | **275** |

**ANSWER: the boundaries are identical.** The longest document is 102 words and
the limit is 200, so both chunkers emit one whole document per chunk. There is
no chunking difference in the comparison at all.

| query | fixed | sentence |
|---|---:|---:|
| refund policy for enterprise customers | 0.4714 | 0.4714 |
| how do I reset my password | 0.0000 | 0.0000 |
| what is the SLA uptime guarantee | 0.3539 | **0.3463** |
| data retention and deletion | 0.4389 | **0.4368** |
| api rate limits | 0.4366 | **0.4354** |

Three get worse; none improves.

**MECHANISM: `text.split(".")` shatters every decimal, version and hostname.**

```text
99.9%  ->  "99."  "9%"              1.3.   ->  "1."  "3."
0.1%   ->  "0."   "1%"              2.0.   ->  "2."  "0."
99.99% ->  "99."  "99%"             status.acme.com -> "status." "acme." "com"
```

Six tokens destroyed, eleven junk tokens created. The whitespace word count goes
up by 8, so the chunker does not preserve its input — and with the boundaries
identical, that tokenisation is the entire measured difference.

**FINDING: the splitter cannot enforce its own budget.** The length test fires
only when the current chunk is already non-empty, so a single 300-word sentence
becomes one 300-word chunk against a `max_chunk_tokens` of 200. No sample
document is long enough to trigger it, which is why the comparison never sees
it.
