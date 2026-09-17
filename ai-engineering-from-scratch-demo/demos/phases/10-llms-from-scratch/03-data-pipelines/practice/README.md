<!-- generated:start -->
# 10-llms-from-scratch / 03-data-pipelines

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/03-data-pipelines/) · upstream spec
`phases/10-llms-from-scratch/03-data-pipelines/docs/en.md`

```bash
uv run demo practice run 03-data-pipelines --ex 1
uv run demo explain 03-data-pipelines --ex 1
uv run pytest demos/phases/10-llms-from-scratch/03-data-pipelines
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy: Add language detection to the cleaning pipeline using a simple heuristic (character set… | code | T0 | `ex01_stage_one_already_did_it_with_a_regex.py` |
| 2 | Medium: Implement exact deduplication using SHA-256 hashes alongside the MinHash near-dedupli… | code | T0 | `ex02_the_near_duplicate_is_the_one_it_misses.py` |
| 3 | Hard: Build a perplexity-based quality filter. Train a small bigram language model on Wikiped… | code | T0 | `ex03_the_filter_ranks_the_duplicates_best.py` |
<!-- generated:end -->

## Answers

`code/main.py` is a six-stage pre-training pipeline: clean, quality-filter,
deduplicate, tokenize, pack, load. The three exercises add a language filter, a
second deduplicator, and a quality filter — and each one lands next to machinery
that is already doing that job, badly, under a different name. Stage 1 is
already a language filter. The near-duplicate detector catches only the exact
duplicate. The quality filter ranks the duplicates highest.

All three are **T0**: `code/main.py` imports nothing outside the standard
library, and neither do these.

### 1 — stage one already did it, spelled as a character class

The heuristic is the one the exercise names: the share of a document's
characters that are ASCII, cut at 0.9. Run it in both places on the same
nine-language probe corpus.

| language | ASCII ratio before | after `clean_text` | chars |
|---|---:|---:|---|
| Chinese | 0.00 | **0.00** | 33 → **0** |
| Russian | 0.12 | 1.00 | 57 → 1 |
| Arabic | 0.17 | 1.00 | 76 → 1 |
| Korean | 0.27 | 1.00 | 22 → 1 |
| Turkish | **0.89** | 1.00 | 72 → 64 |
| German | 0.99 | 1.00 | 67 → 66 |
| Spanish | 0.99 | 1.00 | 76 → 75 |
| French, English | 1.00 | 1.00 | unchanged |

**ANSWER: placed before `clean_text`, it removes 5 of 9** — and Turkish is one
of them, at the same threshold that keeps French and English at 1.00. A
character-set test measures diacritic density, not language, and Turkish carries
more of it than the cut allows. That is the heuristic's real failure mode, and
it is not the one the exercise is worried about.

**FINDING: placed where the exercise says — "to the cleaning pipeline" — it
removes nothing that still has text in it.** `clean_text`'s third line is

```python
text = re.sub(r"[^\x20-\x7E\n]", "", text)
```

so afterwards 8 of the 9 documents score exactly **1.0**, by construction. The
ninth scores 0.0 only because it has no characters left at all. A detector added
after this line cannot fail to say English.

**FINDING: stage 1 is already the language filter, and it deletes rather than
rejects.** The non-Latin documents arrive at `quality_filter` holding one
character — the full stop — and are removed by `min_words=50` and reported as
*low quality*. The pipeline filters language in its first line under the name
cleaning, and reports it two stages later under a third name.

**FINDING: Latin-script non-English is corrupted rather than removed.**

```text
German   'künstlichen'  ->  'knstlichen'
Spanish  'también'      ->  'tambin'
Turkish  'öğrenme'      ->  'renme'
```

These documents keep 94–99% of their characters, pass every downstream filter,
and reach the tokenizer misspelled. The pipeline deletes what it cannot read and
silently damages what it half can — and the half-damaged case is the one that
survives to training.

### 2 — the near-duplicate is the one it misses

The corpus plants two duplicates and names them. Both methods run on the 13
documents `run_pipeline` hands to `deduplicate`.

| | removed |
|---|---:|
| SHA-256 exact | **1** |
| MinHash + LSH, threshold 0.8 | **1** |
| MinHash + LSH, threshold 0.5 | 1 |
| MinHash + LSH, threshold 0.3 | 1 |

**ANSWER: both catch one document, and it is the same document.** `near_dup_2`
is byte-identical to `base_docs[1]` — Jaccard 1.0000. One line of hashing finds
it; so do 128 hash functions across 16 bands.

**FINDING: the document planted as the *near*-duplicate is missed at every
threshold.** `near_dup_1` is a rewrite of `base_docs[0]` and scores **0.2703**.
At that score, "near-duplicate" and "unrelated document" are the same bucket,
and no threshold the pipeline would tolerate separates them.

**MECHANISM: `get_shingles`' `k=5` is the knob.**

| shingle size | 1 (words) | 2 | 3 | 5 (the default) |
|---|---:|---:|---:|---:|
| Jaccard, `base_docs[0]` vs `near_dup_1` | **0.7778** | 0.5543 | 0.4356 | **0.2703** |

A rewrite that changes one word destroys five 5-grams. `k=5` together with
`threshold=0.8` asks for near-verbatim text — the two documents agree on 78% of
their word types and the configuration calls them unrelated.

**FINDING: neither method contains the other.**

```text
get_shingles("a b c")  ->  set()
```

A document under `k` words has no shingles at all, so four short documents
holding two identical pairs are deduplicated to four by MinHash and to two by
SHA-256. In the other direction, a copy differing only in whitespace has a
different digest and identical shingles: MinHash removes it, exact does not. The
exercise asks which method catches more; on this corpus the honest answer is
that their intersection is everything either one catches, and their symmetric
difference is where the interesting cases live.

### 3 — the filter ranks the duplicates best

A bigram model with add-1 smoothing, trained on the 13 documents and scoring
each of them:

| rank | perplexity | document |
|---:|---:|---|
| 1 | **92.5** | `base_docs[1]` — Deep learning… |
| 2 | **92.5** | `near_dup_2` — *byte-identical to rank 1* |
| 3 | 110.1 | `near_dup_1` |
| 4 | 115.8 | `base_docs[0]` — *the pair of rank 3* |
| … | | |
| 9 | 143.1 | the HTML document, beginning `TitleMachine` |
| 11 | **145.9** | Natural language processing — *cut* |
| 12 | **152.9** | Generative adversarial networks — *cut* |
| 13 | **159.4** | Computer vision — *cut* |

**ANSWER: the bottom 20% is three ordinary encyclopedic documents.** The corpus
runs 92.5 to 159.4 — a factor of 1.72, end to end, with no gap anywhere to cut
at. The 20% is doing all the work; the distribution is not.

**FINDING: the four documents a deduplicator targets are the four this filter
protects.** Ranks 1 and 2 are the exact-duplicate pair. Ranks 3 and 4 are the
near-duplicate pair. A document repeated in the training set is maximally
predictable by a model trained on that set, so perplexity computed this way
*rewards* redundancy. It is a redundancy detector wearing a quality filter's
name, and it is pointed the wrong way.

**FINDING: the one genuinely damaged document is not removed.** `clean_text`
strips `</h1><p>` without leaving a separator, so the HTML document reaches the
filter beginning `TitleMachine`. It ranks 9th of 13 — mid-pack — and the cut
does not reach it. A single welded word costs almost nothing against a metric
measuring how well the rest of the corpus predicts a document.

**FINDING: "filtered vs unfiltered" measures the vocabulary.** Held-out
perplexity on three unseen sentences:

| scored under | unfiltered | filtered | change |
|---|---:|---:|---:|
| each model's own vocabulary | 294.1 | 234.2 | **+20.4%** |
| the shared vocabulary (327) | 294.1 | 295.0 | **−0.3%** |

Add-α smoothing divides by `alpha * V`. Dropping three documents drops the
vocabulary from 327 types to 258 — 21% smaller — which raises every probability
for free. The 20% "improvement" is the denominator shrinking; once the
vocabulary is held fixed, filtering buys nothing at all. The comparison the
exercise asks for, performed the way it is naturally performed, cannot tell the
two effects apart.
