<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 26-relation-extraction-kg

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/26-relation-extraction-kg/) · upstream spec
`phases/05-nlp-foundations-to-advanced/26-relation-extraction-kg/docs/en.md`

```bash
uv run demo practice run 26-relation-extraction-kg --ex 1
uv run demo explain 26-relation-extraction-kg --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/26-relation-extraction-kg
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run the pattern extractor in `code/main.py` on 5 news-article sentences. Hand-check pre… | code | T0 | `ex01_precision_is_no_higher_than_recall.py` |
| 2 | Medium. Use REBEL (or a small LLM) on the same sentences. Compare triples. Which extractor ha… | code | T0 | `ex02_eight_lines_of_regex_beat_the_comparison.py` |
| 3 | Hard. Build the AEVS pipeline: extract with LLM + verify spans against source. Measure halluc… | code | T0 | `ex03_the_verify_step_checks_offsets_not_claims.py` |
<!-- generated:end -->

## Answers

The extractor is 8 for 8 on the document the lesson ships with it. On thirteen
news sentences it is 50% precise, its own verification step cannot reject
anything, and eight lines of regex beat every comparison the exercises propose.

All three run at **T0** — `code/main.py` imports only `re` and `collections`.

### 1 — Precision is no higher than recall

13 sentences, 11 hand-labelled gold triples, 10 predicted, 5 correct:

| | value |
|---|---:|
| precision | **0.5000** |
| recall | **0.4545** |
| false positives | 5 |
| false negatives | 6 |

**ANSWER: the closing note's "high precision, low recall" is half right.** The
two numbers are within one triple of each other.

**MECHANISM: four of the five false positives are denied or hedged clauses.**
"The company denied that Carl Reed was born in Ohio" yields `(Carl Reed, place of
birth, Ohio)`. The patterns match a clause and never look at what governs it.

**FINDING: the fifth false positive is also a false negative.** "Steve Jobs
founded Apple Computer Inc" returns the object `Apple`, because the object group
is `([A-Z][A-Za-z]+(?: Inc)?)` — which admits *Apple Inc* but not *Apple Computer
Inc*. One truncation counts on both sides.

**FINDING: the six misses are syntax the pattern cannot express** — a passive, a
relative clause, a coordinated subject carrying two triples, a coordinated
object, and the truncation.

**MECHANISM: both directions are the same regex property.** Every pattern
requires subject, verb and object adjacent — which excludes any paraphrase with
material between them and admits any clause containing that adjacency, however
embedded.

**CONTROL: on `main()`'s own document it is 8 for 8.** Eight sentences in the
contiguous active form, each with a two-word name and a one-word object.

### 2 — Eight lines of regex beat the comparison

`transformers` is absent, so REBEL cannot be run — but the question behind it
can be answered by moving the shipped extractor along its own frontier. A
**guard** drops matches in sentences with a denial or hedge cue; a
**relaxation** lets the object run over consecutive capitalised words and adds a
passive and a relative-clause form.

| variant | precision | recall | F1 |
|---|---:|---:|---:|
| as shipped | 0.5000 | 0.4545 | 0.4762 |
| + guard | **0.8333** | 0.4545 | 0.5882 |
| + relaxation | 0.6667 | **0.7273** | 0.6957 |
| both | **1.0000** | **0.7273** | **0.8421** |

**ANSWER: the shipped extractor is dominated by a variant built from its own six
patterns.** Precision doubles, recall rises 60%, F1 goes 0.4762 → 0.8421.

**MECHANISM: the two edits are independent.** The guard moves precision and
leaves recall *exactly* at 0.4545 — it removes only triples that were wrong. The
relaxation moves recall and drags precision up as a side effect, because two new
matches replace truncated objects.

**FINDING: there is no operating curve to compare against.** A triple carries
`evidence, object, relation, span, subject` — no confidence, no threshold. Each
variant is a single point, and comparing two systems at one unknown operating
point each has no answer that survives editing a regex.

### 3 — The verify step checks offsets, not claims

**ANSWER: `extract` returns 10 triples and `verify` keeps 10 — a rejection rate
of 0.0000**, and it is 0.0000 for every possible input.

**MECHANISM: both conditions are tautologies for a regex match.** `span` comes
from `m.start(), m.end()` and `evidence` from `m.group(0)` of the same match, so
`text[s:e] != evidence` compares a string against a slice of itself.

Applied to seven hand-written candidates of the kinds a language model produces:

| candidate | verdict | correct? |
|---|---|---|
| faithful | kept | ✓ |
| entity absent from the text | **rejected** | ✓ |
| subject given as a partial name (`Tim`) | kept | ✗ |
| relation fabricated, entities real | kept | ✗ |
| evidence not matching its offsets | **rejected** | ✓ |
| offsets off by one, claim correct | **rejected** | ✗ |
| lifted from a denied clause | kept | ✗ |

**MECHANISM: offsets are checked to the character and content is not checked at
all.** A span one byte off is rejected although the claim is right; a triple
whose relation was invented is kept because its offsets are right.

**FINDING: a denied clause verifies, because it really is in the text.**
Verifying provenance is not verifying the assertion — exercise 1's four hedged
false positives all survive this step.

**CONTROL: it cannot be applied to the source it exists for.** `verify` reads
`t["span"]`, which a triple from a language model does not have — `KeyError:
'span'`.
