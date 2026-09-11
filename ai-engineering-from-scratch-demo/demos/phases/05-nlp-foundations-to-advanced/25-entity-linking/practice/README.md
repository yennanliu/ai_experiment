<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 25-entity-linking

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/25-entity-linking/) · upstream spec
`phases/05-nlp-foundations-to-advanced/25-entity-linking/docs/en.md`

```bash
uv run demo practice run 25-entity-linking --ex 1
uv run demo explain 25-entity-linking --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/25-entity-linking
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Implement the prior+context disambiguator in `code/main.py` on 10 ambiguous mentions (P… | code | T0 | `ex01_the_contexts_quote_the_knowledge_base.py` |
| 2 | Medium. Encode 50 ambiguous mentions with a sentence transformer. Embed each candidate's desc… | code | T0 | `ex02_no_lexical_scorer_can_do_better_here.py` |
| 3 | Hard. Build a 1k-entity domain KB (e.g. employees + products in your company). Implement NER… | code | T0 | `ex03_every_error_is_the_popularity_prior.py` |
<!-- generated:end -->

## Answers

The disambiguator scores 100% on the eleven cases the lesson ships with it. Every
finding here comes from asking what that number is made of: the contexts quote
the knowledge base, the scoring function is interchangeable, and on realistic
text the whole system collapses onto its popularity prior.

All three run at **T0** — `code/main.py` imports only `re` and `collections`.

### 1 — The contexts quote the knowledge base

Eleven cases from `main()`, and eleven hand-labelled cases about the same
entities written without reusing the KB's wording:

| set | with prior | without prior | argmax-prior only |
|---|---:|---:|---:|
| `main()`'s own (quoting) | **11 / 11** | 10 / 11 | 4 / 11 |
| hand-labelled (paraphrase) | **5 / 11** | **7 / 11** | 4 / 11 |

**MECHANISM: the lesson's contexts quote the KB.** "Paris is the capital of
France and home to the Eiffel Tower" shares 6 of the 9 tokens in `Q90`'s
description. The accuracy being measured is how the evaluation sentences were
written.

**FINDING: the prior's contribution changes sign.** It rescues one case on the
quoting set and costs two on the paraphrase set, where it pulls four mentions to
the most popular reading. Whether the prior helps is a property of the eval.

**MECHANISM: the case it rescues has no context signal at all.** "Jordan scored
45 points against the Lakers" shares exactly one token with each of the four
Jordan descriptions — and that token is `jordan`. All four Jaccard scores tie and
the prior returns the most popular entity, which is what a system with no context
model would have said.

**CONTROL: `use_prior=False` is a uniform prior, not no prior.** It sets every
candidate to 1.0, so the score becomes `jac + 0.1` and the ranking is Jaccard
alone. And an unrecognised mention returns `(None, 0.0)` — there is no NIL.

### 2 — No lexical scorer can do better here

`sentence_transformers` is absent, so the encoder arm cannot be built. The rest
of the family can:

| scorer | quoting | paraphrase (prior / flat) |
|---|---:|---:|
| Jaccard | 11 | 5 / 7 |
| overlap count | 11 | 6 / 6 |
| Dice | 11 | 6 / 7 |
| coverage of description | 11 | 6 / 7 |

**ANSWER: the choice of scoring function is invisible on the lesson's set and
worth at most one case on the other.**

**FINDING: on 8 of 11 paraphrase contexts, no candidate shares a single content
token** once the mention's own name is removed. Every lexical score is zero,
every candidate ties, and the prior decides. **That is a bound on the whole
family** — any function of token overlap returns the same ranking when the
overlap is empty for every candidate. On the quoting set only 1 of 11 is in that
state.

**FINDING: the encoder is the only arm that could differ**, and those eight cases
say why: linking "the United Center" to *Chicago Bulls*, or "burst its banks" to
*Seine river*, needs knowledge token overlap cannot represent at any weighting.

**CONTROL: the KB caps the exercise.** 5 aliases over 14 entities — "50 ambiguous
mentions" means resampling five distinct problems.

### 3 — Every error is the popularity prior

The lesson ships no NER, so the detector is alias-string matching. 18 sentences,
14 gold mentions:

| | count | correctly handled |
|---|---:|---:|
| linkable (entity in KB) | 6 | **4** |
| entity not in KB (gold NIL) | 4 | **0** |
| surface not in alias index | 4 | **0** (undetectable) |
| sentences with no entity | 4 | 4 |

**ANSWER: precision 0.4000, recall 0.2857.**

**MECHANISM: recall is capped before linking begins.** Alias matching finds 10 of
14 gold mentions; `Microsoft`, `Berlin`, `Java` and `Amazon` have no entry, so
**0.7143 is the ceiling** whatever the disambiguator does.

**FINDING: precision is destroyed by the missing NIL.** `disambiguate` returns a
candidate whenever the alias resolves, so all four out-of-KB mentions come back
confidently linked — Jordan Peterson → the basketball player, Paris Ontario →
Paris France, Apple Records → Apple Inc, Washington the governor → George
Washington.

**FINDING: all six errors are the same answer.** `Q41421`, `Q28865`, `Q41421`,
`Q90`, `Q312`, `Q23` are exactly the argmax-prior reading of their alias. The two
genuine disambiguation failures and the four missing NILs are one event.

**MECHANISM: the pair of numbers hides which component failed.** The detector
produces 0 spurious mentions, so the recall loss is entirely detection and the
precision loss entirely linking.

**CONTROL: the requested KB is 71× the shipped one, and nothing scales.**
`ALIAS_INDEX` and `PRIORS` are hand-written dictionaries with 5 and 14 entries.
