<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 06-named-entity-recognition

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/06-named-entity-recognition/) · upstream spec
`phases/05-nlp-foundations-to-advanced/06-named-entity-recognition/docs/en.md`

```bash
uv run demo practice run 06-named-entity-recognition --ex 1
uv run demo explain 06-named-entity-recognition --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/06-named-entity-recognition
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Implement `bio_to_spans` (the inverse of `spans_to_bio`) and verify round-trip consiste… | code | T0 | `ex01_the_round_trip_that_holds_and_the_one_that_does_not.py` |
| 2 | Medium. Train the sklearn-crfsuite CRF above on the CoNLL-2003 English NER dataset. Report pe… | code | T1 | `ex02_token_accuracy_flatters_what_entity_f1_does_not.py` |
| 3 | Hard. Fine-tune `distilbert-base-cased` on a domain-specific NER dataset (medical, legal, or… | code | T1 | `ex03_the_split_is_worth_more_than_the_model.py` |
<!-- generated:end -->

## Answers

Three exercises about the gap between a number and what it means. Exercise 1
asks for an inverse that exists in one direction only. Exercise 2 names
`seqeval` without saying why, and the why is a factor of nine. Exercise 3 asks
you to document data leakage checks, and the leak turns out to be worth more
than the model, the features and the architecture combined.

Exercise 1 is **T0** (the lesson's `code/main.py` is pure Python); 2 and 3 are
**T1** (scikit-learn).

### 1 — The round trip that holds, and the one that does not

**ANSWER: spans → BIO → spans is the identity on all 10 sentences.** Including
the empty one, and including `Google Google Google`, which comes back as three
separate `ORG` spans — BIO2's `B-` restarts a span where an `I-` would have
merged them.

**MECHANISM: the other direction is not the identity.**

| | |
|---|---:|
| length-4 label sequences over 2 entity types | 625 |
| fixed points of BIO → spans → BIO | **153** |
| rewritten | **472** |
| well-formed IOB2 sequences, counted independently by the transition rule | **153** |

The two counts are derived separately and agree. `bio_to_spans ∘ spans_to_bio`
is not an inverse pair; it is a **projection onto valid IOB2**.

**FINDING: every rewrite is a silent deletion.** All 472 changed sequences
contain an `I-` tag, and `['O','O','O','I-A']` comes back `['O','O','O','O']` —
no exception, no warning, no return value to check. An entity is simply gone.

**FINDING: CoNLL-2003 is IOB1, so this drops gold entities on exercise 2's own
dataset.** In IOB1 a lone `I-` legitimately opens an entity.
`bio_to_spans(['a','b'], ['I-ORG','I-ORG'])` returns `[]` — not mis-bounded,
absent.

**CONTROL: `spans_to_bio` validates nothing either.**

| span list | comes back as |
|---|---|
| `[(2, 2, 'ORG')]` zero width | `[(2, 3, 'ORG')]` |
| `[(3, 1, 'ORG')]` reversed | `[(3, 4, 'ORG')]` |
| `[(0, 3, 'ORG'), (1, 2, 'GPE')]` overlapping | `[(0, 1, 'ORG'), (1, 2, 'GPE')]` |

The later span writes over the earlier one's labels, so the three-token `ORG`
comes back one token wide.

### 2 — Token accuracy flatters what entity F1 does not

`sklearn_crfsuite`, `seqeval`, `datasets` and `pycrfsuite` are all absent and
CoNLL-2003 is not downloadable, so the exercise's "~84 F1" is neither confirmed
nor denied. What is buildable is the reason it names `seqeval`.

The corpus is 144 sentences over 24 entities (half multi-token) from six frames;
the model is logistic regression over the lesson's own `token_features` — the
feature function the CRF would use — with **no test entity appearing in
training**.

**ANSWER: the same predictions, scored two ways.**

| | token accuracy | entity F1 |
|---|---:|---:|
| the model | **0.8302** | **0.0952** |
| answer `O` everywhere | 0.7547 | 0.0000 |

The model is **0.0755** above a model that never finds anything.

**FINDING: multi-token entities score exactly zero.**

| | gold spans | recovered | spurious | F1 | token accuracy |
|---|---:|---:|---:|---:|---:|
| single-token | 24 | 6 | 18 | **0.2500** | 0.8750 |
| multi-token | 24 | **0** | 54 | **0.0000** | 0.7931 |

The aggregate 0.0952 is carried entirely by the single-token half. A per-token
classifier has no representation of "this tag follows that tag", so a
three-token name has to come out right three times independently *and* agree on
its boundaries. Transition features are the whole of what a CRF adds over this
model.

**FINDING: the errors are over-prediction, not silence.** 78 proposed spans
against 48 real — 72 spurious, 42 missed, precision 0.0769, recall 0.125.

### 3 — The split is worth more than the model

`transformers`, `spacy`, `torch` and `datasets` are all absent and no domain NER
corpus is in the checkout, so the fine-tune and the spaCy comparison are not
attempted. The third instruction — "document data leakage checks" — is
measurable, and it turns out to be the one worth the most.

**ANSWER: the same model on the same 144 sentences.**

| split | entity overlap train/test | token accuracy | entity F1 |
|---|---:|---:|---:|
| by **sentence** (what `train_test_split` gives you) | 23 of 24 | 1.0000 | **1.0000** |
| by **entity** | 0 of 24 | 0.8302 | **0.0952** |

Nothing else changed — not the data, not the features, not the classifier. The
leak is worth **0.9048 F1**, more than any architecture choice in the exercise.

**FINDING: the lesson's own gazetteer is the same leak in its purest form.**
`rule_based_ner` scores F1 **0.8571** on the half its gazetteer was written
around and **0.0000** on the other half — 8 of the 12 ORG names are in
`ORG_GAZETTEER` and none of the held-out four. It is not 1.0 on the first half
either, for the same reason: `GPE_GAZETTEER` is missing two of the six countries
there. Its score *is* its coverage of the test set, which is a fact about the
list rather than about the method.

**MECHANISM: `rule_based_ner` emits only `B-`, so its maximum span width is 1.**
Every branch appends a `B-` label and none appends `I-`. Feed it all 15
gazetteer words in a row and it returns **15 separate entities**; the set of span
widths it can produce is `[1]`. `New York City` is three GPEs or none.

**CONTROL: a quarter of this corpus is unrepresentable at any coverage.** 36 of
the 144 sentences carry a multi-token entity and the gazetteer recovers **0**.
Adding every name in the corpus to it would leave that at zero, because the
limit is the label set rather than the vocabulary — the one failure a leakage
check cannot flatter.

### A note on file lengths

The three files run 121 / 150 / 137 lines of code, over D14's 120-line target
and at or under its 150-line ceiling. Exercise 1's overrun is the 10-sentence
fixture the exercise asks for plus the exhaustive 625-sequence sweep; exercise
2's is the corpus, the feature pipeline and a scorer that computes entity F1 the
way `seqeval` would, since `seqeval` is not installed. Exercise 3 imports
exercise 2's corpus, trainer and scorer via `practice.load_module` rather than
duplicating them, and spends its own lines on the two splits and the gazetteer.
