<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 21-nli-textual-entailment

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/21-nli-textual-entailment/) · upstream spec
`phases/05-nlp-foundations-to-advanced/21-nli-textual-entailment/docs/en.md`

```bash
uv run demo practice run 21-nli-textual-entailment --ex 1
uv run demo explain 21-nli-textual-entailment --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/21-nli-textual-entailment
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `facebook/bart-large-mnli` on 20 hand-crafted (premise, hypothesis, label) triples… | code | T0 | `ex01_the_trap_it_names_is_the_one_it_survives.py` |
| 2 | Medium. Compare the zero-shot template `"This text is about {label}"` against `"The topic is… | code | T0 | `ex02_the_swing_is_the_templates_word_count.py` |
| 3 | Hard. Build a RAG faithfulness checker: atomic-claim decomposition + NLI per claim. Evaluate… | code | T0 | `ex03_one_wrong_word_can_never_fail_a_claim.py` |
<!-- generated:end -->

## Answers

`transformers`, `torch` and `datasets` are all absent, so all three exercises run
against the lesson's own `predict_nli` — lexical overlap of content words plus a
negation flag. Every finding here is about what those two features can and cannot
represent, and in each exercise the answer is decided before any data is chosen.

All three run at **T0** — `code/main.py` imports only `re` and `collections`.

### 1 — The trap it names is the one it survives

20 triples, balanced across the three classes, with the exercise's own trap among
them. **15/20.**

| trap class | correct |
|---|---|
| negation ("I did not eat the cake" vs "I ate the cake") | **6 / 7** |
| argument swap ("The doctor saw the lawyer" vs the reverse) | **0 / 3** |

**ANSWER: the subsequence trap the exercise names does not break it.** Negation
is one of the two features `predict_nli` reads.

**MECHANISM: what breaks it is word order, and completely.** `lexical_overlap`
compares *sets* of content words, so permuting either side leaves the label
unchanged in **400 of 400** shuffles. Every argument swap comes back as
entailment at overlap 1.00.

**FINDING: negation is parity, not scope, so two of them cancel.**
`has_negation` counts `without`, so "Nobody left the building without a badge"
and "Somebody left without a badge" both carry one, the parities match, and the
pair comes out entailment.

**FINDING: neutral is unreachable once overlap reaches 0.5.** The label is two
bits — overlap at 0.5, negation parity — and both branches above the threshold
return entailment or contradiction. No hypothesis that reuses half the premise's
content words can be neutral, whatever it says.

**CONTROL: two thirds of the labels need no premise.** Answering contradiction
when the hypothesis contains a negation and neutral otherwise scores **13/20**
without reading the premise, against 15/20 for the classifier — the
hypothesis-only artefact NLI benchmarks are known for.

### 2 — The swing is the template's word count

16 headlines written to the four AG News classes, without naming them:

| template | overlap range | argmax accuracy |
|---|---|---:|
| `This text is about {label}` | `[0.0]` | 4 / 16 |
| `The topic is {label}` | `[0.0]` | 4 / 16 |
| `{label}` | `[0.0]` | 4 / 16 |

**ANSWER: the swing is 0.000, and so is every score.** The class name never
appears in the headline, and lexical overlap is the only evidence there is.
Argmax then returns whichever label is first in the list, so accuracy is exactly
the `World` share of the corpus — reordering `LABELS` would change it and nothing
about the templates would have changed.

On five headlines that *do* contain their class name:

| template | ceiling for a 1-token label | entailment |
|---|---:|---:|
| `This text is about {label}` | 0.33 | **1 / 5** |
| `The topic is {label}` | 0.50 | 4 / 5 |
| `{label}` | 1.00 | **5 / 5** |

**MECHANISM: the template's own words sit in the denominator.**
`lexical_overlap` divides by the hypothesis's content-word count, so a template's
ceiling is `k / (k + e)` for a `k`-token label and `e` added content words.
`predict_nli` thresholds at 0.50, which falls between the first two rows. The
prompt is a threshold crossing in disguise.

**FINDING: label length enters the same denominator.** `Sci/Tech` is two tokens,
so under `This text is about {label}` it reaches 0.50 and returns entailment
while every one-token label maxes at 0.33 and returns neutral. One template, two
verdicts, identical evidence.

**CONTROL: the swing has no value until the decision rule is named.** The same
five headlines score **0 points of swing** under argmax and **80** under the
entailment threshold.

### 3 — One wrong word can never fail a claim

6 contexts, each with a faithful answer and an unfaithful twin differing by
exactly one content word. Decomposed into atomic claims:

| threshold | TP | FP | FN | TN | accuracy |
|---:|---:|---:|---:|---:|---:|
| **0.50** (the lesson's) | 5 | **5** | 1 | 1 | **0.500** |
| 0.67 | 3 | 1 | 3 | 5 | 0.667 |
| 0.75 | 3 | 1 | 3 | 5 | 0.667 |
| 0.90 | 3 | **0** | **3** | 6 | 0.750 |
| 1.00 | 3 | 0 | 3 | 6 | 0.750 |

**ANSWER: 5 false positives, 1 false negative, accuracy at chance.**

**MECHANISM: every false positive is a one-word substitution scoring above
threshold** — `21 percent` for `12 percent` at 0.80, `Chile` for `Brazil` at
0.50, `1852` for `1843` at 0.50, `rejected` for `approved` at 0.67, `rose` for
`fell` at 0.75.

**MECHANISM: which is arithmetic, not a tuning problem.** A claim of `c` content
words with one wrong word scores `(c-1)/c` — 0.500, 0.750, 0.875, 0.938 for c =
2, 4, 8, 16. That is at least 0.5 for every `c` from 2 up, so **at the
classifier's own threshold a single false word cannot fail a claim, at any
decomposition granularity.**

**FINDING: the two error rates are one knob.** Error counts across the sweep run
6, 4, 4, 3, 3 and never fall below 3. False positives reach zero only where half
the faithful answers are rejected: catching a substitution means demanding every
content word literally, which is exactly what a paraphrase does not do.

**FINDING: the decomposition the exercise leads with barely participates.**
Undecomposed accuracy is identical at both ends of the sweep; at 0.5 the split
only trades one false positive for one false negative.

**CONTROL: the one false negative is a faithful paraphrase, which is the same
failure.** "It took two attempts" restates "the second launch attempt" and shares
no content word with it, scoring 0.00. A missing word is what a substitution
looks like and what a paraphrase looks like — one signal for two situations.
