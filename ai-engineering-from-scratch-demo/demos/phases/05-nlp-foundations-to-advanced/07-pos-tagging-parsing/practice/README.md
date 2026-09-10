<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 07-pos-tagging-parsing

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/07-pos-tagging-parsing/) · upstream spec
`phases/05-nlp-foundations-to-advanced/07-pos-tagging-parsing/docs/en.md`

```bash
uv run demo practice run 07-pos-tagging-parsing --ex 1
uv run demo explain 07-pos-tagging-parsing --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/07-pos-tagging-parsing
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Using the most-frequent-tag baseline on a small tagged corpus (e.g., NLTK's Brown subse… | code | T0 | `ex01_the_baseline_measures_the_corpus.py` |
| 2 | Medium. Train the bigram HMM above and report per-tag precision/recall. Which tags does the H… | code | T0 | `ex02_the_hmm_gain_is_exactly_the_hard_tokens.py` |
| 3 | Hard. Use spaCy's dependency parse to extract subject-verb-object triples from a 1000-sentenc… | code | T0 | `ex03_passives_do_not_fail_they_reverse.py` |
<!-- generated:end -->

## Answers

Three exercises about a number that belongs to the data rather than the method.
Exercise 1's "~85%" is one point on a curve indexed by out-of-vocabulary rate.
Exercise 2's HMM beats the baseline by exactly the tokens where the word alone
is not enough, and by nothing else. Exercise 3's failure taxonomy names the
right constructions and the wrong failure — two of the three break by *emitting*
a wrong triple, not by staying silent.

All three run at **T0**: the lesson's `code/main.py` imports only `math` and
`collections`.

### 1 — The baseline measures the corpus

NLTK is not installed and Brown is not downloadable, so the corpus is 56
sentences generated from 7 tag templates over 30 word types, 5 of them
deliberately ambiguous. The template *is* the gold tag sequence, so the labels
are exact rather than annotated.

**ANSWER: 0.8674 at a 30% training split, 0.9103 at 70%.**

| training fraction | 0.3 | 0.5 | 0.7 |
|---|---:|---:|---:|
| most-frequent-tag accuracy | **0.8674** | 0.9062 | 0.9103 |
| OOV rate | 0.0552 | 0.0156 | 0.0000 |

The exercise's ~85% is in there, at one split.

**MECHANISM: three groups, and their weighted mean *is* the headline.** At the
30% split:

| group | accuracy | tokens |
|---|---:|---:|
| seen, unambiguous | **1.0000** | 127 |
| ambiguous | 0.6364 | 44 |
| out-of-vocabulary | 0.2000 | 10 |

Weighted mean **0.8674**, against the measured 0.8674. The headline carries no
information the three parts do not.

**MECHANISM: the first group is right by construction.** A word with a single
training tag has that tag as its most frequent one, so the baseline returns it.
That group cannot be improved and cannot be lost — which is why the headline is
decided entirely by the other two.

**FINDING: the OOV group is the default tag.** Every unseen word gets `NOUN`,
the corpus's most frequent tag, and is right when that lands — 2 of 10 here. The
OOV rate falling with the split is most of what moves the number.

**FINDING: the ambiguous group is the only place a better method has room.** The
five ambiguous words (`fast`, `light`, `run`, `walk`, `watch`) account for 44
held-out tokens at 0.6364. The baseline takes the majority reading every time
and has no way to take the other one.

### 2 — The HMM gain is exactly the hard tokens

**ANSWER: per-tag precision/recall at the 30% split** — everything above 0.96
except ADV recall:

| tag | P | R | support |
|---|---:|---:|---:|
| ADJ | 1.000 | 1.000 | 29 |
| ADP | 1.000 | 1.000 | 5 |
| ADV | 1.000 | **0.909** | 11 |
| AUX | 1.000 | 1.000 | 12 |
| DET | 0.975 | 1.000 | 39 |
| NOUN | 0.982 | 0.982 | 57 |
| VERB | 0.964 | 0.964 | 28 |

The mistakes are `NOUN→DET`, `VERB→NOUN`, `ADV→VERB` — **one apiece**. "Which
tags does it confuse most" has no strong answer at this size, and three
singletons should not be ranked.

**MECHANISM: the gain is an arithmetic identity, not an assertion.**

| group | baseline | HMM |
|---|---:|---:|
| seen, unambiguous (127 tokens) | 1.0000 | **1.0000** |
| ambiguous (44) | 0.6364 | **0.9773** |
| out-of-vocabulary (10) | 0.2000 | **0.8000** |
| **overall** | 0.8674 | **0.9834** |

At the 70% split, where nothing is out of vocabulary, it is 0.9103 → **1.0000**.
Context buys the tokens where the word alone is not enough, and nothing else.

**FINDING: `train_hmm` counts transitions into `<EOS>` and `viterbi` never reads
them.** The lesson's own TRAIN gives `{ADJ: 1, ADV: 1, NOUN: 3, VERB: 1}` into
`<EOS>`, and its sentences end on `[ADJ, ADV, NOUN, VERB]`. `viterbi` takes its
answer from `max(V[n-1])` with no final transition, so `The the the` decodes to
`['DET', 'ADP', 'DET']` — ending on **DET**, which no training sentence ends on.

**CONTROL: the tie-break depends on set order, which Python varies per
process.** `viterbi` opens with `list(tags)` on a `set` and its argmax keeps the
first of equal scores. On a two-tag corpus with identical statistics that decides
the answer outright: order `['X','Y']` returns `X`, order `['Y','X']` returns
`Y`, from the same model on the same token.

**CONTROL: on the lesson's own model it does not fire.** All **5040** orderings
of its seven tags give one distinct output. Exact ties do not arise between sums
of real-valued log probabilities, so the hazard is latent — which is why it
survives.

### 3 — Passives do not fail, they reverse

`spacy`, `nltk` and `stanza` are absent and there is no 1000-sentence sample, so
the labelled set is 12 sentences carrying 15 triples with **gold** POS tags —
four to each failure class the exercise names, plus a control class of plain
actives. No tagging error is in these numbers.

**ANSWER: 0.6000 recall, with 5 spurious triples.**

| class | recall | gold triples | spurious |
|---|---:|---:|---:|
| active (control) | **1.0000** | 4 | 0 |
| passive | **0.0000** | 3 | **3** |
| coordination | 0.5000 | 6 | 0 |
| elided subject | **1.0000** | 2 | **2** |
| **all** | **0.6000** | 15 | 5 |

**FINDING: passives do not return nothing — all three come back reversed.**
`the dog was chased by the cat` has gold `(cat, chased, dog)` and the extractor
returns `(dog, chased, cat)`. The class produces *exactly as many triples as it
should* and gets every one backwards: three confident facts asserting the
opposite of the sentence. A consumer counting extraction failures sees zero
silent drops here, which is why the exercise asks for 50 labelled triples rather
than a coverage count.

**FINDING: coordination halves recall by construction.**
`the cat and the dog chased the mouse` has two gold triples; the extractor
returns `[('dog', 'chased', 'mouse')]` — the nearer conjunct only. That is 3 of
6 across the class, and it is not a tuning problem: "nearest noun" returns one
noun.

**FINDING: elided subjects score perfect recall and still invent two triples.**
`the cat sat and watched the dog` returns
`[('cat', 'sat', 'dog'), ('cat', 'watched', 'dog')]` — the real triple plus one
giving the intransitive `sat` an object it does not have. The exercise lists it
as a failure class; by recall it is the joint best.

**CONTROL: all 5 spurious triples come from the passive and elided classes.**
Coordination and the active control produce none. The taxonomy names the right
constructions and the wrong failure — precision, not recall, is where two of the
three land.

### A note on file lengths

The three files run 149 / 135 / 149 lines of code, over D14's 120-line target
and under its 150-line ceiling. The overrun is fixture in all three cases: the
exercises name Brown, a bigram HMM's confusion matrix, and 50 manually labelled
triples, and none of the three ships. Exercise 2 imports exercise 1's corpus,
ambiguity set and bucketing via `practice.load_module` rather than rebuilding
them.
