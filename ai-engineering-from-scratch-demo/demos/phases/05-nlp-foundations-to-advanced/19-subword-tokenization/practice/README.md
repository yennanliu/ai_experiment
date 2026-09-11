<!-- generated:start -->
# 05-nlp-foundations-to-advanced / 19-subword-tokenization

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/05-nlp-foundations-to-advanced/19-subword-tokenization/) · upstream spec
`phases/05-nlp-foundations-to-advanced/19-subword-tokenization/docs/en.md`

```bash
uv run demo practice run 19-subword-tokenization --ex 1
uv run demo explain 19-subword-tokenization --ex 1
uv run pytest demos/phases/05-nlp-foundations-to-advanced/19-subword-tokenization
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Train a 500-merge BPE on `code/main.py`'s tiny corpus. Encode three held-out words. How… | code | T0 | `ex01_no_held_out_word_can_reach_one_token.py` |
| 2 | Medium. Compare token counts on 100 English Wikipedia sentences between `cl100k_base`, `o200k… | code | T0 | `ex02_the_ratio_it_reports_is_memorisation.py` |
| 3 | Hard. Train the same corpus with BPE, Unigram, and WordPiece. Measure downstream accuracy whe… | code | T0 | `ex03_the_choice_moves_less_than_one_example.py` |
<!-- generated:end -->

## Answers

Three exercises about a BPE implementation that, on the corpus each of them
specifies, is not a subword tokenizer. Run to the merge count exercise 1 asks
for, `train_bpe` converges to a word-level tokenizer over the training
vocabulary — which is why every held-out word splits, why the compression ratio
exercise 2 asks for is exactly 1.0000, and why the algorithm choice exercise 3
asks about moves less than one test example.

All three run at **T0** — `code/main.py` imports only `re` and `collections`.

### 1 — No held-out word can reach one token

`train_bpe(corpus, num_merges=500)`, then encode the four words `main()` holds
out:

| held-out word | pieces at 183 merges |
|---|---|
| tokenizable | `tokeniz` `a` `ble</w>` |
| unlearnable | `u` `n` `learnable</w>` |
| foxhound | `fox` `h` `o` `u` `n` `d</w>` |
| languages | `languag` `e` `s</w>` |

**ANSWER: 0 produce exactly one token, 4 produce more — and none can.**
`encode_bpe` returns a single symbol only when `word + "</w>"` is itself a
training word. Sweeping every merge count from 0 to 183 finds no count at which
any of the four reaches one.

**MECHANISM: the exercise asks for 500 merges and the corpus supplies 183.**
`train_bpe` breaks as soon as `pair_counts` is empty. Past that point the
requested vocabulary size is not a parameter of anything.

**FINDING: at convergence BPE has become a word-level tokenizer.** The final
vocabulary is exactly the 52 distinct corpus words with `</w>` appended, and
every one of them encodes to a single token. The 500-merge model the exercise
asks for is the word tokenizer that subword tokenization exists to replace.

**FINDING: the reported vocabulary size falls as training continues.**

| merges | reported vocab | shippable vocab (alphabet + merges) | held-out pieces |
|---:|---:|---:|---:|
| 0 | 27 | 27 | 31 (at 30) |
| **105** | **84** | 132 | — |
| 150 | 77 | 177 | — |
| 183 | **52** | **210** | **15** |

`train_bpe` returns the symbols left in the *final* decomposition, not the
alphabet plus the merge list, so a piece that gets merged away stops being
counted. **The reported number and the tokenizer's quality move in opposite
directions.**

**CONTROL: the lesson predicts the right outcome for the wrong reason.**
`main()` closes with "with a tiny toy corpus, most held-out words will split" and
tags a one-token result `OK`. It is all held-out words, not most, and corpus size
is not why.

### 2 — The ratio it reports is memorisation

`tiktoken` and `sentencepiece` are absent, so the two OpenAI arms cannot be
measured. The third cannot be built.

**ANSWER: vocab=32k is not a setting this corpus can honour.** 20 encyclopedic
sentences admit 566 merges before `pair_counts` runs dry — a total vocabulary of
**593** against the **32,000** requested. Vocabulary size is a ceiling the corpus
sets, not a parameter you pass.

| sentences | reachable vocabulary |
|---:|---:|
| 6 | 258 |
| 13 | 431 |
| 20 | 593 |

About 23 tokens per added sentence, so 32k needs on the order of **1,400**
sentences, not 100.

**FINDING: the ratio the exercise reports is memorisation.**

| merges | train tokens/word | held-out tokens/word |
|---:|---:|---:|
| 0 | 6.4735 | 6.5316 |
| 100 | 3.1755 | 3.7342 |
| 200 | 2.4939 | 3.3924 |
| 400 | 1.6776 | 3.2532 |
| **566** | **1.0000** | **3.1519** |

BPE run to exhaustion is a word tokenizer over the corpus vocabulary, so it
scores exactly 1.0000 on its own training text. 53 of 67 held-out word types are
unseen — the case subword tokenization exists for, and the case never reported.
**The two sides start equal and diverge at every step.**

**FINDING: the two sides do not encode the same string, so no ratio compares
them.** `word_counts` matches `[a-zA-Z]+`, dropping 80 non-alphabetic characters
including every numeral — `196`, `1969`, `8,849`, `5,500,000`, `1928`. Chars-per-token
here is computed over **1341** surviving characters of **1670**; tiktoken is
byte-level and lossless.

### 3 — The choice moves less than one example

The lesson ships BPE only. WordPiece is one changed expression — score a pair by
`freq(ab) / (freq(a) · freq(b))` — so it reuses `init_vocab`, `pair_counts` and
`merge_pair` untouched. Unigram shares no step and is not buildable from this
code. 30 reviews, leave-one-out Naive Bayes on bag-of-pieces:

| merges | BPE acc / F1 | WordPiece acc / F1 | leader |
|---:|---|---|---|
| 0 (chars) | 0.9333 / 0.9375 | 0.9333 / 0.9375 | tie |
| 25 | 0.9000 / 0.8966 | 0.9333 / 0.9333 | wordpiece |
| 50 | 0.9333 / 0.9333 | 0.9333 / 0.9333 | tie |
| 100 | 0.9000 / 0.9032 | 0.9333 / 0.9333 | wordpiece |
| 200 | 0.9667 / 0.9677 | 0.9667 / 0.9677 | tie |
| 146 (ceiling) | 0.9667 / 0.9677 | 0.9667 / 0.9655 | bpe |

**ANSWER: no.** Everything lands inside **6.67 points of accuracy** — two reviews
out of thirty. At matched merge counts the two algorithms differ by at most one
review, **and the sign flips**.

**MECHANISM: that is not the two algorithms agreeing.** Their merge lists differ
from the first entry — BPE takes `d` + `</w>`, WordPiece takes `f` + `u` — and
share 32 merges out of 146. Two nearly disjoint tokenizers give the same
classifier.

**FINDING: one point of F1 is below the grid this experiment runs on.** With 30
held-out items accuracy moves in steps of 3.33 points, so "more than 1 point" is
not a value the measurement can return. One point needs at least **100** held-out
items to be on the grid at all.

**CONTROL: a tokenizer that learned nothing is already within seven points of the
best.** Zero merges is character-level and scores 0.9375 F1 against 0.9677. The
headroom the algorithm choice competes for is seven points wide before any
algorithm is chosen.
