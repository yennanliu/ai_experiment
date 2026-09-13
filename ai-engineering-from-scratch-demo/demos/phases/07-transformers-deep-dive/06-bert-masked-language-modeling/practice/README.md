<!-- generated:start -->
# 07-transformers-deep-dive / 06-bert-masked-language-modeling

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/07-transformers-deep-dive/06-bert-masked-language-modeling/) · upstream spec
`phases/07-transformers-deep-dive/06-bert-masked-language-modeling/docs/en.md`

```bash
uv run demo practice run 06-bert-masked-language-modeling --ex 1
uv run demo explain 06-bert-masked-language-modeling --ex 1
uv run pytest demos/phases/07-transformers-deep-dive/06-bert-masked-language-modeling
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py` and print the mask distribution across 10,000 tokens. Confirm ~15% a… | code | T0 | `ex01_ten_thousand_tokens_cannot_confirm_the_eighty.py` |
| 2 | Medium. Implement whole-word masking: if a word is tokenized into subwords, mask all subwords… | code | T0 | `ex02_two_thirds_of_the_masks_leak_from_a_sibling.py` |
| 3 | Hard. Train a tiny (2-layer, d=64) BERT on 10,000 sentences from a public dataset. Fine-tune… | code | T0 | `ex03_the_decoder_gets_six_point_seven_times_the_signal.py` |
<!-- generated:end -->

## Answers

The lesson is 147 lines of masking rules and no model — its only "model",
`toy_predict`, returns a uniform distribution. All three exercises ask for a
measurement, and in two of the three the measurement they name cannot move. All
three are **T0**; pure stdlib, as the lesson is.

### 1 — ten thousand tokens is too few to confirm the number it asks about

The lesson's own `distribution_check`, same seed, same vocabulary, at three sizes:

| tokens | selected | → `[MASK]` | → random | → unchanged |
|---:|---:|---:|---:|---:|
| **10,000** (the exercise) | 15.710% | **78.167%** | 11.076% | 10.757% |
| 100,000 (`main()`) | 15.033% | 79.804% | 10.151% | 10.045% |
| 1,000,000 | 15.013% | **80.021%** | 9.942% | 10.037% |

**FINDING: the exercise's sample size cannot confirm its own second number.** At
10,000 tokens only ~1,500 are selected, so 1σ on the 80% is **1.03 points** — a
±3.1 point window at 3σ. The measured 78.17% sits 1.8σ low: unremarkable, and
also indistinguishable from a real 78% rule. `main()` runs the identical check at
**ten times** the exercise's sample size, which is presumably why.

**FINDING: `create_mlm_batch` hangs for `vocab_size ≤ 4`.** The random-replacement
branch loops `while rand_id in SPECIAL_IDS or rand_id == t`. With ids 0, 1, 2
reserved there is no legal replacement for `t = 3`. Probed under a 0.3 s alarm:
`vocab_size=4` never returns, `vocab_size=5` does. The toy vocab is 20 words, so
the lesson never meets its own edge.

**FINDING: 15% is a rate over *eligible* positions.** The loop `continue`s on
specials, so on a sequence shaped like `main()`'s — `[CLS]` + 9 words + `[SEP]` —
only 9 of 11 positions can be selected and the expected rate is **12.3%**.
`distribution_check` never generates a special, which is why *its* number is 15.

**CONTROL: only 13.5% of tokens are actually changed**, 1.5% are labelled and
left visible, and **12%** ever become `[MASK]` — that last figure being the one
the pretrain/finetune mismatch is really about.

### 2 — two thirds of token-level masks leak from a sibling

500 synthetic sentences, word lengths drawn 55/25/13/7 over 1–4 subwords (mean
1.71, 45% multi-subword), 11,085 tokens:

| | token-level | whole-word |
|---|---:|---:|
| tokens selected | 14.163% | 13.893% |
| **masks with a surviving sibling** | **64.27%** | **0.00%** |
| per-sentence mask count, CV | 0.589 | 0.825 |

**ANSWER: whole-word masking removes a 64% leak.** Under `create_mlm_batch`, two
thirds of masked subwords sit in a word that still shows an unmasked subword —
the answer spelled out beside the question. Under `whole_word_mlm` it is zero by
construction: the 80/10/10 branch is drawn per span and applied to every token
in it.

**FINDING: the prescribed metric provably cannot move.** `toy_predict` returns
`1/V` at every position and every vocabulary entry, so MLM accuracy is exactly
0.005 under both schemes, for any corpus, at any mask probability. The exercise
asks whether whole-word masking improves a number that is a constant.

**CONTROL: both schemes select the same share of tokens.** Selection is per word
and independent of word length, so the token rate is unbiased — whole-word
masking is not harder because it masks *more*.

**FINDING: what changes is the variance.** The per-sentence mask count is
**1.40×** lumpier, because one draw now commits up to four positions. Same rate,
noisier gradient: that is the price of removing the leak.

### 3 — the decoder gets 6.7× the supervised positions

Nothing in the setup exists. `torch`, `datasets` and `transformers` all return
`None` from `find_spec`; there is no SST-2 and no network to fetch one; the
module's 12 symbols include no encoder and no training loop. `CLS_ID` appears on
exactly **two lines** of the whole lesson — its own definition and `SPECIAL_IDS`
— so every masking path *skips* it and nothing ever pools it. There is no `[CLS]`
representation to fine-tune. The comparison is therefore decided where it is
decided: on the supervision arithmetic, at L=128.

| per sequence | MLM | causal LM | ratio |
|---|---:|---:|---:|
| supervised positions | 19.2 | 128 | **6.67×** |
| mean context per supervised position | 127 | 64.5 | 1.97× *(BERT)* |
| **supervised context-tokens** | 2,438 | 8,256 | **3.39×** |

**ANSWER: at matched parameters the decoder-only baseline wins on signal.** The
position ratio is `1/mask_prob` and does not depend on model size, so "at matched
params" fixes everything except the one quantity that differs. BERT's
compensation — bidirectional context — is real and worth 1.97×, about a third of
what the position count gives away. Net, the decoder leads **3.39×**. That ratio,
not the architecture, is why decoder-only pretraining won the scaling race; both
models are the same transformer with a different mask.

**CONTROL: 12% of pretraining tokens carry a symbol that never occurs
downstream**, and **85%** of every sequence is pushed through the encoder to
produce no gradient at all. The compute is spent; the supervision is not. A
causal LM has neither gap.
