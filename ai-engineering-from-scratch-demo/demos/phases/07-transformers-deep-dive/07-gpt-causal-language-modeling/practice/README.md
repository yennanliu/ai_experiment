<!-- generated:start -->
# 07-transformers-deep-dive / 07-gpt-causal-language-modeling

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/07-transformers-deep-dive/07-gpt-causal-language-modeling/) · upstream spec
`phases/07-transformers-deep-dive/07-gpt-causal-language-modeling/docs/en.md`

```bash
uv run demo practice run 07-gpt-causal-language-modeling --ex 1
uv run demo explain 07-gpt-causal-language-modeling --ex 1
uv run pytest demos/phases/07-transformers-deep-dive/07-gpt-causal-language-modeling
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py` and verify the causal attention matrix is lower-triangular after sof… | code | T0 | `ex01_the_guard_is_for_a_row_the_mask_cannot_make.py` |
| 2 | Medium. Implement beam search for width 4. Compare perplexity of beam-4 vs greedy on 10 short… | code | T0 | `ex02_beam_does_not_always_win_even_on_its_own_score.py` |
| 3 | Hard. Implement speculative decoding: use a tiny 2-layer model as the draft and a 6-layer mod… | code | T0 | `ex03_a_two_layer_draft_against_six_is_break_even.py` |
<!-- generated:end -->

## Answers

The lesson is 178 lines of masking, loss and sampling with **no model** — its
logits come from `rng.gauss`. Exercise 1 asks to verify a property of the mask;
Exercises 2 and 3 both ask to compare decoders, which needs a scorer, so one is
built: a bigram table over the lesson's own 20-token vocabulary, through its own
`softmax`. All three are **T0**.

### 1 — the guard is for a row this mask cannot make

**ANSWER: verified exactly where exactness is available.** Row 3 is
`[0.269, 0.232, 0.439, 0.060, 0.0, 0.0]` — weight in columns 0–3 and nowhere
else. Every upper-triangle entry is the float `0.0`, not an underflow, because
`apply_softmax_row` *writes* `0.0` for a `-inf` entry instead of exponentiating
it. The row sums are within **1.1e-16** of 1, which is as exact as adding `i+1`
floats gets.

**FINDING: row 0 is exactly one-hot, entropy exactly 0.** Position 0 can attend
only to itself. The first token of a causal sequence takes nothing from the
sequence — which is why a prompt is not optional, and why row 0 is fixed before
any weight is learned.

**FINDING: the module's two softmaxes agree to `0.0` on every causal row.**
`math.exp(-inf)` is already `0.0`, so the ordinary `softmax` handles a masked row
correctly. They differ on exactly one input: an all-`-inf` row, where `softmax`
computes `-inf − (−inf)` and returns **nan** while `apply_softmax_row` returns
zeros.

**FINDING: `causal_mask` cannot produce that row.** Its diagonal is `0.0` at
every position, so every row keeps a finite entry. The guard is **dead code for
the mask the lesson ships** — it is for *padding* masks, which this lesson does
not have, and the softmax that would actually meet one is the unguarded one.

**CONTROL: what is verified is the mask, not attention.** `demo_causal_mask`
feeds `rng.gauss(0, 1)` straight in — no `Q`, no `K`, no `1/√dk` — so
triangularity would hold for any scores at all.

### 2 — beam does not always win, not even on its own score

**ANSWER: no — and the counterexample is on beam's own objective.** Over **300**
bigram models, beam-4's best hypothesis scores *lower* in total log-probability
than greedy's **2 times (0.67%)**. Greedy's prefix can be pushed out of the beam
by four better-scoring prefixes whose own continuations are worse.

**ANSWER: width is not monotone either.**

| width | 1 (greedy) | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| total log-prob | −22.420 | **−18.674** | −18.769 | −18.684 | −18.684 |

Width 2 beats width 4. Widening the beam changes *which* prefixes survive, and a
surviving prefix is not the same thing as a better one.

**FINDING: 10 prompts cannot find that.** On the exercise's own sample beam wins
8, ties 2, loses 0, mean gain **+1.66 nats**. At a 0.67% failure rate, 10 draws
are expected to see none.

**FINDING: the perplexity the exercise names does not depend on the decoder.**
`cross_entropy_shifted(logits_per_pos, target_ids)` takes a model's logits and a
*reference* sequence — 22.70 here. No decoding strategy appears in the formula
and none can: perplexity is a property of a model on text it did not choose.

**CONTROL: what *can* be compared is circular.** Perplexity of each decoder's own
output under the model that produced it: greedy **4.885**, beam **4.255**. Beam
maximises exactly the reciprocal of that by construction — a definition, not a
result — and it still loses 2 times in 300.

### 3 — a 2-layer draft against a 6-layer verifier is break-even

Speedup is counted in **verifier forward passes**, which is what a wall-clock
number reports once both models sit on the same device. The draft/verifier cost
ratio the exercise fixes is `c = 2/6 = 1/3`.

**ANSWER: outputs match verifier greedy 100/100, at every agreement level**
(α = 0.761, 0.531, 0.299). By construction, not luck: the loop accepts the
longest prefix where the draft's argmax equals the verifier's, then emits the
verifier's own next token. **Correctness does not depend on draft quality.**

**ANSWER: speed does.** With `E[tokens per call] = (1 − α^(g+1))/(1 − α)` and
speedup `E / (1 + g·c)`:

| α | g=1 | g=2 | g=4 | g=8 | best |
|---:|---:|---:|---:|---:|---:|
| 0.761 | 1.32× | **1.40×** | 1.34× | 1.04× | 1.40× |
| 0.531 | 1.15× | 1.09× | 0.88× | 0.58× | 1.15× |
| 0.299 | 0.97× | 0.83× | 0.61× | 0.39× | **0.97×** |

At the lowest acceptance rate **no lookahead reaches 1.0** — the configuration
the exercise names can lose.

**FINDING: the cost ratio caps it before the acceptance rate does.** A 4-token
lookahead at `c = 1/3` spends **1.33 verifier equivalents** drafting before one
token is verified, so even perfect acceptance caps the speedup at
`(g+1)/(1 + g·c)` = **2.14×**. The same α = 0.761 at `c = 1/20` gives **2.60×**
at `g = 4` and 2.73× at `g = 8`. Production drafts are 10–20× cheaper than the
verifier; 3× is not enough, and 2-of-6 layers is the number that makes the
exercise's own configuration marginal.

**CONTROL: a bigger lookahead is not a bigger win.** `E[tokens]` saturates at
`1/(1 − α) = 4.18` while drafting cost grows linearly in `g`, so at `c = 1/3` the
optimum is `g = 2` and every larger lookahead is worse.
