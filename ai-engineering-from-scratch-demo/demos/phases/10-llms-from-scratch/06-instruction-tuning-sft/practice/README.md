<!-- generated:start -->
# 10-llms-from-scratch / 06-instruction-tuning-sft

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/06-instruction-tuning-sft/) · upstream spec
`phases/10-llms-from-scratch/06-instruction-tuning-sft/docs/en.md`

```bash
uv run demo practice run 06-instruction-tuning-sft --ex 1
uv run demo explain 06-instruction-tuning-sft --ex 1
uv run pytest demos/phases/10-llms-from-scratch/06-instruction-tuning-sft
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add system prompt support. Modify `tokenize_instruction_pair` to accept a system message and… | code | T1 | `ex01_the_system_prompt_pushes_the_answer_out.py` |
| 2 | Implement data mixing. Create a function that takes an SFT dataset and a raw text corpus, the… | code | T1 | `ex02_five_percent_of_examples_is_not_five_percent.py` |
| 3 | Build a data quality scorer. For each instruction-response pair, compute: (a) response length… | code | T1 | `ex03_the_two_filters_point_in_opposite_directions.py` |
| 4 | Implement multi-turn conversation training. Extend the tokenization to handle 3-turn conversa… | code | T1 | `ex04_the_mask_never_closes.py` |
| 5 | Compare learning rates. Train the same model three times with lr=1e-4, lr=2e-5, and lr=1e-6.… | code | T1 | `ex05_the_learning_rate_does_nothing.py` |
<!-- generated:end -->

## Answers

`code/main.py` imports lesson 04's `MiniGPT` and wraps it in an SFT pipeline:
instruction/response tokenisation, a loss mask that covers only the response,
masked cross-entropy, a training loop, generation, and a forgetting metric. Four
of the five exercises end in "compare the loss" or "compare the metric" — and
none of those comparisons can return anything, because of the four lines in the
middle of the training loop.

All five are **T1** on the `math` group (`uv sync --extra math`).

### The fact the other four exercises rest on

`sft_train` builds `dlogits`, masks it with the loss mask, divides by the
response-token count — and then updates:

```python
block.ffn.W1 -= lr * np.random.randn(*block.ffn.W1.shape) * 0.01
```

`dlogits` appears nowhere on the right-hand side. **The update is Gaussian noise
scaled by the learning rate.** The proof is one line: run `sft_train` on
`INSTRUCTION_DATA` and on eight examples of `"zzzz"`/`"qqqq"` from the same seed,
and every weight matches **bit for bit**. A trainer whose output does not depend
on its input is not training. Only `ffn.W1` and `ffn.W2` move at all; `W_q`,
`W_out`, `ln1`, `ln2`, `token_embed` and `pos_embed` are frozen exactly as they
are in lesson 04.

### 1 — the system prompt evicts the response

| | token lengths (8 examples) | over `seq_len=64` | response tokens supervised |
|---|---|---:|---:|
| no system prompt | 64, 106, 111, 56, 97, 112, 61, 117 | **5 / 8** | 245 of 468 |
| `"You are a poet. "` | 80, 122, 127, 72, 113, 128, 77, 133 | **8 / 8** | **128** |

**FINDING: five examples already overflow before a system prompt exists.**
`sft_train` does `tokens = tokens[:seq_len]` and the response is at the end.

**ANSWER: one 16-byte prompt pushes all eight over**, and supervised response
tokens fall 245 → 128 — **48% less supervision** for the same eight examples.

**MECHANISM: the two ends belong to different owners.** The system prompt is
masked out (no gradient, costs budget); the response is masked in (the only
thing that contributes). Prepending to a fixed window spends the response's
budget on tokens that cannot be learned from.

**FINDING: the verification the exercise asks for cannot be performed.** The
five prompts do produce five distinct token sequences — but see above.

### 2 — 5% of examples is 9.5% of trained tokens

The mix is built at exactly 5% of examples — the eight pairs repeated 19 times,
so that 8 raw examples is 5.0% of 160 — and the share that matters is counted on
the batches `sft_train` receives, after its own `seq_len=64` truncation and
shift.

| | supervised | total | share |
|---|---:|---:|---:|
| one raw example | 61 | 63 | 96.8% |
| eight pairs, untruncated | 468 | 724 | 64.6% |
| eight pairs, as the trainer sees them | **245** | **493** | **49.7%** |

**ANSWER: 5% of examples is 9.5% of trained tokens** — `8 × 61 / (8 × 61 + 19 ×
245)`, nearly double the ratio that was set.

**FINDING: the trainer's truncation removes 48% of the supervised signal.**
`sft_train` cuts each example to 64 tokens, and it cuts from the end, which is
where the response is. Half the tokens SFT exists to train on never reach the
loss, and the cut happens inside the trainer where the dataset cannot show it.

**MECHANISM: the ratio is set on examples and the effect lands on tokens.**
Masking is the point of SFT and it is also what makes the two example types
incomparable units. The gap widens as the supervised share falls — which is
exactly what the truncation does to it.

**FINDING: both arms move in the sixth decimal.** Three epochs move the metric
from 5.524290 to 5.524293 on pure SFT and to 5.524272 on the mix. The update is
`lr * np.random.randn(...)` and the gradient is discarded, so the two arms are
the same random walk. And the held-out loss sits at 5.5243 against `ln(256) =
5.5452` — the model is at the uniform baseline before training and stays there.

### 3 — the two filters point in opposite directions

| response | bytes | ratio | byte diversity | verdict |
|---|---:|---:|---:|---|
| "15 multiplied by 7 is 105." | 26 | 1.04 | 0.654 | keep |
| "World War II ended in 1945." | 27 | 1.15 | 0.593 | keep |
| "The capital of France is Paris." | 31 | 0.97 | 0.581 | keep |
| "…Python, Rust, and TypeScript." | 61 | 0.54 | 0.393 | keep |
| "Gravity is the force that…" | 71 | 0.45 | 0.282 | **DROP** |
| "Waves crash on the shore…" | 78 | 0.38 | 0.282 | **DROP** |
| "Photosynthesis converts sunlight…" | 84 | 0.30 | 0.274 | **DROP** |
| "Machine learning is a field where…" | 90 | 0.27 | 0.233 | **DROP** |

**ANSWER: the filter drops 4 of the lesson's own 8, and none of them for being
short.** Every response is ≥26 bytes, so `length < 10` fires zero times.

**FINDING: the survivors are the four shortest and the dropped are the four
longest.** Byte diversity is `unique/total` over the ~40 characters English
actually uses, so it falls as a response grows: the correlation between length
and diversity here is **−0.99**. The length filter removes short answers; the
diversity filter, standing beside it, removes long ones.

**MECHANISM: 0.3 is a word-level threshold on a byte-level tokenizer.** On word
tokens the same eight score **0.909 to 1.000** — all far above 0.3. On bytes
they score 0.233 to 0.654. On words the filter passes everything; on bytes it
halves the dataset. The number is not wrong, it is in the wrong units, and the
exercise says only "tokens".

**FINDING: the "final loss" difference is which example was measured last** —
5.5011 vs 5.5042, a gap of 3.1e-03 against a within-run spread of 1.7e-02.

### 4 — the mask never closes

```text
tokens  253  65 254 255  66 | 253  67 254 255  68 | 253  69 254 255  70
mask      0   0   0   0   1 |   1   1   1   0   1 |   1   1   1   0   1
                          B         C                     E
```

**ANSWER: all three assistant turns are covered** — and so is everything else
from the first response onward. Of the 9 masked-in positions, 3 are the
assistant bytes the exercise wants, **2 are the user bytes C and E**, and 4 are
turn markers.

**MECHANISM: `in_response` is a latch.** `create_loss_mask` sets it `True` at
the first `RESP_START` and has no branch that clears it, so after position 4 the
only zeros left are the `continue` on a later `RESP_START`, at positions 8
and 13.

**CONTROL: the single-turn mask is exactly right**, which is why this survives —
a single-turn conversation has nothing after its response, so every check the
lesson runs passes. The objective the latch produces is not a noisier version of
instruction tuning but the opposite one: a model trained to write both halves of
the conversation, the failure response masking exists to prevent.

### 5 — the learning rate does nothing

| lr | first logged loss | last logged loss |
|---|---:|---:|
| 1e-6 | 5.5011 | 5.5011 |
| 2e-5 | 5.5011 | 5.5011 |
| 1e-4 | 5.5011 | 5.5011 |
| 1e-3 | 5.5011 | 5.5011 |

**ANSWER: 5.5011 everywhere**, a spread of 6.2e-05 over a 1000× range of `lr`.
None of the exercise's three predictions occurs — no rapid initial descent, no
overfitting at 1e-4, no sweet spot at 2e-5. "The 1e-6 run should barely move" is
right, for a reason that applies just as well to the other three.

The mechanism, the proof and the frozen-weight list are at the top of this file.
