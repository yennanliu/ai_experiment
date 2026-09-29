<!-- generated:start -->
# 19-capstone-projects / 35-gpt-model-assembly

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/35-gpt-model-assembly/) · upstream spec
`phases/19-capstone-projects/35-gpt-model-assembly/docs/en.md`

```bash
uv run demo practice run 35-gpt-model-assembly --ex 1
uv run demo explain 35-gpt-model-assembly --ex 1
uv run pytest demos/phases/19-capstone-projects/35-gpt-model-assembly
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Untie the LM head from the token embedding and recount parameters. Verify the delta is 50257… | code | T1 | `ex01_untying_adds_38_597_376_and_the_lessons_dedup_counter_is_a_no_op_but_state_dict_double_counts.py` |
| 2 | Replace the learned position embedding with a sinusoidal table computed at construction time.… | code | T1 | `ex02_the_sinusoidal_swap_drops_786_432_but_its_table_is_35x_the_token_embedding_and_drowns_the_token.py` |
| 3 | Add a `greedy=True` flag to generation that skips sampling and picks argmax. Confirm the sequ… | code | T1 | `ex03_greedy_repeats_across_runs_and_processes_and_top_k_1_already_was_greedy_while_the_tied_model_copies_97pct.py` |
| 4 | Add a `repetition_penalty` knob that divides the logit of any token in the prompt or generate… | code | T1 | `ex04_a_penalty_of_5_cuts_greedy_repeats_40_to_0_but_dividing_raises_the_logit_of_44pct_of_history_tokens.py` |
| 5 | Add `top_p` (nucleus) sampling next to `top_k`. Two-line check that the sum of probabilities… | code | T1 | `ex05_the_nucleus_mass_check_holds_at_0_9_but_exceeds_is_the_wrong_test_and_fails_at_top_p_1.py` |
<!-- generated:end -->

## Answers

Every exercise builds the lesson's own `GPTModel` from `code/main.py`. Exercises
1 and 2 use the 124M reference configuration; exercises 3 to 5 use the
lesson's tiny demo configuration (vocab 512, context 64, d_model 64, 4 heads,
2 layers), always built at seed 0 and never trained. External sources were
read on 2026-09-29: arXiv:1706.03762 (section 3.4), arXiv:1904.09751
(section 3.1) and Hugging Face's `logits_process.py`.

### 1 — untying adds 38,597,376, the lesson's dedup counter is a no-op, and state_dict double-counts

**The delta is exactly 38,597,376 = 50257 x 768.** The tied model has
124,439,808 parameters and the untied one 163,037,184. The shared matrix is
31.0% of the tied model. As a rounded number it is 38.6M, so "38 million"
is a truncation.

`count_parameters` deduplicates by `id(param)`, but `model.parameters()`
already yields a shared tensor once, so a plain sum gives the same
124,439,808. The count goes wrong on paths the lesson never mentions:

| tied model, counted by | parameters |
|---|---:|
| lesson's `count_parameters` | 124,439,808 |
| `sum(p.numel() for p in parameters())` | 124,439,808 |
| `named_parameters(remove_duplicate=False)` | 163,037,184 |
| `state_dict()` | 163,037,184 |

The lesson's arithmetic is also off: a block has 7,087,872 parameters, not
"roughly 7 million", so twelve make 85.1M rather than 84M. The lesson's own
rounded pieces sum to 122.8M, not 124M.

### 2 — the sinusoidal swap drops 786,432, but its table is 35x the token embedding and drowns the token

**It forwards, and the count drops by exactly 786,432 = 1024 x 768** (from
124,439,808 to 123,653,376). The table is a buffer computed in `__init__`,
put in `pos_embed`'s place. The obvious `nn.Embedding.from_pretrained(table,
freeze=True)` drops **0**, because a frozen weight is still an
`nn.Parameter` and the counter ignores `requires_grad`.

The swap also changes what the model sees. The lesson draws token rows at std
0.02 (norm 0.554), while every sinusoidal row has norm sqrt(384) = 19.60,
which is 35x larger. The original Transformer multiplies embeddings by
sqrt(d_model); this lesson does not.

| last token swapped, cosine of last-position logits | learned positions | sinusoidal |
|---|---:|---:|
| mean over 16 sequences | 0.754 | 0.999 |

### 3 — greedy repeats across runs and processes, top_k=1 already was greedy, and the tied model copies 97%

**Greedy decoding is deterministic.** Three in-process runs with the global
RNG reseeded, and two fresh Python processes, all give the same 45 tokens.
Unseeded sampling at the demo's settings does not.

The flag was already there in another form. The reference `generate` with
`top_k=1` matches greedy on 20/20 seeds, and so does temperature 0.01. At
temperature 0.1 none match. `temperature=0` raises ValueError, although the
lesson says near-zero temperature "collapses to greedy".

What greedy produces is the more telling result: `[1, 2, 3, 4, 5]` followed
by `5` forty times. With tied weights, an untrained model scores the current
token's own embedding row highest. Over 200 random prompts greedy copies the
last token 97.0% of the time; untied, 1.5%.

### 4 — a penalty of 5 cuts greedy repeats from 40 to 0, but dividing raises the logit of 44% of history tokens

**Yes: values above one reduce repeats**, on the demo prompt, greedy, 40 new
tokens:

| penalty | 1.0 | 1.2 | 1.5 | 2.0 | 5.0 |
|---|---:|---:|---:|---:|---:|
| repeated tokens / 40 | 40 | 39 | 37 | 15 | 0 |

Sampling 100 sequences at temperature 0.1 gives a mean of 26.7 repeats at
penalty 1.0 and 2.1 at 2.0.

The rule as the exercise words it has a sign bug. Dividing a negative logit
by a constant above one moves it toward zero, which raises it: 44% of the
penalised history logits went up. Hugging Face's
`RepetitionPenaltyLogitsProcessor` multiplies negative scores instead
(`torch.where(score < 0, score * penalty, score / penalty)`). Under that rule
0% rise, and the same batch repeats slightly less (2.00 against 2.06).

### 5 — the nucleus mass check holds at 0.9, but "exceeds" is the wrong test and fails at top_p = 1

**The two-line check passes on all 200 next-token distributions at top_p 0.5,
0.9 and 0.95.** The nucleus keeps 224, 446 and 477 of 512 tokens on average.
Generation chains `top_k` and then `top_p`.

Nucleus sampling is defined with ">= p" (Holtzman et al., eq. 2), and the
kept mass can equal p. Two equal logits at top_p 0.5 keep one token of mass
exactly 0.5, which fails a strict "exceeds". At top_p 1.0 every token is
kept, but in float64 their probabilities sum to just under 1, and the strict
check fails on 130 of 200 prompts on the machine this was measured on (the
exact count depends on the platform's rounding).

On this near-uniform model the knobs do not compare: top_k = 20 keeps 5.5% of
the mass, top_p = 0.9 keeps 446 tokens, and chained they keep 18. The
lesson's `top_k_filter` also keeps every tie: top_k = 2 over four equal
logits keeps 4, although its own test is called
`test_top_k_filter_keeps_exactly_k`.
