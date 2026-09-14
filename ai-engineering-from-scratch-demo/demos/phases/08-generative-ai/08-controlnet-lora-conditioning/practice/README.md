<!-- generated:start -->
# 08-generative-ai / 08-controlnet-lora-conditioning

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/08-generative-ai/08-controlnet-lora-conditioning/) · upstream spec
`phases/08-generative-ai/08-controlnet-lora-conditioning/docs/en.md`

```bash
uv run demo practice run 08-controlnet-lora-conditioning --ex 1
uv run demo explain 08-controlnet-lora-conditioning --ex 1
uv run pytest demos/phases/08-generative-ai/08-controlnet-lora-conditioning
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. In `code/main.py`, vary the LoRA rank `r` from 1 to 4. At what rank does the LoRA exact… | code | T0 | `ex01_rank_two_and_the_r_one_row_is_eckart_young.py` |
| 2 | Medium. Train two separate LoRAs on two target transforms. Load them together and show their… | code | T0 | `ex02_it_never_breaks_because_there_is_nothing_to_break.py` |
| 3 | Hard. Use diffusers to stack: SDXL-base + Canny-ControlNet (weight 0.8) + a style LoRA (α 0.8… | code | T0 | `ex03_the_stack_is_two_knobs_on_one_line.py` |
<!-- generated:end -->

## Answers

The lesson is 111 lines: a LoRA fitted to a known weight delta, and a
zero-initialised gate on a side signal. Exercises 1 and 2 are **T0**; exercise 3
names a four-part SDXL stack that cannot be built here, so it ships the
scaled-down runnable `DESIGN D11` requires.

The module contains **no activation function** — `matmul_mat_vec`, `outer`,
`zeros` and `randn_matrix` are all linear — and that fact decides exercises 2
and 3.

### 1 — rank 2, and the r=1 row is Eckart–Young

`main()` builds a **rank-1** delta, so a rank-2 target has to be constructed. It
is assembled from orthonormal pairs with singular values chosen as **3.0** and
**2.0**, which makes the answer known before the sweep runs.

| r | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| residual mse | 4.053 | **4.8e-31** | 4.2e-31 | 5.4e-31 |

**ANSWER: r = 2, exactly.** Ranks 3 and 4 buy nothing — there is nothing left to
fit.

**FINDING: the r=1 row is the optimum for its rank, not a failure.** Eckart–Young
says the best rank-1 approximation leaves exactly `2.0² = 4.0`; the trainer
reaches **4.053**, within **1.3%** of a bound it cannot beat. The sweep measures a
theorem.

**FINDING: readability depends on the delta's size.** The same `train_lora` at the
same `lr=0.01`, given singular values 8.96 and 4.99, returns **nan** at r=1 rather
than that delta's bound of 24.9. `lr=0.001` recovers 23.5.

### 2 — it never breaks, because there is nothing to break

**ANSWER: additive to 8.9e-16.** The two adapters are merged by concatenating
their factors and pushed through the lesson's own `lora_forward` in one call;
that agrees with the sum of the separate contributions to machine precision.

**FINDING: a property of this code, not a result about LoRA.**
`(W + α·BA)x` is affine in the adapters, so stacking is the distributive law.
The module exposes **none** of `tanh`, `leaky`, `sigmoid`, `relu` — there is no
nonlinearity for linearity to break through. In a real transformer the adapters
sit either side of softmax and a nonlinear FFN, which is where the interaction
the exercise asks about actually lives.

**CONTROL:** each adapter alone matches its own rank-2 target to **1.4e-13** and
**7.8e-10**, so this is additivity between two adapters that work.

### 3 — the trade-off curve is bowed, and no nonlinearity is needed for that

`diffusers`, `torch`, `transformers` are all absent, so neither the pipeline nor
FID's Inception network can be built. What runs is the structure: two weighted
adapters on one frozen layer, two competing objectives, swept across the weights.

**ANSWER: a real trade-off** — style adherence **−1.315 → 0.000** against control
adherence **0.000 → −1.315** — with the front **0.465** off the chord.

**FINDING: the bow needs no nonlinearity, only a squared error.** The stack is
exactly linear in its weights, which makes a straight front the natural guess —
and it is wrong. Adherence is a *squared* error, so each objective is quadratic in
the weights even though the model is not. A bowed FID-versus-adherence curve is
evidence about the **metric**, not about interaction inside the network.

**FINDING: the curve is exactly a conic** — an exact quadratic fit in the sweep
parameter leaves a worst residual of **2.2e-16**.

**FINDING: orthogonal targets bow it *more*, not less** — 0.465 → **0.760**.
Overlap is not the source of the tension; one layer satisfying two unrelated
targets is harder than two related ones, not easier.
