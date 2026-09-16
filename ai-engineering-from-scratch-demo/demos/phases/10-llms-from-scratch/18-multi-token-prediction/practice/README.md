<!-- generated:start -->
# 10-llms-from-scratch / 18-multi-token-prediction

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/18-multi-token-prediction/) · upstream spec
`phases/10-llms-from-scratch/18-multi-token-prediction/docs/en.md`

```bash
uv run demo practice run 18-multi-token-prediction --ex 1
uv run demo explain 18-multi-token-prediction --ex 1
uv run pytest demos/phases/10-llms-from-scratch/18-multi-token-prediction
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Show the per-depth loss decreases monotonically as the synthetic signal s… | code | T0 | `ex01_the_depth_two_loss_rises_as_the_signal_strengthens.py` |
| 2 | Compute the parameter overhead for a dense 70B model (hidden 8192, 80 layers) with D=1 MTP mo… | code | T0 | `ex02_the_counter_has_no_moe_term.py` |
| 3 | Implement D=2 in the toy: add a second MTP module that takes h^(1) and predicts `t_{i+2}`. Ve… | code | T0 | `ex03_the_last_module_is_never_used.py` |
| 4 | Switch the toy to parallel MTP (Gloeckle-style): add D output heads on top of the main hidden… | code | T0 | `ex04_the_sequential_path_is_teacher_forced.py` |
| 5 | Use the trained MTP module as an EAGLE-style draft: call module k to propose `t_{i+k}` at inf… | code | T0 | `ex05_fifty_percent_is_the_floor_of_the_measurement.py` |
<!-- generated:end -->

## Answers

`code/main.py` is a stdlib toy for DeepSeek-V3's sequential MTP: a module that
folds the previous hidden state with the next token's embedding, a joint loss
over D depths, and a parameter counter. Each of the five exercises finds that
the quantity it asks for is either produced by something other than the
mechanism named, or is the floor of its own metric.

All five are **T0** on **no** dependency group (stdlib only).

### 1 — the depth-2 loss rises as the signal strengthens

| noise | L₁ | L₂ |
|---:|---:|---:|
| 0.50 | 3.600 | 3.187 |
| 0.30 | 3.563 | 3.203 |
| 0.15 | 3.513 | 3.235 |
| 0.05 | 3.489 | 3.287 |
| 0.01 | **3.492** | **3.319** |

`ln(32) = 3.466` is the uniform-guess reference.

**ANSWER: neither depth decreases monotonically, and depth 2 moves the wrong
way.**

**MECHANISM: the backbone carries the current token and the target is the next
one.** `mtp_loss` scores `shared_head_logits(h_i, E)` against `tokens[i+1]`
while `main` builds `h_i = rms_norm(E[tokens[i]] + noise)`. Removing the noise
makes the head converge on `tokens[i]` — the one token that is certainly not the
answer.

**FINDING: shift the backbone by one and depth 1 obeys.** 2.877 → 1.988,
monotone, 1.48 nats below the reference. Depth 2 still rises.

**FINDING: a fixed pattern does make both converge**, to 2.339 and 3.311, on a
sequence with one distinct token in it.

### 2 — the counter has no MoE term

| model | main | per MTP module | overhead |
|---|---:|---:|---:|
| 70B dense (8192/28672) | 78.9B | **1.040B** | **1.3%** |
| DeepSeek-V3 shape | 37.6B | **0.653B** | 1.7% |
| mini GPT (768/3072) | 211.6M | 10.0M | 4.7% |

**ANSWER: 1.04B and 1.3% for the dense 70B — and 0.653B where the exercise
expects 14B**, a factor of **21**. The explanation the exercise supplies is
correct and the counter it supplies cannot express it.

**MECHANISM: `per_mtp` is `h² + 4h² + 3·h·ff` and none of those is an expert
count.** DeepSeek's actual MoE MLP — 256 routed + 1 shared at 44.0M each — gives
a module of **11.58B**, and the 257 experts are **97.8%** of it.

**FINDING: the overhead is small because the backbone is large.** Same module,
mini-GPT shape: 4.7%.

**MECHANISM: the module is one decoder layer plus one projection** — `1.0125`
layers out of 80.

### 3 — the last module is never used

```text
losses                       depth 1   depth 2
as shipped                    3.5129    3.2347
modules[1] replaced entirely  3.5129    3.2347   ← bit-identical
modules[0] replaced           3.5129    3.3433
D=1, two unrelated modules    3.579055 both      ← the module is not in the loss
```

**MECHANISM: the loop scores before it advances.** `mtp_loss` reads the depth-`k`
loss off `h^(k-1)` and applies `modules[k-1]` afterwards, so the last module's
output is never scored.

**FINDING: this is off by one against equations 19-21**, where
`p_{i+k} = OutHead(h_i^(k))` takes the prediction from the depth-`k` module's
*output*.

**FINDING: the accounting charges D modules and the loss reads D−1.** At the D=1
the parameter table uses throughout: 1 charged, 0 used.

### 4 — the sequential path is teacher-forced

**ANSWER: sequential's depth-2 loss is lower in 32 of 40 seeds** (mean gap
+0.155, sd 0.191) — with every parameter in both arms untrained.

**MECHANISM: the sequential path is fed the true next token.**

```python
h_prev = mtp_forward(h_prev, E[tokens[i + k]], modules[k - 1])
#                              ^^^^^^^^^^^^ the label whose loss was just computed
```

That is teacher forcing, which is why an untrained module beats an untrained
head.

**FINDING: the exercise's stated reason is the one thing the toy does not do.**
"Conditions on the intermediate predictions" describes feeding depth 1 its own
output; the code feeds it the label — Lesson 15's exposure bias, with the sign
that flatters the method.

**FINDING: depth 1 is identical in both schemes** to twelve decimal places on
all 40 seeds. The comparison needs D ≥ 2 to exist.

### 5 — fifty percent is the floor of the measurement

| | rate |
|---|---:|
| mean acceptance | **52.2%** (≥50% on 40/40 seeds) |
| depth 1 (draft vs itself) | **100.0%** |
| depth 2 (the only real comparison) | **4.5%** |
| the main model's own next-token accuracy | **3.4%** |
| uniform draw at vocab 32 | 3.1% |

**ANSWER: the exercise's threshold is cleared on every seed by an untrained
module agreeing with a backbone that guesses.**

**MECHANISM: half the comparisons compare a hidden state with itself.** At `k=1`
the draft reads `backbone[i]` and so does the main model, so `1/D` of the
comparisons score 100% before anything is measured. At D=2 the floor is exactly
50% — and 50% is the pass mark.

**FINDING: acceptance against a wrong prediction is not speculative decoding.**
Leviathan acceptance compares *distributions* and corrects the difference; this
compares two argmaxes and reports a match rate.
