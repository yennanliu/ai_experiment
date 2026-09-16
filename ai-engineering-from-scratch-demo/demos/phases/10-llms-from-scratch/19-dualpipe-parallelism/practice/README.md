<!-- generated:start -->
# 10-llms-from-scratch / 19-dualpipe-parallelism

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/19-dualpipe-parallelism/) · upstream spec
`phases/10-llms-from-scratch/19-dualpipe-parallelism/docs/en.md`

```bash
uv run demo practice run 19-dualpipe-parallelism --ex 1
uv run demo explain 19-dualpipe-parallelism --ex 1
uv run pytest demos/phases/10-llms-from-scratch/19-dualpipe-parallelism
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py` on `(P=8, micro_batches=16, schedule=dualpipe)` and `(P=8, micro_batches=1… | code | T0 | `ex01_the_two_fractions_have_different_denominators.py` |
| 2 | Sketch the schedule table for `(P=4, micro_batches=8, schedule=dualpipe)` by hand. Mark each… | code | T0 | `ex02_the_1f1b_formula_checks_out_and_dualpipes_has_no_schedule.py` |
| 3 | Read Figure 5 of the DeepSeek-V3 technical report (arXiv:2412.19437). Identify the overlap wi… | code | T0 | `ex03_the_overlap_is_a_string.py` |
| 4 | Compute the 2x parameter overhead of DualPipe for a 70B dense model with P=8 pipeline stages… | code | T0 | `ex04_the_moe_overhead_is_larger_until_expert_parallelism_arrives.py` |
| 5 | Compare DualPipe to Chimera (a competing bidirectional scheduler from 2021). Identify the two… | code | T0 | `ex05_the_model_records_one_of_the_two_properties.py` |
<!-- generated:end -->

## Answers

`code/main.py` is 145 lines of closed-form arithmetic: four bubble-fraction
formulas, a summary table and a GPU-hour conversion. There is no schedule, no
communication model and no parameter count anywhere in it, and the five
exercises ask for all three.

All five are **T0** on **no** dependency group (stdlib only).

> Exercises 3 and 5 are scaffolded as prose items. They are built here as
> **audits of what the model can represent**, because both ask about quantities
> — communication overlap, and the properties that separate two schedulers —
> that have to exist in the code for the question to have a checkable answer.

### 1 — the two fractions have different denominators

| schedule | bubble | numerator / denominator |
|---|---:|---|
| 1F1B | 30.43% | 14 / 46 **chunks** |
| Zero Bubble | 11.29% | 7 / 62 **sub-chunks** |
| DualPipe | 5.45% | 3 / 55 **sub-chunks** |

**ANSWER: 24.98 points, 249,802 GPU-hours per million.**

**FINDING: the denominators are different units.** `bubble_zero_bubble`'s own
docstring says its total counts B/W sub-chunks; `bubble_1f1b`'s counts whole
forward-or-backward chunks. `gpu_hours_recovered` subtracts one whole's fraction
from another's and multiplies by wall-clock hours.

**FINDING: "per million tokens" is not computable here.** No function takes a
token count, a batch size or a sequence length.

**FINDING: every schedule's bubble shrinks to zero as M grows** — 63.6% → 0.17%
for 1F1B. Every numerator is constant in P; every denominator is linear in M.

### 2 — the 1F1B formula checks out; DualPipe's has no schedule

```text
r0   .  .  . F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7  .  .  .
r1   .  . F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 B6 B7  .  .  .  .
r2   . F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 B7  .  .  .  .  .
r3  F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7  .  .  .  .  .  .
              ↑ t=3, the first slot with every rank busy (= P−1)
```

**ANSWER: t = 3.** Every rank is busy from t=3 to t=15 — 13 of 22 slots, 59.1%.

**FINDING: the 1F1B formula is exact to twelve decimal places** on all six shapes
tried. `2(P−1) / (2M + 2(P−1))` is a count of that table.

**FINDING: nothing lets the same check run on DualPipe.** Its bubble is asserted
by a docstring, not derived from a schedule.

**MECHANISM: `(P−1)//2` makes the bubble a step function of P** — exactly
**0.000%** at P=2, and identical numerators at P=3 and P=4.

### 3 — the overlap is a string

**ANSWER: communication appears once, as `minimal` / `partial` / `full`.** It is
printed and nothing consumes it.

**MECHANISM: every bubble function takes exactly `(P, M)`.** A schedule that
overlaps all-to-all perfectly and one that overlaps none of it return the same
number.

**FINDING: `ScheduleStats` is declared and never constructed.** `summarize`
returns bare tuples, so `scales_with_micro_batches` is assigned nowhere.

**FINDING: DualPipeV's cost is a stipulated constant** — `1.2 ×` DualPipe's, to
machine precision, at every shape. Its docstring says so.

### 4 — the MoE overhead is larger until expert parallelism arrives

| model | params | P | one stage | DualPipe's extra copy |
|---|---:|---:|---:|---:|
| 70B dense | 70B | 8 | 8.75B | **8.75B** |
| 671B MoE | 671B | 16 | 41.94B | **41.94B** |

**ANSWER: 4.8× *larger*, not smaller** — on the two configurations the exercise
names.

**FINDING: expert parallelism is what makes the claim true, and it is not in the
model.** Experts are **656B of 671B (97.8%)**:

| EP | MoE stage | as a share of the dense stage |
|---:|---:|---:|
| 1 | 41.94B | 479% |
| 8 | 6.04B | 69% |
| 64 | **1.55B** | **18%** |

**MECHANISM: DualPipe replicates the stage; EP shrinks the stage.** Nothing
about DualPipe is cheaper on an MoE — the MoE *stage* is cheaper.

**FINDING: no function in the lesson takes a parameter count.**

### 5 — the model records one of the two properties

| schedule | bubble | param_copies | comm_overlap |
|---|---:|---:|---|
| 1F1B | 30.43% | 1 | minimal |
| Zero Bubble | 11.29% | 1 | partial |
| **DualPipe** | 5.45% | **2** | **full** |
| DualPipeV | 6.55% | 1 | partial |

**ANSWER: `param_copies = 2` is Chimera's own 2021 trick**, so it cannot be a
property DualPipe added. `comm_overlap = "full"` is the only field that separates
DualPipe from the rest — and it is a string. The fine-grained chunk split that
makes the overlap possible has no field at all.

**FINDING: DualPipeV is the honest test and it fails it.** It drops the second
copy and keeps `partial` overlap at 1.2× the bubble — so the table says a
Chimera-memory schedule loses a fifth of DualPipe's advantage.

**MECHANISM: DualPipe and DualPipeV share a formula** (3/55 both) up to the
stipulated 1.2.

**FINDING: `scales_with_micro_batches` is assigned nowhere** — and as Exercise 1
measures, it would read the same for all four anyway.
