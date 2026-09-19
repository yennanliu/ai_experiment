<!-- generated:start -->
# 12-multimodal-ai / 07-open-weight-vlm-recipes

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/07-open-weight-vlm-recipes/) · upstream spec
`phases/12-multimodal-ai/07-open-weight-vlm-recipes/docs/en.md`

```bash
uv run demo practice run 07-open-weight-vlm-recipes --ex 1
uv run demo explain 07-open-weight-vlm-recipes --ex 1
uv run pytest demos/phases/12-multimodal-ai/07-open-weight-vlm-recipes
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Read MM1 Section 3.2. For a fixed 2B LLM at budget 50M images, which encoder wins? Would the… | code | T0 | `ex01_the_table_has_no_llm_axis_to_flip_along.py` |
| 2 | Cambrian-1 finds that concatenating DINOv2 + SigLIP outperforms either alone on vision-centri… | code | T0 | `ex02_the_concat_beats_the_mean_everywhere_and_the_max_twice.py` |
| 3 | Your target is a mobile UI agent on a 2B LLM. Pick encoder, connector, resolution, and data m… | code | T0 | `ex03_the_picker_returns_nothing_and_the_weights_change_nothing.py` |
| 4 | Molmo ships 4B and 72B models. The 4B is competitive with closed 7B VLMs; the 72B beats Llama… | code | T0 | `ex04_the_plateau_is_a_property_of_the_benchmark.py` |
| 5 | Design an ablation table to isolate data-mix quality from encoder quality on a 7B VLM. How ma… | code | T0 | `ex05_four_runs_and_the_lesson_ran_nine_of_the_wrong_ones.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib, and all five read the lesson's own tables — parsed
back out of `compare_encoders`, `compare_data`, `axis_impact` and `RECIPES`
rather than retyped, so a change upstream shows up as a failing check rather
than as a stale number here.

The through-line is that this lesson ships an *evidence base* and four of the
five exercises ask a question that evidence base cannot answer: it has no LLM
axis (1), no cell off both axes (5), no recipe at the requested size (3), and no
point past the plateau it hypothesises (4). Exercise 2 is the exception — its
answer is sitting in the table, and it contradicts the exercise's premise.

### 1 — the table has no LLM axis to flip along

**ANSWER: no encoder wins, and there is no LLM axis to flip along.**
`compare_encoders` is fixed at one 7B LLM and names no image budget, so neither
2B nor 13B appears in it. Within the one setting it does report:

| benchmark | winner |
|---|---|
| MMMU | InternViT-6B @ 448 |
| CV-Bench | SigLIP + DINOv2 concat |
| DocVQA | InternViT-6B @ 448 |

Five encoders, three answers.

**FINDING: encoder choice matters 4.3× more for DocVQA than for MMMU.** The same
five span **6.0** points of MMMU, 11.0 of CV-Bench and **26.0** of DocVQA. A
single "image encoder: 20% of variance" figure is task-blind by construction.

**FINDING: the axis decomposition sums to 115%.** The weights are
`[60, 20, 5, 10, 15, 5]`. `axis_impact` computes the total, prints a note saying
they are "rebased from ~115% to ~100% after rounding", and then prints the
unrebased numbers. The rebasing is described and not performed.

**FINDING: an additive decomposition cannot hold the question.** "Would the
answer flip at 13B?" *is* an encoder × LLM-size interaction, and the model has
six additive terms with nothing that multiplies two of them.

### 2 — the concat beats the mean everywhere and the max twice

| | MMMU | CV-Bench | DocVQA |
|---|---:|---:|---:|
| SigLIP SO400m/14 @ 384 | 41.0 | 60.0 | **75.0** |
| DINOv2 ViT-g/14 @ 224 | 37.0 | 65.0 | 52.0 |
| SigLIP + DINOv2 concat | **42.0** | **67.0** | 74.0 |
| **vs. the better part** | **+1.0** | **+2.0** | **−1.0** |
| **vs. the mean of the parts** | +3.0 | +4.5 | **+10.5** |

**ANSWER: CV-Bench gains, MMMU gains slightly, DocVQA loses.** The exercise's
premise is that the concat "adds no signal on MMMU"; the lesson's own table says
**+1.0**, and the benchmark that actually goes backwards is DocVQA.

**FINDING: against the mean of its parts the concat wins everywhere, by up to
ten times the margin it has against the max.** Concatenating two encoders
reliably beats averaging them. Beating the *better* one is a different question,
and it is the only one that matters when you have to pick.

**FINDING: the DocVQA gap is where the two encoders disagree most** — 23.0
points, against 4.0 and 5.0. That is exactly where the concat fails to beat the
better part: adding a weak encoder's features costs something, and it costs most
where it is weakest.

**FINDING: "vision-centric benchmarks gain" restates which encoder is which.**
DINOv2 beats SigLIP on CV-Bench and nowhere else, and CV-Bench is where the
concat beats both parts by most.

### 3 — the picker returns nothing, and the weights change nothing

**ANSWER: `pick_recipe(2, "agent")` returns zero candidates.** The smallest
recipe in the table is MM1-3B, so the picker prints its header and stops.

| budget | 2B | 3B | 7B | 8B | 13B | 30B | 72B |
|---|---:|---:|---:|---:|---:|---:|---:|
| candidates | **0** | 1 | 4 | 5 | 7 | 8 | 9 |

At 3B the pick is forced; the scoring function first has something to choose
between at 7B. **The budget filter, not the score, is the picker.**

**FINDING: the four task profiles produce identical rankings.** At budget 10 and
at 80, `balanced` / `ocr` / `agent` / `reasoning` all return the same top-3 in
the same order. At 80B Molmo-72B is the maximum on all three benchmarks at once,
so no weighting can move it. At 10B nothing dominates, and the ranking holds
anyway because the benchmarks have unequal spreads — 6.7, 9.8 and **30.4**
points — so a 1.2-against-0.8 weight cannot overcome a 30-point DocVQA gap. The
narrowest margin across the four profiles is **10.9**.

**ANSWER: so the choice has to be argued from the tables by hand.**

| axis | choice | the table |
|---|---|---|
| encoder | SigLIP SO400m/14 | `compare_encoders`: +2.5 MMMU, +5.0 DocVQA over CLIP at equal token count |
| resolution | AnyRes 672 | `axis_impact`: visual-token count is 60%, the largest axis; UI screenshots are the OCR case |
| data | PixMo-style dense human captions | `compare_data`: 40.0 → 45.3 MMMU |
| connector | MLP-2 | `axis_impact`: 5%, the smallest axis — **the justification is a table saying the question does not matter** |

The LLM is the axis with no answer: at 2B there is no row.

### 4 — the plateau is a property of the benchmark

| family | LLM step | ΔMMMU | ΔCV-Bench | ΔDocVQA |
|---|---|---:|---:|---:|
| Molmo 7B → 72B | 1.01 decades | **+8.8** | +8.0 | **+1.1** |
| MM1 3B → 30B | 1.00 decades | **+6.1** | +5.0 | +12.0 |

**ANSWER: the plateau is not visible, and it is not one number.** About +7 MMMU
per decade of LLM, with no sign of flattening — and on the *same two Molmo
checkpoints*, DocVQA moves **8.0× less**.

**FINDING: DocVQA has plateaued at 7B and MMMU has not by 72B.** Molmo goes
92.4 → 93.5 on DocVQA with 6.5 points of headroom left, and 45.3 → 54.1 on MMMU
with 45.9 still above it.

**FINDING: the lesson's own plateau figure is its own largest model.**
`axis_impact` says "7B → 70B plateau around MMMU 55"; the highest MMMU anywhere
in the table is **54.1**. The hypothesis is stated at a point the evidence
reaches and does not pass, so nothing here can falsify it.

**FINDING: the data axis is worth as much as the LLM axis, and is weighted
less.** `compare_data` moves MMMU 40.0 → 47.0 at a fixed encoder and a fixed 7B
LLM — **+7.0**, against the +8.8 a decade of LLM buys — while the decomposition
weights data mix at 10% and LLM size at 15%.

### 5 — four runs, and the lesson ran nine of the wrong ones

**ANSWER: four — a 2 × 2 factorial.** Three runs (a baseline plus one move along
each axis) measure both main effects and *assume* additivity; the fourth cell
turns the interaction from an assumption into a measurement.

**FINDING: the lesson runs 9 configurations and cannot separate anything.**
`compare_encoders` is 5 encoders at one data mix; `compare_data` is 5 mixes at
one encoder. Nine cells laid out as a cross, **zero** of them off both axes at
once. Nine runs buy what three would.

**FINDING: the lesson has three numbers for the same cell.** SigLIP + 7B +
LLaVA-Inst & ShareGPT4V:

| source | MMMU |
|---|---:|
| `compare_encoders` | 41.0 |
| `compare_data` | 42.0 |
| `RECIPES`, "Prismatic-7B default" | 40.0 |

A **2.0**-point spread — **80%** of the +2.5 encoder effect the first table
exists to report — in the baseline the two tables were supposed to share. Their
deltas cannot be added.

**ANSWER: the four settings are {SigLIP SO400m, CLIP L/14} × {LLaVA-Inst-150k,
PixMo}.** The encoder pair is the widest measured gap that does *not* change the
visual token count, so the comparison is not secretly a resolution ablation; the
data pair is the distilled-versus-human-caption contrast that is the whole Molmo
thesis. **2.5** points against **5.3** — close enough in size that an
interaction of either sign would change the conclusion, which is the argument
for spending the fourth run.
