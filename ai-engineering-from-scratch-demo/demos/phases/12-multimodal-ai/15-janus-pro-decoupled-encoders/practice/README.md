<!-- generated:start -->
# 12-multimodal-ai / 15-janus-pro-decoupled-encoders

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/15-janus-pro-decoupled-encoders/) · upstream spec
`phases/12-multimodal-ai/15-janus-pro-decoupled-encoders/docs/en.md`

```bash
uv run demo practice run 15-janus-pro-decoupled-encoders --ex 1
uv run demo explain 15-janus-pro-decoupled-encoders --ex 1
uv run pytest demos/phases/12-multimodal-ai/15-janus-pro-decoupled-encoders
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Janus-Pro-7B beats DALL-E 3 on GenEval. Explain why a 7B open model can match a frontier prop… | code | T0 | `ex01_five_of_the_twenty_nine_points_are_the_parameter_count.py` |
| 2 | Implement a router function: given prompt text, classify as `understand` or `generate`. How d… | code | T0 | `ex02_ambiguous_means_both_and_neither_and_the_pipeline_pays_for_both.py` |
| 3 | JanusFlow replaces the VQ path with rectified flow. What does the transformer body now output… | explain | T0 | prose, below |
| 4 | Propose a fourth task the Janus-Pro architecture could handle with one more decoupled encoder… | explain | T0 | prose, below |
| 5 | Read Janus-Pro Section 4.2 on data scaling. Which data stage contributes most to the T2I qual… | code | T0 | `ex05_four_axes_moved_and_the_table_has_no_ablation.py` |
<!-- generated:end -->

## Answers

Three code solutions and two prose ones. The code ones all measure the lesson's
own artefacts — its router against a labelled set, and its two-model table
against the scaling rate Lesson 12.07 established — and two of the three find the
same shape: the table ranks inputs and the router ranks keywords, and neither
tells you about the outcome it is being asked about.

### 1 — five of the 29.8 points are the parameter count

| | Janus | Janus-Pro | gain | relative |
|---|---:|---:|---:|---:|
| MMMU | 30.5 | 60.3 | **+29.8** | **+97.7%** |
| GenEval | 0.61 | 0.80 | +0.19 | +31.1% |

**ANSWER: because generation benchmarks test compliance and understanding
benchmarks test knowledge.** GenEval scores a checklist — counts, colours,
relative positions — which an instruction-following 7B scores as well as a
frontier model. MMMU scores what the *language model* knows, which is where
parameters and pretraining still buy points.

**FINDING: the same step buys 3.1× more in understanding than in generation.**

**FINDING: only 5.1 of the 29.8 MMMU points are the parameter increase.** 1.3B →
7B is 0.731 decades, and at Lesson 12.07's measured **+7 MMMU per decade** that
is **17.2%** of the gain. The other **82.8%** is the data.

**FINDING: GenEval has 0.20 of headroom left against MMMU's 39.7.** A scoreboard
with a model at 80% of its ceiling cannot separate a 7B from a frontier model by
much, whatever either knows — so the asymmetry the exercise asks about is partly a
property of the two benchmarks, not only of the two models.

### 2 — "ambiguous" means both *and* neither, and the pipeline pays for both

**ANSWER: 9 of 14 (64.3%), with the five misroutes in exactly two classes.**

| misroute | route says | why |
|---|---|---|
| "Show me the contents of the **draw**er" | generate | substring |
| "With**draw** the overlay…" | generate | substring |
| "Is this a cat?" | ambiguous | no keyword |
| "A photo of a dog at sunset" | ambiguous | no keyword |
| "Count the cars" | ambiguous | no keyword |

**FINDING: `ambiguous` conflates "both" with "neither".** Of the five prompts it
returns for, **2** score 1–1 and **3** score 0–0. One case wants both encoders;
the other wants a default. `route` returns the same string for them.

**FINDING: and `run_pipeline` runs both encoders for all five.** Its `else`
branch encodes with SigLIP *and* VQ and calls the body twice, so the 60% that
mean "neither" pay double for nothing.

**FINDING: the substring bug is load-bearing.** A word-boundary matcher takes the
router **down**, 64.3% → **50.0%**: "paint" inside "painting" and "what" inside
"Somewhat" are two of its nine correct answers. And it rescues none of the five
misroutes — the "draw" collisions merely become `ambiguous` instead of
`generate`, because without that substring they have no keyword at all. The
matcher is a bad stemmer; the fix is the keyword list.

**ANSWER: so the way to handle ambiguity is to stop producing it accidentally.**
Split the third verdict: `both` for a 1–1 score, `default` for 0–0. Route
`default` on whether an image is attached rather than on the text, and keep the
dual-encoder path for the genuine `both` — 2 prompts here rather than 5, a
**60%** cut in double-encoded requests.

### 3 — what JanusFlow changes in the body and the loss

Drawing on **JanusFlow — the rectified flow variant**.

**The body stops emitting a distribution and starts emitting a vector.** Two
changes, and the second follows from the first:

| | Janus-Pro | JanusFlow |
|---|---|---|
| generation head output | a softmax over the **VQ codebook** (K-wide) | a **velocity vector** per latent position (d-wide) |
| generation loss | cross-entropy on discrete indices | **MSE on the velocity** — flow matching |
| sampling | masked/autoregressive token emission | an **ODE solve** over the latent |

Concretely, the body now predicts `v = z₁ − z₀` at a sampled time `t` given
`z_t = (1−t)z₀ + t·z₁` — which is exactly the three lines Lesson 12.13's
`two_loss_step` already implements, arrived at from the other side of the phase.

**Three consequences worth naming.**

1. **The tokenizer ceiling goes away, and a different one arrives.** Lesson
   12.14's exercise 4 shows the VQ path's reconstruction is essentially
   independent of codebook size at 768 dimensions. Flow matching removes that
   bound entirely — and replaces it with the VAE's, which is a continuous
   bottleneck rather than a discrete one and is where the remaining loss lives.
2. **The two losses stop sharing a unit.** Cross-entropy is in nats and MSE is in
   squared latent units, so the weighting question that Lesson 12.13's exercise 1
   measures — magnitude balance against contribution balance, 0.1 against 0.2333 —
   arrives here as a new hyperparameter that Janus-Pro did not have.
3. **The decoupling survives intact.** JanusFlow keeps SigLIP for understanding;
   only the generation tower and head change. That is the point of the
   architecture: the reason for two encoders was that understanding wants
   semantic features and generation wants reconstructive ones, and swapping what
   "reconstructive" means does not touch the other half.

### 4 — a fourth decoupled encoder

Drawing on **Decoupled visual encoding**, whose argument is that one encoder
cannot serve two purposes because the two want different features.

**Proposal: a metric-geometry tower — depth, MiDaS- or Depth-Anything-style.**

**Why depth rather than segmentation**, which is the more obvious suggestion.
Segmentation is largely *downstream* of semantic features: DINOv2 and SigLIP
patches already support linear-probe segmentation, so a segmentation tower mostly
duplicates what the understanding tower nearly has. Depth is a genuinely third
kind of feature. A semantic encoder is trained to be invariant to viewpoint; a
reconstruction encoder is trained to preserve appearance; a depth encoder needs
features that are invariant to *texture and albedo* and sensitive to *geometry* —
which is the opposite of what the reconstruction tower keeps and orthogonal to
what the semantic one does.

**What it unlocks that neither existing tower can.** Occlusion order. "Put the mug
behind the laptop" requires knowing what is in front of what, and neither a
semantic embedding nor a VQ code carries it — the semantic tower knows there is a
mug and a laptop, the reconstruction tower knows their pixels, and neither knows
which is nearer. Same for "how far is the doorway", scene-consistent compositing,
and any editing instruction whose correctness is a 3D fact.

**What it costs, and where the real cost is.** Not 25% of the body — this is a
new *encoder*, not a block duplication, so the arithmetic is Lesson 12.13's
exercise 3 in reverse: a MiDaS-class ViT is a few hundred million parameters plus
a projection into the body, and the shared transformer is untouched. The real
cost is routing. Exercise 2 measures the existing three-way router at **64.3%**
on a keyword design; a fourth destination makes that worse, and the router is
already the weakest component in the pipeline. A depth tower would need the
routing question answered properly — by a learned classifier or by an explicit
task tag — before the tower itself is worth building.

### 5 — four axes moved, and the table has no ablation

| stage | Janus | Janus-Pro | relative | absolute |
|---|---|---|---:|---:|
| 1, alignment | 72M pairs | 90M pairs | +25.0% | +18M |
| **2, unified** | **26M pairs** | **72M pairs** | **+176.9%** | **+46M** |
| 3, instruction | 1.2M inst | 1.4M inst | +16.7% | +0.2M |

**ANSWER: stage 2**, and it leads on both the relative and the absolute reading.

**FINDING: stage 3's percentage is in a different unit.** Stages 1 and 2 count
image-text *pairs*; stage 3 counts *instructions*. 200,000 instructions against
46,000,000 pairs is a **230×** difference that the percentage column hides by
giving all three rows the same format.

**FINDING: four axes moved and one outcome was measured.** Three data stages plus
a **5.4×** parameter increase separate the two columns, in a table with **6** rows
and **0** ablations. Attributing the +0.19 GenEval to stage 2 is not identified by
this evidence — the same table would look identical if stage 2 contributed nothing
and the parameters did all of it. Lesson 12.07's exercise 5 puts the minimum at
four runs for two axes; this is four axes and two runs.

**FINDING: even the one subtractable axis cannot be subtracted.** 1.3B → 7B is
0.731 decades, and Lesson 12.07 measured +7 **MMMU** per decade — but it has no
GenEval-per-decade row, because neither of its two families varied LLM size at a
fixed generation benchmark. So the honest answer is that this table ranks the
*inputs* and says nothing about which of them produced the output.
