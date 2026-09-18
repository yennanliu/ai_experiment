<!-- generated:start -->
# 12-multimodal-ai / 14-show-o-discrete-diffusion-unified

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/14-show-o-discrete-diffusion-unified/) · upstream spec
`phases/12-multimodal-ai/14-show-o-discrete-diffusion-unified/docs/en.md`

```bash
uv run demo practice run 14-show-o-discrete-diffusion-unified --ex 1
uv run demo explain 14-show-o-discrete-diffusion-unified --ex 1
uv run pytest demos/phases/12-multimodal-ai/14-show-o-discrete-diffusion-unified
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Masked discrete diffusion samples in ~16 steps. Why not 1? What breaks if you unmask everythi… | code | T0 | `ex01_one_step_and_eight_agree_on_all_sixteen_tokens.py` |
| 2 | Inpainting is free with masked diffusion. Propose a product use case (real or hypothetical) w… | explain | T0 | prose, below |
| 3 | Cosine schedule vs linear schedule: trace the number of unmasked tokens per step for T=8. Whi… | code | T0 | `ex03_the_floor_is_doing_the_balancing_not_the_cosine.py` |
| 4 | A 512x512 Show-o image is 1024 tokens. At vocab K=16384, the model emits 1024 * log2(16384) =… | code | T0 | `ex04_doubling_the_codebook_buys_one_bit_and_nine_hundredths_of_a_percent.py` |
| 5 | Read LlamaGen (arXiv:2406.06525). How is LlamaGen's class-conditional autoregressive image mo… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

Three code solutions and two prose ones. Exercise 1's result is the one worth
reading first: the lesson's sampling loop and the model it samples from are
completely decoupled, and running the loop proves it.

### 1 — one step and eight agree on all sixteen tokens

**ANSWER: because one step samples every position from its own marginal.**
Masked diffusion models the joint as a product of per-position conditionals given
whatever is visible. At step 0 nothing is visible, so unmasking everything draws
all 16 positions **independently** — each patch plausible alone, the picture
jointly incoherent. The steps exist so that each round of decisions becomes
context for the next.

**FINDING: the lesson's sampler cannot show this.** `mock_logits` reads the index
`i` and `tokens[i]`, and nothing else — no neighbour, no window, no attention.

**FINDING: so one step and eight produce identical output** — **16 of 16**
matching tokens, both returning the deterministic cycle `(prompt_seed + i) % 8`:

```
[3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2]
```

Eight forward passes buy exactly nothing, which is the strongest available
demonstration that the sampling loop and the model are separate concerns.

**FINDING: the `+3.0` self-bias is a lock, not a condition.** The one place a
decided token is read is to add 3.0 to *that same position's* already-chosen
value — a commitment device that prevents a revisit. Show-o's real gain comes
from positions informing each other; this toy implements only the half that stops
a position changing its mind.

### 2 — where inpainting beats a specialist

Drawing on **Tasks in one checkpoint**, which is where the lesson lists the four
inference modes and notes that inpainting "comes for free from the masked-
prediction training".

**Start with the honest constraint.** On *fill quality alone*, a specialist —
SDXL-inpaint, LaMa — beats Show-o, for exactly the reason Lesson 12.12 measures:
a unified loss lands near the specialist everywhere and past it nowhere by much.
So a use case that is only "fill this hole well" is a case the specialist wins.
The case has to be one where something *other than fill quality* dominates.

**The case: instruction-driven redaction with verification in the loop.**
Screenshots from a support-ticket pipeline, where a region containing personal
data has to be replaced with plausible non-identifying content that still looks
like the surrounding UI.

```
1. VQA      : "Does this screenshot contain a personal name or email?"  -> yes, two regions
2. mask     : token positions covering those regions
3. inpaint  : "fill with a generic placeholder label matching the surrounding style"
4. VQA      : "Does the result contain a personal name or email?"       -> re-run 2-3 if yes
```

**Why the specialist structurally cannot do this, rather than doing it worse.**
Three reasons, and only the third is about quality:

1. **Steps 1 and 4 are not inpainting.** A specialist fills a hole; it cannot say
   whether there is a hole to fill or whether the fill worked. You need a second
   model, which means two checkpoints resident — at 7B each in bf16, **26 GiB**
   against 13 — and a protocol between them that has to agree on coordinates.
2. **The constraint is semantic, not textural.** "Must not be a name" is a
   property of what the region *means*. A specialist inpainter is conditioned on
   surrounding pixels and, at best, a text prompt fed through a frozen text
   encoder that was never trained to enforce a negative constraint. Show-o's
   text conditioning and its text head are the same weights, so the constraint is
   a prompt rather than a pipeline stage.
3. **The loop is where the cost is, and the loop is cheap here.** A fill is 8
   forward passes at the lesson's T=8; a verification is one. Three rounds of
   fill-and-check is 27 passes through one set of weights and one KV cache. The
   two-model version pays a model switch per round, and the switch — not the
   fill — is what sets the latency.

**What would falsify the case.** If the verification step is reliable enough to
run once, the loop disappears and the specialist's better fill wins on the
remaining axis. So the case is strongest exactly where verification is
*unreliable* — which is also where it is most needed, and is the honest reason
redaction is the example rather than, say, object removal from product photos.

### 3 — the floor is doing the balancing, not the cosine

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | σ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cosine | 2 | 2 | 2 | 2 | 3 | 3 | 1 | 1 | **0.707** |
| linear | 2 | 3 | **4** | 3 | 2 | 1 | 1 | — | **1.030** |

**ANSWER: cosine**, and it uses all 8 steps where linear finishes in 7 having
spent a quarter of its budget on one step.

**FINDING: the first two cosine steps are the floor, not the cosine.** Raw
`1 − cos` gives keep ratios of **0.0192** and **0.0761**, both under the
`max(0.15, …)` clamp in `sample`. Without it the first step would unmask
`int(16 × 0.0192) = 0` tokens and `max(1, …)` would make it one — so the
schedule being compared does not control its own opening.

**FINDING: linear is front-loaded exactly where the model knows least.** Its peak
is step 2, unmasking **4** tokens with **11** still hidden; cosine's is step 4, 3
tokens with 8 hidden. The schedule that commits hardest does so with the least
context — which is the argument for cosine, visible in the traces rather than in
the curves.

**FINDING: the trailing `while` loop never fires for either.** Both reach a keep
ratio of 1.0 at the last step and clear the pool. It exists for a schedule that
does not end at 1.0 — and cosine only just does, because `cos(π/2)` is 6.1e-17
rather than zero.

### 4 — doubling the codebook buys one bit and 0.09% of a percent

**ANSWER: 438.9×** — 14,336 bits (1.75 KiB) against 6,291,456 (768 KiB).

| | bits/token | total bits | ratio |
|---|---:|---:|---:|
| Show-o, K=16384 | 14 | 14,336 | **438.9×** |
| Chameleon (12.11), K=8192 | 13 | 13,312 | **472.6×** |

**FINDING: Show-o's larger codebook compresses 7.1% *less*.** Doubling K costs
exactly one bit per token, by construction.

**FINDING: one bit of per-patch capacity is worth 0.09% of the reconstruction
error.** A vector quantizer's distortion scales as K^(−2/d) in MSE, and a 16×16×3
patch has **d = 768**. Doubling K multiplies the RMS error by **0.99910**.
Halving the error would need **2⁷⁶⁸** times the codebook.

**ANSWER: so what the compression buys is not fidelity.** At 768 dimensions the
codebook size is nearly irrelevant to reconstruction, and 438.9× is a rate at
which no photographic detail survives either way. What the budget buys is *a
sequence a language model can predict* — 1,024 positions over a 14-bit alphabet,
against 786,432 bytes no autoregressive model can emit. The ratio is the price of
admission, not a quality setting.

### 5 — how LlamaGen differs from Show-o

Drawing on **Where Show-o sits**, which is where the lesson places both models in
its 2026 taxonomy.

**Start with the lesson's own row, because it is wrong.** It lists:

> Discrete tokens + masked diffusion: Show-o, MaskGIT, **LlamaGen**, Muse.

LlamaGen does not belong in that bucket. Its paper is *"Autoregressive Model
Beats Diffusion: Llama for Scalable Image Generation"* — it is plain next-token
prediction in raster-scan order, with a Llama architecture and no masking
schedule at all. It belongs in the row above it, next to Chameleon and Emu3.

**The three differences that follow from that, in order of consequence:**

1. **Generation order.** LlamaGen emits tokens left-to-right, top-to-bottom, one
   at a time, each conditioned on every token before it. Show-o unmasks tokens in
   **confidence** order — the sampler picks *which* positions to commit at each
   step, and the schedule decides how many. LlamaGen's order is fixed by the
   raster; Show-o's is chosen at inference from the model's own certainty.
2. **Number of forward passes.** LlamaGen is N passes for N tokens; Show-o is T
   passes for any N. At 1,024 tokens and the lesson's T=8 that is 1,024 against
   **8** — a 128× difference in calls, at the cost of the independence problem in
   exercise 1, which is what forces T above 1 in the first place. This is the same
   trade Lesson 12.13's exercise 4 measures for Transfusion, arrived at from the
   discrete side.
3. **What conditioning is available.** LlamaGen as published is
   **class-conditional** — a single class embedding prepended to the sequence, on
   ImageNet — so the conditioning channel is one token wide. Show-o's conditioning
   is a text prefix in the same vocabulary as its output, which is what makes
   T2I, VQA and inpainting the same checkpoint rather than three.

**The thing they share, which is why the confusion is easy.** Both are discrete:
both run over a VQ-tokenised image and both are trained with cross-entropy over a
codebook. Exercise 4's arithmetic applies unchanged to both — the tokenizer
ceiling is the same ceiling. The difference is entirely in the *order* the tokens
are produced and in how wide the conditioning channel is, and neither of those is
visible in the loss.
