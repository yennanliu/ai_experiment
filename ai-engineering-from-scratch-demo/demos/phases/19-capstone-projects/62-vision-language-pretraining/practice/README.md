<!-- generated:start -->
# 19-capstone-projects / 62-vision-language-pretraining

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/62-vision-language-pretraining/) · upstream spec
`phases/19-capstone-projects/62-vision-language-pretraining/docs/en.md`

```bash
uv run demo practice run 62-vision-language-pretraining --ex 1
uv run demo explain 62-vision-language-pretraining --ex 1
uv run pytest demos/phases/19-capstone-projects/62-vision-language-pretraining
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Replace InfoNCE with SigLIP-style sigmoid pair loss and compare convergence on the mock corpus. | code | T1 | `ex01_with_the_papers_minus_10_bias_siglip_reaches_31pct_in_batch_accuracy_where_infonce_reaches_73pct.py` |
| 2 | Add a hard-negative mining step: every other batch, select the hardest off-diagonal pair from… | code | T1 | `ex02_hard_negative_mining_does_not_make_the_contrastive_loss_drop_faster.py` |
| 3 | Add an image-text matching binary head on top of the joint embedding (true/false: do these ma… | code | T1 | `ex03_the_matching_head_stays_at_ln_2_for_300_steps_while_cosine_already_ranks_69pct_of_captions_first.py` |
| 4 | Replace the mock corpus with caption-id sequences drawn from a Markov chain whose transition… | code | T1 | `ex04_the_markov_captions_end_4_6x_higher_because_the_mock_captions_were_already_a_plus_3_counter.py` |
| 5 | Train the same model with `lm_weight = 0` and again with `lm_weight = 1`. Compare contrastive… | code | T1 | `ex05_the_lm_loss_does_not_regress_ranking_and_never_reaches_the_projection_the_doc_says_it_trains.py` |
<!-- generated:end -->

## Answers

Every exercise imports and runs the lesson's `code/main.py`: a 2-block ViT
encoder, a two-layer projector, a mean-pooled text encoder and a
cross-attention decoder, trained with InfoNCE plus a captioning loss on 200
synthetic image-caption pairs (batch 16, Adam at 5e-4). Each solution
replays the lesson's training loop step for step; exercise 1 checks that
it matches `ref.train` exactly. A single batch's loss is noisy, so
ranking is measured on 12 fixed 16-pair blocks of the corpus. All runs are
on CPU, seeded, with torch pinned to one thread. External source read on
2026-09-29: the SigLIP paper, Algorithm 1 and Section 3
(https://arxiv.org/pdf/2303.15343).

### 1 — with the paper's -10 bias SigLIP reaches 31% in-batch accuracy where InfoNCE reaches 73%

**SigLIP converges more slowly on the mock corpus.** The sigmoid loss from
the paper's Algorithm 1 replaces `info_nce_loss` inside the model's forward
pass. Everything else stays the same. Image-to-text top-1 accuracy inside
16-pair blocks (chance 0.0625):

| step | 0 | 50 | 300 |
|---|---:|---:|---:|
| InfoNCE (lesson) | 0.083 | 0.151 | 0.729 |
| SigLIP, b = -10 (paper) | 0.083 | 0.094 | 0.312 |
| SigLIP, b = -ln 15 | 0.083 | 0.146 | 0.651 |

**The paper's bias of -10 causes most of the gap.** The paper picked it for
batches of thousands, where almost every pair is a negative. At batch 16,
Adam at 5e-4 moves the bias only to -9.975 by step 50 and to -9.852 by step
300. Starting it at the 1-in-16 prior, -ln 15, closes most of the gap.
**The lesson's "tau" is the inverse temperature:** the printed
`initial tau: 14.286` is the logit scale 1/0.07.

### 2 — hard-negative mining does not make the contrastive loss drop faster

**No.** The hardest off-diagonal entry is image i scored against caption j.
Items i and j are appended to the next batch on every other step, so the
pair meets again as negatives. Over seeds 0-4 at 50 steps, the fixed-block
contrastive loss is lower with mining on 1 seed in 5, and on average it is
0.010 higher. The training loss over the last 10 steps is higher with mining
on all 5 seeds.

**The printed training loss penalises mining for batch size alone.** Mined
batches hold 17-18 pairs, and chance-level InfoNCE is ln N, so 18 pairs
start ln(18/16) = 0.118 higher. On seed 0 the mined odd steps average 2.849
against 2.733 on its even steps; without mining it is 2.764 against 2.734.
**The doc's loss values are not what the run prints.** The doc says the loss
starts near ln 16 = 2.77 and drops toward 2.4. The run prints 3.046 at step
0, and the final 2.362 comes from a single batch: step 45 printed 2.668, and
the mean of the last 10 steps is 2.541.

### 3 — the matching head stays at ln 2 for 300 steps while cosine already ranks 69% of captions first

**The three-loss model trains, but the ITM head learns nothing in 300
steps.** The head is a linear layer over [z_img, z_txt, z_img * z_txt]. As in
BLIP, it is trained with binary cross-entropy on one positive and one hard
negative per image.

| seed | ITM loss, first -> step 300 | ITM accuracy | cosine top-1 |
|---|---|---:|---:|
| 0 | 0.692 -> 0.694 | 0.487 | 0.688 |
| 1 | 0.698 -> 0.692 | 0.547 | 0.786 |

**The head only learns once ranking is already solved.** At 700 steps its
accuracy is 0.727, and by then cosine top-1 is 1.000 and the contrastive
loss is 0.015. Its inputs are unit-vector coordinates, so its logits stay
near zero for hundreds of steps. BLIP's ITM head reads a fused
cross-attention encoder, not the two embeddings being ranked. **The third
loss does not consistently help ranking.** After 300 steps it changes the
fixed-block contrastive loss from 0.615 to 0.793 on seed 0 and from 0.672 to
0.621 on seed 1.

### 4 — the Markov captions end 4.6x higher, because the mock captions were already a +3 counter

**No, the captioning loss drops less.** Captions are drawn from one of 4
transition tables, chosen by the image's SHA-256 mod 4. Each token has 2
equally likely successors, so the best possible loss is ln 2 = 0.693.

| LM loss | step 50 | step 300 |
|---|---:|---:|
| lesson's mock corpus | 4.78 | 0.14 |
| Markov corpus | 5.41 | 0.65 |

**The mock captions are not random.** The doc calls them "random caption
ids", but in all 200 of them every token is the previous one plus 3 (mod
511). Their floor is 0, so the exercise's prediction runs the wrong way.
**The Markov model memorises instead of learning the chain.** Its training
loss of 0.652 is below the chain's own floor. On fresh captions for the same
images it scores 5.47, near uniform (6.24). That holds even though 507 of
the 1,244 distinct fresh transitions also appear in training. **Only the
Markov corpus makes the decoder read the image.** Scoring each caption
against the wrong image moves the mock loss from 0.142 to 0.152, and the
Markov loss from 0.652 to 0.813.

### 5 — the LM loss does not regress ranking, and never reaches the projection the doc says it trains

**It does not regress.** Mean fixed-block contrastive loss over seeds 0-4:

| | 50 steps | 300 steps |
|---|---:|---:|
| lm_weight = 0 | 2.521 | 0.596 |
| lm_weight = 1 | 2.517 | 0.585 |
| seeds where lm 1 is lower | 4 / 5 | 3 / 5 |

The largest gap between the two at 50 steps is 0.016. At 300 steps the seeds
alone range from 0.351 to 0.706.

**The LM gradient reaches the encoder but not the projection.** On one batch
at init, the LM loss sends the ViT encoder a gradient of total size 190.6,
and sends exactly 0 to the projector and the text encoder. The decoder reads
the encoder's patch tokens, not the projected embedding. The doc's table
says LM affects "Encoder + projection + decoder", and its prose says "only
the decoder receives LM-loss gradient"; both are wrong. **With lm_weight = 0
the caption loss stays above uniform,** at 6.39-6.42 against ln 512 = 6.238.
With lm_weight = 1 it is 4.78-4.82 after 50 steps.
