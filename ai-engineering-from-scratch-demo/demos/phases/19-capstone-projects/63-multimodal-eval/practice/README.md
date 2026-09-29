<!-- generated:start -->
# 19-capstone-projects / 63-multimodal-eval

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/63-multimodal-eval/) · upstream spec
`phases/19-capstone-projects/63-multimodal-eval/docs/en.md`

```bash
uv run demo practice run 63-multimodal-eval --ex 1
uv run demo explain 63-multimodal-eval --ex 1
uv run pytest demos/phases/19-capstone-projects/63-multimodal-eval
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add CIDEr to the captioning metrics. CIDEr uses TF-IDF weighting on n-grams, which rewards in… | code | T1 | `ex01_cider_scores_a_caption_with_no_shared_token_0_where_bleu4_gives_it_97pct_of_the_untrained_score.py` |
| 2 | Implement soft-accuracy VQA: multiple human answers per question, accuracy is `min(human_coun… | code | T1 | `ex02_the_trained_model_scores_0_on_exact_match_and_soft_accuracy_below_the_1_in_128_chance_level.py` |
| 3 | Add a NaN-safe variant of `bleu4` that handles empty generated sequences without crashing. | code | T1 | `ex03_bleu4_never_crashes_on_an_empty_caption_but_raises_on_521_of_2000_inputs_with_no_references.py` |
| 4 | Compute mean reciprocal rank (MRR) alongside R@K. MRR is sensitive to where the correct item… | code | T1 | `ex04_training_lifts_image_to_text_mrr_from_0_083_to_0_104_while_r_at_1_falls_from_0_02_to_0.py` |
| 5 | Run the eval on the model at five checkpoints during training (step 0, 10, 20, 30, 40, 50) an… | code | T1 | `ex05_held_out_loss_falls_at_all_5_checkpoints_but_only_bleu4_tracks_it_and_vqa_ends_below_chance.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's own `main()` as shipped: 50 training steps of
lesson 62's model on the mock corpus, evaluated before and after on the
50-sample suite. One lesson function at a time (`bleu4`, `vqa_exact_match`,
`recall_at_k`, `evaluate` or `sample_batch`) is wrapped to record what the
run scores, and restored afterwards. External definitions were read on
2026-09-29 from pycocoevalcap's `cider_scorer.py`, GT-Vision-Lab's
`vqaEval.py` and NLTK's `bleu_score.py`.

### 1 — CIDEr scores a caption with no shared token 0, where BLEU-4 gives it 97% of the untrained score

**CIDEr-D (the COCO variant) moves from 0.0354 to 0.335 on the shipped run,
against a ceiling of 5.79**, where the ceiling means each image's first
reference used as its caption. A caption equal to its only reference scores
exactly 10.

| caption | BLEU-4 (lesson) | CIDEr-D |
|---|---:|---:|
| 8 tokens found in no reference | 0.1186 | 0 |
| untrained model | 0.1220 | 0.0354 |
| trained model | 0.1730 | 0.335 |

BLEU-4's smoothing floor makes the untrained model look 70% as good as the
trained one; CIDEr puts it at 11%. The trained captioner has collapsed to 4
distinct captions for 50 images, and 22 of them score CIDEr 0. CIDEr also
cannot score a one-image eval: every idf is log 1 = 0, so even a perfect
caption scores 0.

### 2 — the trained model scores 0 on exact match and soft accuracy, below the 1/128 chance level

**Before training the model scores 0.02 exact match, 0.0333 soft (stated
formula) and 0.032 soft (official); after training it scores 0 on all three.**
Each question is given 10 labelled annotations: 6 x the answer and 2 x each
variant caption's first token. The exercise's formula is not quite VQA v2's.
The official code averages the formula over the 10 leave-one-out subsets of 9
annotators:

| annotators matching | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| `min(n / 3, 1)` | 0.333 | 0.667 | 1.0 | 1.0 |
| vqaEval.py | 0.3 | 0.6 | 0.9 | 1.0 |

The suite's VQA cannot be answered. The answer is the caption's first token,
which depends on the sample index. The question is random tokens, and the
image carries only a brightness class (i % 7), so any image-only rule tops
out at 0.14. The lesson says "VQA improving above random", but its run goes
from 0.02 to 0.0, below 1/128.

### 3 — `bleu4` never crashes on an empty caption, but raises on 521 of 2,000 inputs with no references

**`bleu4_safe` wraps the lesson's `bleu4`. It drops empty references and
returns 0.0 when nothing usable is left. On 2,000 seeded edge-case inputs it
raises 0 times and always returns a value in [0, 1].** The shipped `bleu4`
already returns 0.0 on an empty caption and cannot produce NaN. What it
crashes on is an empty reference list: 521 raises, 52 of them with an empty
caption as well. Against only empty references it scores an 8-token caption
0.1349.

The lesson's smoothing is not Chen and Cherry method 1, as it says. Method 1
adds epsilon = 0.1 to a zero numerator; the lesson adds 1 to the numerator
and the denominator, which is method 2 in NLTK.

| | lesson (add 1) | method 1 (eps 0.1) |
|---|---:|---:|
| untrained | 0.122 | 0.0188 |
| trained | 0.173 | 0.0955 |
| no shared token | 0.1186 | 0.0137 |

Where no precision is zero, the two agree exactly. A perfect caption shorter
than 4 tokens scores 0.

### 4 — training lifts image-to-text MRR from 0.083 to 0.104 while R@1 falls from 0.02 to 0

**MRR, image-to-text 0.083 -> 0.1036, text-to-image 0.1007 -> 0.1528.** The
ranks use the same `topk` test as `recall_at_k` and reproduce its R@K
exactly. Median rank improves from 26 to 18 and from 26 to 16, but most
queries stay beyond rank 10 (42 -> 37 and 40 -> 31). R@1_i2t moves the other
way, from 0.02 to 0.0. Its one hit before training is a tie: image 1's
caption appears twice, both copies score 0.149, and `topk(1)` happened to
return the diagonal.

Only 39 of the 50 captions are distinct, so a perfect oracle similarity
scores R@1 0.78 under the lesson's `recall_at_k` and MRR 0.89 (1.0 with ties
broken favourably, 0.78 against). All 50 eval captions also appear verbatim in
the training corpus, although the lesson calls the suite held out.

### 5 — held-out loss falls at all 5 checkpoints, but only BLEU-4 tracks it, and VQA ends below chance

**The loss falls at every checkpoint, and only BLEU-4 follows it on all 5
intervals.** The shipped `evaluate` runs inside the lesson's loop, and the
final metrics equal an unhooked run.

| step | 0 | 10 | 20 | 30 | 40 | 50 |
|---|---:|---:|---:|---:|---:|---:|
| held-out loss | 9.583 | 8.716 | 8.471 | 8.046 | 7.675 | 7.283 |
| BLEU-4 | 0.122 | 0.128 | 0.132 | 0.137 | 0.152 | 0.173 |
| R@10_t2i | 0.20 | 0.34 | 0.34 | 0.38 | 0.38 | 0.38 |
| R@1_i2t | 0.02 | 0.02 | 0.04 | 0 | 0 | 0 |
| VQA EM | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0 |

Intervals tracked out of 5: BLEU-4 5, R@10_i2t 3, R@5 and R@10_t2i 2, R@1 1,
VQA 0. The lesson says being above the random baseline "is what the demo
checks". `main()` has no assert or baseline in it, and ends with R@1_i2t at
0.0 (baseline 0.02) and VQA at 0.0 (baseline 1/128).
