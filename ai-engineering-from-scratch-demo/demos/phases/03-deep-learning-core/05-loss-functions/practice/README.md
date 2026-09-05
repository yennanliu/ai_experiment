<!-- generated:start -->
# 03-deep-learning-core / 05-loss-functions

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/03-deep-learning-core/05-loss-functions/) · upstream spec
`phases/03-deep-learning-core/05-loss-functions/docs/en.md`

```bash
uv run demo practice run 05-loss-functions --ex 1
uv run demo explain 05-loss-functions --ex 1
uv run pytest demos/phases/03-deep-learning-core/05-loss-functions
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement Huber loss (smooth L1 loss), which is MSE for small errors and MAE for large errors… | code | T0 | `ex01_huber_outliers.py` |
| 2 | Add focal loss to the binary classification training loop. Create an imbalanced dataset (90%… | code | T0 | `ex02_focal_imbalance.py` |
| 3 | Implement triplet loss with semi-hard negative mining. Generate 2D embedding data for 5 class… | code | T0 | `ex03_semi_hard_triplets.py` |
| 4 | Run the MSE vs cross-entropy comparison but track gradient magnitudes at each layer during tr… | code | T0 | `ex04_mse_vs_cross_gradients.py` |
| 5 | Implement KL divergence loss and verify that minimizing KL(true \|\| predicted) gives the sam… | code | T0 | `ex05_kl_and_soft_targets.py` |
<!-- generated:end -->

## Answers

Four of the five exercises ask a comparative question, and in each one the
comparison as posed cannot answer it — because the two arms are measured in
different units, or on different networks, or against a claim that is stated
backwards. The exception is exercise 1, where the claim is true and the margin is
71×.

**1 — Huber's test error is 71× lower, measured in MSE's own units.**

Mean squared error against clean `sin(x)`, over 3 seeds:

| trained with | error | ratio |
|---|---:|---:|
| MSE | 0.13844 | — |
| Huber | **0.00194** | **71.5×** (51.4× at the worst seed) |

**MECHANISM: 3 rows in 60 own a quarter of the MSE gradient.** The corrupted 5%
carry **28.8%** of `Σ|dL/dŷ|` at the zero start under MSE, against 5.8% under
Huber, whose per-row gradient is clipped at `δ/n = 0.0083`.

**MECHANISM: inside δ, Huber *is* MSE at half the step size.** At `δ = 1e9`,
`lr = 1.0`, it reproduces the MSE run at `lr = 0.5` to a weight deviation of
**exactly 0.0** — bit for bit. On outlier-free targets it costs 1.44× MSE's error,
which is the gap a half-rate run leaves after 1000 epochs.

**CONTROL: halving MSE's learning rate does not recover it** — MSE at `lr = 0.25`
scores 0.12440, still **64.3×** worse than Huber. The win is the clip, not the
smaller effective step it implies.

Both arms descend the loss they claim to: worst gap to a central difference over
four residuals spanning the `δ = 0.5` kink is **2.4e-10**.

**2 — focal loss at γ=2 does not move minority recall.**

| | recall over 90 positives | AUC |
|---|---:|---:|
| BCE | 0.700 | 0.9536 |
| focal (γ=2) | 0.711 | 0.9613 |

No single seed differs by more than **1 of 30** positives.

**FINDING: focal moves calibration, not detection.** Mean probability on negatives
0.056 → **0.158**; on positives 0.680 → **0.629**. Both classes drift toward 0.5 and
the 0.5 cut still lands between them.

**FINDING: the lesson's `(1−p_t)^γ` weights the loss, not the gradient.** The docs
quote 0.0100 at `p_t = 0.9` ("ignored") and 0.8100 at 0.1 ("full gradient
signal"). The *gradient* factors are **0.0290** and **1.2245** (a central difference
agrees to 2.9e-11), and the factor peaks at 1.2246 (`p_t = 0.1040`) before decaying
back to 1.0000 as `p_t → 0`. **Focal only ever damps.**

**CONTROL: α moves recall, and it moves it by moving the threshold.** `α = 0.75` at
`γ = 0` lifts recall to **0.811** while AUC goes 0.9536 → 0.9553 — and simply letting
plain BCE call its own top 152 scores positive recovers **0.822**.

**3 — semi-hard mining converges in fewer epochs and more work.**

| seed | 3 | 4 | 5 |
|---|---:|---:|---:|
| epochs to zero violations — semi-hard | **21** | **21** | **24** |
| — random | 29 | 23 | 33 |
| gradient steps taken — semi-hard | 597 | 495 | 641 |
| — random | **391** | **328** | **393** |

1.1–1.4× fewer epochs, 1.5–1.6× more steps. **Per step random selection is ahead;
only per epoch is semi-hard.**

**MECHANISM: mining raises the hit rate, not the gradient.** Violating triplets
found in the first epoch, out of 100 anchors: semi-hard **[75, 77, 68]** against
random **[37, 34, 43]**. The hardest negative farther than the positive is by
construction near the margin, so it almost always violates — the same update,
asked for twice as often.

**FINDING: the "still farther than the positive" clause almost never binds.** The
semi-hard set is empty on **11 of 45,000** anchor draws — five blobs two units
apart leave almost every negative farther than the positive. The exercise's
wording says nothing about the case where it does bind; this solution skips the
anchor.

**CONTROL: both rules end in the same place, and it is structure not scale.** 1-NN
label agreement 0.370 → **1.000** under both. The semi-hard arm's mean embedding
norm grew 3.12 → 4.06 and random's did not move at all — but the lesson's own
cosine-based `contrastive_loss`, which no rescaling can touch, falls **7.567 →
0.007** and **0.015**. The margin is met by structure.

**4 — cross-entropy's gradient is larger by exactly 1/(2p(1−p)), and uncertainty is where that is *smallest*.**

Evaluate both losses on the **same** parameters and the ratio is exact:

```
|d(BCE)/dz| / |d(MSE)/dz|  =  |p − t| / (2|p − t|·p(1−p))  =  1/(2p(1−p))
```

Over 200 epochs × 200 points × 2 arms the measured ratio never leaves that closed
form by more than **1.4e-15** relative. The sigmoid derivative cancels in
cross-entropy — `d/dz` is exactly `p − t` — and MSE keeps it.

**FINDING: the exercise has the direction backwards.** `1/(2p(1−p))` is
**minimised** at `p = 0.5` — the most uncertain the model can be — where it is
exactly **2.0000**. It grows without bound as the model becomes confidently wrong,
reaching **1.5e+31** in this run. The per-epoch curve agrees: the layer-1 ratio
*rises* from 2.72 at epoch 0 to 3.95 at epoch 199.

Cross-entropy's advantage is largest where the model is most **certain**, and at
its floor where it is most uncertain.

**FINDING: two separately trained arms stop being a comparison at the first
update.** Train them as the exercise says and the layer-1 ratio starts at 1.58 and
falls to **1.08** — below the 2.0 the closed form guarantees on shared weights.
The lesson's net updates after every sample, so after one point you are comparing
two different networks, not two losses.

**CONTROL: the two arms end in the same place, and their losses are not comparable
numbers** — mse 0.0048 / 100.0% against bce 0.0134 / 99.5%. The loss column is in
different units in each row.

**5 — KL is cross-entropy plus a constant, and only one-hot makes the constant zero.**

With a one-hot target, over 200 random 5-class logit vectors:

| | worst gap |
|---|---:|
| KL(onehot ‖ softmax) vs `categorical_cross_entropy` | **8.9e-16** |
| central difference of KL vs `cce_gradient` | **1.4e-09** (the quotient's own O(h²)) |

**MECHANISM: `KL(q‖p) = CE(q,p) − H(q)`.** The gradient is `p − q` under *both*
targets — worst deviation 1.4e-09 one-hot, 1.7e-09 soft — because `H(q)` does not
depend on the logits.

**FINDING: with soft targets the loss moves and the gradient does not.**
Cross-entropy exceeds KL by exactly the teacher's entropy — 0.1369 nats here, with
a worst mismatch of **3.6e-15** over 200 teachers. A blunter teacher (H = 1.5750)
reports a *smaller* KL for the same student, so **a distillation run can only be
compared to itself**, never to another teacher's.

**MECHANISM: the temperature the technique needs is not in this formula.**
`d/dz KL(q ‖ softmax(z/T)) = (softmax(z/T) − q)/T` exactly — worst deviation
1.7e-09 at T = 1, 2, 4. That `1/T` is why distillation multiplies the loss by `T²`.
The exercise's "teacher's softmax output" leaves T at 1, so a student trained this
way never sees the flattened targets the technique is named for.

**CONTROL: KL is zero at a perfect match and is not symmetric.** Feeding the
teacher's own log-probabilities back as logits gives **2.2e-16**; swapping the
arguments moves the value by up to **10.3113 nats**. `KL(true ‖ predicted)` — the
direction the exercise names — is the one that penalises a student for missing
mass the teacher put somewhere.
