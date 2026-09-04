<!-- generated:start -->
# 01-math-foundations / 06-probability-and-distributions

Solutions to all 4 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/06-probability-and-distributions/) · upstream spec
`phases/01-math-foundations/06-probability-and-distributions/docs/en.md`

```bash
uv run demo practice run 06-probability-and-distributions --ex 1
uv run demo explain 06-probability-and-distributions --ex 1
uv run pytest demos/phases/01-math-foundations/06-probability-and-distributions
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement inverse transform sampling for the exponential distribution. Verify by sampling 10,… | code | T0 | `ex01_inverse_transform_sampling.py` |
| 2 | Build a joint distribution table for two loaded dice. Compute the marginal distributions and… | code | T0 | `ex02_joint_distribution_dice.py` |
| 3 | Compute the cross-entropy loss for a 5-class classifier that outputs logits `[2.0, 0.5, -1.0,… | code | T1 | `ex03_cross_entropy_vs_torch.py` |
| 4 | Write a function that takes a list of log probabilities and returns the most likely sequence,… | code | T0 | `ex04_log_probability_sequence.py` |
<!-- generated:end -->

## Notes

**Exercise 1 — why a z-score and not a percentage.** "Compare the histogram to
the true PDF" needs a statistic. A fixed relative-error threshold is the wrong
one: a bin's count is Binomial(N, p), so its noise scales as 1/√expected, and a
tail bin will legitimately miss by 15% while a central bin misses by 2%. The
solution uses a per-bin z-score against 3σ, plus a Kolmogorov-Smirnov distance
that needs no binning at all — the second matters because bin widths are a free
parameter, and a test with a free parameter can be tuned until it passes.

**Exercise 3 — the float32 trap.** `torch.tensor([2.0, 0.5, ...])` on Python
floats silently produces **float32**, and the resulting loss differs from a
float64 computation by 1.8e-8. That is almost always the explanation when a
from-scratch loss "disagrees" with PyTorch in the 8th decimal. The solution
compares in float64 and asserts the float32 gap separately, so the discrepancy is
documented rather than absorbed into a loose tolerance.

The same exercise shows why the log-sum-exp shift exists: on logits near 1000,
the naive softmax-then-log route raises `OverflowError`, while subtracting
`max(logits)` — mathematically a no-op — returns the exact answer.

**Exercise 4 — the 50-word sentence is a deliberate trap, and it does not
spring.** 0.01^50 = 1e-100 is comfortably inside float64, so multiplying
probabilities directly still works at the size the exercise specifies. It stops
working at n = **162**, where the product hits exactly 0.0 and the log route
merely reads −746. The exercise is worth doing precisely because 50 words is
below the cliff: the naive method looks fine right up until it silently returns
zero for every candidate.
