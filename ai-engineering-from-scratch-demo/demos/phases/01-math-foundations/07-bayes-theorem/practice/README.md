<!-- generated:start -->
# 01-math-foundations / 07-bayes-theorem

Solutions to all 4 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/07-bayes-theorem/) · upstream spec
`phases/01-math-foundations/07-bayes-theorem/docs/en.md`

```bash
uv run demo practice run 07-bayes-theorem --ex 1
uv run demo explain 07-bayes-theorem --ex 1
uv run pytest demos/phases/01-math-foundations/07-bayes-theorem
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Multiple tests. A patient tests positive twice on independent tests (both 99% accurate, disea… | code | T0 | `ex01_sequential_tests.py` |
| 2 | Smoothing impact. Run the spam classifier with smoothing values of 0.01, 0.1, 1.0, and 10.0.… | code | T0 | `ex02_smoothing_impact.py` |
| 3 | Add features. Extend the NaiveBayes class to also use message length (short/long) as a featur… | code | T0 | `ex03_length_feature.py` |
| 4 | MAP by hand. Given observed data (7 heads in 10 coin flips), compute the MAP estimate of the… | code | T0 | `ex04_map_vs_mle.py` |
<!-- generated:end -->

## Answers to the questions the exercises ask

**1 — what is P(sick) after two positives?** **0.495.** One positive test on a
1-in-10,000 disease leaves P(sick) at 0.0098 — still ~99% false alarms, which is
the base-rate result the lesson is built around. The second test multiplies the
*odds* by 99 again, and the solution shows chaining is exactly equivalent to one
update with the likelihood ratio squared (9801). Three positives are needed to
pass 90%.

**2 — how do the top word probabilities change, and what happens at
smoothing=0?** Smoothing moves probability mass from words a class contains to
words it does not:

| α | P('free' \| spam) | P('deadline' \| spam), unseen |
|---:|---:|---:|
| 0.01 | 0.2463 | 0.00082 |
| 0.1 | 0.2183 | 0.00704 |
| 1.0 | 0.1176 | 0.02941 |
| 10.0 | 0.0560 | 0.04310 |

At α=10 the two are nearly equal and the classifier stops discriminating: P(spam)
for `"free money deadline"` falls from 0.9989 to 0.5927, most of the way back to
the prior.

With **α=0** and a ham-only word, `_log_likelihood` raises
`ValueError: expected a positive input, got 0.0`. Worth being precise about what
breaks: the word's likelihood is 0, so `log(0)` fails, and the *entire document
score* is lost — not just that word's contribution. The model does not become
overconfident, it becomes uncomputable.

**3 — the length feature has to be smoothed too.** In this training data spam is
100% short (4 short, 0 long). Without smoothing, `P(long|spam) = 0`, so any long
document scores `log(0)` for spam — exercise 2's failure, reintroduced through the
new feature. The solution smooths the length counts the same way, and the feature
then shifts P(spam) by up to 0.071, in the direction it predicts on every probe.

**4 — MAP vs MLE.** MAP = **8/12 = 0.6667**, MLE = 0.7; the Beta(2,2) prior pulls
0.033 toward 0.5. The relationship is exact rather than vague: MAP is the
count-weighted blend of the MLE and the prior mean, and Beta(α,β) is worth
precisely `α+β−2 = 2` pseudo-observations. Two consequences fall out — a flat
Beta(1,1) prior reproduces the MLE exactly (MLE *is* MAP under a uniform prior),
and the prior's influence decays to 4e-5 by n=10,000.
