<!-- generated:start -->
# 02-ml-fundamentals / 14-naive-bayes

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/14-naive-bayes/) · upstream spec
`phases/02-ml-fundamentals/14-naive-bayes/docs/en.md`

```bash
uv run demo practice run 14-naive-bayes --ex 1
uv run demo explain 14-naive-bayes --ex 1
uv run pytest demos/phases/02-ml-fundamentals/14-naive-bayes
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Smoothing experiment. Train MultinomialNB on text data with alpha values of 0.01, 0.1, 1.0, 1… | code | T0 | `ex01_high_alpha_loses_to_the_prior_not_the_data.py` |
| 2 | Feature independence test. Take a real text dataset. Pick two words that are obviously correl… | code | T0 | `ex02_correlated_words_cost_confidence_not_accuracy.py` |
| 3 | Bernoulli implementation. Extend the code with a BernoulliNB class. Convert bag-of-words to b… | code | T0 | `ex03_bernoulli_wins_on_bursts_not_short_text.py` |
| 4 | NB vs Logistic Regression. Train both on text data. Start with 100 training samples and incre… | code | T0 | `ex04_lr_only_overtakes_when_words_are_copied.py` |
| 5 | Spam filter. Build a complete spam classifier: tokenize raw email text, build vocabulary, cre… | code | T0 | `ex05_rare_spam_halves_recall_at_92_accuracy.py` |
<!-- generated:end -->

## Answers

The lesson's `code/naive_bayes.py` contains a from-scratch `MultinomialNB` and
`GaussianNB`, plus `make_text_data`, which draws every word count as an
independent Poisson given the class. That is exactly Naive Bayes' own model, and
at about 362 words per document it separates the two classes perfectly. Several
answers below therefore thin the documents (keep a random fraction of their
tokens) so the comparison has something to measure. Exercise 2 needs real text,
so it uses this curriculum's own lesson prose. Exercise 5 needs raw email, so it
writes deterministic synthetic email.

### 1 — high alpha hurts once word evidence falls under the prior

| alpha | 0.01 | 0.1 | 1 | 10 | 100 | 1000 | 1e6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| lesson setup (362 words/doc, 900 train) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| short text (11 words/doc, 40 train) | 0.990 | 0.994 | **0.997** | 0.996 | 0.966 | 0.498 | 0.498 |

**ANSWER: on the lesson's own setup there is no peak.** On short, scarce text
accuracy peaks at alpha = 1.

**FINDING: high alpha hurts through the prior.** As alpha grows, `(count +
alpha)/(total + alpha V)` tends to `1/V` for both classes, so the word evidence
shrinks as `1/alpha`: its median times alpha is 21.4, 26.2, 26.9 and 27.0 for
alpha from 10 to 10^4. The log prior stays fixed. Forty training documents split
21/19 give a prior gap of **0.100 nats**. At alpha = 1000 the evidence is 0.027,
and every test document gets the majority class. The lesson's 452/448 split has
a gap of 0.0089, small enough to survive a million-fold smoothing.

**CONTROL:** probabilities degrade before accuracy does. Held-out log loss is 0
at alpha = 10^4 and 0.426 at 10^6, while accuracy stays at 1.000.

### 2 — correlated words cost confidence, not accuracy

The corpus is every 20+ word prose paragraph in the `docs/en.md` files of phase
02 (678 paragraphs) and phase 03 (424 paragraphs). Probabilities count the
paragraphs that contain each word.

| pair | class | P(w1)·P(w2) | P(both) | lift |
|---|---|---:|---:|---:|
| learning / rate | phase 02 | 0.0042 | 0.027 | **6.3** |
| learning / rate | phase 03 | — | — | 4.5 |
| machine / learning | phase 02 (13 paragraphs) | — | — | 8.7 |

**ANSWER: the independence assumption is off by 4–9×.**

**FINDING: accuracy doesn't notice.** Merging every "learning rate" into one
token changes **1 of 300** held-out predictions (0.930 → 0.933).

**FINDING: confidence does.** 90% of held-out paragraphs get P > 0.99, and those
are right **97.0%** of the time, three times the error rate their probabilities
claim.

**CONTROL:** the lesson's `feature_log_prob_` matches sklearn's within 8.9e-16.

### 3 — Bernoulli wins on bursty counts, not on short text

| data | train docs | Multinomial | Bernoulli |
|---|---:|---:|---:|
| lesson, 364 words/doc | 2000 | 1.000 | 1.000 |
| 7.3 words/doc | 2000 | 0.986 | 0.985 |
| same, 5% of present words repeated 20× | 2000 | 0.953 | **0.985** |
| 3.6 words/doc | 20 | **0.773** | 0.527 |
| 3.6 words/doc | 2000 | 0.928 | 0.929 |

**ANSWER: Bernoulli wins when counts are bursty.** The repetitions carry no
class signal, and only Multinomial counts them.

**FINDING: Bernoulli doesn't win on the short text the doc recommends it for.**
With 20 training documents of 3.6 words, it loses by 25 points. Each of the
~196 absent words casts a vote, and with that little data those votes are
mostly noise.

**CONTROL:** the joint log-likelihoods match `sklearn.naive_bayes.BernoulliNB`
within 5.2e-12.

### 4 — LR overtakes NB only once words are correlated

All runs use documents thinned to 1% of their tokens (3.8 words), averaged over
3 seeds and tested on 4000 documents.

| n | 100 | 300 | 1000 | 3000 | 10000 |
|---|---:|---:|---:|---:|---:|
| NB, lesson data | **0.908** | 0.924 | 0.935 | 0.937 | 0.938 |
| LR, lesson data | 0.890 | 0.924 | 0.934 | 0.939 | 0.939 |
| NB, 10 words copied 8× | 0.872 | 0.890 | 0.906 | 0.903 | 0.902 |
| LR, 10 words copied 8× | 0.889 | 0.916 | 0.932 | 0.938 | **0.940** |

**ANSWER: on the lesson's data LR never meaningfully overtakes.** It ties NB
from n = 300 on and is never more than 0.0017 ahead. The generator *is* Naive
Bayes' model, so NB is already the right model and LR can only converge to it.

**FINDING: once the independence assumption is broken, LR leads at every
size**, by 3.7 points at 10,000. NB stops improving after 1,000 documents,
because it counts every copy of a word as fresh evidence.

**CONTROL:** on the lesson's data NB leads at n = 100, matching the doc's
"small data: better" row (Ng and Jordan).

### 5 — trained on 0.9% spam, the filter misses half the spam at 92% accuracy

The pipeline is a regex tokenizer, a vocabulary built from the training emails
only, bag-of-words counts and the lesson's `MultinomialNB`. It runs on 2000 test
emails, 14.8% of them spam.

| filter | precision | recall | accuracy |
|---|---:|---:|---:|
| always ham | — | 0 | 0.852 |
| trained at 15% spam | 0.924 | 0.818 | 0.963 |
| trained at 0.9% spam | 0.993 | **0.475** | **0.921** |
| same, prior reset to 50/50 | 0.735 | 0.879 | — |

**ANSWER: precision 0.924, recall 0.818.** Accuracy alone can't separate a
filter from doing nothing, because doing nothing already scores 0.852.

**FINDING: accuracy hides a filter that misses half the spam.** Trained on rare
spam, accuracy falls 4.2 points but recall falls 34. The log-prior gap is
**4.65 nats**, while the median spam email carries only **4.32 nats** of word
evidence. This is the doc's "class imbalance" gotcha, measured.

**CONTROL: the doc's worked example reproduces, but it mislabels the output.**
The lesson's `predict_log_proba([2, 1, 0])` gives −3.108 and −8.841 (the doc
gives −3.109 and −8.838). The doc calls these "log P(spam | email)", but they
are joint log scores: P(spam | email) is **0.9968**, not exp(−3.108) = 0.045.
