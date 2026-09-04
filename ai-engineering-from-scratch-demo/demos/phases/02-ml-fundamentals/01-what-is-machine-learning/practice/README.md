<!-- generated:start -->
# 02-ml-fundamentals / 01-what-is-machine-learning

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/01-what-is-machine-learning/) · upstream spec
`phases/02-ml-fundamentals/01-what-is-machine-learning/docs/en.md`

```bash
uv run demo practice run 01-what-is-machine-learning --ex 1
uv run demo explain 01-what-is-machine-learning --ex 1
uv run pytest demos/phases/02-ml-fundamentals/01-what-is-machine-learning
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Take any dataset (e.g., Iris, Titanic). Split it 70/15/15 into train/validation/test. Explain… | code | T0 | `ex01_three_way_split.py` |
| 2 | List three real-world problems. For each one, identify whether it is classification, regressi… | explain | T0 | prose, below |
| 3 | A model gets 99% accuracy on training data but 60% on test data. Diagnose the problem and lis… | explain | T0 | prose, below |
<!-- generated:end -->

## Prose answers

Exercises 2 and 3 have no runnable deliverable — they ship these answers instead,
each citing the lesson section it draws on. The citation is the gate: it must name
a real heading in `docs/en.md`, which `scripts/audit_practice.py` checks.

### 2 — three problems, classified

Drawing on **The Three Types of Machine Learning** and **Classification vs
Regression**:

| Problem | Task | Supervision |
|---|---|---|
| Flagging a card transaction as fraudulent | classification (binary) | **supervised** — chargebacks give labels, though they arrive weeks late |
| Predicting tomorrow's electricity demand in MW | regression | **supervised** — yesterday's meter readings are the labels |
| Grouping support tickets to discover recurring themes nobody named | clustering | **unsupervised** — the categories are the output, not the input |

Two of these are worth a caveat, because the tidy classification hides the part
that actually decides the project.

The fraud labels are **delayed and incomplete**: a transaction is only known to be
fraudulent once someone disputes it, so recent data is unlabelled and undisputed
fraud is silently labelled legitimate. That is supervised learning with a
systematically wrong negative class.

The ticket clustering has **no ground truth to be right about**. Any evaluation is
either an internal metric (silhouette, which measures geometry rather than
usefulness) or a human reading the clusters. "Unsupervised" is not just a missing
label column; it removes the ability to say you were correct.

### 3 — 99% train, 60% test

Drawing on **Overfitting vs Underfitting** and **The Bias-Variance Tradeoff**.

**Diagnosis: overfitting** — high variance. A 39-point train/test gap means the
model has capacity to memorise the training rows and has used it. The 99% is not
evidence of a good model; it is evidence that the model can reproduce its input,
which a lookup table also does.

Before reaching for fixes, two things are worth ruling out, because they look
identical on this evidence and neither is cured by regularisation:

- **Leakage.** A feature that encodes the target — a row id correlated with class
  order, a timestamp, an aggregate computed over the full dataset before
  splitting. Leakage produces exactly this signature and gets *worse* with more
  capacity.
- **Distribution shift between splits.** If the split was not random — sorted by
  class, or by time — the test set is a different problem, not a held-out sample
  of the same one.

Assuming genuine overfitting, three fixes in the order I would try them:

1. **More training data**, or augmentation if collection is impossible. It is the
   only fix that raises the ceiling rather than trading against it, and the
   train/test gap is the specific symptom it addresses.
2. **Constrain capacity** — regularisation (L2 or L1), fewer parameters, shallower
   trees, early stopping on a validation split. This trades a little training
   accuracy for generalisation, which is the right trade when the gap is 39
   points.
3. **Cross-validation for every decision that follows.** Not a fix for the model
   but for the measurement: with a small test set, one number cannot distinguish
   a real improvement from noise, and exercise 1 measures that selection bias
   directly — 0.127 of inflation from picking the best of 40 identical
   candidates.

What I would *not* do first is switch to a more powerful model. The evidence says
capacity is already excessive.
