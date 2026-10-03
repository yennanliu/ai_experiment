<!-- generated:start -->
# 02-ml-fundamentals / 13-ml-pipelines

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/02-ml-fundamentals/13-ml-pipelines/) · upstream spec
`phases/02-ml-fundamentals/13-ml-pipelines/docs/en.md`

```bash
uv run demo practice run 13-ml-pipelines --ex 1
uv run demo explain 13-ml-pipelines --ex 1
uv run pytest demos/phases/02-ml-fundamentals/13-ml-pipelines
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Build a pipeline that handles a dataset with 3 numeric columns and 2 categorical columns. Use… | code | T0 | `ex01_scratch_encoder_crashes_on_a_missing_category.py` |
| 2 | Deliberately introduce data leakage: fit the scaler on the full dataset before splitting. Com… | code | T0 | `ex02_scaler_leak_moves_no_fold_score.py` |
| 3 | Serialize your pipeline with `joblib.dump`. Load it in a separate script and run predictions.… | code | T0 | `ex03_scratch_pipeline_pickle_needs_its_module.py` |
| 4 | Add a custom transformer to the pipeline that creates polynomial features (degree 2) for the… | code | T0 | `ex04_poly_goes_after_impute_and_buys_noise.py` |
| 5 | Set up MLflow tracking for the pipeline. Run 5 experiments with different hyperparameters. Us… | code | T0 | `ex05_lesson_log_truncates_n_iter_away.py` |
<!-- generated:end -->

## Answers

The lesson's `code/pipeline.py` is 654 lines of numpy: a median imputer, a
scaler, a one-hot encoder, gradient-descent logistic regression, a percentile
tree, and a `FullPipeline` with its own `cross_validate_pipeline`, all run on
`make_mixed_data(500)`. Every exercise runs against that code. Exercises 1, 3
and 5 also build the sklearn version the doc describes. Exercise 5 names MLflow,
which is not a dependency here, so it ships the scaled-down runnable
`DESIGN D11` asks for.

### 1 — sklearn imputes a missing category; the lesson's encoder crashes on it

**ANSWER: 0.758 ± 0.012** under shuffled 5-fold CV for a `ColumnTransformer`
(median + scale on age, income, score; most-frequent + one-hot on city, plan)
feeding logistic regression. The lesson's own `FullPipeline` scores 0.756.

**FINDING: the lesson's pipeline has no categorical imputer, and its encoder
crashes on a missing category.** With 5% of `city` set to None or NaN,
`OneHotEncoder.fit` raises **TypeError** both times, because `sorted(set(...))`
cannot order None or a float against a str. The sklearn pipeline imputes those
rows and scores 0.760.

**CONTROL:** the lesson's data has 26 and 12 missing ages and incomes and no
missing categories. On that data the two preprocessors are the same map: they
match within **5.8e-15** over all 11 columns.

### 2 — a scaler fit on all the data moves no fold score

| run | leaky CV | clean CV |
|---|---:|---:|
| tree, seed 42 | 0.724 | 0.724 |
| tree, seed 99 | 0.712 | 0.712 |
| logistic, seed 42 | 0.756 | 0.756 |
| logistic, seed 99 | 0.752 | 0.752 |

**ANSWER: zero, fold for fold.** A fold's scaler mean sits at most **0.04 sd**
from the full-data mean. Scaling is an affine map that never sees a label, so
the tree's percentile thresholds move with it, and gradient descent reaches the
same accuracy on every fold.

**FINDING: the doc's warning names the wrong leak.** `docs/en.md` says this
leak "inflates accuracy estimates", and the lesson's own `demo_data_leakage`
prints `+0.000`.

**CONTROL: a leak that uses the labels does inflate the score.** On pure noise
(200 rows, 2000 features), choosing the top-20 features on all rows before CV
scores **0.735**. Choosing them inside each fold scores **0.490**, which is
chance.

### 3 — sklearn round-trips bit for bit; the lesson's pipeline cannot load

A loader script runs in a fresh interpreter. **ANSWER: identical:** all 100
test probabilities match with max difference **0.0**.

**FINDING: the from-scratch `FullPipeline` dumps fine, then fails to load
elsewhere with ModuleNotFoundError.** A pickle records a module name and a
class name, not code. Run as `python pipeline.py`, the lesson's classes live in
`__main__`, which a separate script cannot supply. "Deployed as one artifact"
holds only if the code ships with the artifact.

**CONTROL:** importing `code/pipeline.py` under the recorded module name first
reproduces all 100 from-scratch predictions exactly.

### 4 — squares go after imputation and before scaling, and buy noise

Dropping a column costs income **0.064**, score **0.040** and age 0.002 of CV
accuracy, so the degree-2 terms go on income and score.

| where `Square` sits | CV |
|---|---:|
| no squares | 0.758 |
| before impute | 0.766 |
| **impute → square → scale** | **0.766** |
| after scale | 0.754 |

**ANSWER: after imputation, before scaling.**

**FINDING: before imputation it silently builds wrong features.** It does not
crash. `MedianImputer` fills NaN products with their own column medians, so on
all **12** rows with a missing income, `income·score` is not the product of the
imputed values. The CV score happens to tie.

**FINDING: after scaling, the squares go unscaled.** A squared z-score of the
lognormal income reaches **30.4**, and CV drops below the no-square baseline.

**CONTROL: the gain is noise.** +0.008 is 4 rows out of 500, smaller than the
0.012 spread across folds. `make_mixed_data` draws labels from a linear
boundary, so there is no curvature to find.

### 5 — the lesson's own log cuts n_iter off

The five configs come from the lesson's `demo_experiment_tracking`. They are
logged to an MLflow-style file store (`mlruns/0/<run>/params/<key>`) and read
back to rank them.

| run | config | CV |
|---|---|---:|
| 1 | tree, depth 3 | 0.684 |
| 2 | tree, depth 5 | 0.724 |
| 3 | tree, depth 10 | 0.684 |
| 4 | logistic, lr 0.01, n_iter 500 | 0.744 |
| 5 | logistic, lr 0.1, n_iter 1000 | **0.758** |

**ANSWER: run 5.**

**FINDING: the lesson's own table loses n_iter.** It prints `str(config)[:40]`,
and both logistic configs are 48 characters long, so **0 of 2** rows show the
value of n_iter. The file store keeps all 12 parameters.

**FINDING: lr and n_iter act as one knob.** Plain gradient descent from zero
depends only on lr × n_iter here. (0.01, 500), (0.1, 50) and (0.001, 5000)
give identical fold scores (0.744), and so do (0.1, 1000) and (1.0, 100)
(0.758). Runs 4 and 5 compare a step budget of 5 with one of 100.

**CONTROL:** run 5 wins on all 5 CV shuffles tried.
