"""Exercise 5 — a run store keeps every parameter; the lesson's own log cuts n_iter off.

    Set up MLflow tracking for the pipeline. Run 5 experiments with different
    hyperparameters. Use the MLflow UI (`mlflow ui`) to compare runs and pick
    the best model.

Reading of the exercise: `mlflow` is not a dependency here and `mlflow ui` is a web
server, so this ships the scaled-down runnable of DESIGN D11: a tracker that
writes MLflow's file-store layout (`mlruns/0/<run>/params/<key>` holding the
value, `metrics/<key>` holding "timestamp value step") into a temp directory,
and a "UI" that reads every run back from disk and ranks it. The 5 experiments
are the lesson's own 5 configs, parsed out of `demo_experiment_tracking` rather
than retyped, each scored by the lesson's `cross_validate_pipeline` on its
`FullPipeline`.

**ANSWER: run 5, logistic regression with lr=0.1 and n_iter=1000, at 0.758
5-fold CV**, ahead of run 4 (0.744) and the depth-5 tree (0.724).

**FINDING: the lesson's own experiment log loses n_iter.** Its table prints
`str(config)[:40]`; both logistic configs are 48 characters long, so n_iter is
cut off in both rows (one keeps "'n_ite", the other "'n_iter"), and 0 of the 2
printed rows show its value. The two runs that differ in it therefore look like
they differ only in lr. The file store keeps all 12 parameters of the 5 runs.

**FINDING: lr and n_iter are one hyperparameter here.** The lesson's logistic
regression is plain gradient descent from zero, and only the product lr x n_iter
matters at these step sizes: (0.01, 500), (0.1, 50) and (0.001, 5000) give
identical fold scores, as do (0.1, 1000) and (1.0, 100). Runs 4 and 5 compare a
budget of 5 with a budget of 100, not two learning rates.

**CONTROL: the pick is stable.** Run 5 is best on all 5 CV shuffles tried
(seeds 0 to 4).

Structure: `log_runs` writes the store; `read_runs` is the UI.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import io
import tempfile
from pathlib import Path

import numpy as np

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "13-ml-pipelines"
NUM, CAT = ["age", "income", "score"], ["city", "plan"]
SAME_BUDGET = ((0.01, 500), (0.1, 50), (0.001, 5000)), ((0.1, 1000), (1.0, 100))


def lesson_configs(ref):
    """The `configs = [...]` literal inside the lesson's demo_experiment_tracking."""
    tree = ast.parse(inspect.getsource(ref.demo_experiment_tracking))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "configs")
    return ast.literal_eval(node.value)


def score(ref, data, config, seed=42):
    """The lesson's 5-fold CV of its FullPipeline around the model `config` names."""
    model = {"tree": lambda: ref.DecisionTreeSimple(config["max_depth"]),
             "logistic": lambda: ref.LogisticRegressionSimple(config["lr"], config["n_iter"])}
    return ref.cross_validate_pipeline(
        lambda: ref.FullPipeline(model[config["model"]](), NUM, CAT), data, seed=seed)


def log_runs(root, ref, data, configs):
    """MLflow file-store layout: one file per param, metric lines 'timestamp value step'."""
    for i, config in enumerate(configs, 1):
        run = Path(root, "mlruns", "0", f"run{i}")
        (run / "params").mkdir(parents=True)
        (run / "metrics").mkdir()
        for key, value in config.items():
            (run / "params" / key).write_text(str(value), encoding="utf-8")
        acc = float(np.mean(score(ref, data, config)))
        (run / "metrics" / "mean_accuracy").write_text(f"0 {acc!r} 0", encoding="utf-8")


def read_runs(root):
    """What the UI shows: every run's params and metrics, read back from disk."""
    return {run.name: ({p.name: p.read_text(encoding="utf-8") for p in (run / "params").iterdir()},
                       {m.name: float(m.read_text(encoding="utf-8").split()[1])
                        for m in (run / "metrics").iterdir()})
            for run in sorted(Path(root, "mlruns", "0").iterdir())}


def stability(ref, data, configs):
    """(best config index per CV seed 0-4, fold scores per equal-budget group)."""
    winners = [int(np.argmax([np.mean(score(ref, data, c, s)) for c in configs]))
               for s in range(5)]
    budgets = [[score(ref, data, {"model": "logistic", "lr": lr, "n_iter": n}) for lr, n in group]
               for group in SAME_BUDGET]
    return winners, budgets


def solve():
    ref = parity.load_reference(PHASE, LESSON, "pipeline")
    data, configs = ref.make_mixed_data(500), lesson_configs(ref)
    with tempfile.TemporaryDirectory() as root:
        log_runs(root, ref, data, configs)
        runs = read_runs(root)
    best = max(runs, key=lambda r: runs[r][1]["mean_accuracy"])
    winners, budgets = stability(ref, data, configs)
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        ref.demo_experiment_tracking()  # the lesson's own experiment table
    rows = [line for line in printed.getvalue().splitlines() if "'logistic'" in line][:2]
    return {"acc": {r: m["mean_accuracy"] for r, (_, m) in runs.items()}, "best": best,
            "best_params": runs[best][0], "n_params": sum(len(p) for p, _ in runs.values()),
            "lengths": [len(str(c)) for c in configs if c["model"] == "logistic"],
            "table_rows": rows, "winners": winners,
            "budgets_equal": all(g[0] == other for g in budgets for other in g),
            "budgets": [float(np.mean(g[0])) for g in budgets]}


def verify(result):
    acc, rows = result["acc"], result["table_rows"]
    shown = sum("'n_iter': " in r for r in rows)
    return [
        practice.Check(
            "ANSWER: run 5 (logistic, lr=0.1, n_iter=1000) is the best of the 5",
            result["best"] == "run5" and result["best_params"].get("n_iter") == "1000",
            "5-fold CV read back: " + ", ".join(f"{r} {a:.3f}" for r, a in acc.items()),
        ),
        practice.Check(
            "FINDING: the lesson's own log truncates the config and loses n_iter",
            len(rows) == 2 and shown == 0 and min(result["lengths"]) > 40,
            f"the logistic configs are {result['lengths']} characters and the lesson prints "
            f"str(config)[:40]; {shown} of {len(rows)} printed rows show n_iter's value. The store "
            f"keeps all {result['n_params']} parameters",
        ),
        practice.Check(
            "FINDING: only lr x n_iter matters, so lr and n_iter are one knob",
            result["budgets_equal"],
            f"(0.01, 500), (0.1, 50), (0.001, 5000) score {result['budgets'][0]:.3f} on every "
            f"fold and (0.1, 1000), (1.0, 100) score {result['budgets'][1]:.3f}: runs 4 and 5 "
            "compare budgets of 5 and 100",
        ),
        practice.Check(
            "CONTROL: the pick survives reshuffling the folds",
            result["winners"] == [4] * 5,
            f"best run index per CV seed 0-4: {[w + 1 for w in result['winners']]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
