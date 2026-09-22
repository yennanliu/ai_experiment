"""Exercise 1 — the polish stage does not polish anything.

    Run `code/main.py` with default parameters. What fraction of loop runs
    produce a "clean" paper? What fraction produce a paper with an
    experiment-failure flaw the figure critique polished over?

Reading of the exercise: the second question names a mechanism, so it is
answered by checking the mechanism before quoting a number for it. Runs are
20000 rather than the shipped 1000, seeded at the lesson's own DEFAULT_SEED,
because a 1000-trial share moves in the second digit and every claim below is
stated to four.

**ANSWER: 18.5% of runs are clean, and 9.8% carry an experiment flaw --
but the figure critique polished over none of them.** Of **34.4%** of runs
that reach submission, **53.9%** are clean and **46.1%** carry a flaw:
**9.8%** of all runs have an experiment flaw and **6.1%** a novelty flaw
alone.

**FINDING: `polish_masks_weakness` is inert.** `run_one` assigns
`polished_hides_weakness` and never reads it -- it is **1** of exactly
**2** names in the function that appear once. Setting the parameter to
**0.0**, **0.7** or **1.0** leaves submissions at **0.3439** and the flawed
share at **0.4613**, identical to four digits, because the `random.random()`
call happens whatever the value is. One of the config's **6** fields changes
nothing.

**FINDING: the flaw is decided at the retry, not at the polish.** Every
retry-recovered experiment carries a residual flaw by construction, so the
share of papers reaching polish with an experiment flaw is
`f*r / (1 - f(1-r))` = **0.2848**, which the simulation reproduces to
**0.2848**. The stage the exercise asks about is downstream of the decision.

**FINDING: the biggest filter has no quality signal in it.** Abandonment
splits **19.2%** at the experiment, **11.9%** at the writeup and **34.5%** at
the internal reviewer -- the coin-flip reviewer discards more than the other
two stages together, and its accept probability is independent of both flaw
flags. It is the loop's largest filter and it filters at random.

Structure: `census()` runs one config and buckets the outcomes; `reaching()`
is the closed form for the same share.
"""

from __future__ import annotations

import ast
import inspect
import random

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "05-ai-scientist-v2"

TRIALS, SEED, STAGES = 20000, 42, ("experiment", "writeup", "internal_review")


def census(ref, config, trials=TRIALS):
    random.seed(SEED)
    outcomes = [ref.run_one(config) for _ in range(trials)]
    submitted = [row for row in outcomes if row.submitted]
    return outcomes, submitted


def share(rows, trials, predicate):
    return round(sum(predicate(row) for row in rows) / trials, 4)


def reaching(config):
    """P(experiment flaw | the run reached the polish stage), in closed form."""
    failure, recovery = config.experiment_failure, config.retry_recovery
    return round(failure * recovery / (1 - failure * (1 - recovery)), 4)


def inert_sweep(ref):
    """Submissions and flawed share at three settings of the polish parameter."""
    rows = []
    for value in (0.0, 0.7, 1.0):
        outcomes, submitted = census(ref, ref.LoopConfig(polish_masks_weakness=value))
        rows.append((share(submitted, len(outcomes), lambda _row: True),
                     share(submitted, len(submitted), lambda row: row.polished_but_flawed)))
    return rows


def write_only_names(ref):
    """Names that appear exactly once in run_one -- assigned and never read."""
    tree = ast.parse(inspect.getsource(ref.run_one))
    names = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
    return sorted(name for name in set(names) if names.count(name) == 1)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    config = ref.LoopConfig()
    outcomes, submitted = census(ref, config)
    sweep = inert_sweep(ref)
    return {
        "submitted": share(submitted, len(outcomes), lambda _row: True),
        "clean": share(submitted, len(outcomes), lambda row: row.polished_ok),
        "clean_of_submissions": share(submitted, len(submitted), lambda r: r.polished_ok),
        "flawed_of_submissions": share(submitted, len(submitted), lambda r: r.polished_but_flawed),
        "experiment_flawed": share(submitted, len(outcomes), lambda r: r.has_experiment_flaw),
        "novelty_only": share(submitted, len(outcomes),
                              lambda r: r.has_novelty_flaw and not r.has_experiment_flaw),
        "write_only": write_only_names(ref),
        "sweep": sweep,
        "config_fields": len(ref.LoopConfig.__dataclass_fields__),
        "predicted_flaw": reaching(config),
        "measured_flaw": share(submitted, len(submitted), lambda r: r.has_experiment_flaw),
        "abandoned": [share(outcomes, len(outcomes),
                            lambda row, s=stage: row.abandoned_stage == s) for stage in STAGES],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 18.5% of runs are clean and 9.8% carry an experiment flaw",
            all([result["submitted"] == 0.3439, result["clean"] == 0.1852,
                 result["clean_of_submissions"] == 0.5387,
                 result["flawed_of_submissions"] == 0.4613,
                 result["experiment_flawed"] == 0.0979,
                 result["novelty_only"] == 0.0607]),
            f"{result['submitted']:.1%} of runs submit, of which "
            f"{result['clean_of_submissions']:.1%} are clean and "
            f"{result['flawed_of_submissions']:.1%} flawed -- "
            f"{result['clean']:.1%} of all runs clean, {result['experiment_flawed']:.1%} "
            f"with an experiment flaw, {result['novelty_only']:.1%} novelty only",
        ),
        practice.Check(
            "FINDING: polish_masks_weakness is inert",
            all([result["write_only"] == ["LoopConfig", "polished_hides_weakness"],
                 result["sweep"] == [(0.3439, 0.4613)] * 3,
                 result["config_fields"] == 6]),
            f"run_one assigns polished_hides_weakness and never reads it, and the "
            f"parameter at 0.0, 0.7 and 1.0 gives {result['sweep']} -- identical to "
            f"four digits, so 1 of the config's {result['config_fields']} fields "
            "changes nothing",
        ),
        practice.Check(
            "FINDING: the flaw is decided at the retry, not at the polish",
            all([result["predicted_flaw"] == 0.2848,
                 result["measured_flaw"] == result["predicted_flaw"]]),
            f"f*r / (1 - f(1-r)) predicts {result['predicted_flaw']} of submissions "
            f"carry an experiment flaw and the simulation gives "
            f"{result['measured_flaw']} -- the polish stage is downstream of the "
            "decision",
        ),
        practice.Check(
            "FINDING: the biggest filter has no quality signal in it",
            all([result["abandoned"] == [0.192, 0.1187, 0.3454],
                 result["abandoned"][2] > result["abandoned"][0] + result["abandoned"][1]]),
            f"abandonment splits {result['abandoned']} across {list(STAGES)} -- the "
            "coin-flip reviewer discards more than the other two stages together, and "
            "accepts independently of both flaw flags",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
