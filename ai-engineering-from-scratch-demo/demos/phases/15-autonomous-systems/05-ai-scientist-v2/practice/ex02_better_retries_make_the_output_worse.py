"""Exercise 2 — better retries make the output worse.

    The defaults already use Beel et al.'s 42% / 25%. Re-run with
    `--experiment-failure 0.20 --novelty-mislabel 0.10` and then with
    `--experiment-failure 0.60 --novelty-mislabel 0.40`. How does the
    polished-but-flawed share shift between the two runs?

Reading of the exercise: "how does it shift" is answered twice, because the
two runs disagree about which direction is worse depending on the
denominator -- the flawed *share of submissions* and the flawed *share of
runs* move together here, while the submission rate moves the other way. All
three are reported, and then the closed form is checked against all of them.

**ANSWER: from 20.8% to 66.7% of submissions, a 3.2x.** As a share of all
runs it goes **8.0%** to **20.8%**, a 2.6x. Submissions themselves fall the
other way, **38.5%** to **31.2%**, so the optimistic configuration ships more
papers *and* a smaller fraction of bad ones -- the two effects do not trade
off, they compound.

**FINDING: the closed form reproduces all three settings.** The flawed share
is `1 - (1-m)(1 - f*r/(1 - f(1-r)))`, giving **0.2088**, **0.4636** and
**0.6712** against simulated **0.2076**, **0.4613** and **0.6672** -- worst
gap **0.0040**, which is the Monte Carlo error on 20000 trials.

**FINDING: the two knobs act on different denominators.**
`novelty_mislabel` never gates a stage: it flips a flag and the run continues
either way. `experiment_failure` both gates and flaws -- it removes runs at
the experiment stage *and* marks the survivors. So raising it lowers the
submission count while raising the flaw rate, which is why the submission
rate falls from 38.5% to 31.2% while nothing in the config mentions
submissions.

**FINDING: `retry_recovery` is the lever nobody is asked to move, and it
points the wrong way.** A recovered experiment becomes a *flawed submission*;
an unrecovered one is abandoned. Sweeping it at the Beel defaults, the flawed
share climbs monotonically from **0.2500** at 0.0 to **0.5650** at 1.0.
Making the agent better at repairing its own failed experiments makes the
papers worse, and it is the one parameter of the six that the exercises never
touch.

Structure: `flawed_share()` is the closed form; `measure()` runs the same
configuration through the shipped `run_one`.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "05-ai-scientist-v2"

TRIALS, SEED = 20000, 42
SETTINGS = ((0.20, 0.10), (0.42, 0.25), (0.60, 0.40))     # the two runs, and the default
RECOVERIES = (0.0, 0.25, 0.55, 0.8, 1.0)


def flawed_share(failure, mislabel, recovery=0.55):
    """`1 - (1-m)(1 - f*r/(1 - f(1-r)))`, the flawed share of submissions."""
    reaching = failure * recovery / (1 - failure * (1 - recovery))
    return round(1 - (1 - mislabel) * (1 - reaching), 4)


def measure(ref, config):
    random.seed(SEED)
    outcomes = [ref.run_one(config) for _ in range(TRIALS)]
    submitted = [row for row in outcomes if row.submitted]
    flawed = sum(row.polished_but_flawed for row in submitted)
    return (round(len(submitted) / TRIALS, 4), round(flawed / len(submitted), 4),
            round(flawed / TRIALS, 4))


def sweep(ref, failure=0.42, mislabel=0.25):
    return [measure(ref, ref.LoopConfig(experiment_failure=failure,
                                        novelty_mislabel=mislabel,
                                        retry_recovery=value))[1]
            for value in RECOVERIES]


def column(runs, index):
    return [row[index] for row in runs]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = [measure(ref, ref.LoopConfig(experiment_failure=f, novelty_mislabel=m))
            for f, m in SETTINGS]
    predicted = [flawed_share(f, m) for f, m in SETTINGS]
    recovered, measured = sweep(ref), column(runs, 1)
    return {
        "settings": [list(pair) for pair in SETTINGS],
        "submissions": column(runs, 0),
        "flawed_of_submissions": measured,
        "flawed_of_runs": column(runs, 2),
        "share_shift": round(measured[2] / measured[0], 1),
        "run_shift": round(runs[2][2] / runs[0][2], 1),
        "predicted": predicted,
        "worst_gap": round(max(abs(a - b) for a, b in zip(predicted, measured)), 4),
        "closed_form_recoveries": [flawed_share(0.42, 0.25, value) for value in RECOVERIES],
        "measured_recoveries": recovered,
        "monotone": all(a <= b for a, b in zip(recovered, recovered[1:])),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 20.8% to 66.7% of submissions, a 3.2x",
            all([result["flawed_of_submissions"] == [0.2076, 0.4613, 0.6672],
                 result["share_shift"] == 3.2, result["run_shift"] == 2.6,
                 result["submissions"] == [0.3846, 0.3439, 0.3124]]),
            f"across {result['settings']} the flawed share of submissions runs "
            f"{result['flawed_of_submissions']} -- a {result['share_shift']}x -- and of "
            f"all runs {result['flawed_of_runs']}, while the submission rate moves the "
            f"other way, {result['submissions']}",
        ),
        practice.Check(
            "FINDING: the closed form reproduces all three settings",
            all([result["predicted"] == [0.2088, 0.4636, 0.6712],
                 result["worst_gap"] <= 0.005]),
            f"1 - (1-m)(1 - f*r/(1 - f(1-r))) gives {result['predicted']} against "
            f"simulated {result['flawed_of_submissions']}, worst gap "
            f"{result['worst_gap']}",
        ),
        practice.Check(
            "FINDING: the two knobs act on different denominators",
            all([result["submissions"][0] > result["submissions"][2],
                 result["flawed_of_submissions"][0] < result["flawed_of_submissions"][2]]),
            f"novelty_mislabel flips a flag and the run continues; experiment_failure "
            f"removes runs and marks the survivors, so the submission rate falls "
            f"{result['submissions'][0]} to {result['submissions'][2]} while the flaw "
            "rate triples",
        ),
        practice.Check(
            "FINDING: retry_recovery points the wrong way",
            all([result["monotone"], result["measured_recoveries"][0] == 0.2475,
                 result["measured_recoveries"][-1] == 0.5625,
                 result["closed_form_recoveries"] == [0.25, 0.365, 0.4636, 0.5251, 0.565]]),
            f"sweeping recovery over {list(RECOVERIES)} takes the flawed share "
            f"{result['measured_recoveries']} against a closed form of "
            f"{result['closed_form_recoveries']} -- a recovered experiment becomes a "
            "flawed submission where an unrecovered one is abandoned",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
