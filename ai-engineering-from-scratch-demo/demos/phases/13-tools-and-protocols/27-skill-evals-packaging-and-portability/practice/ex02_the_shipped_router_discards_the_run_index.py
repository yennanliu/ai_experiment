"""Exercise 2 — the shipped router discards the run index.

    Run a five-run baseline and treatment comparison. Report every per-task
    regression even if the average improves.

Reading of the exercise: "even if the average improves" says the deliverable
is the per-task list and the average is the distractor, so the comparison is
computed per case first and summarized second. Building a treatment that
improves on average while regressing somewhere needs a router that varies
between runs -- and the shipped one cannot, which is the first thing worth
reporting.

**ANSWER: the average improves and three cases regress.** Mean per-case pass
rate goes from **0.82** to **0.92**, and `near-trigger` (**-0.4**),
`near-package` (**-0.2**) and `pos-manifest` (**-0.2**) each lose ground.
Reporting the mean alone would describe a **+0.10** win over a change that
made **3** of **10** cases worse.

**FINDING: the shipped router discards the run index.** `KeywordRouter.__call__`
begins `del run_index`, so five runs are five copies of one run and every
rate it can produce is **0.0** or **1.0**. A five-run baseline against this
router measures nothing that one run did not; the flakiness has to be
injected to be studied.

**FINDING: five runs gives six possible rates, so the smallest regression is
0.2.** A case that flips on one run of five reads exactly the same as a case
that is genuinely 80% reliable. The resolution of the instrument is **0.2**,
and **2** of the **3** regressions here are exactly one run wide.

**FINDING: the pooled metrics hide what the per-case rates show.** Precision
moves **0.8636** to **0.8889** and recall **0.76** to **0.96** across the
same runs -- both up, while three cases got worse.
`classification_metrics` flattens every run of every case into one list, so
a two-run regression is two entries in a pool of fifty.

Structure: `rates()` is the per-case view and `pooled()` the aggregate, run
over identical observations so the difference is presentation, not data.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "27-skill-evals-packaging-and-portability"
RUNS = 5
PROMPTS = (
    ("pos-release", "run the release gate over the skill package", True),
    ("pos-trigger", "score the skill trigger cases for this bundle", True),
    ("pos-manifest", "verify the bundle manifest before release", True),
    ("pos-portability", "check bundle portability for the skill package", True),
    ("pos-recall", "report trigger recall for this package", True),
    ("near-trigger", "trigger the deployment pipeline for this release", False),
    ("near-package", "install the package dependencies before the release", False),
    ("near-bundle", "evaluate the vendor bundle pricing", False),
    ("near-notes", "publish the release notes for this version", False),
    ("near-eval", "evaluate model response quality on our eval set", False),
)


class Recorded:
    """A router that answers from a per-run table, which the shipped one cannot."""

    def __init__(self, table):
        self.table = table

    def __call__(self, prompt, run_index=0):
        return self.table[prompt][run_index]


TERMS = ("skill", "package", "bundle", "trigger", "portability", "release", "manifest")
T, F = True, False
BASELINE = ((T, T, T, T, T), (T, T, T, T, F), (T, T, T, T, T), (T, T, F, F, F),
            (T, T, T, F, F), (F, F, F, F, F), (F, F, F, F, F), (T, T, F, F, F),
            (T, F, F, F, F), (F, F, F, F, F))
TREATMENT = ((T, T, T, T, T), (T, T, T, T, T), (T, T, T, T, F), (T, T, T, T, T),
             (T, T, T, T, T), (F, F, F, T, T), (F, F, F, F, T), (F, F, F, F, F),
             (F, F, F, F, F), (F, F, F, F, F))


def rates(ref, cases, router):
    return ref.repeated_run_rates(cases, router, RUNS)


def pooled(ref, cases, router):
    observations = ref.repeated_run_observations(cases, router, RUNS)
    return ref.trigger_report_from_observations(cases, observations)["metrics"]


def mean(values):
    return round(sum(values) / len(values), 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cases = tuple(ref.TriggerCase(case_id, prompt, expected)
                  for case_id, prompt, expected in PROMPTS)
    baseline = Recorded({prompt: runs for (_, prompt, _), runs in zip(PROMPTS, BASELINE)})
    treatment = Recorded({prompt: runs
                          for (_, prompt, _), runs in zip(PROMPTS, TREATMENT)})

    before, after = rates(ref, cases, baseline), rates(ref, cases, treatment)
    deltas = {case_id: round(after[case_id] - before[case_id], 4) for case_id in before}
    steady = ref.repeated_run_rates(cases, ref.KeywordRouter(TERMS, 2), RUNS)
    return {
        "before": before, "after": after, "deltas": deltas,
        "mean_before": mean(before.values()), "mean_after": mean(after.values()),
        "mean_delta": round(mean(after.values()) - mean(before.values()), 4),
        "regressions": sorted(case_id for case_id, delta in deltas.items() if delta < 0),
        "one_run_wide": sorted(case_id for case_id, delta in deltas.items()
                               if delta == -0.2),
        "steady_values": sorted(set(steady.values())),
        "resolution": round(1 / RUNS, 4), "possible_rates": RUNS + 1,
        "pooled_before": pooled(ref, cases, baseline),
        "pooled_after": pooled(ref, cases, treatment),
        "observations": len(cases) * RUNS,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the average improves and three cases regress",
            all([result["mean_after"] > result["mean_before"],
                 result["mean_before"] == 0.82, result["mean_after"] == 0.92,
                 result["regressions"] == ["near-package", "near-trigger",
                                           "pos-manifest"]]),
            f"the mean per-case pass rate goes from {result['mean_before']} to "
            f"{result['mean_after']}, a gain of {result['mean_delta']}, while "
            f"{result['regressions']} each lose ground "
            f"({[result['deltas'][name] for name in result['regressions']]}). The mean "
            f"alone would describe a win over a change that made "
            f"{len(result['regressions'])} of {len(result['deltas'])} cases worse",
        ),
        practice.Check(
            "FINDING: the shipped router discards the run index",
            all([result["steady_values"] == [0.0, 1.0], result["possible_rates"] == 6]),
            f"KeywordRouter.__call__ begins 'del run_index', so five runs are five copies "
            f"of one run and the only rates it produces are {result['steady_values']}. A "
            "five-run baseline against it measures nothing a single run did not, which is "
            "why the flakiness here had to be injected",
        ),
        practice.Check(
            "FINDING: five runs gives six possible rates, so the smallest regression is 0.2",
            all([result["resolution"] == 0.2, result["possible_rates"] == RUNS + 1,
                 len(result["one_run_wide"]) == 2]),
            f"{RUNS} runs admit {result['possible_rates']} distinct rates, so a case that "
            f"flips on one run reads identically to a case that is genuinely 80% reliable. "
            f"{len(result['one_run_wide'])} of {len(result['regressions'])} regressions "
            f"here are exactly one run wide: {result['one_run_wide']}",
        ),
        practice.Check(
            "FINDING: the pooled metrics hide what the per-case rates show",
            all([result["observations"] == 50,
                 result["pooled_after"]["precision"] > result["pooled_before"]["precision"],
                 result["pooled_after"]["recall"] > result["pooled_before"]["recall"],
                 len(result["regressions"]) == 3]),
            f"across the same {result['observations']} observations precision moves "
            f"{result['pooled_before']['precision']} to "
            f"{result['pooled_after']['precision']} and recall "
            f"{result['pooled_before']['recall']} to {result['pooled_after']['recall']} -- "
            f"both up, while {len(result['regressions'])} of 10 cases got worse. "
            "classification_metrics flattens every run of every case into one list, so a "
            "regression of two runs is two entries in a pool of 50",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
