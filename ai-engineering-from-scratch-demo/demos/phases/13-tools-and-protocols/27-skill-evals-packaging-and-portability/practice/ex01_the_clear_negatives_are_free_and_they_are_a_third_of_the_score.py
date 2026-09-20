"""Exercise 1 — the clear negatives are free, and they are a third of the score.

    Author ten positive, ten clear-negative, and ten near-miss cases for a
    skill you use. Split them before editing the description.

Reading of the exercise: "split them before editing" is a rule about order,
so the three groups are frozen as separate constants and every measurement is
reported per group rather than pooled. Doing that immediately shows why the
split matters: the shipped metrics have one false-positive counter, and two
of the three groups feed it for entirely different reasons.

**ANSWER: 30 cases in three frozen groups, scored apart.** The shipped
`KeywordRouter` gets **10** of **10** positives, **10** of **10** clear
negatives and **6** of **10** near misses -- precision **0.714**, recall
**1.0**, accuracy **0.867**. Pooled, that reads like a strong router; split,
it is a router with one failure mode.

**FINDING: the clear negatives are free, and they are a third of the score.**
They share **0** vocabulary with the skill, so no plausible router fails
them; dropping them takes accuracy from **0.867** to **0.80**. A third of
the set moves the number and cannot move the design.

**FINDING: `classification_metrics` has one false-positive counter for two
different failures.** The **4** false positives are all near misses and the
metric cannot say so, because `TriggerCase` has **3** fields and none of them
is a group. The split has to survive as an id convention, which is exactly
how it gets lost.

**FINDING: tuning after looking is what the ordering rule forbids, and the
cost is visible.** Raising the router threshold to **3** fixes all **4** of the
**4** near misses and breaks **2** positives -- recall **1.0** to **0.8** for
precision **0.714** to **1.0**. A set split after the fact cannot tell that
trade from an improvement.

Structure: `score()` runs one group at a time, so every number in this file
is attributable to a group rather than to the set.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "27-skill-evals-packaging-and-portability"
POSITIVE = (
    "evaluate this skill package before release",
    "measure trigger precision for the skill bundle release",
    "check bundle portability for the skill package",
    "run the release gate over the skill package",
    "evaluate bundle portability before the release",
    "score the skill trigger cases before the package release",
    "verify the bundle manifest before release",
    "report trigger recall for this package",
    "gate the release on package and bundle evidence",
    "evaluate whether the skill bundle is release ready",
)
CLEAR_NEGATIVE = (
    "book a meeting room for thursday",
    "summarize the customer interview notes",
    "what is the weather in taipei tomorrow",
    "translate this paragraph into japanese",
    "draft a birthday message for a colleague",
    "explain recursion to a new hire",
    "order lunch for six people",
    "find the nearest pharmacy",
    "convert these euros to yen",
    "write a haiku about rain",
)
NEAR_MISS = (
    "install the package dependencies before the release",
    "evaluate the vendor bundle pricing",
    "trigger the deployment pipeline for this release",
    "package the design assets and release them to marketing",
    "publish the release notes for this version",
    "evaluate model response quality on our eval set",
    "measure api latency precision in the dashboard",
    "recall what we decided about the shipping date",
    "check whether the python module imports cleanly",
    "gate the deployment on the smoke tests",
)
TERMS = ("skill", "package", "bundle", "trigger", "portability", "evaluate", "release")


def cases(ref, prefix, prompts, expected):
    return tuple(ref.TriggerCase(f"{prefix}-{index}", prompt, expected)
                 for index, prompt in enumerate(prompts))


def score(ref, group, router):
    report = ref.evaluate_triggers(group, router)
    passed = sum(case["passed"] for case in report["cases"])
    return {"passed": passed, "total": len(group), "metrics": report["metrics"]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    groups = {"positive": cases(ref, "pos", POSITIVE, True),
              "clear": cases(ref, "clr", CLEAR_NEGATIVE, False),
              "near": cases(ref, "near", NEAR_MISS, False)}
    every = tuple(case for group in groups.values() for case in group)
    router = ref.KeywordRouter(TERMS, threshold=2)
    strict = ref.KeywordRouter(TERMS, threshold=3)

    per_group = {name: score(ref, group, router) for name, group in groups.items()}
    pooled = ref.evaluate_triggers(every, router)["metrics"]
    without_clear = ref.evaluate_triggers(groups["positive"] + groups["near"],
                                          router)["metrics"]
    tuned = ref.evaluate_triggers(every, strict)["metrics"]
    tuned_groups = {name: score(ref, group, strict) for name, group in groups.items()}
    return {
        "sizes": {name: len(group) for name, group in groups.items()},
        "passed": {name: report["passed"] for name, report in per_group.items()},
        "pooled": pooled, "without_clear": without_clear, "tuned": tuned,
        "false_positives": pooled["false_positive"],
        "near_false_positives": per_group["near"]["total"] - per_group["near"]["passed"],
        "clear_false_positives": per_group["clear"]["total"] - per_group["clear"]["passed"],
        "case_fields": list(vars(ref.TriggerCase)["__dataclass_fields__"]),
        "tuned_passed": {name: report["passed"] for name, report in tuned_groups.items()},
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: thirty cases in three frozen groups, scored apart",
            all([result["sizes"] == {"positive": 10, "clear": 10, "near": 10},
                 result["passed"] == {"positive": 10, "clear": 10, "near": 6},
                 result["pooled"]["precision"] == 0.7143,
                 result["pooled"]["recall"] == 1.0,
                 result["pooled"]["accuracy"] == 0.8667]),
            f"the shipped router passes {result['passed']} of {result['sizes']} -- "
            f"precision {result['pooled']['precision']}, recall "
            f"{result['pooled']['recall']}, accuracy {result['pooled']['accuracy']}. "
            "Pooled that reads like a strong router; split, it is a router with exactly "
            "one failure mode",
        ),
        practice.Check(
            "FINDING: the clear negatives are free, and they are a third of the score",
            all([result["passed"]["clear"] == 10, result["clear_false_positives"] == 0,
                 result["without_clear"]["accuracy"] == 0.8,
                 result["pooled"]["accuracy"] > result["without_clear"]["accuracy"]]),
            f"the clear negatives share no vocabulary with the skill, so the router takes "
            f"{result['passed']['clear']} of 10 and contributes "
            f"{result['clear_false_positives']} false positives. Dropping them moves "
            f"accuracy from {result['pooled']['accuracy']} to "
            f"{result['without_clear']['accuracy']}: a third of the set moves the number "
            "and cannot move the design",
        ),
        practice.Check(
            "FINDING: one false-positive counter for two different failures",
            all([result["false_positives"] == 4,
                 result["near_false_positives"] == result["false_positives"],
                 len(result["case_fields"]) == 3,
                 "group" not in result["case_fields"]]),
            f"the {result['false_positives']} false positives are all near misses and the "
            f"metric cannot say so: TriggerCase carries {result['case_fields']} with no "
            "group. The split survives only as an id convention, which is exactly how it "
            "gets lost between the authoring and the dashboard",
        ),
        practice.Check(
            "FINDING: tuning after looking is what the ordering rule forbids",
            all([result["tuned_passed"]["near"] == 10, result["tuned_passed"]["positive"] == 8,
                 result["tuned"]["recall"] == 0.8, result["tuned"]["precision"] == 1.0,
                 result["tuned_passed"]["clear"] == 10]),
            f"raising the threshold to 3 takes near misses from {result['passed']['near']} "
            f"to {result['tuned_passed']['near']} and positives from "
            f"{result['passed']['positive']} to {result['tuned_passed']['positive']} -- "
            f"recall {result['pooled']['recall']} to {result['tuned']['recall']} for "
            f"precision {result['pooled']['precision']} to {result['tuned']['precision']}. "
            "A set split after the fact cannot tell that trade from an improvement",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
