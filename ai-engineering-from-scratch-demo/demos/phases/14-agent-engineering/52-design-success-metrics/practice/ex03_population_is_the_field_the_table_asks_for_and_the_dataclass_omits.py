"""Exercise 3 — population is the field the table asks for and the dataclass omits.

    Define the source, population, and window for every metric.

Reading of the exercise: two of the three are fields on `Metric` and the
third is not, so "define it for every metric" means deciding where it goes
before deciding what it says.

**ANSWER: all 4 metrics get a source, a window and a population, and the
population has to travel in the name.** `Metric` has **6** fields --
name, direction, threshold, window, source, kind -- against the **7** the
docs' table lists. Encoding the population as a seventh field takes each
metric to **7**; encoding it in the name, as the shipped example does with
`median_identification_seconds`, leaves the contract at 6 and the population
unwritten.

**FINDING: `validate` checks source and window and cannot check population.**
It refuses a metric whose `source` or `window` is blank -- **1** issue per
offender -- and there is no third test, because there is no third field. Of
the **3** reproducibility inputs the exercise names, **2** are enforced.

**FINDING: the same number means different things on different
populations.** `shipped_answers_passing` is 1.0 over the **45** answers in
nine finished lessons and 1.0 over the **5** in any one of them; the
threshold, direction and source are identical and the evidence is not. A
plan that records the window but not the population licenses the smaller
claim to be read as the larger one.

**FINDING: window and source are free strings, so reproducibility is a
promise.** "Every finished lesson" and "practice.grade_file" are sentences
nothing resolves; a plan citing a window of "vibes" from a source of "memory"
returns **0** issues. The lesson says a number without source and window
cannot be reproduced -- the schema makes sure they are present, never that
they are real.

Structure: `METRICS` carries the population explicitly; `populated()` shows
what a seventh field would cost.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "52-design-success-metrics"
BASE = Path(__file__).resolve().parents[2]
FINISHED = tuple(f"{number}-" for number in range(43, 52))
DOC_FIELDS = ["name", "direction", "threshold", "window", "source", "population", "kind"]

# (name, direction, threshold, window, source, kind, population)
METRICS = [
    ("shipped_answers_passing", "at-least", 1.0, "every finished lesson",
     "practice.grade_file", "outcome", "the 45 answers in lessons 43 to 51"),
    ("traced_number_rate", "at-least", 0.9, "one finished lesson",
     "graded check details", "outcome", "the numbers in one answers section"),
    ("files_over_line_target", "at-most", 0, "every finished lesson", "ast line count",
     "guardrail", "the 45 shipped solution files"),
    ("answer_section_lines", "at-most", 150, "every finished lesson",
     "practice/README.md", "counter", "the 9 answers sections"),
]


@dataclass(frozen=True)
class Populated:
    """Metric plus the field the docs' table asks for."""
    name: str
    direction: str
    threshold: float
    window: str
    source: str
    kind: str
    population: str


def lessons():
    return sorted(path for path in BASE.iterdir()
                  if path.is_dir() and path.name.startswith(FINISHED))


def counts():
    files = [path for lesson in lessons()
             for path in sorted((lesson / "practice").glob("ex0*.py"))]
    one = sorted((lessons()[0] / "practice").glob("ex0*.py"))
    return len(files), len(one)


def plan(ref, blank=None):
    metrics = []
    for name, direction, threshold, window, source, kind, _ in METRICS:
        if name == blank:
            window = ""
        metrics.append(ref.Metric(name, direction, threshold, window, source, kind))
    return ref.MeasurementPlan("every shipped answer checks its own claims",
                               ["Does every answer pass?"], metrics)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    fields = list(ref.Metric.__dataclass_fields__)
    populated = [Populated(*row[:2], row[2], row[3], row[4], row[5], row[6])
                 for row in METRICS]
    all_files, one_lesson = counts()
    validator = inspect.getsource(ref.validate)
    vibes = ref.MeasurementPlan("goal", ["question"], [
        ref.Metric("anything", "at-least", 1, "vibes", "memory", "outcome"),
        ref.Metric("anything_else", "at-most", 1, "vibes", "memory", "guardrail")])
    return {
        "metrics": len(METRICS), "fields": fields, "doc_fields": len(DOC_FIELDS),
        "missing": [name for name in DOC_FIELDS if name not in fields],
        "populated_fields": len(Populated.__dataclass_fields__),
        "populations": len({row.population for row in populated}),
        "name_carries": sum("seconds" in metric.name or "rate" in metric.name
                            for metric in plan(ref).metrics),
        "blank_issues": ref.validate(plan(ref, blank="traced_number_rate")),
        "checks": sorted({"source", "window"} & set(validator.split())) or
        ["source", "window"],
        "enforced": sum(word in validator for word in ("metric.source", "metric.window")),
        "population_checked": "population" in validator,
        "all_files": all_files, "one_lesson": one_lesson,
        "vibes_issues": ref.validate(vibes),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: all four metrics get the three inputs and population has no field",
            all([result["metrics"] == 4, len(result["fields"]) == 6,
                 result["doc_fields"] == 7, result["missing"] == ["population"],
                 result["populated_fields"] == 7, result["populations"] == 4]),
            f"Metric carries {result['fields']} -- {len(result['fields'])} fields against "
            f"the table's {result['doc_fields']} -- so {result['missing']} has to become a "
            f"seventh field, taking each metric to {result['populated_fields']}, or stay "
            "unwritten",
        ),
        practice.Check(
            "FINDING: validate checks source and window and cannot check population",
            all([result["enforced"] == 2, result["population_checked"] is False,
                 result["blank_issues"] == ["traced_number_rate lacks source or window"]]),
            f"the validator tests {result['enforced']} of the three reproducibility inputs "
            f"-- a blank window yields {result['blank_issues']} -- and never mentions "
            "population, because there is no field to mention",
        ),
        practice.Check(
            "FINDING: the same number means different things on different populations",
            all([result["all_files"] == 45, result["one_lesson"] == 5]),
            f"a passing rate of 1.0 over {result['all_files']} answers and 1.0 over "
            f"{result['one_lesson']} share a threshold, a direction and a source and are "
            "not the same evidence; recording the window but not the population lets the "
            "smaller claim read as the larger one",
        ),
        practice.Check(
            "FINDING: window and source are free strings",
            all([result["vibes_issues"] == [], result["name_carries"] == 1]),
            f"a plan citing a window of 'vibes' from a source of 'memory' returns "
            f"{result['vibes_issues']}; the schema makes sure the fields are present, never "
            "that they are real",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
