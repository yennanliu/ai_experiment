"""Exercise 2 — exit criteria is one string in a list of five.

    Write pilot exit criteria that include a stop decision.

Reading of the exercise: the pilot's control list already contains the words
"exit criteria", so the exercise is asking for the thing the string stands
for -- three named paths, each with a measured condition, one of which ends
the pilot.

**ANSWER: three paths -- expand, revise, stop -- with the measured run
landing on revise.** Expand needs every shipped answer passing and a traced
rate of at least 0.9; stop fires on any answer failing or a traced rate below
0.75; revise is the remainder. The pilot's real numbers -- **45** of **45**
answers passing and a traced rate of **0.875** -- satisfy neither, so the
pilot continues in its bounded form rather than expanding.

**FINDING: the control is a string and the criteria are data.**
`required_controls("pilot")` returns **5** strings, of which "exit criteria"
is **1**; `plan` returns **3** keys and none of them can hold a threshold, a
window or a decision. A pilot whose exit criteria are three sentences in a
document and a pilot whose criteria are unwritten produce identical plans.

**FINDING: the stop path is the one that needs the number written first.**
Expand and revise can both be argued after the fact; stop is the decision
somebody will want to relitigate, so its condition -- a traced rate below
**0.75** -- is the one that has to exist before the run. The lesson's own
warning about moving thresholds applies to exactly one of the three branches.

**FINDING: the pilot has no duration, so "bounded" is unenforceable.** The
docs ask for bounded duration and bounded authority; `BuildDecision` carries
**6** fields and **0** of them is a date, a window or a budget. A pilot that
runs for a week and one that has been running for a year are the same record.

Structure: `EXITS` is the rule; `pilot_values()` reads the real run.
"""

from __future__ import annotations

import re
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "53-prototype-pilot-or-production"
BASE = Path(__file__).resolve().parents[2]
FINISHED = tuple(f"{number}-" for number in range(43, 53))
SAMPLE = "47-outcomes-before-output"

EXITS = {
    "expand": "every shipped answer passes and the traced rate is at least 0.9",
    "revise": "everything else: the pilot continues in its bounded form",
    "stop": "any shipped answer fails, or the traced rate falls below 0.75",
}


def decide(passing, total, traced):
    if passing < total or traced < 0.75:
        return "stop"
    if traced >= 0.9:
        return "expand"
    return "revise"


def lessons():
    return sorted(path for path in BASE.iterdir()
                  if path.is_dir() and path.name.startswith(FINISHED))


def traced_rate(folder):
    answers = (folder / "README.md").read_text(encoding="utf-8").split("## Answers", 1)[1]
    claimed = sorted(set(re.findall(r"\b\d+(?:\.\d+)?%?\b", answers)))
    details = " ".join(check.detail for path in sorted(folder.glob("ex0*.py"))
                       for check in practice.grade_file(path).checks)
    return round(sum(value in details for value in claimed) / len(claimed), 3)


def pilot_values():
    files = [path for lesson in lessons()
             for path in sorted((lesson / "practice").glob("ex0*.py"))]
    return {"total": len(files),
            "passing": sum(practice.grade_file(path).status == "pass" for path in files),
            "traced": traced_rate(BASE / SAMPLE / "practice")}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    values = pilot_values()
    pilot = ref.BuildDecision("Would a reader trust a generated answer?",
                              True, True, 3, True, False)
    document = ref.plan(pilot)
    controls = ref.required_controls("pilot")
    return {
        **values, "paths": len(EXITS),
        "decision": decide(values["passing"], values["total"], values["traced"]),
        "stop_threshold": 0.75, "expand_threshold": 0.9,
        "stage": document["stage"], "keys": sorted(document),
        "controls": len(controls),
        "exit_strings": sum("exit" in control for control in controls),
        "holds_threshold": any(any(character.isdigit() for character in control)
                               for control in controls),
        "fields": list(ref.BuildDecision.__dataclass_fields__),
        "duration_fields": [name for name in ref.BuildDecision.__dataclass_fields__
                            if name in ("duration", "until", "budget", "window")],
        "would_stop": decide(values["passing"] - 1, values["total"], values["traced"]),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: expand, revise and stop, with the real run landing on revise",
            all([result["paths"] == 3, result["total"] == 50, result["passing"] == 50,
                 result["traced"] == 0.875, result["decision"] == "revise"]),
            f"{result['passing']} of {result['total']} shipped answers pass and the traced "
            f"rate is {result['traced']}, which clears the stop threshold of "
            f"{result['stop_threshold']} and misses the expand threshold of "
            f"{result['expand_threshold']}, so the pilot {result['decision']}s",
        ),
        practice.Check(
            "FINDING: the control is a string and the criteria are data",
            all([result["controls"] == 5, result["exit_strings"] == 1,
                 result["holds_threshold"] is False, len(result["keys"]) == 3,
                 result["stage"] == "pilot"]),
            f"required_controls returns {result['controls']} strings of which "
            f"{result['exit_strings']} is 'exit criteria', and plan returns "
            f"{result['keys']} -- nothing that can hold a threshold, a window or a decision",
        ),
        practice.Check(
            "FINDING: the stop path is the one that needs the number written first",
            all([result["would_stop"] == "stop", result["stop_threshold"] == 0.75]),
            f"one failing answer moves the same run from {result['decision']!r} to "
            f"{result['would_stop']!r}; expand and revise can be argued afterwards, so the "
            "threshold that has to exist before the run is the stop one",
        ),
        practice.Check(
            "FINDING: the pilot has no duration, so 'bounded' is unenforceable",
            all([len(result["fields"]) == 6, result["duration_fields"] == []]),
            f"BuildDecision carries {result['fields']} and "
            f"{len(result['duration_fields'])} of them is a date, a window or a budget, so "
            "a pilot running a week and one running a year are the same record",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
