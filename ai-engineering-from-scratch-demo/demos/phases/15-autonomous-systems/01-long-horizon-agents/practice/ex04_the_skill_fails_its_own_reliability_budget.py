"""Exercise 4 — the skill fails the reliability budget it sells.

    Pick one production agent workflow you know. Estimate the median
    trajectory length in tool calls. Multiply by your best guess of per-step
    reliability. Is the resulting end-to-end number honest with your users?

Reading of the exercise: "one you know" invites a number from memory, and a
number from memory cannot be checked. The one production workflow whose step
list can be *read* is the one this lesson ships for production use --
`outputs/skill-horizon-reality-check.md` -- so the trajectory is counted from
its own obligations rather than recalled, and "your users" are the people it
hands a go/no-go memo to.

**ANSWER: 17 obligations, and at the skill's own three rates the workflow
finishes 41.8%, 84.3% and 91.8% of the time.** The skill names **5** produce
steps, **3** hard rejects, **3** refusal rules and **6** memo fields; each is
at minimum one action, so 17 is a floor on the trajectory, not a generous
estimate. Run through the lesson's own `end_to_end_reliability` at the 0.95,
0.99 and 0.995 the skill itself prescribes, **1** of **3** scores "coin flip
or worse", **2** score "fragile" and **0** score "ok".

**FINDING: the memo reports every reliability number except its own.** Its
**6** output fields carry an end-to-end table for the user's task and a
go/hold/no-go verdict; **0** of them carry the reliability of the verdict. The
word "reliability" appears **4** times in the skill and every one is about the
thing being judged.

**FINDING: no trajectory at the skill's own rates clears the skill's own
bar.** At 0.95, 0.99 and 0.995 per-step, the lengths that still reach a 95%
end-to-end -- the lesson's "ok" flag -- are **1**, **5** and **10** steps. The
skill is 17. It is longer than its own quality bar permits at every rate it
tells the user to tabulate.

**FINDING: all six of its gates point outward.** **3** hard rejects and **3**
refusal rules, every one of them a condition on the user's task -- horizon
ratio, budget, irreversible actions. **0** mention the memo. A workflow that
refuses deployments for having no reliability argument ships without one.

Structure: `obligations()` counts the skill's own step list; `budget()` runs
that count through the lesson's compounding at the rates the skill names.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "01-long-horizon-agents"

SKILL = "outputs/skill-horizon-reality-check.md"
RATES = (0.95, 0.99, 0.995)     # the three per-step rates the skill's step 3 names
OK = 0.95                       # the lesson's own "ok" flag threshold
GATES = ("Hard rejects", "Refusal rules")
FLAGS = ((0.5, "coin flip or worse"), (0.8, "not production"), (OK, "fragile"))


def skill_text():
    return (parity.lesson_dir(PHASE, LESSON) / SKILL).read_text(encoding="utf-8")


def bullets(text, heading):
    """The `- ` lines under one `Heading:` line, up to the next heading."""
    body = text.split(f"{heading}:\n", 1)[1].split("\n\n", 1)[0]
    return [line[2:] for line in body.splitlines() if line.startswith("- ")]


def obligations(text):
    counts = {"produce": len(re.findall(r"(?m)^\d+\. \*\*", text)),
              "memo": len(bullets(text, "Return a short memo with"))}
    for heading in GATES:
        counts[heading] = len(bullets(text, heading))
    return counts


def flag(score):
    """The label the lesson's own compounding table would print for a score."""
    return next((name for bound, name in FLAGS if score < bound), "ok")


def gate_text(text):
    """Every hard reject and refusal rule, as one list."""
    return [line for heading in GATES for line in bullets(text, heading)]


def naming_memo(lines):
    return [line for line in lines if "memo" in line.lower()]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    text = skill_text()
    counts = obligations(text)
    steps = sum(counts.values())
    scores = [round(ref.end_to_end_reliability(rate, steps), 3) for rate in RATES]
    memo, gates = bullets(text, "Return a short memo with"), gate_text(text)
    return {
        "counts": counts,
        "steps": steps,
        "rates": list(RATES),
        "scores": scores,
        "flags": [flag(score) for score in scores],
        "ok_scores": sum(flag(score) == "ok" for score in scores),
        "memo_fields": memo,
        "memo_self_reported": naming_memo(memo),
        "says_reliability": text.count("reliability"),
        "ok_lengths": [ref.max_steps_for_target(rate, OK) for rate in RATES],
        "gates": len(gates),
        "gates_naming_memo": naming_memo(gates),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 17 obligations scoring 41.8%, 84.3% and 91.8% end-to-end",
            all([result["steps"] == 17, result["counts"]["produce"] == 5,
                 result["counts"]["memo"] == 6,
                 result["scores"] == [0.418, 0.843, 0.918],
                 result["ok_scores"] == 0]),
            f"the skill's {result['steps']} obligations {result['counts']} run through "
            f"the lesson's compounding at {result['rates']} give {result['scores']} -- "
            f"{result['flags']}, and {result['ok_scores']} of 3 reach 'ok'",
        ),
        practice.Check(
            "FINDING: the memo reports every reliability number except its own",
            all([len(result["memo_fields"]) == 6, result["memo_self_reported"] == [],
                 result["says_reliability"] == 4]),
            f"{len(result['memo_self_reported'])} of the {len(result['memo_fields'])} "
            f"output fields describes the memo itself, and all "
            f"{result['says_reliability']} uses of 'reliability' are about the task "
            "being judged",
        ),
        practice.Check(
            "FINDING: no length at the skill's own rates clears the skill's own bar",
            all([result["ok_lengths"] == [1, 5, 10],
                 min(result["ok_lengths"]) < result["steps"],
                 max(result["ok_lengths"]) < result["steps"]]),
            f"a 95% end-to-end allows {result['ok_lengths']} steps at "
            f"{result['rates']}, against the skill's {result['steps']} -- longer than "
            "its own quality bar permits at every rate it tells the user to tabulate",
        ),
        practice.Check(
            "FINDING: all six of its gates point outward",
            all([result["gates"] == 6, result["gates_naming_memo"] == [],
                 result["counts"]["Hard rejects"] == 3,
                 result["counts"]["Refusal rules"] == 3]),
            f"{result['gates']} gates -- {result['counts']['Hard rejects']} hard "
            f"rejects and {result['counts']['Refusal rules']} refusal rules -- and "
            f"{len(result['gates_naming_memo'])} of them names the memo, so the "
            "workflow refuses deployments for having no reliability argument and "
            "ships without one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
