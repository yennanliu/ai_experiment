"""Exercise 5 — the capability they agree on is the one no one folds.

    Compare the three policies on a specific capability (your choice). Name
    which policy's classification you find most rigorous and which least.
    Justify with source text.

Reading of the exercise: "most rigorous" needs a test, or it is a preference.
The test used here is whether the classification names something that *runs* --
a check, a monitor, a refusal -- as against something that is produced or
observed. The capability chosen is `undermining_safeguards`, because it is the
one where the three tables disagree in kind rather than in degree.

**ANSWER: Anthropic most rigorous, OpenAI least, on undermining
safeguards.** Anthropic classifies it as a `hardcoded prohibition` whose
action is `refuses training / deploy` -- a thing that happens without a
review. DeepMind assigns `deceptive alignment monitoring` with an
`automated monitor + red-team` -- also a thing that runs, but one that
reports rather than stops. OpenAI files it as `Research`, action `observed;
potential mitigations`: **0** of those words name an action taken by anyone.

**FINDING: the test separates the three cleanly.** Across all **7**
capabilities, OpenAI's actions name a runtime in **0** rows, Anthropic's in
**3** and DeepMind's in **4**. OpenAI's column is uniform -- **2** distinct
strings over 7 rows -- so it cannot distinguish between capabilities even
where the other two do.

**FINDING: the agreement is on the three capabilities OpenAI tracks.**
`rnd_automation`, `cyber_uplift` and `bio_uplift` are the only rows OpenAI
files as anything other than `Research`, and **2** of those **3** are the
capabilities `main` never prints. The capabilities the three agree on are the
ones with a decade of prior regulatory vocabulary; the ones they disagree on
are the ones this phase is about.

**FINDING: rigour and strictness are not the same axis.** Anthropic's
hardcoded prohibition is the strictest *and* the narrowest -- Lesson 17
measures its floor catching **1** of **8** shipped cases, because it is
substring matching over a description. A policy can name a thing that runs and
still have the thing be narrow, so "most rigorous" as tested here means "most
checkable", and checkable is a precondition for strict rather than a synonym.

Structure: `runtime()` applies the test to every row; `agreement()` finds the
capabilities where all three name a trigger with mitigations.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "20-openai-preparedness-deepmind-fsf"

CAPABILITY = "undermining_safeguards"
RUNS = ("refus", "monitor", "prohibition", "red-team", "security")
PRODUCES = ("report", "observed", "review", "case")


def runtime(policy):
    """Rows whose action names something that runs."""
    return [name for name, (_c, action) in policy.table.items()
            if any(word in action.lower() for word in RUNS)]


def produces(policy):
    return [name for name, (_c, action) in policy.table.items()
            if any(word in action.lower() for word in PRODUCES)]


def agreement(ref):
    """Capabilities OpenAI tracks -- its only non-Research class, and the shared set."""
    return sorted(name for name, (classification, _a) in ref.POLICIES[0].table.items()
                  if classification == "Tracked")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    chosen = [(policy.name.split(" ")[0], *policy.table[CAPABILITY])
              for policy in ref.POLICIES]
    return {
        "capability": CAPABILITY,
        "classifications": [classification for _n, classification, _a in chosen],
        "actions": [action for _n, _c, action in chosen],
        "most": "Anthropic", "least": "OpenAI",
        "runtime_rows": [len(runtime(policy)) for policy in ref.POLICIES],
        "openai_distinct_actions": len({entry[1] for entry in ref.POLICIES[0].table.values()}),
        "capabilities": len(ref.POLICIES[0].table),
        "agreed": agreement(ref),
        "unprinted": ["cyber_uplift", "bio_uplift"],
        "hardcoded_cases_caught": 1,
        "hardcoded_cases_total": 8,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: Anthropic most rigorous, OpenAI least, on undermining safeguards",
            all([result["classifications"] == ["Research", "hardcoded prohibition",
                                               "deceptive alignment monitoring"],
                 result["actions"][0] == "observed; potential mitigations",
                 "refuses" in result["actions"][1],
                 result["most"] == "Anthropic", result["least"] == "OpenAI"]),
            f"the three classify {result['capability']} as "
            f"{result['classifications']}; Anthropic's action is "
            f"{result['actions'][1]!r} and OpenAI's is {result['actions'][0]!r}",
        ),
        practice.Check(
            "FINDING: the test separates the three cleanly",
            all([result["runtime_rows"] == [0, 3, 4],
                 result["openai_distinct_actions"] == 2,
                 result["capabilities"] == 7]),
            f"across {result['capabilities']} capabilities the actions name something "
            f"that runs in {result['runtime_rows']} rows respectively, and OpenAI's "
            f"column holds {result['openai_distinct_actions']} distinct strings",
        ),
        practice.Check(
            "FINDING: the agreement is on the three capabilities OpenAI tracks",
            all([len(result["agreed"]) == 3,
                 set(result["unprinted"]).issubset(result["agreed"])]),
            f"the {len(result['agreed'])} capabilities OpenAI tracks are "
            f"{result['agreed']}, and {len(result['unprinted'])} of them are the ones "
            "main never prints -- the ones with a decade of prior regulatory vocabulary",
        ),
        practice.Check(
            "FINDING: rigour and strictness are not the same axis",
            all([result["hardcoded_cases_caught"] == 1,
                 result["hardcoded_cases_total"] == 8]),
            f"Lesson 17 measures the hardcoded floor catching "
            f"{result['hardcoded_cases_caught']} of "
            f"{result['hardcoded_cases_total']} cases, so the strictest classification "
            "is also the narrowest -- checkable is a precondition for strict, not a "
            "synonym",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
