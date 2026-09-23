"""Exercise 1 — two capabilities are verifiable, and two are never printed.

    Run `code/main.py`. Confirm the diff tool's output matches the policies
    for at least two capabilities you can verify against the source documents.

Reading of the exercise: the source documents are not in this repository, so
"verify" has to mean verify against something present. The lesson's own
headline block restates the classifications for exactly two capabilities, and
those two are therefore the confirmable set -- which is worth saying, because
the tables hold **21** entries and **6** of them have in-lesson corroboration.

**ANSWER: long-range autonomy and undermining safeguards, 5 of 6 fields
verbatim.** Five of the six classifications the headline prints appear in the
tables word for word; the sixth -- DeepMind on long-range autonomy -- is
summarised as `domain-folded` where the table says `folded into ML R&D / Cyber
domains`. The remaining **5** capabilities have no statement anywhere in the
lesson to check them against, so the confirmable set is the set the lesson
happens to repeat.

**FINDING: two capabilities are in every table and never printed.**
`cyber_uplift` and `bio_uplift` appear in all **3** policies -- **6** entries
-- and `main` diffs **5** of the **7** keys. They are also the two where the
three policies agree most closely, which is either a reason to omit them or a
reason not to, and the module does not say which.

**FINDING: OpenAI's table has seven capabilities and two actions.** Every
`Research` row carries the identical string `observed; potential mitigations`
and every `Tracked` row carries `Capabilities + Safeguards Reports; SAG
review` -- **2** distinct classifications and **2** distinct actions across
**7** rows. The other two policies carry **6** distinct actions each. In the
OpenAI column the action is a restatement of the classification; in the other
two it is information.

**FINDING: one capability draws three different verbs.**
`undermining_safeguards` is `Research` (observe), a `hardcoded prohibition`
(refuse) and `deceptive alignment monitoring` (detect). Observe, refuse,
detect are not points on a scale, so "which policy is stricter here" has no
answer -- and it is the capability Lesson 17 makes a hardcoded floor for and
Lesson 14 a tripwire for, which is the same disagreement one layer down.

Structure: `restated()` pulls the classifications the headline prints;
`spread()` counts distinct classifications and actions per policy.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "20-openai-preparedness-deepmind-fsf"

RESTATED = ("long_range_autonomy", "undermining_safeguards")


def headline(ref):
    """The classifications `main` prints in its summary block."""
    source = inspect.getsource(ref.main)
    return source[source.index("HEADLINE"):]


def restated(ref):
    """Which table classifications the headline reproduces verbatim, and which it does not."""
    text = headline(ref).lower()
    verbatim, paraphrased = [], []
    for capability in RESTATED:
        for policy in ref.POLICIES:
            classification = policy.table[capability][0].lower()
            target = verbatim if classification.split(" (")[0][:18] in text else paraphrased
            target.append(f"{policy.name.split(' ')[0]}:{capability}")
    return verbatim, paraphrased


def printed(ref):
    source = inspect.getsource(ref.main)
    keys = list(ref.POLICIES[0].table)
    return [key for key in keys if f'"{key}"' in source], keys


def spread(policy):
    return (len({entry[0] for entry in policy.table.values()}),
            len({entry[1] for entry in policy.table.values()}))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shown, keys = printed(ref)
    verbatim, paraphrased = restated(ref)
    return {
        "policies": len(ref.POLICIES),
        "capabilities": len(keys),
        "entries": len(keys) * len(ref.POLICIES),
        "verifiable": list(RESTATED),
        "matched": len(verbatim), "checkable": len(verbatim) + len(paraphrased),
        "paraphrased": paraphrased,
        "printed": len(shown),
        "unprinted": [key for key in keys if key not in shown],
        "spread": [list(spread(policy)) for policy in ref.POLICIES],
        "openai_actions": spread(ref.POLICIES[0])[1],
        "other_actions": [spread(policy)[1] for policy in ref.POLICIES[1:]],
        "verbs": [policy.table["undermining_safeguards"][0] for policy in ref.POLICIES],
        "same_keys": len({tuple(sorted(policy.table)) for policy in ref.POLICIES}) == 1,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: two capabilities verifiable, 5 of 6 fields matching verbatim",
            all([result["matched"] == 5, result["checkable"] == 6,
                 result["verifiable"] == list(RESTATED),
                 result["paraphrased"] == ["DeepMind:long_range_autonomy"],
                 result["entries"] == 21]),
            f"the headline restates {len(result['verifiable'])} capabilities across "
            f"{result['policies']} policies, of which {result['matched']} of "
            f"{result['checkable']} appear verbatim and "
            f"{result['paraphrased']} is summarised as 'domain-folded' -- against "
            f"{result['entries']} entries with no in-lesson corroboration at all",
        ),
        practice.Check(
            "FINDING: two capabilities are in every table and never printed",
            all([result["printed"] == 5, result["capabilities"] == 7,
                 result["unprinted"] == ["cyber_uplift", "bio_uplift"],
                 result["same_keys"]]),
            f"main diffs {result['printed']} of {result['capabilities']} keys, leaving "
            f"{result['unprinted']} in all {result['policies']} tables and out of the "
            "output",
        ),
        practice.Check(
            "FINDING: OpenAI's table has seven capabilities and two actions",
            all([result["openai_actions"] == 2, result["other_actions"] == [6, 6],
                 result["spread"][0] == [2, 2]]),
            f"OpenAI carries {result['spread'][0][0]} classifications and "
            f"{result['openai_actions']} actions across {result['capabilities']} rows, "
            f"against {result['other_actions']} actions for the other two -- its action "
            "column restates its classification",
        ),
        practice.Check(
            "FINDING: one capability draws three different verbs",
            all([len(set(result["verbs"])) == 3,
                 result["verbs"][1] == "hardcoded prohibition"]),
            f"undermining_safeguards is {result['verbs']} -- observe, refuse and detect "
            "are not points on a scale, so 'which policy is stricter here' has no "
            "answer",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
