"""Exercise 1 — the clean deployment gets no kill switch.

    Run `code/main.py`. Feed in three synthetic deployments at different
    scales. Confirm the four-risk tags match what you would expect; identify
    one case where the tool under- or over-tags.

Reading of the exercise: "one case" is an invitation to find a single edge,
and the shipped three already contain two -- one at each end of the scale. So
both are reported, plus the fourth deployment that separates them.

**ANSWER: the tags match, and the under-tag is the internal frontier agent.**
The three shipped deployments tag **0**, **3** and **4** of the four risks,
which is the ordering the exercise expects. Flipping `public_facing` to False
on the frontier agent -- an internal ML research agent with harmful
capabilities and **48** hours of autonomy -- drops it to **3** tags, because
`malicious_use` requires `handles_harmful_capabilities and public_facing`.
Insider misuse is unrepresentable in this tagger.

**FINDING: the over-tag is at the other end, and it is an under-tag in
disguise.** The clean deployment tags **0** risks, so it receives **0** of the
**14** mitigations -- including the kill switches and canary tokens of Lesson
14 -- and the output prints "check sub-levers manually", the only branch with
no list. A scoped internal agent with **1.0** hour of autonomy is exactly the
deployment most teams actually run, and the tool's advice for it is to think
of something.

**FINDING: organizational risk has four sub-levers and three fields.** The
mitigation list names safety culture, independent audit, multi-layered
defenses and information security -- **4** -- and `Deployment` carries **3**
of them. Safety culture, the one the lesson's headline calls the lever
practitioners actually pull, has no field, so the tag can never fire on it
and can never clear because of it.

**FINDING: the rogue-AI trigger is an absolute number of hours.** It fires at
`agent_autonomy_hours >= 4.0`, and the mid deployment sits exactly on it. That
is the same shape Lesson 19 priced: a threshold in hours against a horizon
that doubles roughly every seven months, so the boundary between "tagged" and
"not" moves without anyone editing it.

Structure: `DEPLOYMENTS` rebuilds the three the module ships plus the internal
variant; `tags()` reads the shipped tagger.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "22-cais-caisi-societal-risk"

SUB_LEVERS = ("safety culture", "independent audit", "multi-layered defenses",
              "information security")
AUTONOMY_GATE = 4.0


def deployments(ref):
    build = ref.Deployment
    return {
        "low": build("internal refactor helper", False, False, False,
                     True, True, True, 1.0),
        "mid": build("public coding agent", True, False, True,
                     True, True, False, 4.0),
        "high": build("autonomous ML research agent", True, True, True,
                      False, False, False, 48.0),
        "internal_high": build("the same agent, internal", False, True, True,
                               False, False, False, 48.0),
    }


def tags(ref, deployment):
    return ref.tag(deployment)


def mitigations_for(ref, names):
    return sum(len(ref.MITIGATIONS[name]) for name in names)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    built = deployments(ref)
    tagged = {name: tags(ref, deployment) for name, deployment in built.items()}
    fields = [field.name for field in dataclasses.fields(ref.Deployment)]
    org = ref.MITIGATIONS["organizational_risks"]
    return {
        "counts": [len(tagged[name]) for name in ("low", "mid", "high")],
        "high_tags": tagged["high"],
        "internal_tags": tagged["internal_high"],
        "dropped": sorted(set(tagged["high"]) - set(tagged["internal_high"])),
        "clean_tags": tagged["low"],
        "clean_mitigations": mitigations_for(ref, tagged["low"]),
        "total_mitigations": sum(len(rules) for rules in ref.MITIGATIONS.values()),
        "clean_autonomy": built["low"].agent_autonomy_hours,
        "sub_levers": list(SUB_LEVERS),
        "org_mitigations": len(org),
        "fields": fields,
        "culture_field": [name for name in fields if "culture" in name],
        "gate": AUTONOMY_GATE,
        "mid_autonomy": built["mid"].agent_autonomy_hours,
        "on_the_gate": built["mid"].agent_autonomy_hours == AUTONOMY_GATE,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 0, 3 and 4 tags, and the under-tag is the internal agent",
            all([result["counts"] == [0, 3, 4], len(result["high_tags"]) == 4,
                 len(result["internal_tags"]) == 3,
                 result["dropped"] == ["malicious_use"]]),
            f"the three shipped deployments tag {result['counts']} risks; turning the "
            f"frontier agent internal drops {result['dropped']}, because malicious_use "
            "requires public_facing -- insider misuse is unrepresentable",
        ),
        practice.Check(
            "FINDING: the clean deployment gets no mitigations at all",
            all([result["clean_tags"] == [], result["clean_mitigations"] == 0,
                 result["total_mitigations"] == 14,
                 result["clean_autonomy"] == 1.0]),
            f"a scoped agent with {result['clean_autonomy']} hour of autonomy tags "
            f"{len(result['clean_tags'])} risks and receives "
            f"{result['clean_mitigations']} of the {result['total_mitigations']} "
            "mitigations -- including no kill switch -- under a 'check sub-levers "
            "manually' line",
        ),
        practice.Check(
            "FINDING: organizational risk has four sub-levers and three fields",
            all([result["org_mitigations"] == 4, len(result["sub_levers"]) == 4,
                 result["culture_field"] == []]),
            f"the mitigation list names {result['org_mitigations']} sub-levers "
            f"{result['sub_levers']} and Deployment carries "
            f"{len(result['culture_field'])} field for safety culture -- the lever the "
            "headline calls the one practitioners pull",
        ),
        practice.Check(
            "FINDING: the rogue-AI trigger is an absolute number of hours",
            all([result["gate"] == 4.0, result["on_the_gate"],
                 result["mid_autonomy"] == 4.0]),
            f"the tag fires at {result['gate']} hours and the mid deployment sits "
            "exactly on it -- a threshold in hours against a horizon that doubles every "
            "seven months, so the boundary moves without an edit",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
