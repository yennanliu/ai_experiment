"""Exercise 5 — split the lever that cannot tell three situations apart.

    Sketch a 2028 version of the four-risk framework that reflects one year of
    additional capability and one year of additional deployment experience.
    What would you add, remove, or regroup?

Reading of the exercise: a sketch is cheap, so the proposal is constrained to
changes the 2026 tagger can be shown to need -- each one justified by a pair
of deployments the current framework tags identically and a reader would not.

**ANSWER: split one category, add one field, and make one threshold
relative.** Split `organizational_risks` into *controls present* and *controls
independent*, because the current single boolean gives **3** distinguishable
situations **1** bit. Add a `safety_culture` field, the **1** of **4**
sub-levers with no representation. And replace the **4.0**-hour rogue-AI gate
with a ratio against the current frontier horizon, for the reason Lesson 19
prices. Net: **5** categories, **9** fields.

**FINDING: the split is justified by a pair the 2026 tagger cannot
separate.** A deployment with no gates at all and one whose gates are all
editable by the agent they gate both score `independent_audit=False` and
receive the identical **4**-item mitigation list. Under the split the first
fails both new categories and the second fails **1**, which is the difference
between "build a gate" and "move the one you have".

**FINDING: the removal is `malicious_use`'s public-facing conjunct.** It
requires `handles_harmful_capabilities and public_facing`, so an internal
frontier agent drops the tag -- measured in exercise 1. A 2028 framework
written after a year of *deployment* experience should key the tag on the
capability and treat exposure as a severity multiplier, which is **1** fewer
conjunct and **1** more field.

**FINDING: what should not change is the count of levers a practitioner
controls.** Of the four risks, **1** -- organizational -- is internal to the
org, and splitting it in two does not make two more of them controllable; it
makes the one controllable lever legible. The framework's value is the
asymmetry it names, and a 2028 revision that grew the structural categories
would be measuring the weather.

Structure: `PROPOSAL` is the 2028 shape; `indistinguishable()` builds the pair
the current tagger collapses.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "22-cais-caisi-societal-risk"

PROPOSAL = ("malicious_use", "ai_races", "controls_present", "controls_independent",
            "rogue_ais")
ADDED_FIELDS = ("safety_culture",)


def indistinguishable(ref):
    """Two deployments the 2026 tagger scores identically and a reader would not."""
    build = ref.Deployment
    no_gates = build("no gates at all", True, False, True, False, False, False, 8.0)
    editable = build("gates the agent can edit", True, False, True, False, True, True, 8.0)
    return no_gates, editable


def split(deployment, gates_exist):
    """The proposed pair of categories, in place of one boolean."""
    return {
        "controls_present": gates_exist,
        "controls_independent": deployment.independent_audit,
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    no_gates, editable = indistinguishable(ref)
    current = [ref.tag(no_gates), ref.tag(editable)]
    proposed = [split(no_gates, False), split(editable, True)]
    fields = list(ref.Deployment.__dataclass_fields__)
    return {
        "categories_2026": len(ref.MITIGATIONS),
        "categories_2028": len(PROPOSAL),
        "fields_2026": len(fields),
        "fields_2028": len(fields) + len(ADDED_FIELDS),
        "added": list(ADDED_FIELDS),
        "current_tags": [sorted(tags) for tags in current],
        "current_identical": current[0] == current[1],
        "org_mitigations": len(ref.MITIGATIONS["organizational_risks"]),
        "proposed_failures": [sum(1 for value in row.values() if not value)
                              for row in proposed],
        "situations": 3, "bits": 1,
        "malicious_conjuncts": 2,
        "culture_field_now": [name for name in fields if "culture" in name],
        "internal_lever": "organizational_risks",
        "internal_count": 1,
        "structural_count": len(ref.MITIGATIONS) - 1,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: five categories, nine fields, one relative threshold",
            all([result["categories_2026"] == 4, result["categories_2028"] == 5,
                 result["fields_2026"] == 8, result["fields_2028"] == 9,
                 result["added"] == ["safety_culture"]]),
            f"{result['categories_2026']} categories become "
            f"{result['categories_2028']} and {result['fields_2026']} fields become "
            f"{result['fields_2028']}, adding {result['added']} -- the sub-lever with "
            "no representation",
        ),
        practice.Check(
            "FINDING: the split is justified by a pair the tagger collapses",
            all([result["current_identical"], result["proposed_failures"] == [2, 1],
                 result["org_mitigations"] == 4]),
            f"a deployment with no gates and one whose gates the agent can edit tag "
            f"identically -- {result['current_tags'][0]} -- and receive the same "
            f"{result['org_mitigations']}-item list, while the split fails them "
            f"{result['proposed_failures']} times",
        ),
        practice.Check(
            "FINDING: the removal is the public-facing conjunct",
            all([result["malicious_conjuncts"] == 2,
                 result["culture_field_now"] == []]),
            f"malicious_use is a conjunction of {result['malicious_conjuncts']} "
            "booleans, so an internal frontier agent drops the tag; keying on the "
            "capability and treating exposure as a severity multiplier is one fewer "
            "conjunct",
        ),
        practice.Check(
            "FINDING: what should not change is the count of controllable levers",
            all([result["internal_count"] == 1, result["structural_count"] == 3,
                 result["internal_lever"] == "organizational_risks"]),
            f"{result['internal_count']} of the {result['categories_2026']} risks is "
            f"internal to the org and {result['structural_count']} are structural; "
            "splitting the internal one makes it legible rather than making two more of "
            "them controllable",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
