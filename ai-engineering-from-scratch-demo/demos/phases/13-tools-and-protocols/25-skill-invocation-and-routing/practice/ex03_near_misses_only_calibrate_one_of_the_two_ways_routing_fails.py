"""Exercise 3 — near misses only calibrate one of the two ways routing fails.

    Write ten near misses for a deployment skill. Each prompt must share
    vocabulary with the skill while belonging to a different workflow.

Reading of the exercise: "shares vocabulary" and "different workflow" are
two conditions that have to be checked separately, so the set is validated
for overlap mechanically and for workflow by construction -- ten prompts,
ten named workflows, no two the same. The set is then used for the thing a
near-miss set is for, which is locating a threshold, and that is where the
shipped demo turns out to be on the wrong side of it.

**ANSWER: ten near misses, ten workflows, and they move the threshold from
0.15 to 0.30.** Every prompt shares at least **2** tokens with the skill; at
the demo's **0.15** threshold **6** of **10** falsely activate. The highest
near miss scores **0.2667** and the lowest true positive **0.3846**, so a
**0.30** threshold takes the set to **0** of **10** false and **3** of **3**
true.

**FINDING: padding a near miss makes it pass.** `relevance_score` divides by
the union, so appending eleven unrelated words to the hardest prompt drops it
from **0.1111** to **0.0741**. The eval then passes because the prompt was
long rather than because it was different, which means a near-miss set has
to be length-controlled or it measures the wrong thing.

**FINDING: the skill's name is scored as part of its description.**
`relevance_score` tokenizes `f"{name} {description}"`, so renaming
`deploy-service` to `ship-service` moves every score with no description
change -- the top near miss to **0.25** and the top true positive to
**0.5833**. A rename is a routing change.

**FINDING: near misses only calibrate one of the two ways routing fails.**
They are high-overlap and wrong. The other direction is low-overlap and
right: `what happened during the outage last night` is exactly what
`incident-triage` exists for and scores **0.0**, which no threshold admits.
A set of ten near misses and no far hits tunes one boundary and leaves the
other unmeasured.

Structure: `sweep()` counts both error kinds at a threshold, so "the demo is
mis-tuned" and "0.30 is right" are the same measurement read twice.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "25-skill-invocation-and-routing"
NAME = "deploy-service"
DESCRIPTION = "Deploy a merged release branch of a service to the production environment."
NEAR_MISSES = {
    "rollback": "roll back the production deployment from last night",
    "documentation": "write the deploy runbook section about the release branch",
    "code-review": "review this pull request that merged into the release branch",
    "performance": "explain why the production database migration is slow",
    "access-request": "grant me production access for the on-call rotation",
    "finance": "estimate the cost of the production environment this month",
    "local-dev": "deploy a test fixture into the local development environment",
    "changelog": "summarize what changed in the service since the last release",
    "paging": "page the on-call engineer about a production incident",
    "refactor": "rename the service in the deployment manifest",
}
TRUE_POSITIVES = ("deploy the merged release branch to production",
                  "ship the service to the production environment now",
                  "release branch is merged, deploy the service")
PADDING = " and also please consider the weather forecast tomorrow morning here"
FAR_HIT = "what happened during the outage last night"


def scores(ref, skill, prompts):
    return [round(ref.relevance_score(prompt, skill), 4) for prompt in prompts]


def sweep(ref, skill, threshold):
    """Both error kinds at one threshold, counted on the same set."""
    near = scores(ref, skill, NEAR_MISSES.values())
    true = scores(ref, skill, TRUE_POSITIVES)
    return {"false_activations": sum(score >= threshold for score in near),
            "true_activations": sum(score >= threshold for score in true),
            "near": near, "true": true}


def shared_tokens(ref, skill, prompt):
    return len(ref._tokens(prompt) & ref._tokens(f"{skill.name} {skill.description}"))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    skill = ref.SkillMetadata(NAME, DESCRIPTION)
    renamed = ref.SkillMetadata("ship-service", DESCRIPTION)
    triage = ref.SkillMetadata("incident-triage",
                               "Triage an incident timeline and separate evidence "
                               "from hypotheses.")
    demo, tuned = sweep(ref, skill, 0.15), sweep(ref, skill, 0.30)
    hardest = min(NEAR_MISSES.values(), key=lambda prompt: ref.relevance_score(prompt, skill))
    routed = ref.route_request(
        (skill,), ref.InvocationRequest(ref.Actor.MODEL, list(NEAR_MISSES.values())[1]),
        ref.CorePolicyAdapter(ref.InvocationPolicy(model_threshold=0.15)))
    return {
        "prompts": len(NEAR_MISSES), "workflows": len(set(NEAR_MISSES)),
        "min_shared": min(shared_tokens(ref, skill, prompt)
                          for prompt in NEAR_MISSES.values()),
        "demo": demo, "tuned": tuned,
        "top_near": max(demo["near"]), "low_true": min(demo["true"]),
        "routed_activated": routed.activated, "routed_skill": routed.skill_name,
        "hardest": round(ref.relevance_score(hardest, skill), 4),
        "padded": round(ref.relevance_score(hardest + PADDING, skill), 4),
        "renamed_near": max(scores(ref, renamed, NEAR_MISSES.values())),
        "renamed_true": max(scores(ref, renamed, TRUE_POSITIVES)),
        "top_true": max(demo["true"]),
        "far_hit": round(ref.relevance_score(FAR_HIT, triage), 4),
        "far_hit_shared": shared_tokens(ref, triage, FAR_HIT),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: ten near misses, ten workflows, and they move the threshold to 0.30",
            all([result["prompts"] == 10, result["workflows"] == 10,
                 result["min_shared"] == 2,
                 result["demo"]["false_activations"] == 6,
                 result["demo"]["true_activations"] == 3,
                 result["tuned"]["false_activations"] == 0,
                 result["tuned"]["true_activations"] == 3,
                 result["top_near"] == 0.2667, result["low_true"] == 0.3846]),
            f"{result['prompts']} prompts across {result['workflows']} workflows each share "
            f"at least {result['min_shared']} tokens with the skill. At the demo's 0.15 "
            f"threshold {result['demo']['false_activations']} of {result['prompts']} "
            f"falsely activate; the top near miss is {result['top_near']} and the lowest "
            f"true positive {result['low_true']}, so 0.30 gives "
            f"{result['tuned']['false_activations']} false and "
            f"{result['tuned']['true_activations']} true",
        ),
        practice.Check(
            "FINDING: padding a near miss makes it pass",
            all([result["padded"] < result["hardest"], result["hardest"] == 0.1111,
                 result["padded"] == 0.0741, result["routed_activated"],
                 result["routed_skill"] == NAME]),
            f"relevance_score divides by the union, so eleven unrelated words drop the "
            f"hardest prompt from {result['hardest']} to {result['padded']}. The eval then "
            f"passes because the prompt was long rather than different -- and the "
            f"unpadded runbook prompt really does route to {result['routed_skill']!r} at "
            "the demo threshold, so the failure is not hypothetical",
        ),
        practice.Check(
            "FINDING: the skill's name is scored as part of its description",
            all([result["renamed_near"] == 0.25, result["renamed_true"] == 0.5833,
                 result["top_near"] == 0.2667, result["top_true"] == 0.6364]),
            f"relevance_score tokenizes name and description together, so renaming "
            f"{NAME!r} to 'ship-service' moves the top near miss from "
            f"{result['top_near']} to {result['renamed_near']} and the top true positive "
            f"from {result['top_true']} to {result['renamed_true']}, with the description "
            "untouched. A rename is a routing change",
        ),
        practice.Check(
            "FINDING: near misses only calibrate one of the two ways routing fails",
            all([result["far_hit"] == 0.0, result["far_hit_shared"] == 0,
                 result["tuned"]["false_activations"] == 0]),
            f"near misses are high-overlap and wrong; the other direction is low-overlap "
            f"and right. {FAR_HIT!r} is what incident-triage exists for and shares "
            f"{result['far_hit_shared']} tokens with it, scoring {result['far_hit']} -- no "
            "threshold admits it. Ten near misses and no far hits tunes one boundary and "
            "leaves the other unmeasured",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
