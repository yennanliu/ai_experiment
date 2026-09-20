"""Exercise 6 — the extension adapter can only subtract.

    Run the same labeled set through core and extension adapters. Explain
    every changed decision.

Reading of the exercise: "explain every changed decision" is a closed task
only if the set of possible changes is known, so the diff is taken first and
then checked against the one direction the adapter is capable of. It turns
out there is only one direction, which makes the explanation short and the
next question -- what counts as `false` -- the interesting one.

**ANSWER: 12 requests, 3 changed decisions -- one denial and two silent
re-routes.** `release-notes` for `HUMAN` becomes a deny
(`user-invocable=false`); the `MODEL` and `AGENT` requests that chose
`archive-cleanup` now choose `incident-triage` and stay activated
(`disable-model-invocation=true`). **0** decisions change the other way and
the other **9** do not move.

**FINDING: subtracting eligibility from an implicit route re-routes, it does
not deny.** `ExtensionPolicyAdapter.allows` calls `super().allows` and
returns early when the core denied, so it can only subtract -- but an
implicit route with one candidate removed still has candidates. The caller
is handed the second-best skill and the ordinary `model activation policy`
reason, and is never told which skill was excluded.

**FINDING: the false-ness test recognises four spellings and fails open on
the rest.** `_extension_false` accepts `False` and the strings `false`,
`False` and `FALSE`; `0`, `"no"`, `"off"`, `"disabled"` and `None` are all
treated as absent, so a plausible typo silently re-enables human invocation.
**4** of **9** values deny.

**FINDING: one field governs two actors and another governs one.**
`disable-model-invocation` is checked for `MODEL` and `AGENT` together,
while `user-invocable=false` is checked for `HUMAN` only -- so an
`APPLICATION` or `HARNESS` caller reaches a skill its own metadata says is
not user-invocable. The metadata names an actor and the check names a
branch, and the two are not the same partition.

Structure: `decisions()` runs the labeled set through one adapter, so the
diff is a comparison of two identical calls differing only in the adapter.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "25-skill-invocation-and-routing"
NOTES, TRIAGE, CLEANUP = "release-notes", "incident-triage", "archive-cleanup"
FALSEY = (False, "false", "False", "FALSE", 0, "no", "off", "disabled", None)


def skills(ref):
    return (ref.SkillMetadata(NOTES, "Draft release notes from merged pull requests.",
                              {"user-invocable": False}),
            ref.SkillMetadata(TRIAGE, "Triage an incident timeline and separate evidence."),
            ref.SkillMetadata(CLEANUP, "Archive and clean up stale incident timeline data.",
                              {"disable-model-invocation": True}))


def labeled(ref):
    """One request per (actor, target) pair worth distinguishing."""
    explicit = [(ref.Actor.HUMAN, NOTES), (ref.Actor.HUMAN, TRIAGE),
                (ref.Actor.HUMAN, CLEANUP), (ref.Actor.APPLICATION, NOTES),
                (ref.Actor.HARNESS, NOTES), (ref.Actor.HARNESS, CLEANUP),
                (ref.Actor.SKILL, NOTES), (ref.Actor.SKILL, CLEANUP)]
    requests = [ref.InvocationRequest(actor, "", explicit_name=name,
                                      caller_name=TRIAGE, depth=1)
                for actor, name in explicit]
    for actor in (ref.Actor.MODEL, ref.Actor.AGENT):
        requests.append(ref.InvocationRequest(actor, "archive stale incident timeline data"))
        requests.append(ref.InvocationRequest(actor, "draft release notes from pull requests"))
    return requests


def decisions(ref, adapter):
    return [ref.route_request(skills(ref), request, adapter) for request in labeled(ref)]


def label(request, decision):
    actor = request.actor.value
    return f"{actor}:{request.explicit_name or decision.skill_name}"


def human_denied(ref, policy, value):
    skill = ref.SkillMetadata(NOTES, "Draft release notes.", {"user-invocable": value})
    return not ref.ExtensionPolicyAdapter(policy).allows(skill, ref.Actor.HUMAN)[0]


def diff(requests, core, extended):
    """Every decision whose activation or chosen skill moved between adapters."""
    return [(label(request, before), before, after)
            for request, before, after in zip(requests, core, extended)
            if (before.activated, before.skill_name) != (after.activated, after.skill_name)]


def moves(changed):
    """The skill pairs for changes that stayed activated -- the silent half."""
    return [(before.skill_name, after.skill_name) for _, before, after in changed
            if before.activated and after.activated]


def tally(changed):
    was = [before.activated for _, before, _ in changed]
    now = [after.activated for _, _, after in changed]
    moved = moves(changed)
    return {"grants": sum(now) - len(moved), "revocations": sum(was) - len(moved),
            "reroutes": len(moved), "rerouted_from": sorted({old for old, _ in moved}),
            "rerouted_to": sorted({new for _, new in moved}),
            "changed": [name for name, _, _ in changed],
            "reasons": sorted({after.reason for _, _, after in changed})}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    policy = ref.InvocationPolicy(
        model_threshold=0.15, allow_skill=True, skill_caller_allowlist=(TRIAGE,),
        harness_allowlist=(NOTES, CLEANUP), application_allowlist=(NOTES, CLEANUP))
    requests = labeled(ref)
    changed = diff(requests, decisions(ref, ref.CorePolicyAdapter(policy)),
                   decisions(ref, ref.ExtensionPolicyAdapter(policy)))
    locked = ref.SkillMetadata(NOTES, "Draft release notes.", {"user-invocable": False})
    extension = ref.ExtensionPolicyAdapter(policy)
    return {
        "requests": len(requests), "unchanged": len(requests) - len(changed),
        "denies": [value for value in FALSEY if human_denied(ref, policy, value)],
        "values": len(FALSEY),
        "human_locked": extension.allows(locked, ref.Actor.HUMAN)[0],
        "application_locked": extension.allows(locked, ref.Actor.APPLICATION)[0],
        "harness_locked": extension.allows(locked, ref.Actor.HARNESS)[0],
        "model_pair": [extension.allows(skills(ref)[2], actor)[0]
                       for actor in (ref.Actor.MODEL, ref.Actor.AGENT)],
        **tally(changed),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: twelve requests, three changes -- one denial, two silent re-routes",
            all([result["requests"] == 12, len(result["changed"]) == 3,
                 result["revocations"] == 1, result["reroutes"] == 2,
                 result["grants"] == 0, result["unchanged"] == 9,
                 sorted(result["changed"]) == ["agent:archive-cleanup",
                                               f"human:{NOTES}", "model:archive-cleanup"]]),
            f"{result['requests']} requests through both adapters differ on "
            f"{sorted(result['changed'])}: {result['revocations']} allow becomes a deny "
            f"(user-invocable=false), {result['reroutes']} stay activated and change skill, "
            f"and {result['grants']} go the other way. Reasons: {result['reasons']}. The "
            f"other {result['unchanged']} do not move",
        ),
        practice.Check(
            "FINDING: subtracting eligibility from an implicit route re-routes, it does not deny",
            all([result["grants"] == 0, result["reroutes"] == 2,
                 result["rerouted_from"] == [CLEANUP], result["rerouted_to"] == [TRIAGE],
                 result["reasons"].count("model activation policy") == 1,
                 result["reasons"].count("agent activation policy") == 1]),
            f"the adapter can only subtract -- it returns early when the core denied -- but "
            f"an implicit route with one candidate removed still has candidates, so the "
            f"model and agent move from {result['rerouted_from']} to "
            f"{result['rerouted_to']} and stay activated. The caller is handed the "
            "second-best skill and the ordinary eligibility reason, never the exclusion",
        ),
        practice.Check(
            "FINDING: the false-ness test recognises four spellings and fails open on the rest",
            all([result["denies"] == [False, "false", "False", "FALSE"],
                 result["values"] == 9]),
            f"_extension_false accepts {result['denies']} and treats the other "
            f"{result['values'] - len(result['denies'])} of {result['values']} -- 0, 'no', "
            "'off', 'disabled', None -- as absent, so a plausible typo silently re-enables "
            "human invocation. Fail-open is the wrong default for a field whose only job is "
            "to deny",
        ),
        practice.Check(
            "FINDING: one field governs two actors and another governs one",
            all([result["human_locked"] is False, result["application_locked"] is True,
                 result["harness_locked"] is True, result["model_pair"] == [False, False]]),
            "disable-model-invocation is checked for MODEL and AGENT together, while "
            "user-invocable=false is checked for HUMAN only -- so an APPLICATION or HARNESS "
            "caller reaches a skill whose own metadata says it is not user-invocable. The "
            "metadata names an actor and the check names a branch, and those differ",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
