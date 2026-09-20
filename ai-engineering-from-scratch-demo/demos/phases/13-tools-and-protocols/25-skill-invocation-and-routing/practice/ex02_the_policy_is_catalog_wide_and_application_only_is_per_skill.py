"""Exercise 2 — the policy is catalog-wide and application-only is per-skill.

    Add application-only activation to `CorePolicyAdapter`. Prove that human
    and model callers remain denied.

Reading of the exercise: the shipped policy can already deny humans and
models, so the first question is why that is not the answer -- and it is
because `InvocationPolicy` is one object for the whole catalog while
"application-only" is a property of one skill. The proof is therefore run
against a three-skill catalog, where the flag approach and the per-skill
approach give visibly different answers.

**ANSWER: a per-skill rule denies every non-application actor and keeps the
rest of the catalog reachable.** The application-only skill is refused to all
**5** other actors and activated for `APPLICATION`; the other two skills stay
open to humans. Clearing `allow_human` instead denies **3** of **3** skills
to humans to protect **1**.

**FINDING: the exercise names two actors and there are six.** Denying human
and model leaves `AGENT`, `HARNESS` and `SKILL` -- and `AGENT` is a model on
a different branch, so "human and model remain denied" is satisfied by a rule
that still lets an agent in. The rule has to be written as an allowlist of
one actor, not a denylist of two.

**FINDING: the shipped application branch already requires two things.**
`allow_application and skill.name in application_allowlist` means a skill
absent from the allowlist is denied even when applications are permitted, so
the allowlist is the per-skill half that already exists. Application-only is
the same idea pointed the other way, and there is nowhere in
`SkillMetadata` to put it.

**FINDING: implicit routing hides the refusal it acted on.** The three
explicit actors are told `blocked by payment-refund is application-only`;
the model and the agent are told `best match did not meet the host
threshold`, because eligibility filtering runs before scoring and the
filtered-out skill is simply absent. Shrink the catalog to that one skill
and the same request finally names it -- the refusal is only reportable when
there is nothing else to report.

Structure: `ApplicationOnly` is the adapter, `decide()` routes one actor at
it, and `across()` is the catalog-wide comparison the flag approach loses.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "25-skill-invocation-and-routing"
LOCKED = "payment-refund"
OPEN = ("incident-triage", "release-notes")


def build(ref):
    class ApplicationOnly(ref.CorePolicyAdapter):
        """Application-only is a property of a skill, so it is keyed by skill name."""

        def __init__(self, policy, names):
            super().__init__(policy)
            self.names = frozenset(names)

        def allows(self, skill, actor, request=None):
            if skill.name in self.names and actor is not ref.Actor.APPLICATION:
                return False, f"{skill.name} is application-only"
            return super().allows(skill, actor, request)

    return ApplicationOnly


def skills(ref):
    return (ref.SkillMetadata(LOCKED, "Refund a captured payment for a support case."),
            ref.SkillMetadata(OPEN[0], "Triage an incident timeline and separate evidence."),
            ref.SkillMetadata(OPEN[1], "Draft release notes from merged pull requests."))


def request_for(ref, actor, name):
    if actor in (ref.Actor.MODEL, ref.Actor.AGENT):
        return ref.InvocationRequest(actor, "refund a captured payment for this support case")
    return ref.InvocationRequest(actor, "", explicit_name=name,
                                 caller_name="incident-triage", depth=1)


def decide(ref, adapter, actor, name):
    return ref.route_request(skills(ref), request_for(ref, actor, name), adapter)


def across(ref, adapter, actor):
    """Which skills this actor can still reach."""
    return sorted(skill.name for skill in skills(ref)
                  if decide(ref, adapter, actor, skill.name).activated)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    actors = [ref.Actor.HUMAN, ref.Actor.MODEL, ref.Actor.AGENT, ref.Actor.HARNESS,
              ref.Actor.SKILL, ref.Actor.APPLICATION]
    policy = ref.InvocationPolicy(
        model_threshold=0.15, allow_skill=True, skill_caller_allowlist=("incident-triage",),
        harness_allowlist=(LOCKED, *OPEN), application_allowlist=(LOCKED, *OPEN))
    adapter = build(ref)(policy, {LOCKED})
    per_skill = {actor.value: decide(ref, adapter, actor, LOCKED).activated
                 for actor in actors}

    flagged = ref.CorePolicyAdapter(ref.InvocationPolicy(
        allow_human=False, allow_model=False, model_threshold=0.15,
        harness_allowlist=(LOCKED, *OPEN), application_allowlist=(LOCKED, *OPEN)))
    naive = ref.CorePolicyAdapter(policy)
    reasons = {actor.value: decide(ref, adapter, actor, LOCKED).reason for actor in actors}
    shipped_reasons = {actor.value: decide(ref, flagged, actor, LOCKED).reason
                       for actor in actors if actor is not ref.Actor.APPLICATION}
    solo = ref.route_request((skills(ref)[0],), request_for(ref, ref.Actor.MODEL, LOCKED),
                             adapter).reason
    return {
        "per_skill": per_skill,
        "denied": sorted(name for name, ok in per_skill.items() if not ok),
        "reachable_per_skill": across(ref, adapter, ref.Actor.HUMAN),
        "reachable_flagged": across(ref, flagged, ref.Actor.HUMAN),
        "reachable_open": across(ref, naive, ref.Actor.HUMAN),
        "actors": len(actors),
        "agent_without_rule": decide(ref, naive, ref.Actor.AGENT, LOCKED).activated,
        "agent_with_rule": per_skill["agent"],
        "unlisted": ref.CorePolicyAdapter(ref.InvocationPolicy(
            model_threshold=0.15, application_allowlist=())).allows(
                skills(ref)[0], ref.Actor.APPLICATION)[0],
        "reasons": reasons, "shipped_reasons": shipped_reasons,
        "distinct_reasons": len(set(r for a, r in reasons.items() if a != "application")),
        "names_the_skill": sum(LOCKED in reason for reason in reasons.values()),
        "shipped_names_it": sum(LOCKED in reason for reason in shipped_reasons.values()),
        "shipped_distinct": len(set(shipped_reasons.values())), "solo": solo,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a per-skill rule denies every non-application actor, catalog intact",
            all([result["per_skill"]["application"] is True,
                 result["denied"] == ["agent", "harness", "human", "model", "skill"],
                 result["reachable_per_skill"] == sorted(OPEN),
                 result["reachable_flagged"] == [],
                 result["reachable_open"] == sorted((LOCKED, *OPEN))]),
            f"the application-only skill is refused to {result['denied']} and activated for "
            f"application, while humans keep {result['reachable_per_skill']}. Clearing "
            f"allow_human instead leaves humans {result['reachable_flagged']} out of "
            f"{result['reachable_open']} -- three skills denied to protect one",
        ),
        practice.Check(
            "FINDING: the exercise names two actors and there are six",
            all([result["actors"] == 6, result["agent_without_rule"],
                 result["agent_with_rule"] is False]),
            f"denying human and model leaves agent, harness and skill of the "
            f"{result['actors']}, and an agent is a model on a different adapter branch: "
            "without the per-skill rule it activates the refund skill. 'Human and model "
            "remain denied' is satisfiable by a rule that still lets an agent in, so the "
            "rule has to allowlist one actor rather than denylist two",
        ),
        practice.Check(
            "FINDING: the shipped application branch already requires two things",
            all([not result["unlisted"], result["per_skill"]["application"]]),
            "allow_application and skill.name in application_allowlist means a skill absent "
            "from the allowlist is denied even when applications are permitted, so the "
            "allowlist is the per-skill half that already exists. Application-only is the "
            "same idea pointed the other way, and SkillMetadata has nowhere to put it",
        ),
        practice.Check(
            "FINDING: implicit routing hides the refusal it acted on",
            all([result["names_the_skill"] == 3, result["distinct_reasons"] == 2,
                 result["reasons"]["model"] == "best match did not meet the host threshold",
                 result["reasons"]["model"] == result["reasons"]["agent"],
                 LOCKED in result["solo"], result["shipped_names_it"] == 0]),
            f"the three explicit actors are told {result['reasons']['human']!r}; the model "
            f"and the agent are told {result['reasons']['model']!r}, because eligibility "
            f"filtering runs before scoring and the filtered-out skill is simply absent. "
            f"Shrink the catalog to that one skill and the same request answers "
            f"{result['solo']!r} -- the refusal is reportable only when nothing else is",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
