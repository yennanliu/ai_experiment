"""Exercise 5 — the depth is asserted by the caller it is meant to bound.

    Add a maximum composition depth to skill-to-skill requests and detect a
    two-skill cycle.

Reading of the exercise: the shipped policy already has `max_skill_depth`, so
"add a maximum" only means something if the existing one does not bound
anything -- and it does not, because `depth` is a field the caller fills in.
The cycle half is the same defect seen from the other side: `caller_name`
holds one name, which is exactly enough to catch a self-call and one name
short of catching anything else.

**ANSWER: carry the chain, derive the depth from it, and both checks become
possible.** An A -> B -> A request is refused as a cycle and an
A -> B -> C -> D request is refused at depth **4** against a limit of **3**.
The shipped adapter allows the cycle, refuses the fourth hop only while it
declares `depth=4`, and allows the identical call the moment it declares
`depth=1`. Depth stops being a claim and becomes `len(chain)`.

**FINDING: the depth is asserted by the caller it is meant to bound.**
`InvocationRequest.depth` is an integer the requester supplies, so **5**
nested calls each declaring `depth=1` are all allowed and the limit of
**2** is never approached. A bound the bounded party writes is not a bound.

**FINDING: one name catches a self-cycle and nothing longer.**
`request.caller_name == skill.name` refuses A -> A, and A -> B -> A passes
every check the adapter has, because at that point the adapter is looking at
a request from B to A and has never heard of the first hop.

**FINDING: the dataclass default is a value the policy rejects.**
`depth` defaults to **0** and the adapter requires `depth >= 1`, so a skill
request built the obvious way is denied for a reason that is about the
default rather than about composition. The two halves of the contract
disagree about what "no depth given" means.

Structure: `Chained` adds the one field the check needs; `ChainAdapter`
derives depth and cycles from it, so both answers come from the same data.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "25-skill-invocation-and-routing"
A, B, C, D = "release-readiness", "incident-triage", "deploy-service", "changelog-entry"
LIMIT = 3


def build(ref):
    class Chained(ref.InvocationRequest):
        """The shipped request plus the one field the checks actually need."""

        def __init__(self, *args, chain=(), **kwargs):
            super().__init__(*args, **kwargs)
            object.__setattr__(self, "chain", tuple(chain))

    class ChainAdapter(ref.CorePolicyAdapter):
        """Depth is len(chain) and a cycle is a repeat, so neither is the caller's word."""

        def allows(self, skill, actor, request=None):
            if actor is not ref.Actor.SKILL:
                return super().allows(skill, actor, request)
            chain = tuple(getattr(request, "chain", ()) or ())
            if not chain:
                return False, "skill composition requires a call chain"
            if skill.name in chain:
                return False, f"cycle: {' -> '.join(chain)} -> {skill.name}"
            if len(chain) + 1 > self.policy.max_skill_depth:
                return False, f"depth {len(chain) + 1} over limit {self.policy.max_skill_depth}"
            if chain[-1] not in self.policy.skill_caller_allowlist:
                return False, "skill caller allowlist"
            return True, f"chain depth {len(chain) + 1} within {self.policy.max_skill_depth}"

    return Chained, ChainAdapter


def skills(ref):
    return tuple(ref.SkillMetadata(name, f"Handle the {name.replace('-', ' ')} workflow.")
                 for name in (A, B, C, D))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    Chained, ChainAdapter = build(ref)
    policy = ref.InvocationPolicy(allow_skill=True, skill_caller_allowlist=(A, B, C, D),
                                  max_skill_depth=LIMIT, model_threshold=0.15)
    shipped, chained = ref.CorePolicyAdapter(policy), ChainAdapter(policy)
    catalog = skills(ref)

    def route(adapter, target, caller, depth, chain=()):
        request = Chained(ref.Actor.SKILL, "compose", explicit_name=target,
                          caller_name=caller, depth=depth, chain=chain)
        decision = ref.route_request(catalog, request, adapter)
        return decision.activated, decision.reason

    cycle_new = route(chained, A, B, 2, chain=(A, B))
    cycle_old = route(shipped, A, B, 2)
    deep_new = route(chained, D, C, 4, chain=(A, B, C))
    deep_old = route(shipped, D, C, 4)
    deep_forged = route(shipped, D, C, 1)  # the same fourth hop, declaring the first
    forged = [route(shipped, D, C, 1) for _ in range(5)]
    self_call = route(shipped, A, A, 1)
    defaulted = ref.route_request(catalog, ref.InvocationRequest(
        ref.Actor.SKILL, "compose", explicit_name=A, caller_name=B), shipped)
    return {
        "cycle_new": cycle_new, "cycle_old": cycle_old,
        "deep_new": deep_new, "deep_old": deep_old, "deep_forged": deep_forged,
        "limit": LIMIT, "forged": [ok for ok, _ in forged], "forged_calls": len(forged),
        "policy_depth": policy.max_skill_depth, "self_call": self_call,
        "default_depth": ref.InvocationRequest(ref.Actor.SKILL, "").depth,
        "defaulted": (defaulted.activated, defaulted.reason),
        "request_fields": [name for name in
                           vars(ref.InvocationRequest)["__dataclass_fields__"]],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: carry the chain, derive the depth from it, and both checks work",
            all([result["cycle_new"][0] is False,
                 result["cycle_new"][1].startswith("blocked by cycle:"),
                 result["deep_new"][0] is False, "depth 4" in result["deep_new"][1],
                 result["cycle_old"][0] is True, result["deep_old"][0] is False,
                 result["deep_forged"][0] is True]),
            f"an A -> B -> A request is refused with {result['cycle_new'][1]!r} and a "
            f"four-deep chain with {result['deep_new'][1]!r} against a limit of "
            f"{result['limit']}. The shipped adapter allows the cycle, refuses the fourth "
            f"hop only while it declares depth=4, and allows the identical call the moment "
            "it declares depth=1 -- depth stops being a claim and becomes len(chain)",
        ),
        practice.Check(
            "FINDING: the depth is asserted by the caller it is meant to bound",
            all([result["forged"] == [True] * 5, result["forged_calls"] == 5,
                 result["policy_depth"] == LIMIT, "depth" in result["request_fields"]]),
            f"InvocationRequest.depth is an integer the requester supplies, so "
            f"{result['forged_calls']} nested calls each declaring depth=1 are all allowed "
            f"and the limit of {result['policy_depth']} is never approached. A bound the "
            "bounded party writes is not a bound",
        ),
        practice.Check(
            "FINDING: one name catches a self-cycle and nothing longer",
            all([result["self_call"][0] is False,
                 "self-cycle" in result["self_call"][1], result["cycle_old"][0] is True,
                 "caller_name" in result["request_fields"]]),
            f"caller_name == skill.name refuses A -> A with {result['self_call'][1]!r}, and "
            f"A -> B -> A is allowed ({result['cycle_old'][1]!r}) because by then the "
            "adapter is looking at a request from B to A and has never heard of the first "
            "hop. One name is exactly one hop of memory",
        ),
        practice.Check(
            "FINDING: the dataclass default is a value the policy rejects",
            all([result["default_depth"] == 0, result["defaulted"][0] is False,
                 "depth limit" in result["defaulted"][1]]),
            f"depth defaults to {result['default_depth']} and the adapter requires depth "
            f">= 1, so a skill request built the obvious way is denied with "
            f"{result['defaulted'][1]!r} -- a refusal about the default rather than about "
            "composition. The two halves disagree on what 'no depth given' means",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
