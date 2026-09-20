"""Exercise 2 — one malformed allowlist entry denies every origin.

    Add an origin policy that permits `https://registry.example.test` on
    port 443, separately permits port 8443, and rejects redirects to every
    undeclared origin.

Reading of the exercise: "separately permits" is the instruction that says
the port is part of the origin rather than a detail of it, and "rejects
redirects" is a requirement about a sequence, which `ActionRequest` cannot
hold -- it has one `url`. So the policy is two entries and the redirect rule
is a loop that re-reviews every hop, because a chain is only safe if each
link is.

**ANSWER: two entries, and every hop reviewed as if it were the first.**
`https://registry.example.test` alone permits `:443` and denies `:8443`;
adding the second entry permits both. A redirect chain ending at
`https://cdn.example.test` is denied at hop **2** of **2**, and one ending
back inside the allowlist is allowed after **3** reviews.

**FINDING: one malformed allowlist entry denies every origin.** An entry
with a trailing path fails `origin_only` validation inside `review_action`,
which returns `network-policy-shape` -- a denial of the *request*. A typo in
an unrelated entry therefore refuses traffic to the correct ones. Failing
closed is right; failing closed as `network-allowlist` would have been
misleading, and it is not what the rule says.

**FINDING: normalization does most of the work before the allowlist is
consulted.** `https://REGISTRY.example.test.:443/x?y#z` and
`https://registry.example.test/x` both normalize to the same origin, and
`https://registry.example.test:443@evil.test/` is rejected for userinfo
rather than matched on its prefix. Of **5** attempts, **2** die in the
parser as `network-shape`, **1** at the allowlist, and the **2** that are
allowed genuinely are the allowlisted origins.

**FINDING: the request has one URL, so the chain lives outside the model.**
`ActionRequest` has no field for a redirect history and `review_action` is
called once per URL, so "reject redirects" is a property of the caller's
loop. Nothing in a single decision records that it was hop **3**.

Structure: `follow()` is the loop; every element of it is a call to the
shipped `review_action`, so the chain rule adds no new trust.
"""

from __future__ import annotations

import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "26-skill-permissions-sandboxes-and-trust"
REGISTRY = "https://registry.example.test"
BOTH = (REGISTRY, f"{REGISTRY}:8443")
ATTACKS = ("https://REGISTRY.example.test.:443/x?y#z",
           "https://registry.example.test:443@evil.test/",
           "http://registry.example.test/x",
           "https://registry.example.test.evil.test/x",
           "https://registry.example.test:8443/x")


def policy_for(ref, workspace, allowlist):
    return ref.SandboxPolicy(workspace_root=workspace, allowed_kinds=("network",),
                             approval_kinds=(), network_allowlist=allowlist)


def verdict(ref, policy, url):
    decision = ref.review_action(policy, ref.ActionRequest("network", url=url))
    return decision.verdict.value, decision.rule


def follow(ref, policy, chain):
    """Every hop reviewed as if it were the first; the chain is the caller's job."""
    reviews = []
    for url in chain:
        result, rule = verdict(ref, policy, url)
        reviews.append((url, result, rule))
        if result != "allow":
            return reviews
    return reviews


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as temp:
        workspace = pathlib.Path(temp) / "workspace"
        workspace.mkdir(parents=True)
        one = policy_for(ref, workspace, (REGISTRY,))
        two = policy_for(ref, workspace, BOTH)
        broken = policy_for(ref, workspace, (REGISTRY, "https://mirror.example.test/pkgs"))

        away = follow(ref, two, (f"{REGISTRY}/pkg", "https://cdn.example.test/pkg"))
        home = follow(ref, two, (f"{REGISTRY}/pkg", f"{REGISTRY}:8443/pkg",
                                 f"{REGISTRY}/pkg.tar"))
        return {
            "one_443": verdict(ref, one, f"{REGISTRY}/pkg"),
            "one_8443": verdict(ref, one, f"{REGISTRY}:8443/pkg"),
            "two_443": verdict(ref, two, f"{REGISTRY}/pkg"),
            "two_8443": verdict(ref, two, f"{REGISTRY}:8443/pkg"),
            "away": [(result, rule) for _, result, rule in away], "away_hops": len(away),
            "home": [result for _, result, _ in home], "home_hops": len(home),
            "broken": verdict(ref, broken, f"{REGISTRY}/pkg"),
            "attacks": [verdict(ref, two, url) for url in ATTACKS],
            "normalized": ref.normalize_https_origin(ATTACKS[0])
            == ref.normalize_https_origin(f"{REGISTRY}/x"),
            "request_fields": list(vars(ref.ActionRequest)["__dataclass_fields__"]),
            "decision_fields": list(vars(ref.ReviewDecision)["__dataclass_fields__"]),
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: two entries, and every hop reviewed as if it were the first",
            all([result["one_443"] == ("allow", "policy-allow"),
                 result["one_8443"] == ("deny", "network-allowlist"),
                 result["two_443"][0] == "allow", result["two_8443"][0] == "allow",
                 result["away_hops"] == 2, result["away"][1][0] == "deny",
                 result["away"][1][1] == "network-allowlist",
                 result["home"] == ["allow"] * 3, result["home_hops"] == 3]),
            f"one entry gives {result['one_443'][0]!r} on 443 and "
            f"{result['one_8443'][0]!r} on 8443; two entries allow both. A chain to "
            f"cdn.example.test dies at hop {result['away_hops']} with "
            f"{result['away'][1][1]!r}, and a chain that stays inside is allowed after "
            f"{result['home_hops']} reviews",
        ),
        practice.Check(
            "FINDING: one malformed allowlist entry denies every origin",
            all([result["broken"] == ("deny", "network-policy-shape"),
                 result["two_443"] == ("allow", "policy-allow")]),
            f"an entry with a trailing path fails origin_only validation inside "
            f"review_action, which returns {result['broken'][1]!r} -- a denial of the "
            "request. A typo in an unrelated entry refuses traffic to the correct ones. "
            "Failing closed is right, and reporting it as a policy-shape error rather than "
            "an allowlist miss is what makes it debuggable",
        ),
        practice.Check(
            "FINDING: normalization does most of the work before the allowlist",
            all([result["normalized"],
                 [verdict for verdict, _ in result["attacks"]]
                 == ["allow", "deny", "deny", "deny", "allow"],
                 [rule for _, rule in result["attacks"]][1:4]
                 == ["network-shape", "network-shape", "network-allowlist"]]),
            f"uppercase, a trailing dot, an explicit :443, a query and a fragment all "
            f"normalize to the allowlisted origin, and the userinfo form is rejected as "
            f"{result['attacks'][1][1]!r} rather than matched on its prefix. The five "
            f"attempts answer {[v for v, _ in result['attacks']]} -- the two allows are the "
            "two that genuinely are the allowlisted origins",
        ),
        practice.Check(
            "FINDING: the request has one URL, so the chain lives outside the model",
            all(["url" in result["request_fields"],
                 not any("redirect" in name or "chain" in name
                         for name in result["request_fields"]),
                 not any("hop" in name for name in result["decision_fields"])]),
            f"ActionRequest carries {result['request_fields']} -- one url and no history -- "
            f"and ReviewDecision carries {result['decision_fields']}. 'Reject redirects' is "
            "therefore a property of the caller's loop: nothing in a single decision "
            "records that it was the third hop rather than the first",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
