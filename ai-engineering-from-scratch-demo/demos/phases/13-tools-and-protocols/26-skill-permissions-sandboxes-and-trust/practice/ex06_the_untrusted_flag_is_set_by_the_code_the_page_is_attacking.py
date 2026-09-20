"""Exercise 6 — the untrusted flag is set by the code the page is attacking.

    Threat-model a skill that reads web pages and writes pull-request
    comments. Mark every trust and authority boundary.

Reading of the exercise: "mark every boundary" is a list, and a list is only
a threat model if each entry says who enforces it, so each boundary is paired
with a probe against the shipped review and scored enforced or not. Six
boundaries, six probes. The answer is that the module enforces the authority
boundary absolutely and leaves the trust boundary to a caller-set bool.

**ANSWER: six boundaries, three enforced by the module and three by the
caller.** Enforced: the authority boundary (`policy-change` is denied
unconditionally), the origin boundary (an unallowlisted host is denied), and
the secret boundary (a token pasted from a page into a comment is denied).
Not enforced: labelling the page as untrusted, distinguishing reading from
writing, and the fact that read access and exfiltration are the same
permission.

**FINDING: the untrusted flag is set by the code the page is attacking.**
`influenced_by_untrusted_content` is a field on the request. With it the
comment write needs approval; without it the identical write is allowed. The
one control designed for this threat is a self-report by the component the
threat has already reached.

**FINDING: reading pages and posting comments are the same permission.** A
`network` allowlist wide enough to fetch `https://docs.example.test` also
permits a POST to it, because the request has no direction. A skill that may
read the web may send the repository to the web at the same origin, and no
policy in this module separates the two.

**FINDING: the secret scan is the one control that reads the content.**
`contains_secret` inspects the payload, so an API key copied out of a fetched
page into a comment body is denied on content rather than on shape -- the
only place in the module where what was read affects what may be written.
Its **3** patterns are also its whole coverage.

**FINDING: the authority boundary is the one thing that cannot be
negotiated.** `policy-change` is denied before the kind allowlist is even
consulted, so no combination of approval, allowlists or untrusted-content
flags reaches it. That is what "skills are context, not a security boundary"
means in code.

Structure: `BOUNDARIES` pairs each boundary with the probe that decides
whether the module enforces it, so the threat model is a table with evidence.
"""

from __future__ import annotations

import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "26-skill-permissions-sandboxes-and-trust"
PAGE = "https://docs.example.test/guide"
COMMENT = "https://api.github.test/repos/acme/app/issues/7/comments"
TOKEN_BODY = 'Found this in the docs: api_key=sk-live-9f2c41ab'


def probes(ref):
    """Each boundary with the one request that tests whether the module holds it."""
    return {
        "authority": ref.ActionRequest("policy-change", payload="allow everything",
                                       approved=True),
        "origin": ref.ActionRequest("network", url=COMMENT, payload="looks good"),
        "secret": ref.ActionRequest("network", url=PAGE, payload=TOKEN_BODY),
        "provenance": ref.ActionRequest("network", url=PAGE, payload="looks good"),
        "direction": ref.ActionRequest("network", url=PAGE,
                                       payload="exfiltrated repository contents"),
        "labelled": ref.ActionRequest("network", url=PAGE, payload="looks good",
                                      influenced_by_untrusted_content=True),
    }


def verdicts(ref, policy):
    return {name: (ref.review_action(policy, request).verdict.value,
                   ref.review_action(policy, request).rule)
            for name, request in probes(ref).items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as temp:
        workspace = pathlib.Path(temp) / "workspace"
        workspace.mkdir(parents=True)
        policy = ref.SandboxPolicy(
            workspace_root=workspace, allowed_kinds=("read", "network"),
            approval_kinds=(), network_allowlist=("https://docs.example.test",))
        seen = verdicts(ref, policy)
        approved_change = ref.review_action(policy, ref.ActionRequest(
            "policy-change", approved=True, claimed_permissions=("all",)))
        read_then_write = [
            ref.review_action(policy, ref.ActionRequest("network", url=PAGE)).verdict.value,
            ref.review_action(policy, ref.ActionRequest(
                "network", url=PAGE, payload="exfiltrated repository contents"))
            .verdict.value]
        return {
            "boundaries": sorted(seen), "verdicts": {k: v[0] for k, v in seen.items()},
            "rules": {k: v[1] for k, v in seen.items()},
            "enforced": sorted(name for name, (result, _) in seen.items()
                               if result == "deny"),
            "gated": sorted(name for name, (result, _) in seen.items()
                            if result == "require-approval"),
            "allowed": sorted(name for name, (result, _) in seen.items()
                              if result == "allow"),
            "approved_change": (approved_change.verdict.value, approved_change.rule),
            "ignored_claims": approved_change.claimed_permissions_ignored,
            "read_then_write": read_then_write,
            "secret_patterns": len(ref.SECRET_PATTERNS),
            "secret_caught": ref.contains_secret(TOKEN_BODY),
            "secret_missed": ref.contains_secret("ghp_9f2c41ab9f2c41ab9f2c41ab"),
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: six boundaries, three the module enforces and three left to the caller",
            all([len(result["boundaries"]) == 6,
                 result["enforced"] == ["authority", "origin", "secret"],
                 result["gated"] == ["labelled"],
                 result["allowed"] == ["direction", "provenance"],
                 result["rules"]["authority"] == "authority-boundary",
                 result["rules"]["origin"] == "network-allowlist"]),
            f"the six probes answer {result['verdicts']}. The module denies "
            f"{result['enforced']} on its own; {result['gated']} is gated only because the "
            f"caller said so, and {result['allowed']} pass untouched. Three boundaries have "
            "an enforcer and three have a convention",
        ),
        practice.Check(
            "FINDING: the untrusted flag is set by the code the page is attacking",
            all([result["verdicts"]["labelled"] == "require-approval",
                 result["verdicts"]["provenance"] == "allow",
                 result["rules"]["labelled"] == "approval-gate"]),
            f"influenced_by_untrusted_content is a field on the request: with it the "
            f"comment write answers {result['verdicts']['labelled']!r}, without it the "
            f"identical write answers {result['verdicts']['provenance']!r}. The one control "
            "designed for this threat is a self-report by the component the threat has "
            "already reached",
        ),
        practice.Check(
            "FINDING: reading pages and posting comments are the same permission",
            all([result["read_then_write"] == ["allow", "allow"],
                 result["verdicts"]["direction"] == "allow",
                 result["verdicts"]["origin"] == "deny"]),
            f"an allowlist wide enough to fetch the page also permits a payload to it -- a "
            f"bare fetch and a fetch carrying repository contents both answer "
            f"{result['read_then_write'][0]!r}. The request has no direction, so a skill "
            "that may read the web may send the repository to the web at the same origin",
        ),
        practice.Check(
            "FINDING: the secret scan is the one control that reads the content",
            all([result["secret_caught"], not result["secret_missed"],
                 result["secret_patterns"] == 3,
                 result["verdicts"]["secret"] == "deny",
                 result["rules"]["secret"] == "secret-review"]),
            f"contains_secret inspects the payload, so a key copied out of a fetched page "
            f"into a comment is denied under {result['rules']['secret']!r} -- the only "
            f"place where what was read affects what may be written. Its "
            f"{result['secret_patterns']} patterns are also its whole coverage: a bare "
            "ghp_ token matches none of them",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
