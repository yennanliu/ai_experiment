"""Exercise 4 — the CSP can name an origin and cannot pin a build.

    Move the script to `https://static.example.com/app.js`. Add that origin to
    `resourceDomains` and explain the new supply-chain risk.

Reading of the exercise: the risk is argued from what the CSP dictionary can
and cannot express, because that is what fixes the blast radius. So the four
lists are enumerated, the one being opened is changed, and the other three are
left shut on purpose -- the residual exposure is then whatever survives with
`connectDomains` still empty, which turns out to be more than it sounds.

**ANSWER: the inline script moves out and one origin is named.**
`resourceDomains` goes from **0** entries to **1**,
`https://static.example.com`, and the document's inline `<script>` becomes a
`src=` to that origin. The other three lists stay empty.

**THE RISK: the server now vouches for code it does not build, and cannot
name a version of.** The CSP grammar has **4** keys and none of them is an
integrity hash, so the policy authorizes an *origin* rather than an artifact.
Whatever `static.example.com` serves next runs inside the app frame the next
time the resource is read -- no rebuild, no new `resources/read`, no version
in the URL. A `cacheScope: public` resource makes it worse: the host may keep
serving the same HTML while the script behind it changes underneath.

**FINDING: `connectDomains` empty is a real mitigation and not a complete
one.** The imported script cannot open a socket back out, so bulk
exfiltration is blocked at the policy level. It still executes in a document
whose markup contains the user's note titles -- exercise 3's finding -- so
confidentiality now rests on the other three lists staying empty too.
`frameDomains` and `baseUriDomains` are the remaining egress shapes, and both
are **0** only because nobody has needed them yet.

**FINDING: the same origin ends up in two places that cannot disagree
safely.** The URL sits in the HTML and the origin sits in the CSP, and nothing
cross-checks them: pointing the tag at a domain absent from `resourceDomains`
produces a document whose own script the policy forbids. The failure is
silent at the protocol layer and visible only in the frame.

Structure: `rehost` rewrites the document and the policy together, so the two
halves of the change are one function and the mismatch case is the same
function with one argument changed.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "14-mcp-apps"
ORIGIN = "https://static.example.com"
SCRIPT = f"{ORIGIN}/app.js"


def rehost(ref, html, csp, *, origin=ORIGIN, src=SCRIPT):
    """Move the inline script out and open exactly one list for it."""
    document = re.sub(r"<script>.*?</script>", f'<script src="{src}"></script>',
                      html, flags=re.S)
    policy = {key: list(value) for key, value in csp.items()}
    policy["resourceDomains"] = [origin]
    return document, policy


def allowed(policy, src):
    """Would the frame be permitted to load this script?"""
    return any(src.startswith(origin) for origin in policy["resourceDomains"])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = ref.McpAppServer()
    body, headers = ref.make_request("resources/read", 1, {"uri": ref.RESOURCE_URI})
    contents = server.handle(body, headers)[1]["result"]
    original = contents["contents"][0]["text"]
    csp = contents["contents"][0]["_meta"]["ui"]["csp"]

    document, policy = rehost(ref, original, csp)
    mismatched, strict = rehost(ref, original, csp, origin="https://cdn.other.example")
    return {
        "keys": sorted(csp), "before": {k: len(v) for k, v in csp.items()},
        "after": {k: len(v) for k, v in policy.items()},
        "resource_domains": policy["resourceDomains"],
        "inline_gone": "<script>" not in document, "src_present": SCRIPT in document,
        "others_empty": [policy[k] for k in ("connectDomains", "frameDomains",
                                             "baseUriDomains")],
        "integrity_key": [k for k in csp if "integrity" in k.lower() or "hash" in k.lower()],
        "versioned": bool(re.search(r"\bv?\d+\.\d+", SCRIPT)),
        "scope": contents["cacheScope"],
        "titles_in_document": sum(n["title"] in document for n in ref.NOTES),
        "allowed": allowed(policy, SCRIPT),
        "mismatch_allowed": allowed(strict, SCRIPT),
        "mismatch_silent": server.handle(*ref.make_request(
            "resources/read", 2, {"uri": ref.RESOURCE_URI}))[0],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the inline script moves out and exactly one origin is named",
            all([result["before"]["resourceDomains"] == 0,
                 result["after"]["resourceDomains"] == 1,
                 result["resource_domains"] == [ORIGIN],
                 result["inline_gone"], result["src_present"],
                 result["others_empty"] == [[], [], []]]),
            f"resourceDomains goes from {result['before']['resourceDomains']} entries to "
            f"{result['after']['resourceDomains']}, {result['resource_domains']}, the inline "
            f"script is gone and the src is present. The other three lists stay "
            f"{result['others_empty']}",
        ),
        practice.Check(
            "THE RISK: the policy authorizes an origin, and cannot name an artifact",
            all([result["keys"] == ["baseUriDomains", "connectDomains", "frameDomains",
                                    "resourceDomains"],
                 result["integrity_key"] == [], not result["versioned"],
                 result["scope"] == "public"]),
            f"the CSP grammar has {len(result['keys'])} keys, {result['keys']}, and "
            f"{len(result['integrity_key'])} of them is an integrity hash -- so the policy "
            f"authorizes an origin rather than a build, and the URL carries no version "
            f"either. Whatever static.example.com serves next runs on the next read, and a "
            f"{result['scope']!r} resource may keep serving the same HTML while the script "
            "behind it changes",
        ),
        practice.Check(
            "FINDING: connectDomains empty is a real mitigation and not a complete one",
            all([result["others_empty"][0] == [], result["titles_in_document"] == 3]),
            f"the imported script cannot open a socket back out, so bulk exfiltration is "
            f"blocked at the policy level -- and it still executes in a document carrying "
            f"{result['titles_in_document']} of the user's note titles. Confidentiality now "
            "rests on frameDomains and baseUriDomains staying empty too, and both are empty "
            "only because nobody has needed them yet",
        ),
        practice.Check(
            "FINDING: the origin lives in two places that cannot disagree safely",
            all([result["allowed"], not result["mismatch_allowed"],
                 result["mismatch_silent"] == 200]),
            f"the URL sits in the HTML and the origin sits in the CSP with nothing "
            f"cross-checking them: pointing the tag at a domain absent from resourceDomains "
            f"gives allowed={result['mismatch_allowed']} while the server still answers "
            f"HTTP {result['mismatch_silent']}. The failure is silent at the protocol layer "
            "and visible only inside the frame",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
