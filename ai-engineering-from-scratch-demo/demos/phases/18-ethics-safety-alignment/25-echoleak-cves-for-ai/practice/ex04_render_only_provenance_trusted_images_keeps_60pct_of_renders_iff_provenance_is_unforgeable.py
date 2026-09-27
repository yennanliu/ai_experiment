"""Exercise 4 — render only provenance-trusted images; it keeps 60% of renders iff provenance is unforgeable.

    Microsoft's CamoLeak fix disabled image rendering entirely. Propose a
    partial fix that preserves image rendering for trusted sources only.
    Identify the authentication assumption it requires.

Reading of the exercise: the total fix (no rendering) scores 100% blocked in
ex02. The partial fix renders an image only when its source is provenance-
trusted -- the URL was produced by a trusted content channel (system, user, a
first-party CDN), never by retrieved untrusted content. It is graded on the
same 100-render fixture as ex02 plus the reference leak, and the authentication
assumption is tested by flipping one bit: if untrusted content can forge a
trusted provenance tag, the fix collapses to the vendor CSP.

**ANSWER: rendering by source provenance keeps 60% of legitimate renders and
blocks the leak, and it requires that provenance be unforgeable by retrieved
content.**

| policy | legit rendered | reference leak |
|---|---:|---|
| no rendering (Microsoft's fix) | 0% | blocked |
| provenance-trusted, unforgeable | 60% | blocked |
| provenance-trusted, forgeable | 60% | rendered |

The 40% not rendered are the diagrams the agent itself composes from content,
which have no trusted provenance; the 60% kept are images whose URL came from a
trusted channel. The blocked/rendered split for the leak flips entirely on the
one assumption.

**FINDING: the assumption is the whole fix.** With forgeable provenance the
partial fix admits exactly the images the vendor-CSP allowlist admits -- the
leak included -- so it buys nothing. Provenance must be bound by the platform
(a signed capability on the render call), not asserted in the content, which is
the same trust boundary EchoLeak crossed.

Structure: `make_fixture()` tags each render with a provenance source;
`render()` applies the policy; the forgeable case marks the attack trusted.
"""

from __future__ import annotations

import urllib.parse as up

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "25-echoleak-cves-for-ai"
TRUSTED_PROVENANCE = {"system", "user", "first_party_cdn"}


def make_fixture():
    """(url, provenance source, count): the ex02 renders, each tagged by origin."""
    rows = [
        ("https://contoso.sharepoint.com/_layouts/15/thumb.aspx?id=doc7", "first_party_cdn", 40),
        ("https://res.cdn.office.net/assets/icon.png", "first_party_cdn", 20),
        ("https://upload.wikimedia.org/wikipedia/commons/chart.png", "retrieved_content", 25),
        ("https://signed.microsoft.com/img?data=Q4 roadmap", "retrieved_content", 15),
    ]
    return [row for row in rows for _ in range(row[2])]


def render(source, no_render=False):
    """Partial fix: render iff the URL's provenance source is trusted."""
    return not no_render and source in TRUSTED_PROVENANCE


def grade(legit, leak_source, no_render=False):
    kept = sum(render(src, no_render) for _, src, _ in legit)
    return round(100 * kept / len(legit), 1), render(leak_source, no_render)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    state = ref.naive_copilot(ref.State(user_prompt="summarize my recent emails"))
    leak_url, legit = state.tool_calls[0]["url"], make_fixture()
    # the leak URL is composed by the agent from retrieved untrusted content
    return {
        "leak_url": leak_url, "legit": len(legit),
        "no_render": grade(legit, "retrieved_content", no_render=True),
        "unforgeable": grade(legit, "retrieved_content"),
        "forgeable": grade(legit, "user"),
        "leak_host": up.urlsplit(leak_url).hostname,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: provenance rendering keeps 60% and blocks the leak only if unforgeable",
            result["legit"] == 100
            and result["no_render"] == (0.0, False)
            and result["unforgeable"] == (60.0, False)
            and result["forgeable"] == (60.0, True),
            f"(legit rendered %, leak rendered) -- no-render {result['no_render']}, "
            f"unforgeable {result['unforgeable']}, forgeable {result['forgeable']}",
        ),
        practice.Check(
            "FINDING: forgeable provenance renders the same leak host the CSP allowed",
            result["forgeable"][1] and result["leak_host"] == "signed.microsoft.com",
            f"with forgeable provenance the leak to {result['leak_host']} renders again",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
