"""Exercise 2 — dropping the signed proxy host stops the reference leak at 15% false positives.

    The EchoLeak attack bypasses CSP because it exfiltrates via a
    Microsoft-signed URL. Design a deployment that narrows the set of allowed
    exfiltration destinations and measure the legitimate-use false-positive
    rate.

Reading of the exercise: the reference has no legitimate render path (the naive
agent renders only on attack, the defended one never renders), so its
false-positive rate is undefined. A labelled 100-render fixture of what an
assistant legitimately draws stands in for legitimate use: tenant thumbnails,
Office CDN icons, web images, and diagrams drawn through the same signed
endpoint. Four destination allowlists are graded on it, and against the one
leak path the reference produces: its own exfil URL.

**ANSWER: narrow the allowlist to exact hosts and drop the signed proxy.**

| allowlist | legit renders blocked | reference leak URL |
|---|---:|---|
| vendor CSP (domain suffixes) | 0% | allowed |
| exact hosts, no signed proxy | 15% | blocked |
| tenant host only | 60% | blocked |
| no rendering (the CamoLeak fix) | 100% | blocked |

The 15% are the 15 diagrams that go through the proxy. The vendor CSP lets the
leak through because `signed.microsoft.com` matches the `microsoft.com` suffix.

**FINDING: the leak URL and a legitimate diagram are the same URL shape.**
Host, path and query key of the reference's exfil URL equal those of every
diagram render in the fixture. A destination rule cannot tell the two apart; it
can only keep or drop the endpoint, so the cost of the fix is exactly the
legitimate traffic on that endpoint. The cheapest allowlist that stops the leak
is the one that drops only that host.

Structure: `make_fixture()` is the labelled legitimate traffic; `ALLOWLISTS`
maps each policy to a host predicate; `grade()` scores one policy.
"""

from __future__ import annotations

import urllib.parse as up

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "25-echoleak-cves-for-ai"
SUFFIXES = ("microsoft.com", "sharepoint.com", "office.net", "wikimedia.org")
EXACT = {"contoso.sharepoint.com", "res.cdn.office.net", "upload.wikimedia.org"}
ALLOWLISTS = {
    "vendor CSP": lambda host: host.endswith(SUFFIXES),
    "exact hosts, no proxy": lambda host: host in EXACT,
    "tenant only": lambda host: host == "contoso.sharepoint.com",
    "no rendering": lambda host: False,
}


def make_fixture():
    """(label, url, count): 100 labelled legitimate renders."""
    rows = [
        ("tenant thumbnail", "https://contoso.sharepoint.com/_layouts/15/thumb.aspx?id=doc7", 40),
        ("office cdn icon", "https://res.cdn.office.net/assets/icon.png", 20),
        ("web image", "https://upload.wikimedia.org/wikipedia/commons/chart.png", 25),
        ("diagram", "https://signed.microsoft.com/img?data=Q4 roadmap", 15),
    ]
    return [row for row in rows for _ in range(row[2])]


def shape(url):
    parts = up.urlsplit(url)
    return parts.hostname, parts.path, parts.query.split("=")[0]


def grade(allow, legit, leak_url):
    blocked = sum(not allow(up.urlsplit(url).hostname) for _, url, _ in legit)
    return round(100 * blocked / len(legit), 1), allow(up.urlsplit(leak_url).hostname)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    state = ref.naive_copilot(ref.State(user_prompt="summarize my recent emails"))
    leak_url, legit = state.tool_calls[0]["url"], make_fixture()
    return {
        "legit": len(legit), "leak_url": leak_url,
        "table": {name: grade(allow, legit, leak_url) for name, allow in ALLOWLISTS.items()},
        "diagram_shapes": sorted({shape(u) for label, u, _ in legit if label == "diagram"}),
        "leak_shape": shape(leak_url),
    }


def verify(result):
    t = result["table"]
    return [
        practice.Check(
            "ANSWER: exact hosts without the signed proxy block the leak at 15% false positives",
            result["legit"] == 100
            and t == {"vendor CSP": (0.0, True), "exact hosts, no proxy": (15.0, False),
                      "tenant only": (60.0, False), "no rendering": (100.0, False)},
            f"(legit blocked %, reference leak URL allowed) per allowlist: {t}",
        ),
        practice.Check(
            "FINDING: the leak URL and a legitimate diagram are the same URL shape",
            result["diagram_shapes"] == [result["leak_shape"]]
            and result["leak_shape"] == ("signed.microsoft.com", "/img", "data"),
            f"reference leak {result['leak_url']!r} has shape {result['leak_shape']}; "
            f"diagram shapes {result['diagram_shapes']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
