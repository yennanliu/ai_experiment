"""Exercise 1 -- the lesson's registry checks pass 3 of 3 drifted remote URLs, because nothing probes the URL.

    Change the published remote URL while leaving the live server unchanged. Make the registry validation report the exact drift.

Reading of the exercise: "published" is the `server.json` record the
lesson's `Registry` keeps in `entries`; "live" is the unchanged
`build_readonly_server()` process, reachable at its own `url`. Three edits
are made to the published `remotes[0].url` only: a path typo, an
`http://` downgrade, and the destructive server's URL. The lesson's three
validators are run as `Registry.register` runs them, then a drift report
probes the *published* URL with `server/discover` through a tiny in-process
route table (URL -> server) and compares what answers.

**ANSWER: the drift report names the field, the published value and the
live value for all 3 edits.** The path typo and the http downgrade reach no
server, so the report says `remotes[0].url` is published as X, nothing
answers there, and the live endpoint is `https://mcp.internal.example.com/readonly`.
The third URL does answer, but as `com.example/internal-destructive`, so the
report adds the `serverInfo.name` mismatch.

**FINDING: the lesson's registry validation reports 0 issues for all 3.**
`validate_registry_document`, `validate_publisher_namespace` and
`validate_runtime_alignment` return empty lists, because `Registry.register`
calls `server.discover()` on the in-process object and never uses the URL,
and `validate_runtime_alignment` compares only name and version. The doc's
Build It step 7 asks for drift on "the published remote, identity, version".

**FINDING: probing by URL, the lesson's own alignment check catches 1 of 3.**
Fed the discovery that actually answers at the published URL, it flags the
destructive URL (name mismatch) and has nothing to compare for the other two.
"""

from __future__ import annotations

import copy

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "13-mcp-server-with-registry"
EDITS = {
    "path typo": "https://mcp.internal.example.com/readonly-v2",
    "http downgrade": "http://mcp.internal.example.com/readonly",
    "other server": "https://mcp.internal.example.com/destructive",
}


def lesson_issues(ref, doc, live):
    """What `Registry.register` checks, in its order, against the in-process server."""
    issues = ref.validate_registry_document(doc) + ref.validate_publisher_namespace(doc, ref.PUBLISHER_DOMAIN)
    return issues + ref.validate_runtime_alignment(doc, live.discover(ref.request_meta()))


def drift_report(ref, doc, live, routes):
    """Probe every published remote by URL and say exactly what differs."""
    drift = []
    for i, remote in enumerate(doc["remotes"]):
        field = f"remotes[{i}].url"
        target = routes.get(remote["url"])
        if remote["url"] != live.url:
            drift.append({"field": field, "published": remote["url"], "live": live.url})
        if target is None:
            drift.append({"field": field, "published": remote["url"], "live": "no server answers"})
            continue
        info = target.discover(ref.request_meta())["_meta"]["io.modelcontextprotocol/serverInfo"]
        for key in ("name", "version"):
            if info[key] != doc[key]:
                drift.append({"field": f"serverInfo.{key}", "published": doc[key], "live": info[key]})
    return drift


def probe_alignment(ref, doc, routes):
    target = routes.get(doc["remotes"][0]["url"])
    return None if target is None else ref.validate_runtime_alignment(doc, target.discover(ref.request_meta()))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    live, other = ref.build_readonly_server(), ref.build_destructive_server()
    routes = {live.url: live, other.url: other}
    registry = ref.Registry()
    registry.register(live)
    rows = {}
    for label, url in EDITS.items():
        doc = copy.deepcopy(registry.entries[live.name])
        doc["remotes"][0]["url"] = url
        rows[label] = {
            "lesson": lesson_issues(ref, doc, live),
            "report": drift_report(ref, doc, live, routes),
            "probed": probe_alignment(ref, doc, routes),
        }
    clean = drift_report(ref, registry.entries[live.name], live, routes)
    return {"rows": rows, "clean": clean, "live_url": live.url}


def summarize(result):
    rows, live = result["rows"], result["live_url"]
    first = {k: r["report"][0] for k, r in rows.items()}
    return {
        "exact": first == {k: {"field": "remotes[0].url", "published": u, "live": live} for k, u in EDITS.items()},
        "shapes": {k: [d["field"] for d in r["report"]] for k, r in rows.items()},
        "dead": [k for k, r in rows.items() if "no server answers" in str(r["report"])],
        "probed": {k: r["probed"] for k, r in rows.items()},
        "lesson": {k: r["lesson"] for k, r in rows.items()},
    }


def verify(result):
    s = summarize(result)
    exact, shapes, dead, probed = s["exact"], s["shapes"], s["dead"], s["probed"]
    return [
        practice.Check(
            "ANSWER: the drift report names field, published and live value for all 3 URL edits",
            exact
            and shapes
            == {
                "path typo": ["remotes[0].url"] * 2,
                "http downgrade": ["remotes[0].url"] * 2,
                "other server": ["remotes[0].url", "serverInfo.name"],
            }
            and dead == ["path typo", "http downgrade"]
            and result["clean"] == [],
            f"drift fields {shapes}; unreachable: {dead}; unedited record drift {result['clean']}",
        ),
        practice.Check(
            "FINDING: the lesson's registry validation reports 0 issues for all 3 drifted URLs",
            s["lesson"] == {k: [] for k in EDITS},
            f"lesson issues per edit: {s['lesson']}",
        ),
        practice.Check(
            "FINDING: probing by URL, the lesson's alignment check catches 1 of 3",
            probed
            == {
                "path typo": None,
                "http downgrade": None,
                "other server": ["runtime serverInfo.name does not match server.json name"],
            },
            f"validate_runtime_alignment on the discovery at the published URL: {probed}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
