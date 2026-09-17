"""Exercise 5 — four of the five recognized errors are standard JSON-RPC.

    Sketch a dual-era stdio client. Treat a result or recognized modern error
    as modern, and permit fallback to `initialize` only for an unrecognized
    error or timeout.

Reading of the exercise: the classifier is written as stated and then run over
seven probes -- five that the lesson's server actually answers, one that it
answers with silence, and one hand-built legacy reply -- so "recognized" and
"unrecognized" are decided by measurement rather than by taste. The probes are
also scored on a second axis the exercise does not mention: whether the answer
tells the client which protocol version to use next.

**ANSWER: five of the seven probes classify modern and two fall back.** A
result classifies modern; so do -32600, -32601, -32602 and -32022. The two
fallbacks are the notification, which draws no response at all, and a
hand-built `-32000 Method not found`, which is the shape a pre-2026 server
returns.

**FINDING: only one probe proves modernity and yields the version at once.**
`server/discover` at the current version answers both, but a dual-era client
does not know the current version yet -- that is what it is probing for.
`server/discover` at a deliberately wrong revision returns -32022 whose `data`
carries `supported`, so it settles the era and the version in one round trip.
The other three modern probes settle the era and say nothing about the version.

**FINDING: `initialize` answers -32601, which the rule reads as modern.** The
legacy handshake is a *method* on this server's dispatch table -- an unknown
one -- so sending it first does not detect an old server, it detects a JSON-RPC
server. The ordering in the exercise is load-bearing: probe modern first and
fall back only on the unrecognized, because the fallback probe itself returns a
recognized modern error.

**FINDING: a notification is indistinguishable from a timeout.** `handle`
returns `None` before any validation when `"id"` is absent, so a probe sent as
a notification produces exactly the silence that means "legacy" -- and would
send a dual-era client into the fallback against a fully modern server. The
probe must carry an id.

**FINDING: four of the five recognized codes are standard JSON-RPC.** -32600,
-32601, -32602 and -32603 are in the JSON-RPC 2.0 pre-defined block and a
pre-2026 server emits them too; only **-32022** is this protocol's own. So
"recognized modern error" is a reliable era signal for one code out of five,
and for the other four it means "some JSON-RPC server answered".

Structure: `MODERN_CODES` is the recognized set, `PROBES` the seven messages,
`classify` is the rule the exercise states, `version_of` reads the version out
of whichever field carries it, and `LEGACY` is the hand-built old-era reply.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "14-model-context-protocol"
OLD_VERSION = "2025-11-25"
MODERN_CODES = (-32600, -32601, -32602, -32603, -32022)
JSON_RPC_RESERVED = (-32700, -32600, -32601, -32602, -32603)
LEGACY = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "Method not found"}}


def classify(response):
    """The rule: a result or a recognized modern error is modern; else fall back."""
    if response is None:
        return "fallback"
    if "result" in response:
        return "modern"
    return "modern" if response["error"]["code"] in MODERN_CODES else "fallback"


def version_of(response):
    """The protocol version, from a result or from an unsupported-version error."""
    if response is None:
        return None
    if "result" in response:
        return (response["result"].get("supportedVersions") or [None])[0]
    return ((response["error"].get("data") or {}).get("supported") or [None])[0]


def probes(ref):
    meta = lambda version: {"_meta": ref.request_metadata(protocol_version=version)}  # noqa: E731
    return {
        "discover@current": {"jsonrpc": "2.0", "id": 1, "method": "server/discover",
                             "params": meta(ref.PROTOCOL_VERSION)},
        "discover@wrong": {"jsonrpc": "2.0", "id": 1, "method": "server/discover",
                           "params": meta(OLD_VERSION)},
        "no params": {"jsonrpc": "2.0", "id": 1, "method": "server/discover"},
        "initialize": {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": meta(ref.PROTOCOL_VERSION)},
        "notification": {"jsonrpc": "2.0", "method": "server/discover",
                         "params": meta(ref.PROTOCOL_VERSION)},
        "not an object": "hello",
    }


def code_of(response):
    return None if response is None or "result" in response else response["error"]["code"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    server = ref.MCPServer("practice")
    server.tools.update(ref.server.tools)
    answers = {name: server.handle(message) for name, message in probes(ref).items()}
    answers["legacy -32000"] = LEGACY
    verdicts = {name: classify(response) for name, response in answers.items()}
    versions = {name: version_of(response) for name, response in answers.items()}
    return {
        "probes": len(answers), "verdicts": verdicts,
        "modern": sorted(n for n, v in verdicts.items() if v == "modern"),
        "fallback": sorted(n for n, v in verdicts.items() if v == "fallback"),
        "codes": {name: code_of(response) for name, response in answers.items()},
        "with_version": sorted(n for n, v in versions.items() if v),
        "declared": ref.PROTOCOL_VERSION, "wrong_version": versions["discover@wrong"],
        "notification": answers["notification"],
        "recognized": len(MODERN_CODES),
        "standard": sorted(set(MODERN_CODES) & set(JSON_RPC_RESERVED)),
        "era_specific": sorted(set(MODERN_CODES) - set(JSON_RPC_RESERVED)),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: five of the seven probes classify modern and two fall back",
            all([len(result["modern"]) == 5, result["fallback"] ==
                 ["legacy -32000", "notification"], result["probes"] == 7]),
            f"{result['modern']} classify modern and {result['fallback']} fall back. The "
            f"codes behind the verdicts are {result['codes']} -- a result, then -32602, "
            "-32022 and -32601 and -32600, against silence and a pre-2026 -32000",
        ),
        practice.Check(
            "FINDING: only one probe proves modernity and yields the version at once",
            all([result["with_version"] == ["discover@current", "discover@wrong"],
                 result["wrong_version"] == result["declared"]]),
            f"{result['with_version']} are the two probes that return a version, and the "
            "first needs the answer to ask the question. server/discover at a deliberately "
            f"wrong revision returns -32022 whose data carries {result['wrong_version']!r}, "
            "settling the era and the version in one round trip",
        ),
        practice.Check(
            "FINDING: initialize answers -32601, which the rule reads as modern",
            all([result["codes"]["initialize"] == -32601,
                 result["verdicts"]["initialize"] == "modern"]),
            f"the legacy handshake is an unknown method on this server's dispatch table, so "
            f"it answers {result['codes']['initialize']} and classifies "
            f"{result['verdicts']['initialize']!r}. Sending it first detects a JSON-RPC "
            "server, not an old one -- the exercise's ordering is load-bearing",
        ),
        practice.Check(
            "FINDING: a notification is indistinguishable from a timeout",
            all([result["notification"] is None,
                 result["verdicts"]["notification"] == "fallback"]),
            "`handle` returns None before any validation when 'id' is absent, so a probe "
            f"sent as a notification produces {result['notification']} -- exactly the "
            "silence that means legacy -- and sends a dual-era client into the fallback "
            "against a fully modern server. The probe must carry an id",
        ),
        practice.Check(
            "FINDING: four of the five recognized codes are standard JSON-RPC",
            all([len(result["standard"]) == 4, result["era_specific"] == [-32022],
                 result["recognized"] == 5]),
            f"{result['standard']} are in the JSON-RPC 2.0 pre-defined block and a pre-2026 "
            f"server emits them too; {result['era_specific']} is this protocol's own. "
            "'Recognized modern error' is an era signal for one code of five, and for the "
            "other four it means some JSON-RPC server answered",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
