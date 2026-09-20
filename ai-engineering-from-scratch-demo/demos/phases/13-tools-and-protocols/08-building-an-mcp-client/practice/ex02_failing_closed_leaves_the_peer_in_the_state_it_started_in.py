"""Exercise 2 — failing closed leaves the peer in the state it started in.

    Allowlist a fake legacy server, make its bounded `initialize` probe time
    out, and prove the peer stays `unknown` and unavailable.

Reading of the exercise: the allowlist is set, the probe is made to time out,
and "stays unknown" is checked against every field of the peer rather than the
one the exercise names -- because a half-written peer is the failure mode a
bounded probe exists to prevent, and `era` alone would not show it. The bound
itself is then checked, since "bounded" is a claim about the argument the
transport receives.

**ANSWER: `bounded legacy probe failed closed`, and every field is untouched.**
`era="unknown"`, `available=False`, `protocol_version=None`, `capabilities={}`,
`server_info={}`, `tools=[]` -- the same values `add_server` created. The peer
is not partially connected; it is not connected.

**FINDING: the bound is passed to the transport, and it is the legacy one.**
The discovery probe arrives with `discovery_timeout_ms` and the `initialize`
probe with `legacy_probe_timeout_ms`; configured to **1,000** and **250** the
transport sees exactly `[1000, 250]`. They are separate settings because they
bound different risks -- an unreachable server and a server that answers slowly
enough to hold a startup open.

**FINDING: a timeout and a refusal are the same outcome, and a malformed reply
is not.** `TimeoutError` and `ConnectionError` both become `failed closed`; an
`initialize` that answers with the wrong protocol revision becomes `unsupported
legacy protocol revision` instead. The client distinguishes "could not ask" from
"asked and did not like the answer", and leaves the peer unavailable either way.

**FINDING: and the allowlist is checked before the probe, not after.** Without
`allow_legacy`, the same dead transport raises `legacy compatibility is not
allowlisted` and the transport sees **1** message -- the discovery probe alone.
The allowlist is not a filter on the result; it is a gate on whether the second
request is sent at all.

Structure: `dead` is a transport that fails on the legacy probe, `run` connects
one peer and returns the peer plus the failure, and `fields` reads every
connection-bearing attribute off the peer.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "08-building-an-mcp-client"
DISCOVERY_MS, LEGACY_MS = 1_000, 250


def dead(on_legacy):
    """Discovery is unreachable; the legacy probe fails with `on_legacy`."""
    seen = []

    def transport(message, timeout_ms=None):
        seen.append((message.get("method"), timeout_ms))
        if message.get("method") == "server/discover":
            raise ConnectionError("discovery unreachable")
        raise on_legacy("legacy probe failed")

    transport.seen = seen
    return transport


def wrong_revision(ref):
    """A legacy server that answers initialize with a revision the client refuses."""
    seen = []

    def transport(message, timeout_ms=None):
        seen.append((message.get("method"), timeout_ms))
        if message.get("method") == "server/discover":
            raise ConnectionError("discovery unreachable")
        if message.get("method") == "initialize":
            return {"jsonrpc": "2.0", "id": message["id"],
                    "result": {"protocolVersion": "2024-01-01",
                               "capabilities": {},
                               "serverInfo": {"name": "old", "version": "0.1"}}}
        return None

    transport.seen = seen
    return transport


def run(ref, transport, allow_legacy=True):
    client = ref.MultiServerClient(discovery_timeout_ms=DISCOVERY_MS,
                                   legacy_probe_timeout_ms=LEGACY_MS)
    client.add_server("L", transport, allow_legacy=allow_legacy)
    failure = None
    try:
        client.connect_all()
    except RuntimeError as error:
        failure = str(error)
    return client.peers["L"], failure


def fields(peer):
    return {"era": peer.era, "available": peer.available,
            "protocol_version": peer.protocol_version,
            "capabilities": peer.capabilities, "server_info": peer.server_info,
            "tools": peer.tools}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    timing_out = dead(TimeoutError)
    peer, failure = run(ref, timing_out)
    fresh = ref.Peer(name="L", transport=timing_out)
    refused, refused_failure = run(ref, dead(ConnectionError))
    malformed, malformed_failure = run(ref, wrong_revision(ref))
    blocked_transport = dead(TimeoutError)
    _, blocked_failure = run(ref, blocked_transport, allow_legacy=False)
    return {
        "failure": failure, "peer": fields(peer), "pristine": fields(fresh),
        "unchanged": fields(peer) == fields(fresh),
        "timeouts": [timeout for _, timeout in timing_out.seen],
        "methods": [method for method, _ in timing_out.seen],
        "refused_failure": refused_failure,
        "refused_unchanged": fields(refused) == fields(fresh),
        "malformed_failure": malformed_failure,
        "malformed_unchanged": fields(malformed) == fields(fresh),
        "blocked_failure": blocked_failure,
        "blocked_requests": len(blocked_transport.seen),
        "blocked_methods": [method for method, _ in blocked_transport.seen],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: failed closed, and every field is untouched",
            all([result["failure"] == "L: bounded legacy probe failed closed",
                 result["unchanged"],
                 result["peer"] == {"era": "unknown", "available": False,
                                    "protocol_version": None, "capabilities": {},
                                    "server_info": {}, "tools": []}]),
            f"the client raises {result['failure']!r} and the peer reads {result['peer']} -- "
            "identical to a freshly constructed one. The peer is not partially connected; it "
            "is not connected",
        ),
        practice.Check(
            "FINDING: the bound is passed to the transport, and it is the legacy one",
            all([result["timeouts"] == [DISCOVERY_MS, LEGACY_MS],
                 result["methods"] == ["server/discover", "initialize"]]),
            f"the transport sees {result['methods']} with timeouts {result['timeouts']} -- "
            f"discovery at {DISCOVERY_MS} and initialize at {LEGACY_MS}. They are separate "
            "settings because they bound different risks: an unreachable server, and one "
            "that answers slowly enough to hold a startup open",
        ),
        practice.Check(
            "FINDING: a timeout and a refusal are the same outcome, a bad reply is not",
            all([result["refused_failure"] == result["failure"],
                 result["refused_unchanged"],
                 result["malformed_failure"] ==
                 "L: unsupported legacy protocol revision",
                 result["malformed_unchanged"]]),
            f"TimeoutError and ConnectionError both give {result['failure']!r}; an initialize "
            f"answering with the wrong revision gives {result['malformed_failure']!r}. The "
            "client distinguishes 'could not ask' from 'asked and did not like the answer', "
            "and leaves the peer unavailable either way",
        ),
        practice.Check(
            "FINDING: and the allowlist is checked before the probe, not after",
            all([result["blocked_failure"] ==
                 "L: ConnectionError; legacy compatibility is not allowlisted",
                 result["blocked_requests"] == 1,
                 result["blocked_methods"] == ["server/discover"]]),
            f"without allow_legacy the same dead transport raises "
            f"{result['blocked_failure']!r} and sees {result['blocked_requests']} message, "
            f"{result['blocked_methods']}. The allowlist is not a filter on the result; it "
            "is a gate on whether the second request is sent at all",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
