"""Exercise 1 — one request is sent, and the retry path is the one not taken.

    Make a fake server return `-32022` with no mutually supported version.
    Confirm the client fails instead of sending `initialize`.

Reading of the exercise: the server is configured to support only a version the
client does not, and the confirmation is taken from the transport's own record
of what it received rather than from the exception text. `-32022` is also the
one error code with a *success* path -- it carries the supported list, so a
client that shares a version retries on it -- which makes "no mutually supported
version" the branch that proves the retry is conditional.

**ANSWER: it raises, having sent exactly one message, and `initialize` is not
among them.** The transport records **1** request, `server/discover`. The peer
stays `era="unknown"`, `available=False`, `protocol_version=None`.

**FINDING: the same code with a shared version sends a second modern request,
not a legacy one.** Probe at `2099-01-01` against a server supporting
`2026-07-28`: the client reads `data.supported`, picks the mutual version, and
re-sends **`server/discover`** -- **2** requests, both modern, and the peer
activates. `-32022` is a negotiation signal; only the empty intersection is
fatal.

**FINDING: legacy is never reached, because `-32022` proves the peer is
modern.** `_connect_peer` calls `_probe_legacy` for an unrecognised error code
and for a dead transport, but a recognised modern error means the server spoke
modern well enough to reject the version -- so the client fails closed rather
than falling back. Even with `allow_legacy=True` the transport still sees **0**
`initialize` messages.

**FINDING: and the failure is the same whether or not legacy is allowlisted.**
Both configurations raise `no mutually supported modern version`, so the
allowlist does not widen what this peer can become. It governs the fallback
path, and this peer never enters it.

Structure: `Recorder` wraps a fake server and keeps every method it saw,
`connect` runs the client's own `connect_all` and returns the failure, and
`ONLY_FUTURE` / `SHARED` are the two version lists.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "08-building-an-mcp-client"
ONLY_FUTURE = ["2099-01-01"]
SHARED = ["2026-07-28"]


class Recorder:
    """A transport that delegates to a fake server and remembers the methods."""

    def __init__(self, server):
        self.server = server
        self.methods = []

    def __call__(self, message, timeout_ms=None):
        self.methods.append(message.get("method"))
        return self.server(message, timeout_ms)


def connect(ref, supported_versions, probe_version, allow_legacy=False):
    """Connect one peer, returning the recorder, the peer and any failure."""
    server = ref.ModernFakeServer("s", [ref.tool("notes_list", "List notes.")],
                                  supported_versions=list(supported_versions))
    recorder = Recorder(server)
    client = ref.MultiServerClient(probe_version=probe_version)
    client.add_server("s", recorder, allow_legacy=allow_legacy)
    failure = None
    try:
        client.connect_all()
    except RuntimeError as error:
        failure = str(error)
    return recorder, client.peers["s"], failure


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    future_recorder, future_peer, future_failure = connect(ref, ONLY_FUTURE, "2026-07-28")
    shared_recorder, shared_peer, shared_failure = connect(ref, SHARED, "2099-01-01")
    legacy_recorder, _, legacy_failure = connect(
        ref, ONLY_FUTURE, "2026-07-28", allow_legacy=True)
    return {
        "failure": future_failure,
        "methods": future_recorder.methods, "requests": len(future_recorder.methods),
        "sent_initialize": "initialize" in future_recorder.methods,
        "era": future_peer.era, "available": future_peer.available,
        "version": future_peer.protocol_version,
        "shared_failure": shared_failure,
        "shared_methods": shared_recorder.methods,
        "shared_requests": len(shared_recorder.methods),
        "shared_era": shared_peer.era, "shared_available": shared_peer.available,
        "shared_version": shared_peer.protocol_version,
        "legacy_failure": legacy_failure,
        "legacy_initialize": legacy_recorder.methods.count("initialize"),
        "recognized": sorted(ref.RECOGNIZED_MODERN_ERRORS),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: it raises after one request, and initialize is not among them",
            all([result["failure"] == "s: no mutually supported modern version",
                 result["methods"] == ["server/discover"], result["requests"] == 1,
                 not result["sent_initialize"], result["era"] == "unknown",
                 result["available"] is False, result["version"] is None]),
            f"the transport records {result['requests']} request, {result['methods']}, and "
            f"the client raises {result['failure']!r}. The peer stays era={result['era']!r}, "
            f"available={result['available']}, protocol_version={result['version']}",
        ),
        practice.Check(
            "FINDING: the same code with a shared version sends a second modern request",
            all([result["shared_failure"] is None,
                 result["shared_methods"] == ["server/discover", "server/discover"],
                 result["shared_requests"] == 2, result["shared_era"] == "modern",
                 result["shared_available"], result["shared_version"] == "2026-07-28"]),
            f"probing at 2099-01-01 against a server supporting {SHARED[0]}, the client reads "
            f"data.supported, picks the mutual version and "
            f"re-sends {result['shared_methods']} -- {result['shared_requests']} requests, "
            f"both modern -- activating at {result['shared_version']!r}. -32022 is a "
            "negotiation signal; only the empty intersection is fatal",
        ),
        practice.Check(
            "FINDING: legacy is never reached, because -32022 proves the peer is modern",
            all([result["legacy_initialize"] == 0,
                 result["recognized"] == [-32022, -32021, -32020]]),
            f"_connect_peer probes legacy for an unrecognised error code or a dead transport, "
            f"but {result['recognized']} are recognised modern errors -- the server spoke "
            f"modern well enough to reject the version. Even with allow_legacy=True the "
            f"transport sees {result['legacy_initialize']} initialize messages",
        ),
        practice.Check(
            "FINDING: the failure is the same whether or not legacy is allowlisted",
            result["legacy_failure"] == result["failure"],
            f"both configurations raise {result['failure']!r}, so the allowlist does not "
            "widen what this peer can become. It governs the fallback path, and this peer "
            "never enters it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
