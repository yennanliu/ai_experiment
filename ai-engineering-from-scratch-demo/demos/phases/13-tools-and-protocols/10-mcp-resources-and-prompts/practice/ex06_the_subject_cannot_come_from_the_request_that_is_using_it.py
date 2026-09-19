"""Exercise 6 — the subject cannot come from the request that is using it.

    Add an authorization subject to the read handler and prove a cache entry
    cannot cross subjects.

Reading of the exercise: "add a subject" is only meaningful once you say where
it comes from, and the request is the one place it cannot come from --
`_meta.clientInfo` is a name the client picks for itself. So the subject is a
parameter supplied beside the message, and the proof that a cache entry cannot
cross subjects is run against the cache that *can*, so the two differ by the
key and nothing else.

**ANSWER: `read(params, subject)` against a grant table, and a cache keyed on
`(subject, uri)`.** Alice is granted both notes and mallory one; after both
read what they may, the keyed cache holds **3** entries and mallory's lookup
of alice's note misses. Keyed on the URI alone the same sequence holds **2**
entries and mallory is served alice's note verbatim.

**FINDING: `clientInfo` is an identity claim, not an authorization subject.**
`validate_request_meta` accepts any `clientInfo.name`, so two requests
differing only in that field are both valid — a caller naming itself
`user:alice` is believed exactly as far as it asked to be. The subject has to
arrive beside the message, from whatever authenticated the transport.

**FINDING: the hint was always there and the parameter never was.**
`resources_read` returns `cacheScope: "private"` for every note and its
signature is `('params',)` — one argument, no subject. The result has named
the requirement since the first read; nothing in the handler could satisfy it.

**FINDING: a distinct denial code is an enumeration oracle.** Answering
`-32003` for a note that exists but is not granted and `-32602` for one that
does not exist lets mallory tell the two apart in **1** request. Answering
`-32602` for both makes the responses **identical**, and costs an
unauthorized caller the difference between "not yours" and "not there".

Structure: `Reader` is the handler plus the grant table; `SubjectCache` takes
the key function as a switch, so the crossing and non-crossing caches are the
same code.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "10-mcp-resources-and-prompts"
ALICE, MALLORY = "user:alice", "user:mallory"
NOTE_1, NOTE_2, MISSING = "notes://note-1", "notes://note-2", "notes://note-9"
GRANTS = {ALICE: {NOTE_1, NOTE_2}, MALLORY: {NOTE_2}}
FORBIDDEN, UNKNOWN = -32003, -32602


class Reader:
    """The lesson's read, with a subject supplied beside the message."""

    def __init__(self, ref, disclose=False):
        self.ref, self.disclose = ref, disclose

    def read(self, params, subject):
        uri = params.get("uri")
        if uri not in GRANTS.get(subject, set()):
            exists = uri in self.ref.NOTES
            if exists and self.disclose:
                raise self.ref.RpcError(FORBIDDEN, "Forbidden resource URI", {"uri": uri})
            raise self.ref.RpcError(UNKNOWN, "Unknown or invalid resource URI", {"uri": uri})
        return self.ref.resources_read(params)


class SubjectCache:
    """A host cache whose key is the whole question."""

    def __init__(self, by_subject=True):
        self.entries, self.by_subject = {}, by_subject

    def key(self, subject, uri):
        return (subject, uri) if self.by_subject else uri

    def put(self, subject, uri, result):
        self.entries[self.key(subject, uri)] = result

    def get(self, subject, uri):
        return self.entries.get(self.key(subject, uri))


def denial(reader, uri, subject):
    try:
        reader.read({"uri": uri}, subject)
    except reader.ref.RpcError as exc:
        return (exc.code, exc.message)
    return None


def fill(ref, reader, cache):
    """Every subject reads everything it is granted, in a fixed order."""
    for subject in (ALICE, MALLORY):
        for uri in sorted(GRANTS[subject]):
            cache.put(subject, uri, reader.read({"uri": uri}, subject))
    return cache


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    reader, leaky_reader = Reader(ref), Reader(ref, disclose=True)
    keyed = fill(ref, reader, SubjectCache())
    crossing = fill(ref, reader, SubjectCache(by_subject=False))
    named = ref.request_meta(client_name=ALICE)
    return {
        "granted": reader.read({"uri": NOTE_1}, ALICE)["contents"][0]["uri"],
        "denied": denial(reader, NOTE_1, MALLORY),
        "keyed_entries": len(keyed.entries), "crossing_entries": len(crossing.entries),
        "keyed_cross": keyed.get(MALLORY, NOTE_1),
        "crossing_cross": crossing.get(MALLORY, NOTE_1) is not None,
        "client_name": named["io.modelcontextprotocol/clientInfo"]["name"],
        "any_name_valid": ref.validate_request_meta({"_meta": named}) is None,
        "read_signature": list(
            ref.resources_read.__code__.co_varnames[:ref.resources_read.__code__.co_argcount]),
        "scope": ref.resources_read({"uri": NOTE_2})["cacheScope"],
        "leaky": [denial(leaky_reader, NOTE_1, MALLORY), denial(leaky_reader, MISSING, MALLORY)],
        "uniform": [denial(reader, NOTE_1, MALLORY), denial(reader, MISSING, MALLORY)],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: read takes a subject, and a cache keyed on (subject, uri) cannot cross",
            all([result["granted"] == NOTE_1, result["denied"][0] == UNKNOWN,
                 result["keyed_entries"] == 3, result["keyed_cross"] is None,
                 result["crossing_entries"] == 2, result["crossing_cross"]]),
            f"alice is granted both notes and mallory one; after both read what they may the "
            f"keyed cache holds {result['keyed_entries']} entries and mallory's lookup of "
            f"alice's note returns {result['keyed_cross']}. Keyed on the URI alone the same "
            f"sequence holds {result['crossing_entries']} entries and mallory is served "
            "alice's note",
        ),
        practice.Check(
            "FINDING: clientInfo is an identity claim, not an authorization subject",
            all([result["client_name"] == ALICE, result["any_name_valid"]]),
            f"validate_request_meta accepts a clientInfo.name of {result['client_name']!r} "
            "without complaint, so a caller naming itself alice is believed exactly as far as "
            "it asked to be. The subject has to arrive beside the message, from whatever "
            "authenticated the transport",
        ),
        practice.Check(
            "FINDING: the hint was always there and the parameter never was",
            all([result["read_signature"] == ["params"], result["scope"] == "private"]),
            f"resources_read's signature is {result['read_signature']} -- one argument, no "
            f"subject -- and it returns cacheScope {result['scope']!r} for every note. The "
            "result has named the requirement since the first read; nothing in the handler "
            "could satisfy it",
        ),
        practice.Check(
            "FINDING: a distinct denial code is an enumeration oracle",
            all([result["leaky"][0][0] == FORBIDDEN, result["leaky"][1][0] == UNKNOWN,
                 result["leaky"][0] != result["leaky"][1],
                 result["uniform"][0] == result["uniform"][1]]),
            f"answering {FORBIDDEN} for a note that exists but is not granted and "
            f"{UNKNOWN} for one that does not exist lets mallory tell them apart in one "
            f"request. Answering {UNKNOWN} for both makes the responses identical, "
            f"{result['uniform'][0]}, costing an unauthorized caller the difference between "
            "'not yours' and 'not there'",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
