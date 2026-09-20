"""Exercise 3 — zero is a ttl, and a truthiness test turns it into a default.

    Change one resource to `cacheScope: "private"` with `ttlMs: 0`, add a
    host-level no-store policy, and explain the threat that justifies both
    controls.

Reading of the exercise: "both controls" is the load-bearing phrase, so each
one is ablated separately and the leak that survives is named. The threat is
then whatever needs *both* ablations to be closed -- a threat that only one
control stops would not justify the other. The host policy is built first,
because the server has no cache and therefore cannot demonstrate anything on
its own.

**THREAT: a shared host cache serving a private resource.** The two controls
close two different reuses of one stored copy. `cacheScope: "private"` stops
the copy reaching a *second subject*; `ttlMs: 0` stops the *same* subject
being served it again after the underlying note has changed or their access
has been revoked. Neither implies the other, so a host that honours one and
not the other still leaks -- in a different direction each time.

**ANSWER: the sensitive read returns `cacheScope: "private"`, `ttlMs: 0`, and
the host stores nothing.** The strict host holds **1** entry after reading
both notes: the ordinary one. The no-store path is the ttl, not the scope.

**FINDING: `ttlMs: 0` is not `ttlMs` absent, and `or` cannot tell them
apart.** A host reading `result.get("ttlMs") or 60000` assigns the sensitive
note **60000** — the strictest instruction the protocol has becomes the
default, silently, because `0` is falsy. `result.get("ttlMs", 60000)` gives
**0**. One character of Python is the whole control.

**FINDING: the two ablations leak in different directions.** Keyed on the URI
alone, the ordinary private note read by `alice` is served to `mallory`.
Honouring the scope but not the zero, `alice` is served her own stale copy
after the note's text has changed. Each ablation leaves the other control
working and still loses something.

**FINDING: neither control is enforced by the server, or could be.**
`resources_read` emits the hints and the module has **0** cache-shaped names
once its dunders are set aside, so both live entirely in the host. The server states a policy it has no
machinery to apply, which is why the exercise asks for the host half.

Structure: `HostCache` takes the two controls as switches, so the strict host
and each ablation are the same code under different flags.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "10-mcp-resources-and-prompts"
SENSITIVE, ORDINARY = "notes://note-1", "notes://note-2"
DEFAULT_TTL = 60_000
ALICE, MALLORY = "user:alice", "user:mallory"


def read(ref, uri):
    """The lesson's own read, with the sensitive note downgraded to no-store."""
    result = ref.resources_read({"uri": uri})
    return result | {"ttlMs": 0, "cacheScope": "private"} if uri == SENSITIVE else result


class HostCache:
    """The host-level policy, with each control switchable for ablation."""

    def __init__(self, honour_zero=True, key_subject=True):
        self.entries, self.honour_zero, self.key_subject = {}, honour_zero, key_subject

    def ttl(self, result):
        if self.honour_zero:
            return result.get("ttlMs", DEFAULT_TTL)
        return result.get("ttlMs") or DEFAULT_TTL  # 0 is falsy, so 0 becomes the default

    def key(self, subject, uri):
        return (subject, uri) if self.key_subject else uri

    def store(self, subject, uri, result):
        if self.ttl(result) == 0:
            return False
        self.entries[self.key(subject, uri)] = result
        return True

    def get(self, subject, uri):
        return self.entries.get(self.key(subject, uri))


def text_of(result):
    return result["contents"][0]["text"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    original = ref.NOTES[SENSITIVE]["text"]
    sensitive, ordinary = read(ref, SENSITIVE), read(ref, ORDINARY)

    strict = HostCache()
    stored = [strict.store(ALICE, SENSITIVE, sensitive), strict.store(ALICE, ORDINARY, ordinary)]

    falsy = HostCache(honour_zero=False)
    falsy.store(ALICE, SENSITIVE, sensitive)

    shared = HostCache(key_subject=False)
    shared.store(ALICE, ORDINARY, ordinary)

    stale = HostCache(honour_zero=False)
    stale.store(ALICE, SENSITIVE, sensitive)
    try:
        ref.NOTES[SENSITIVE]["text"] = "# Decision\n\nRevised after review."
        served = text_of(stale.get(ALICE, SENSITIVE))
        live = text_of(read(ref, SENSITIVE))
    finally:
        ref.NOTES[SENSITIVE]["text"] = original
    return {
        "hints": {k: sensitive[k] for k in ("ttlMs", "cacheScope")},
        "ordinary_hints": {k: ordinary[k] for k in ("ttlMs", "cacheScope")},
        "stored": stored, "entries": len(strict.entries),
        "strict_ttl": strict.ttl(sensitive), "falsy_ttl": falsy.ttl(sensitive),
        "falsy_entries": len(falsy.entries),
        "cross_subject": shared.get(MALLORY, ORDINARY) is not None,
        "cross_subject_strict": strict.get(MALLORY, ORDINARY) is None,
        "served_stale": served != live, "served": served, "live": live,
        "caches": [n for n in dir(ref)
                   if "cache" in n.lower() and not n.startswith("__")],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the sensitive read is private with ttlMs 0, and the host stores nothing",
            all([result["hints"] == {"ttlMs": 0, "cacheScope": "private"},
                 result["stored"] == [False, True], result["entries"] == 1,
                 result["strict_ttl"] == 0]),
            f"the sensitive read returns {result['hints']} and the strict host refuses it, "
            f"keeping {result['entries']} entry after reading both notes -- the ordinary one, "
            f"whose hints are {result['ordinary_hints']}. The no-store path is the ttl and "
            "not the scope",
        ),
        practice.Check(
            "FINDING: ttlMs 0 is not ttlMs absent, and `or` cannot tell them apart",
            all([result["falsy_ttl"] == DEFAULT_TTL, result["strict_ttl"] == 0,
                 result["falsy_entries"] == 1]),
            f"a host reading `result.get('ttlMs') or {DEFAULT_TTL}` assigns the sensitive note "
            f"{result['falsy_ttl']} and stores it, while `result.get('ttlMs', {DEFAULT_TTL})` "
            f"gives {result['strict_ttl']}. The strictest instruction the protocol has becomes "
            "the default, silently, because 0 is falsy",
        ),
        practice.Check(
            "FINDING: the two ablations leak in different directions",
            all([result["cross_subject"], result["cross_subject_strict"],
                 result["served_stale"]]),
            f"keyed on the URI alone, {ORDINARY} read by alice is served to mallory, which "
            f"keying on the subject prevents. Honouring the scope but not the zero, alice is "
            f"served {result['served'].splitlines()[-1]!r} after the note became "
            f"{result['live'].splitlines()[-1]!r}. Each ablation leaves the other control "
            "working and still loses something",
        ),
        practice.Check(
            "FINDING: neither control is enforced by the server, or could be",
            all([result["caches"] == [], result["ordinary_hints"]["cacheScope"] == "private"]),
            f"resources_read emits the hints and the module has {len(result['caches'])} "
            f"cache-shaped names, so both controls live entirely in the host. Even the "
            f"ordinary note is already {result['ordinary_hints']['cacheScope']!r} -- the "
            "server states a policy it has no machinery to apply, which is why the exercise "
            "asks for the host half",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
