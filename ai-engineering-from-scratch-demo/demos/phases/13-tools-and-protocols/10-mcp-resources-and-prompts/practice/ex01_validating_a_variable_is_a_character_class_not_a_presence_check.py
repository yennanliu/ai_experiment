"""Exercise 1 — validating a variable is a character class, not a presence check.

    Add a `notes://projects/{project}/notes/{id}` resource template and
    validate both variables.

Reading of the exercise: "validate both variables" is read as a question about
which strings the template can and cannot have produced, not as a null check.
The test is therefore a URI that satisfies a presence check and still cannot
have come from this template -- `notes://projects/a/b/notes/1` -- because a
template that accepts it is not a grammar, it is a prefix match.

**ANSWER: `[^/]+` for both variables, matched after decoding.** `apollo` / `7`
round-trips through expand and parse; an empty project, an empty id, and an
embedded slash are all rejected, **4** rejections against **1** acceptance on
the same pattern.

**FINDING: a presence check does not merely over-accept, it is ambiguous.**
On `notes://projects/x/notes/y/notes/z` a greedy `(.+)` reads
`project="x/notes/y"`, a lazy `(.+?)` reads `project="x"`, and both pass a
non-empty test — so the template no longer determines which values it
accepted. `[^/]+` has one reading or none, and rejects that URI outright, as
it rejects `project="a/b"`. Validation is the character class; the capture
group is just where the class is written.

**FINDING: percent-encoding puts the ambiguity back if you decode too late.**
`notes://projects/a%2Fb/notes/1` satisfies `[^/]+` — `%2F` has no slash in it —
and `unquote` then yields `project="a/b"`, the value the strict pattern was
written to refuse. Decoding after validating validates the wrong string, so
the two steps are ordered and the exercise's "both variables" means both
decoded variables.

**FINDING: the template describes a route this server does not serve, and has
no capability to be advertised under.** `resources_read` is an exact lookup in
`NOTES`, so an expanded URI answers **-32602 Unknown or invalid resource
URI**; `HANDLERS` has no `resources/templates/list`; and `server_discover`'s
resources capability is `{'listChanged': True, 'subscribe': True}` with no
templates key. The grammar is well-formed and unreachable.

Structure: `STRICT`, `LOOSE` and `LAZY` are three readings of the template,
`captured` runs one of them and decodes, `valid` is the character class on its
own, and `expand` is the inverse used for the round trip.
"""

from __future__ import annotations

import re
import urllib.parse

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "10-mcp-resources-and-prompts"
TEMPLATE = "notes://projects/{project}/notes/{id}"
STRICT = re.compile(r"^notes://projects/(?P<project>[^/]+)/notes/(?P<id>[^/]+)$")
LOOSE = re.compile(r"^notes://projects/(?P<project>.+)/notes/(?P<id>.+)$")
LAZY = re.compile(r"^notes://projects/(?P<project>.+?)/notes/(?P<id>.+)$")
CASES = ["notes://projects/apollo/notes/7", "notes://projects//notes/7",
         "notes://projects/apollo/notes/", "notes://projects/a/b/notes/1",
         "notes://projects/apollo/notes/7/extra"]
DOUBLED = "notes://projects/x/notes/y/notes/z"
ENCODED = "notes://projects/a%2Fb/notes/1"


def expand(project, note_id):
    return TEMPLATE.replace("{project}", project).replace("{id}", note_id)


def captured(uri, pattern=STRICT, decode=True):
    """The variables `pattern` finds in `uri`, decoded, before validation."""
    match = pattern.match(uri)
    if match is None:
        return None
    return {k: urllib.parse.unquote(v) if decode else v
            for k, v in match.groupdict().items()}


def valid(values):
    """Both variables are present and hold no separator -- the whole check."""
    return values is not None and all(v and "/" not in v for v in values.values())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    accepted = {uri: valid(captured(uri)) for uri in CASES}
    good = captured(CASES[0])
    try:
        ref.resources_read({"uri": CASES[0]})
        read_error = None
    except ref.RpcError as exc:
        read_error = exc.code
    return {
        "accepted": accepted,
        "good": good, "round_trip": expand(good["project"], good["id"]) == CASES[0],
        "loose_slash": captured(CASES[3], LOOSE, decode=False),
        "loose_valid": valid(captured(CASES[3], LOOSE, decode=False)),
        "greedy": captured(DOUBLED, LOOSE, decode=False),
        "lazy": captured(DOUBLED, LAZY, decode=False),
        "strict_doubled": captured(DOUBLED),
        "encoded_late": captured(ENCODED, decode=False),
        "encoded_late_valid": valid(captured(ENCODED, decode=False)),
        "encoded_decoded": captured(ENCODED),
        "encoded_valid": valid(captured(ENCODED)),
        "read_error": read_error,
        "handlers": sorted(ref.HANDLERS),
        "resources_capability": sorted(ref.server_discover({})["capabilities"]["resources"]),
    }


def verify(result):
    accepted = result["accepted"]
    return [
        practice.Check(
            "ANSWER: [^/]+ for both variables, matched after decoding, accepts one of five",
            all([accepted == dict(zip(CASES, [True, False, False, False, False])),
                 result["good"] == {"project": "apollo", "id": "7"}, result["round_trip"]]),
            f"the strict template accepts {result['good']} and round-trips back to "
            f"{CASES[0]}, while an empty project, an empty id, an embedded slash and a "
            f"trailing segment are all rejected -- {sum(accepted.values())} acceptance "
            f"against {len(accepted) - sum(accepted.values())} rejections",
        ),
        practice.Check(
            "FINDING: a presence check does not merely over-accept, it is ambiguous",
            all([result["greedy"] == {"project": "x/notes/y", "id": "z"},
                 result["lazy"] == {"project": "x", "id": "y/notes/z"},
                 result["strict_doubled"] is None,
                 result["loose_slash"] == {"project": "a/b", "id": "1"},
                 not result["loose_valid"]]),
            f"on {DOUBLED} a greedy (.+) reads {result['greedy']} and a lazy (.+?) reads "
            f"{result['lazy']} -- both non-empty, so the template no longer determines which "
            f"values it accepted. [^/]+ has one reading or none and rejects that URI, as it "
            f"rejects the {result['loose_slash']['project']!r} the greedy pattern captures "
            f"from {CASES[3]}",
        ),
        practice.Check(
            "FINDING: percent-encoding puts the ambiguity back if you decode too late",
            all([result["encoded_late"] == {"project": "a%2Fb", "id": "1"},
                 result["encoded_late_valid"],
                 result["encoded_decoded"] == {"project": "a/b", "id": "1"},
                 not result["encoded_valid"]]),
            f"{ENCODED} satisfies [^/]+ because %2F holds no slash, so validating before "
            f"decoding accepts {result['encoded_late']}; unquoting first yields "
            f"{result['encoded_decoded']} and the same check rejects it. The two steps are "
            "ordered, and 'both variables' means both decoded ones",
        ),
        practice.Check(
            "FINDING: the grammar is well-formed and unreachable",
            all([result["read_error"] == -32602,
                 "resources/templates/list" not in result["handlers"],
                 result["resources_capability"] == ["listChanged", "subscribe"]]),
            f"resources_read is an exact lookup in NOTES, so an expanded URI answers "
            f"{result['read_error']}; HANDLERS is {result['handlers']} with no templates "
            f"method; and the advertised resources capability is "
            f"{result['resources_capability']} with no key to announce a template under",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
