"""Exercise 4 — the scanner reads prose and the digest reads the whole thing.

    Change a tool's `inputSchema` without changing its description. Confirm
    whole-descriptor pinning catches it.

Reading of the exercise: "without changing its description" selects the one
mutation the prose scanner is blind to by construction, so the confirmation
is worth little unless the blindness is measured alongside it. Both halves are
therefore run on the same edited catalog -- what `scan_description` finds, and
what `descriptor_digest` finds -- and then the pin is probed for the changes
it should *not* flag.

**ANSWER: the scanner finds nothing and the pin finds a rug pull.** Adding
`maxLength: 10` to `notes.search`'s query property leaves the description
byte-identical, so `scan_description` returns **0** labels while
`scan_catalog` reports `rug_pull` on `notes.search`. The tool then disappears
from `tools/list`: **3** visible tools become **2**.

**FINDING: pinning the description alone would have missed it.** The
description's own digest is unchanged, and **5** injection patterns match
**0** times. A pin over the field a human reads cannot see a change to the
field a model is constrained by -- which is the whole argument for hashing
the descriptor rather than its prose.

**FINDING: the digest is canonical, so a reordering is not a change.**
`descriptor_digest` dumps with `sort_keys=True`, so rebuilding the same tool
with its keys in a different order produces the identical hash and **0**
findings. The pin distinguishes content from spelling, which is what keeps it
from crying wolf on a serializer change.

**FINDING: one of the catalog's findings can never block anything.**
`_visible_tools` blocks keys containing a dot, and `shadowing` is reported
under the bare tool name -- `search`, from `issues.search` and `notes.search`
-- so it is emitted on every scan and filters nothing. A finding that cannot
block is a log line, and the two colliding tools stay visible.

Structure: `edited` returns a gateway whose catalog has been mutated in one
named way, so the scanner and the pin see exactly the same input.
"""

from __future__ import annotations

import copy
import hashlib

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "15-mcp-security-tool-poisoning"


def edited(ref, mutate):
    gateway = ref.SecurityGateway()
    mutate(gateway.catalog)
    return gateway


def add_max_length(catalog):
    catalog["notes"][0]["inputSchema"]["properties"]["query"]["maxLength"] = 10


def reorder(catalog):
    tool = catalog["notes"][0]
    catalog["notes"][0] = {key: tool[key] for key in reversed(list(tool))}


def listed(ref, gateway):
    body, headers = ref.make_request("tools/list", 1)
    return [tool["name"] for tool in gateway.handle(body, headers)[1]["result"]["tools"]]


def kinds(findings):
    return sorted({(finding.kind, finding.key) for finding in findings})


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    clean = ref.SecurityGateway()
    schema = edited(ref, add_max_length)
    shuffled = edited(ref, reorder)

    before = copy.deepcopy(clean.catalog["notes"][0])
    after = schema.catalog["notes"][0]
    text_digest = [hashlib.sha256(t["description"].encode()).hexdigest()
                   for t in (before, after)]
    return {
        "same_description": before["description"] == after["description"],
        "description_digest_same": text_digest[0] == text_digest[1],
        "scanner": ref.scan_description(after["description"]),
        "patterns": len(ref.INJECTION_PATTERNS),
        "digest_changed": (ref.descriptor_digest(before) != ref.descriptor_digest(after)),
        "schema_findings": kinds(ref.scan_catalog(schema.catalog, schema.approved)),
        "clean_findings": kinds(ref.scan_catalog(clean.catalog, clean.approved)),
        "visible_clean": listed(ref, clean), "visible_schema": listed(ref, schema),
        "reorder_digest_same": (ref.descriptor_digest(before)
                                == ref.descriptor_digest(shuffled.catalog["notes"][0])),
        "reorder_findings": kinds(ref.scan_catalog(shuffled.catalog, shuffled.approved)),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the scanner finds nothing and the pin finds a rug pull",
            all([result["same_description"], result["scanner"] == [],
                 result["digest_changed"],
                 ("rug_pull", "notes.search") in result["schema_findings"],
                 result["visible_clean"] == ["issues.search", "notes.export", "notes.search"],
                 result["visible_schema"] == ["issues.search", "notes.export"]]),
            f"adding maxLength to the query property leaves the description identical, so "
            f"scan_description returns {result['scanner']} while scan_catalog reports "
            f"{[f for f in result['schema_findings'] if f[0] == 'rug_pull']}. The tool then "
            f"disappears from tools/list: {len(result['visible_clean'])} visible become "
            f"{len(result['visible_schema'])}",
        ),
        practice.Check(
            "FINDING: pinning the description alone would have missed it",
            all([result["description_digest_same"], result["scanner"] == [],
                 result["patterns"] == 5]),
            f"the description's own digest is unchanged and all {result['patterns']} "
            "injection patterns match 0 times. A pin over the field a human reads cannot see "
            "a change to the field a model is constrained by -- which is the argument for "
            "hashing the descriptor rather than its prose",
        ),
        practice.Check(
            "FINDING: the digest is canonical, so a reordering is not a change",
            all([result["reorder_digest_same"],
                 result["reorder_findings"] == result["clean_findings"]]),
            "descriptor_digest dumps with sort_keys=True, so rebuilding the same tool with "
            "its keys reversed gives the identical hash and the same findings as the "
            "untouched catalog. The pin distinguishes content from spelling, which keeps it "
            "from crying wolf on a serializer change",
        ),
        practice.Check(
            "FINDING: one of the catalog's findings can never block anything",
            all([("shadowing", "search") in result["clean_findings"],
                 result["visible_clean"] == ["issues.search", "notes.export",
                                             "notes.search"]]),
            f"_visible_tools blocks keys containing a dot, and shadowing is reported under "
            f"the bare name 'search'. It is emitted on every scan -- "
            f"{result['clean_findings']} on an untouched catalog -- and filters nothing: "
            f"both colliding tools stay in {result['visible_clean']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
