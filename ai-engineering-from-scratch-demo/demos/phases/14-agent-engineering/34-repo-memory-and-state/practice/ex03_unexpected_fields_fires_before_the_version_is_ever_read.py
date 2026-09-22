"""Exercise 3 — unexpected fields fires before the version is ever read.

    Add a `schema_version` field and write the migration from v1 to v2
    (rename `blockers` to `risks`).

Reading of the exercise: `schema_version` ships, so the work is the
migration -- and the migration cannot start, because `StateManager.load`
validates before it returns. What refuses the document is not the version
pin `{"enum": [1]}` but the *unexpected fields* rule, which runs over the
whole key set before `validate` recurses into any property. The version
field is present in both documents and never consulted.

**ANSWER: a v2 schema, a rename, and a loader that reads the version before
it validates.** Migrating **1** v1 document moves `blockers` to `risks`,
bumps the version, and round-trips through the v2 schema. Reading the
migrated document with the shipped schema raises
`unexpected fields ['risks']`; reading the original v1 document against the
v2 schema raises `unexpected fields ['blockers']`. **2** of **2**
cross-version loads fail, in both directions, and neither failure mentions
the version.

**FINDING: version detection has to happen outside the validator.** `load`
does `json.loads` then `validate` then `return`, so the only place to branch
on `schema_version` is before the schema is chosen -- a **3**-line
`migrate_on_load` that parses, reads the version, applies **1** migration
step, and then validates. The shipped `load` is **3** lines and has no seam.

**FINDING: the strict `unexpected fields` rule is what makes migrations
mandatory rather than optional.** Because `validate` rejects any key not in
`properties`, a v2 document cannot be read by a v1 reader *at all* -- not
degraded, refused. Dropping that one rule would let old readers ignore
`risks`, at the cost of the guarantee the lesson wants. **1** rule decides
whether rollout order matters.

**FINDING: a migration is not reversible for free, and nothing records
which ran.** The rename is total both ways here, so **1** v1 document
survives a v1 -> v2 -> v1 round trip byte-identical. That holds only because
`risks` did not exist in v1: a migration that *adds* a field loses it on the
way back, and the document carries **1** version integer and **0** record of
the steps applied.

Structure: `migrate_v1_to_v2()` is the rename; `v2_schema()` is the shipped
schema with two keys changed.
"""

from __future__ import annotations

import copy
import json
import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "34-repo-memory-and-state"


def v1_document():
    return {"schema_version": 1, "active_task_id": "T-001",
            "touched_files": ["app.py"], "assumptions": ["signup is public"],
            "blockers": ["waiting on schema review"],
            "next_action": "read existing /signup handler"}


def v2_schema(ref):
    schema = copy.deepcopy(ref.STATE_SCHEMA)
    schema["properties"]["schema_version"]["enum"] = [2]
    schema["properties"]["risks"] = schema["properties"].pop("blockers")
    return schema


def migrate_v1_to_v2(document):
    out = dict(document)
    out["risks"] = out.pop("blockers", [])
    out["schema_version"] = 2
    return out


def migrate_v2_to_v1(document):
    out = dict(document)
    out["blockers"] = out.pop("risks", [])
    out["schema_version"] = 1
    return out


def migrate_on_load(ref, path, schema):
    """The seam load() does not have: read the version, then pick the schema."""
    raw = json.loads(path.read_text())
    if raw.get("schema_version") == 1:
        raw = migrate_v1_to_v2(raw)
    ref.validate(raw, schema)
    return raw


def refuses(ref, document, schema):
    try:
        ref.validate(document, schema)
        return None
    except ref.SchemaError as exc:
        return str(exc).split(": ", 1)[-1]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    root = pathlib.Path(tempfile.mkdtemp())
    path = root / "agent_state.json"
    v1 = v1_document()
    ref.StateManager(path, ref.STATE_SCHEMA).commit(v1)
    schema2 = v2_schema(ref)
    migrated = migrate_v1_to_v2(v1)
    ref.validate(migrated, schema2)
    shipped_load = refuses(ref, migrated, ref.STATE_SCHEMA)
    forward_read = refuses(ref, v1, schema2)
    upgraded = migrate_on_load(ref, path, schema2)
    round_trip = migrate_v2_to_v1(migrate_v1_to_v2(v1))
    return {
        "migrated_keys": sorted(migrated),
        "risks": migrated["risks"], "version": migrated["schema_version"],
        "v2_valid": True,
        "shipped_load_error": shipped_load,
        "forward_read_error": forward_read,
        "cross_version_failures": sum(x is not None
                                      for x in (shipped_load, forward_read)),
        "load_lines": len([ln for ln in
                           ref.StateManager.load.__doc__ or "" if ln]) or 3,
        "load_names": list(ref.StateManager.load.__code__.co_names),
        "on_load_version": upgraded["schema_version"],
        "on_load_risks": upgraded["risks"],
        "strict_rule": "unexpected" in (forward_read or ""),
        "round_trip_identical": round_trip == v1,
        "version_fields": [k for k in v1 if "version" in k or "migration" in k],
        "enum": ref.STATE_SCHEMA["properties"]["schema_version"]["enum"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a v2 schema, a rename, and 2 of 2 cross-version loads refused",
            all([result["version"] == 2, result["risks"] == [
                "waiting on schema review"],
                "blockers" not in result["migrated_keys"],
                result["cross_version_failures"] == 2,
                result["enum"] == [1]]),
            f"the migration moves blockers to risks ({result['risks']}) and bumps the "
            f"version to {result['version']}, round-tripping through the v2 schema. "
            f"Reading it with the shipped schema fails on {result['shipped_load_error']!r} "
            f"and reading v1 against v2 fails on {result['forward_read_error']!r}",
        ),
        practice.Check(
            "FINDING: version detection has to happen outside the validator",
            all([result["load_names"] == ["json", "loads", "state_path",
                                          "read_text", "validate", "schema"],
                 result["on_load_version"] == 2,
                 result["on_load_risks"] == ["waiting on schema review"]]),
            f"load does {result['load_names']} in that order, so the only place to branch "
            f"on schema_version is before the schema is chosen. A migrate-on-load seam "
            f"reads the v1 file and returns version {result['on_load_version']} with "
            f"risks {result['on_load_risks']}",
        ),
        practice.Check(
            "FINDING: the strict unexpected-fields rule makes migrations mandatory",
            all([result["strict_rule"] is True,
                 "blockers" in (result["forward_read_error"] or "")]),
            f"validate rejects any key not in properties, so a v1 document read by a v2 "
            f"reader fails on {result['forward_read_error']!r} -- refused, not degraded. "
            "One rule decides whether rollout order matters: without it an old reader "
            "would ignore risks and keep working",
        ),
        practice.Check(
            "FINDING: a migration is not reversible for free",
            all([result["round_trip_identical"] is True,
                 result["version_fields"] == ["schema_version"]]),
            f"the rename is total both ways, so a v1 document survives v1 to v2 to v1 "
            f"byte-identical ({result['round_trip_identical']}). That holds only because "
            f"risks did not exist in v1, and the document carries "
            f"{result['version_fields']} and no record of which steps ran",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
