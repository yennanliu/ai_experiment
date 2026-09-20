"""Exercise 2 — a qualified name is not a name the loader can find.

    Change the collision policy from highest precedence to qualified names.
    Preserve both entries in the catalog.

Reading of the exercise: "preserve both entries" is the requirement, so the
new policy is written to produce the same serialization the shipped one does
and the two are measured side by side rather than described. Doing that
surfaces the part the exercise does not mention -- what the rest of the
module does with a name once it is no longer the skill's name.

**ANSWER: every candidate survives, qualified as `scope::name`, and the
catalog grows by the entry it used to drop.** Highest-precedence publishes
**2** entries with **1** collision recorded; qualification publishes **3**
with **0**. The model-facing cost rises from **254** to **403** characters --
**59%** for one duplicate.

**FINDING: a qualified name is not a name the loader can find.**
`_candidate_for` matches on the name and the directory together, so an entry
named `project::evidence-report` raises `KeyError` and both `load_skill_body`
and `load_reference` stop working. The policy is four lines in the catalog
builder and it breaks a function 90 lines away that never mentions scopes.

**FINDING: a qualified name is not a portable skill name either.**
`NAME_PATTERN` admits lowercase kebab-case only, so `project::evidence-report`
fails the same check discovery applies, and a catalog written back out as
packages could not be rediscovered. The qualifier has to live in a second
field, which is what `scope` already is.

**FINDING: the ambiguity is not removed, it is moved.** Under precedence the
duplicate is a `Collision` in the diagnostics and the model sees one name;
under qualification the diagnostics are empty and the model sees two names
that differ by a prefix it was told nothing about. **0** collisions is a
worse signal than one.

Structure: `qualified()` is the new policy, written against the same
`CatalogEntry` and the same serializer so `cost()` compares like with like.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
from dataclasses import asdict

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "24-skill-discovery-and-progressive-disclosure"
PRECEDENCE = ("project", "user")
SKILL = "evidence-report"


def cost(entries):
    """The module's measure, with the install path folded to its last segment.

    `CatalogEntry.directory` is absolute, so the shipped number depends on where
    the bundle happens to live. Comparing two policies needs a measure that does
    not move when the tree does.
    """
    rows = [{**asdict(entry), "directory": entry.directory.rsplit("/", 1)[-1]}
            for entry in entries]
    return len(json.dumps({"entries": rows}, sort_keys=True, separators=(",", ":")))


def qualified(ref, candidates, precedence, budget):
    """Keep every candidate; identity becomes scope::name."""
    rank = {scope: index for index, scope in enumerate(precedence)}
    ordered = sorted(candidates, key=lambda item: (rank[item.scope], item.name))
    return tuple(
        ref.CatalogEntry(name=f"{candidate.scope}::{candidate.name}",
                         description=ref._shorten(candidate.description,
                                                  budget.max_description_chars),
                         scope=candidate.scope, directory=str(candidate.directory))
        for candidate in ordered)


def plant(ref, base, scope, name, description):
    ref._write_skill(base / scope, name, description, f"# {name}\n\nBody for {scope}.")


def failure(call):
    try:
        call()
    except Exception as exc:
        return type(exc).__name__
    return None


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    budget = ref.CatalogBudget()
    with tempfile.TemporaryDirectory() as directory:
        base = pathlib.Path(directory)
        plant(ref, base, "project", SKILL, "Report this project's audit evidence.")
        plant(ref, base, "user", SKILL, "Report evidence for any audit findings.")
        plant(ref, base, "user", "meeting-brief", "Prepare a brief from notes.")
        scopes = [ref.Scope(name, base / name) for name in PRECEDENCE]
        candidates = tuple(c for scope in scopes for c in ref.discover_scope(scope))
        shipped = ref.build_catalog(candidates, PRECEDENCE, budget)
        entries = qualified(ref, candidates, PRECEDENCE, budget)

        broken = failure(lambda: ref.load_skill_body(entries[0], candidates))
        repaired = ref.CatalogEntry(SKILL, entries[0].description, entries[0].scope,
                                    entries[0].directory)
        body = ref.load_skill_body(repaired, candidates)
        return {
            "shipped_entries": len(shipped.entries), "qualified_entries": len(entries),
            "shipped_collisions": len(shipped.collisions), "qualified_collisions": 0,
            "shipped_names": [entry.name for entry in shipped.entries],
            "qualified_names": [entry.name for entry in entries],
            "shipped_cost": cost(shipped.entries), "qualified_cost": cost(entries),
            "growth": round(cost(entries) / cost(shipped.entries) - 1, 2),
            "broken": broken, "repaired": body.splitlines()[0],
            "portable": bool(ref.NAME_PATTERN.fullmatch(entries[0].name)),
            "plain_portable": bool(ref.NAME_PATTERN.fullmatch(SKILL)),
            "scope_field": entries[0].scope,
            "shadowed": list(shipped.collisions[0].shadowed_scopes),
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: every candidate survives as scope::name, and the catalog grows",
            all([result["shipped_entries"] == 2, result["qualified_entries"] == 3,
                 result["shipped_collisions"] == 1, result["qualified_collisions"] == 0,
                 result["qualified_names"] == ["project::evidence-report",
                                               "user::evidence-report", "user::meeting-brief"],
                 result["qualified_cost"] > result["shipped_cost"]]),
            f"highest-precedence publishes {result['shipped_entries']} entries "
            f"{result['shipped_names']} with {result['shipped_collisions']} collision; "
            f"qualification publishes {result['qualified_entries']}, "
            f"{result['qualified_names']}. The model-facing cost rises from "
            f"{result['shipped_cost']} to {result['qualified_cost']} characters, "
            f"{result['growth']:.0%} for one duplicate",
        ),
        practice.Check(
            "FINDING: a qualified name is not a name the loader can find",
            all([result["broken"] == "KeyError", result["repaired"].startswith("#")]),
            f"_candidate_for matches on the name and the directory together, so a "
            f"{result['qualified_names'][0]!r} entry raises {result['broken']} and both "
            f"load_skill_body and load_reference stop working; restoring the plain name "
            f"loads {result['repaired']!r}. Four lines in the catalog builder break a "
            "function that never mentions scopes",
        ),
        practice.Check(
            "FINDING: a qualified name is not a portable skill name either",
            all([not result["portable"], result["plain_portable"]]),
            f"NAME_PATTERN admits lowercase kebab-case, so {result['qualified_names'][0]!r} "
            f"fails the check discovery itself applies while {SKILL!r} passes. A catalog "
            "written back out as packages could not be rediscovered, so the qualifier "
            f"belongs in a second field -- which is what scope={result['scope_field']!r} is",
        ),
        practice.Check(
            "FINDING: the ambiguity is not removed, it is moved",
            all([result["shadowed"] == ["user"], result["qualified_collisions"] == 0,
                 result["qualified_names"][:2] == ["project::evidence-report",
                                                   "user::evidence-report"]]),
            f"under precedence the duplicate is a Collision shadowing {result['shadowed']} "
            f"and the model sees one name; under qualification the diagnostics hold "
            f"{result['qualified_collisions']} and the model sees two names differing by a "
            "prefix it was told nothing about. Zero collisions is the worse signal",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
