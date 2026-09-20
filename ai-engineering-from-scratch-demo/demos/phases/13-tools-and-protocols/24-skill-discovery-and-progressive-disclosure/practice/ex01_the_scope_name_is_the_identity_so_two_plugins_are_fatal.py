"""Exercise 1 — the scope name is the identity, so two plugins are fatal.

    Add a plugin scope and place it between user and built-in precedence.
    Prove the collision result with a test.

Reading of the exercise: "place it between" is a claim about three
neighbours, so one collision is not a proof -- the test drops the winning
scope one at a time and reads the sequence of winners, which pins plugin's
position from both sides at once. Building the fourth scope then exposes what
`Scope` treats as identity, and it is not the root.

**ANSWER: plugin sits third, and removing scopes one at a time walks the
order.** With all four present the winner is `project` and the shadowed list
is `('user', 'plugin', 'builtin')`; dropping the winner each time yields
`project -> user -> plugin -> builtin`. **1** collision is recorded and **4**
candidates went in.

**FINDING: adding a scope is two edits, and forgetting the second is a hard
error.** `resolve_collisions` knows scopes only through the caller's
precedence tuple, so a candidate whose scope is missing from it raises
`CollisionError` rather than defaulting to the bottom. Fail-closed is the
right choice here and it means the policy cannot be inferred from the roots.

**FINDING: the scope *name* is the identity, so two plugin bundles are
fatal.** Two `Scope("plugin", ...)` objects over different roots produce
equal ranks, and equal precedence raises `ambiguous`. A host with more than
one plugin source has to name them apart before discovery, because nothing
downstream can tell them apart.

**FINDING: the shadowing survives in diagnostics and is invisible to the
model.** `CatalogEntry` has **4** fields and none of them is the `selected`
flag the lesson's own example shows; the `collisions` tuple lives outside
`model_dict()`. The model sees one `evidence-report` and no sign that three
others were rejected.

Structure: `winners_in_order()` is the positional proof; `plant()` writes one
named skill per scope so every run has the same four candidates.
"""

from __future__ import annotations

import dataclasses
import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "24-skill-discovery-and-progressive-disclosure"
PRECEDENCE = ("project", "user", "plugin", "builtin")
SKILL = "evidence-report"


def plant(ref, base, scope):
    """One scope root holding one skill of the contested name."""
    root = base / scope
    ref._write_skill(root, SKILL, f"Create the {scope} evidence report when findings land.",
                     f"# {scope} evidence report\n\nRead `references/format.md`.")
    return ref.Scope(scope, root)


def candidates_for(ref, scopes):
    return tuple(candidate for scope in scopes for candidate in ref.discover_scope(scope))


def winners_in_order(ref, scopes):
    """Drop the winning scope repeatedly; the sequence is the precedence."""
    order, remaining = [], list(scopes)
    while remaining:
        winners, _ = ref.resolve_collisions(candidates_for(ref, remaining), PRECEDENCE)
        winner = next(w for w in winners if w.name == SKILL)
        order.append(winner.scope)
        remaining = [scope for scope in remaining if scope.name != winner.scope]
    return order


def failure(call):
    try:
        call()
    except Exception as exc:  # the class is part of what is being reported
        return f"{type(exc).__name__}: {exc}"
    return None


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as directory:
        base = pathlib.Path(directory)
        scopes = [plant(ref, base, scope) for scope in PRECEDENCE]
        every = candidates_for(ref, scopes)
        winners, collisions = ref.resolve_collisions(every, PRECEDENCE)
        order = winners_in_order(ref, scopes)

        missing = failure(lambda: ref.resolve_collisions(every, ("project", "user", "builtin")))
        twin = plant(ref, base / "second", "plugin")
        doubled = candidates_for(ref, [scopes[2], twin])
        ambiguous = failure(lambda: ref.resolve_collisions(doubled, PRECEDENCE))
        ambiguous = ambiguous.split(" at equal")[0] + " at equal precedence"

        catalog = ref.build_catalog(every, PRECEDENCE, ref.CatalogBudget())
        return {
            "candidates": len(every), "winner": winners[0].scope,
            "shadowed": list(collisions[0].shadowed_scopes),
            "collisions": len(collisions), "order": order,
            "missing": missing, "ambiguous": ambiguous,
            "twin_roots": len({str(scopes[2].root), str(twin.root)}),
            "twin_names": len({scopes[2].name, twin.name}),
            "entry_fields": [f.name for f in dataclasses.fields(ref.CatalogEntry)],
            "model_keys": sorted(catalog.model_dict()),
            "report_keys": sorted(catalog.to_dict()),
            "model_entries": len(catalog.model_dict()["entries"]),
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: plugin sits third, and dropping the winner each time walks the order",
            all([result["candidates"] == 4, result["winner"] == "project",
                 result["shadowed"] == ["user", "plugin", "builtin"],
                 result["collisions"] == 1, result["order"] == list(PRECEDENCE)]),
            f"{result['candidates']} candidates produce {result['collisions']} collision won "
            f"by {result['winner']!r} over {result['shadowed']}, and dropping the winner "
            f"each time yields {result['order']} -- which pins plugin between user and "
            "builtin from both sides rather than asserting it from one",
        ),
        practice.Check(
            "FINDING: adding a scope is two edits, and forgetting the second is a hard error",
            result["missing"] is not None and result["missing"].startswith("CollisionError"),
            f"resolve_collisions knows scopes only through the caller's precedence tuple, so "
            f"leaving plugin out of it raises {result['missing']!r} rather than defaulting "
            "to the bottom. Fail-closed is right, and it means the order cannot be inferred "
            "from the roots -- it has to be declared",
        ),
        practice.Check(
            "FINDING: the scope name is the identity, so two plugin bundles are fatal",
            all([result["twin_roots"] == 2, result["twin_names"] == 1,
                 result["ambiguous"] is not None, "ambiguous" in result["ambiguous"]]),
            f"two Scope objects over {result['twin_roots']} roots share "
            f"{result['twin_names']} name, so their ranks tie and resolution raises "
            f"{result['ambiguous']!r}. A host with two plugin sources has to name them apart "
            "before discovery, because nothing downstream can tell them apart",
        ),
        practice.Check(
            "FINDING: the shadowing survives in diagnostics and is invisible to the model",
            all([len(result["entry_fields"]) == 4, "selected" not in result["entry_fields"],
                 result["model_keys"] == ["entries"], "collisions" in result["report_keys"],
                 result["model_entries"] == 1]),
            f"CatalogEntry carries {result['entry_fields']} -- no selected flag, unlike the "
            f"lesson's own example -- and model_dict() exposes {result['model_keys']} while "
            f"the report also carries {result['report_keys']}. The model sees "
            f"{result['model_entries']} evidence-report and no sign three were rejected",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
