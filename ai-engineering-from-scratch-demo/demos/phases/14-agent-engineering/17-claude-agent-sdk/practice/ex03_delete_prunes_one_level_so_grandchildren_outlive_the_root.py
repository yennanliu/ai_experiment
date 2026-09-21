"""Exercise 3 — delete prunes one level, so grandchildren outlive the root.

    Wire `list_subkeys` to render a subagent tree. What does deep nesting
    look like?

Reading of the exercise: `list_subkeys` returns the children of one session,
so a tree is a recursive walk over it -- three lines. Rendering it is where
the shape becomes visible, and the shape turns out to be the thing that
breaks `delete`, which prunes exactly one level.

**ANSWER: a recursive renderer over `list_subkeys`.** A root with **3**
children and **3** grandchildren each renders **13** lines at depths **0**
to **2**, with the session id spelling the path: `root.sub01.sub04`. Deep
nesting looks like a filename, because the id is built by string
concatenation in `spawn_subagents`.

**FINDING: `delete` prunes one level.** Deleting the root removes the root
and its **3** children and leaves the **9** grandchildren in
`list_sessions()` -- **9** of **13** sessions survive a delete of their
ancestor, reachable by nobody, because the `_subkeys` entry that named them
was removed with their parent.

**FINDING: the last component is a global sequence, not a position.**
`_sub_counter` increments on the `Harness`, not on the parent, so the root's
children are `sub01`, `sub02`, `sub03` and the first child's own children are
`sub04`, `sub05`, `sub06`. The suffix says when a subagent was created across
the whole harness, never where it sits under its parent.

**FINDING: a subagent created outside the spawner is an orphan.**
`run_agent` links a parent only when `parent_session` is passed, so calling
it directly adds a session that `list_subkeys` never returns: **1** extra
session in `list_sessions()` and **0** extra nodes in the tree.

Structure: `tree()` is the renderer; the nesting is built with the lesson's
own `spawn_subagents`, called recursively.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "17-claude-agent-sdk"
FANOUT, DEPTH = 3, 2


def build(ref):
    tools = ref.ToolRegistry()
    tools.register(ref.Tool("read_file", "read", ref._read_file_demo))
    return ref.Harness(tools, ref.Hooks(), ref.SessionStore())


def grow(harness, session, fanout, depth):
    """Spawn `fanout` subagents per level, `depth` levels deep."""
    if depth == 0:
        return
    tasks = [(f"work at {session} #{n}", [("read_file", {"path": "x.py"})])
             for n in range(fanout)]
    for run in harness.spawn_subagents(session, tasks):
        grow(harness, run.session_id, fanout, depth - 1)


def tree(store, session, depth=0):
    rows = [("  " * depth) + session]
    for sub in store.list_subkeys(session):
        rows += tree(store, sub, depth + 1)
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    harness = build(ref)
    harness.store.append("root", ref.Turn("user", "start"))
    grow(harness, "root", FANOUT, DEPTH)
    rendered = tree(harness.store, "root")
    children = harness.store.list_subkeys("root")
    grandchildren = [sub for child in children
                     for sub in harness.store.list_subkeys(child)]
    before = len(harness.store.list_sessions())
    harness.store.delete("root")
    survivors = harness.store.list_sessions()
    orphan = build(ref)
    orphan.store.append("root", ref.Turn("user", "start"))
    orphan.run_agent("loose", "no parent", [])
    return {
        "lines": len(rendered), "depths": sorted({row.count("  ") for row in rendered}),
        "children": len(children), "grandchildren": len(grandchildren),
        "path": grandchildren[0], "dots": grandchildren[0].count("."),
        "before": before, "survivors": len(survivors),
        "leaked": sorted(set(survivors) & set(grandchildren)),
        "sibling_ids": [child.split(".")[-1] for child in children],
        "first_branch": [sub.split(".")[-1]
                         for sub in harness.store.list_subkeys(children[0])],
        "orphan_sessions": len(orphan.store.list_sessions()),
        "orphan_tree": len(tree(orphan.store, "root")),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: ten lines, three depths, and the id spells the path",
            all([result["lines"] == 13, result["depths"] == [0, 1, 2],
                 result["children"] == 3, result["grandchildren"] == 9,
                 result["dots"] == 2, result["path"].startswith("root.sub")]),
            f"a root with {result['children']} children and "
            f"{result['grandchildren']} grandchildren renders {result['lines']} lines at "
            f"depths {result['depths']}, with ids like {result['path']!r} -- "
            f"{result['dots']} dots deep. Nesting looks like a filename because the id "
            "is string concatenation in spawn_subagents",
        ),
        practice.Check(
            "FINDING: delete prunes one level",
            all([result["before"] == 13, result["survivors"] == 9,
                 len(result["leaked"]) == 9]),
            f"deleting the root takes the root and its {result['children']} children and "
            f"leaves {result['survivors']} of {result['before']} sessions behind -- "
            f"exactly the {len(result['leaked'])} grandchildren, reachable by nobody, "
            "because the _subkeys entry naming them went with their parent",
        ),
        practice.Check(
            "FINDING: the last component is a global sequence, not a position",
            all([result["sibling_ids"] == ["sub01", "sub02", "sub03"],
                 result["first_branch"] == ["sub04", "sub05", "sub06"]]),
            f"_sub_counter increments on the Harness rather than the parent, so the "
            f"root's children are {result['sibling_ids']} and the first child's own "
            f"children are {result['first_branch']}. The suffix says when a subagent was "
            "created across the whole harness, never where it sits under its parent",
        ),
        practice.Check(
            "FINDING: a subagent created outside the spawner is an orphan",
            all([result["orphan_sessions"] == 2, result["orphan_tree"] == 1]),
            f"run_agent links a parent only when parent_session is passed, so calling it "
            f"directly leaves {result['orphan_sessions']} sessions in the store and "
            f"{result['orphan_tree']} node in the tree. The tree is a view over "
            "_subkeys, not over the sessions that exist",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
