"""Exercise 3 — the order is deterministic because the names happen to be unique.

    Reverse the in-memory tool registry. Confirm `tools/list` still returns the
    same deterministic order.

Reading of the exercise: the registry is reversed in place, `tools/list` is
called on both arrangements, and the two outputs are compared -- then the same
test is run on a registry the lesson does not ship, because "deterministic" is a
claim about all inputs and the shipped registry has a property that makes it
true for free.

**ANSWER: the same order, both ways.** `['notes_list', 'notes_search']` before
and after reversing `TOOLS`, because `dispatch` returns
`sorted(TOOLS, key=lambda tool: tool["name"])` rather than the registry itself.
The registry's own order never reaches the wire.

**FINDING: the determinism comes from the names being unique, not from the
sort.** `sorted` is stable, so equal keys keep their input order. Add a second
tool also called `notes_list` and reversing the registry **changes the
output** -- the duplicate and the original swap places while the sorted names
stay identical. A client diffing `tools/list` by name would see no change and
get a different tool.

**FINDING: nothing in the lesson would catch that.** The registry is a module
list with no uniqueness check, `tools/list` does not deduplicate, and
`tools/call` dispatches on a hard-coded `if name == ...` chain rather than a
lookup -- so a duplicate name is unreachable by call and invisible by list.
Lesson 13.05's linter has the rule this needs; this lesson does not import it.

**FINDING: and only one of the three methods sorts anything.**
`server/discover` returns capabilities and an instruction string, and
`tools/call` returns content -- neither has a collection whose order a client
could depend on. The determinism guarantee covers **1** of the **3** methods,
which is the one where it matters.

Structure: `listed` calls the lesson's own `tools/list` under a temporarily
swapped registry, `with_registry` restores it, and `DUPLICATE` is the entry that
breaks the tie.
"""

from __future__ import annotations

import contextlib
import inspect

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "06-mcp-fundamentals"
DUPLICATE = {"name": "notes_list", "description": "A second tool with a taken name.",
             "inputSchema": {"type": "object", "properties": {}}}


@contextlib.contextmanager
def with_registry(ref, tools):
    original = list(ref.TOOLS)
    ref.TOOLS[:] = tools
    try:
        yield
    finally:
        ref.TOOLS[:] = original


def listed(ref, tools, field="name"):
    with with_registry(ref, tools):
        response = ref.dispatch(ref.make_request(1, "tools/list"))
    return [tool[field] for tool in response["result"]["tools"]]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = list(ref.TOOLS)
    forward = listed(ref, shipped)
    backward = listed(ref, list(reversed(shipped)))
    duped = shipped + [DUPLICATE]
    dup_forward = listed(ref, duped, "description")
    dup_backward = listed(ref, list(reversed(duped)), "description")
    discover = ref.dispatch(ref.make_request(2, "server/discover"))["result"]
    call = ref.dispatch(
        ref.make_request(3, "tools/call", {"name": "notes_list"}))["result"]
    source = inspect.getsource(ref.dispatch)
    return {
        "forward": forward, "backward": backward, "stable": forward == backward,
        "shipped_names": sorted(tool["name"] for tool in shipped),
        "unique": len({tool["name"] for tool in shipped}) == len(shipped),
        "dup_names_forward": listed(ref, duped),
        "dup_names_backward": listed(ref, list(reversed(duped))),
        "dup_forward": dup_forward, "dup_backward": dup_backward,
        "dup_names_agree": listed(ref, duped) == listed(ref, list(reversed(duped))),
        "dup_bodies_agree": dup_forward == dup_backward,
        "sorts": source.count("sorted("),
        "dispatches_by_chain": source.count('name == "'),
        "discover_keys": sorted(k for k in discover if k not in ("_meta", "resultType")),
        "call_keys": sorted(k for k in call if k not in ("_meta", "resultType")),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the same order, both ways",
            all([result["forward"] == ["notes_list", "notes_search"],
                 result["stable"], result["backward"] == result["forward"]]),
            f"tools/list returns {result['forward']} before and after reversing TOOLS, "
            "because dispatch returns sorted(TOOLS, key=name) rather than the registry "
            "itself. The registry's own order never reaches the wire",
        ),
        practice.Check(
            "FINDING: the determinism comes from the names being unique, not the sort",
            all([result["unique"], result["dup_names_agree"],
                 not result["dup_bodies_agree"],
                 result["dup_forward"] != result["dup_backward"]]),
            f"sorted is stable, so equal keys keep their input order. The shipped names "
            f"{result['shipped_names']} are unique, so nothing ties. Add a second "
            f"notes_list and the names still agree ({result['dup_names_forward']}) while the "
            f"bodies do not -- {result['dup_forward']} against {result['dup_backward']}. A "
            "client diffing by name would see no change and get a different tool",
        ),
        practice.Check(
            "FINDING: nothing in the lesson would catch that",
            all([result["dispatches_by_chain"] >= 1, result["unique"]]),
            f"TOOLS is a module list with no uniqueness check, tools/list does not "
            f"deduplicate, and tools/call dispatches on {result['dispatches_by_chain']} "
            "hard-coded name comparisons rather than a lookup -- so a duplicate name is "
            "unreachable by call and invisible by list. Lesson 13.05's linter has the rule "
            "this needs; this lesson does not import it",
        ),
        practice.Check(
            "FINDING: only one of the three methods sorts anything",
            all([result["sorts"] == 1,
                 result["discover_keys"] == ["cacheScope", "capabilities",
                                             "instructions", "supportedVersions", "ttlMs"],
                 result["call_keys"] == ["content", "isError"]]),
            f"dispatch calls sorted {result['sorts']} time. server/discover returns "
            f"{result['discover_keys']} and tools/call returns {result['call_keys']} -- "
            "neither has a collection whose order a client could depend on, so the guarantee "
            "covers the one method where it matters",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
