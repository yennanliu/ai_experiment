"""Exercise 3 — the tree returns langgraph for 93% of its input space.

    **Hard.** Build a decision-tree script `pick_framework.py` that takes a
    short problem description (JSON: `{has_typed_state, has_roles,
    has_dialogue, has_parallel_fanout, needs_resume}`) and returns a
    recommendation with one-sentence justification. Verify it on six cases you
    design yourself.

Reading of the exercise: the lesson already ships the script -- `recommend`
takes exactly that descriptor, plus three more fields, and returns a framework
with a one-sentence reason. So the work is the verification, and the honest way
to verify a decision tree is to enumerate it: 7 boolean fields and four call
counts give **512** descriptors, which is small enough to run every one.
Then the six cases are designed against the seams the enumeration exposes,
rather than against the branches the author was thinking of.

**ANSWER: over all 512 descriptors the tree returns `langgraph` 476 times.**
93.0%. The other four answers share 36 descriptors between them: `autogen` 16,
`plain python` 8, `crewai` 8, `agno` 4. A recommender that gives the same
answer to 93% of its inputs is a default with a questionnaire in front of it --
and the questionnaire is eight fields long.

**FINDING: the smallest-first branch ignores two of the eight fields.** It
tests `total_llm_calls <= 2` and five flags, and never reads `has_typed_state`
or `needs_session_memory`. So a two-call task with a declared state schema and
durable per-user memory is told to use plain Python, which is the one answer
that cannot hold a session.

**FINDING: declaring a state schema removes two frameworks from
consideration.** `crewai` and `autogen` both require `has_typed_state == False`
-- the branches read `p.has_roles and not p.has_typed_state` and
`p.has_dialogue and not p.has_typed_state` -- so the 256 descriptors that have
a state schema can never reach either, whatever their roles or dialogue say. A
CrewAI pipeline with a typed hand-off is unreachable by construction.

**FINDING: the fallback is one equivalence class.** The final `return` is
reachable for exactly **2** of the 512 descriptors: every flag false, with 3 or
8 calls. Its comment calls it a fallback for "any uncertainty about future
state", and the `has_typed_state` branch one line above already catches
everything that is not this.

**ANSWER: on six cases chosen to probe the seams, the tree agrees 3 times.**
It differs on the two-call typed-state case (`plain python`), on a role
pipeline that must remember the user between runs (`crewai`, silently dropping
the session store), and on a single call that needs a human to approve it
(`langgraph`, a graph and a checkpointer for one call). It agrees on the three
cases where a state schema decides the answer. The shipped suite is 7 for 7,
because its cases each exercise one branch in isolation.

Structure: `FLAGS` and `CALLS` span the input space, `enumerate_tree` runs
every descriptor, `reach` reports which fields a given answer requires,
`CASES` is the six hand-designed descriptors with the answer this solution
argues for, and `REASON_PREFIX` identifies the fallback branch by its text.
"""

from __future__ import annotations

import itertools
from collections import Counter

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "17-agent-framework-tradeoffs"
FLAGS = ("has_typed_state", "has_roles", "has_dialogue", "has_parallel_fanout",
         "needs_resume", "needs_human_interrupt", "needs_session_memory")
CALLS = (1, 2, 3, 8)
REASON_PREFIX = "Default for multi-step"
CASES = (
    ("two calls, typed state, durable per-user memory",
     {"total_llm_calls": 2, "has_typed_state": True, "needs_session_memory": True}, "agno"),
    ("role pipeline that also declares a state schema",
     {"total_llm_calls": 4, "has_typed_state": True, "has_roles": True}, "langgraph"),
    ("proposer-critic loop over a typed scratchpad",
     {"total_llm_calls": 10, "has_typed_state": True, "has_dialogue": True}, "langgraph"),
    ("role pipeline that must remember the user between runs",
     {"total_llm_calls": 6, "has_roles": True, "needs_session_memory": True}, "agno"),
    ("fanout to three retrievers inside a debate",
     {"total_llm_calls": 20, "has_parallel_fanout": True, "has_dialogue": True}, "langgraph"),
    ("one call that needs a human to approve it",
     {"total_llm_calls": 1, "needs_human_interrupt": True}, "plain python"),
)


def enumerate_tree(ref):
    """Every descriptor in the input space, with what the tree says about it."""
    rows = []
    for bits in itertools.product((False, True), repeat=len(FLAGS)):
        fields = dict(zip(FLAGS, bits))
        for calls in CALLS:
            answer = ref.recommend(ref.Problem(total_llm_calls=calls, **fields))
            rows.append((fields, calls, answer))
    return rows


def reach(rows, framework):
    """The fields every descriptor reaching this answer agrees on."""
    hits = [fields for fields, _, answer in rows if answer.framework == framework]
    return {flag: hits[0][flag] for flag in FLAGS
            if len({fields[flag] for fields in hits}) == 1}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = enumerate_tree(ref)
    spread = Counter(answer.framework for _, _, answer in rows)
    designed = [(label, ref.recommend(ref.Problem(**fields)).framework, mine)
                for label, fields, mine in CASES]
    typed = [row for row in rows if row[0]["has_typed_state"]]
    return {
        "descriptors": len(rows), "spread": dict(spread),
        "share": round(spread["langgraph"] / len(rows) * 100, 1),
        "reasons": len({answer.reason for _, _, answer in rows}),
        "plain_python_reach": reach(rows, "plain python"),
        "crewai_reach": reach(rows, "crewai"), "autogen_reach": reach(rows, "autogen"),
        "typed_descriptors": len(typed),
        "typed_answers": sorted({row[2].framework for row in typed}),
        "fallback": sum(1 for _, _, answer in rows
                        if answer.reason.startswith(REASON_PREFIX)),
        "designed": designed,
        "agreed": sum(1 for _, tree, mine in designed if tree == mine),
    }


def verify(result):
    spread, designed = result["spread"], result["designed"]
    return [
        practice.Check(
            "ANSWER: over all 512 descriptors the tree returns langgraph 476 times",
            all([result["descriptors"] == 512, spread["langgraph"] == 476,
                 result["share"] == 93.0, sum(spread.values()) == 512]),
            f"{len(FLAGS)} boolean fields and {len(CALLS)} call counts give "
            f"{result['descriptors']} descriptors, and the answers are {spread} -- "
            f"{result['share']}% langgraph. A recommender that gives the same answer to that "
            f"share of its inputs is a default with an eight-field questionnaire in front",
        ),
        practice.Check(
            "FINDING: the smallest-first branch ignores two of the eight fields",
            all(["has_typed_state" not in result["plain_python_reach"],
                 "needs_session_memory" not in result["plain_python_reach"],
                 len(result["plain_python_reach"]) == 5]),
            f"every descriptor reaching plain python agrees on "
            f"{sorted(result['plain_python_reach'])} and on nothing else, so the branch "
            "never reads has_typed_state or needs_session_memory. A two-call task with a "
            "state schema and durable memory is told to use the one answer that cannot hold "
            "a session",
        ),
        practice.Check(
            "FINDING: declaring a state schema removes two frameworks from consideration",
            all([result["crewai_reach"]["has_typed_state"] is False,
                 result["autogen_reach"]["has_typed_state"] is False,
                 result["typed_descriptors"] == 256,
                 "crewai" not in result["typed_answers"],
                 "autogen" not in result["typed_answers"]]),
            f"crewai and autogen both require has_typed_state False, so the "
            f"{result['typed_descriptors']} descriptors that declare a schema answer only "
            f"{result['typed_answers']}. A CrewAI pipeline with a typed hand-off is "
            "unreachable by construction",
        ),
        practice.Check(
            "FINDING: the fallback is one equivalence class",
            all([result["fallback"] == 2, result["reasons"] == 7]),
            f"the final return is reachable for {result['fallback']} of "
            f"{result['descriptors']} descriptors -- every flag false, with 3 or 8 calls -- "
            f"out of {result['reasons']} distinct reasons in the tree. The has_typed_state "
            "branch one line above already catches everything that is not this",
        ),
        practice.Check(
            "ANSWER: on six cases chosen to probe the seams, the tree agrees 3 times",
            all([result["agreed"] == 3, len(designed) == 6,
                 [tree for _, tree, _ in designed][:1] == ["plain python"]]),
            "; ".join(f"{label}: tree {tree}, argued {mine}"
                      for label, tree, mine in designed if tree != mine)
            + f". {result['agreed']} of {len(designed)} agree -- it agrees wherever a state "
            "schema decides the answer, and the shipped suite is 7 for 7 because each of its "
            "cases exercises one branch in isolation",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
