"""Exercise 3 — moving the production write to delegated removes its rationale requirement.

    Mark every decision and justify each locked or bounded choice.

Reading of the exercise: marking is the easy half. The half worth doing is
checking what the marking buys, and the lesson's own build instruction points
at it: move the production-write decision from `locked` to `delegated` and
watch the schema accept it.

**ANSWER: all 6 decisions are marked, the 4 constrained ones carry
rationales, and moving the production write to `delegated` compiles clean
with its rationale deleted.** `validate` requires a rationale only when the
mode is not `delegated`, so the decision whose consequence is a production
write becomes the one decision the document stops asking about. The
`human_checkpoint` list drops from **2** entries to **1** and nothing is
reported.

**FINDING: the mode decides who is asked, and the schema has no opinion about
which mode is right.** `VALID_MODES` holds **3** strings and the check is
membership; consequence, reversibility and authority -- the three things the
lesson says should choose the mode -- appear in the module **0** times. The
decision table in the docs is advice the code cannot apply.

**FINDING: a bounded decision's boundary is its rationale field.**
`compile_contract` publishes `{"question": ..., "boundary": item.rationale}`,
so the same string is the justification and the limit. "Stop after five
sources or two minutes" is a boundary; "production authority stays with the
incident commander" is a reason -- and if a bounded decision is given the
second kind, the contract reports a boundary nobody can check.

**FINDING: justifying every constrained decision is 4 strings and justifying
the delegated ones is the exercise the validator skips.** Of the **6**
decisions here, **2** are delegated: which exception to catch and how to name
the fallback. Both are cheap and reversible, which is the right call, and
both ship with an empty explanation because nothing requires one.

Structure: `DECISIONS` is the marked set; `moved()` performs the lesson's own
experiment.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "51-write-specifications-that-preserve-judgment"

# (question, mode, rationale)
DECISIONS = [
    ("Which exception does the fallback catch?", "delegated", ""),
    ("How is the fallback helper named?", "delegated", ""),
    ("What fills the Chinese slot when the file is missing?", "bounded",
     "the English text, unchanged"),
    ("How many documents may the scaffolder read?", "bounded", "at most two per lesson"),
    ("May the fallback invent Chinese text?", "locked",
     "invented text would ship as the lesson's own words"),
    ("May the scaffolder overwrite an existing manifest?", "locked",
     "a manifest carries hand-written thresholds"),
]


def spec(ref, decisions=None):
    return ref.Specification(
        outcome="every lesson can be scaffolded, whatever languages it ships",
        invariants=["a lesson with no Chinese document still scaffolds"],
        examples=["an English-only lesson produces a manifest with identical en and zh"],
        non_goals=["translating anything"],
        decisions=[ref.Decision(*row) for row in (decisions or DECISIONS)],
        proof=["both manifests, read from the repository"])


def moved(ref):
    """The lesson's experiment: production-write authority marked delegated."""
    rows = [row if row[0] != DECISIONS[4][0] else (row[0], "delegated", "")
            for row in DECISIONS]
    return spec(ref, rows)


def marking(rows=DECISIONS):
    """How the decisions are marked, and which carry a justification."""
    return {"decisions": len(rows),
            "modes": sorted({mode for _, mode, _ in rows}),
            "constrained": sum(mode != "delegated" for _, mode, _ in rows),
            "justified": sum(bool(rationale) for _, mode, rationale in rows
                             if mode != "delegated"),
            "delegated": sum(mode == "delegated" for _, mode, _ in rows),
            "delegated_explained": sum(bool(rationale) for _, mode, rationale in rows
                                       if mode == "delegated")}


def criteria(ref):
    """Whether the module's logic mentions what the docs say should pick a mode."""
    logic = inspect.getsource(ref.validate) + inspect.getsource(ref.compile_contract)
    words = ("consequence", "reversib", "authority")
    return {"criteria_mentioned": sum(word in logic for word in words),
            "criteria_in_data": sum(word in inspect.getsource(ref) for word in words)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    before = ref.compile_contract(spec(ref))
    after = ref.compile_contract(moved(ref))
    bounded = [item.rationale for item in ref.example().decisions
               if item.mode == "bounded"]
    return {
        **marking(), **criteria(ref),
        "before_status": before["status"], "after_status": after["status"],
        "before_checkpoint": len(before["human_checkpoint"]),
        "after_checkpoint": len(after["human_checkpoint"]),
        "after_issues": after["issues"],
        "valid_modes": len(ref.VALID_MODES),
        "boundary_is_rationale": '"boundary": item.rationale' in inspect.getsource(ref),
        "shipped_boundary": bounded[0] if bounded else "",
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: moving the production write to delegated compiles clean",
            all([result["decisions"] == 6, result["constrained"] == 4,
                 result["justified"] == 4, result["after_status"] == "executable",
                 result["after_issues"] == [],
                 result["before_checkpoint"] == 2, result["after_checkpoint"] == 1]),
            f"{result['justified']} of {result['constrained']} constrained decisions carry "
            f"rationales; moving the production write to delegated deletes its rationale, "
            f"drops the human checkpoint from {result['before_checkpoint']} to "
            f"{result['after_checkpoint']} and still compiles {result['after_status']!r}",
        ),
        practice.Check(
            "FINDING: the mode decides who is asked and the schema has no opinion",
            all([result["valid_modes"] == 3, result["criteria_mentioned"] == 0,
                 result["modes"] == ["bounded", "delegated", "locked"]]),
            f"VALID_MODES holds {result['valid_modes']} strings and the check is "
            f"membership; consequence, reversibility and authority appear "
            f"{result['criteria_mentioned']} times in validate and compile_contract -- "
            f"{result['criteria_in_data']} appears only inside the example's own rationale "
            "text -- so the decision table in the docs is advice the code cannot apply",
        ),
        practice.Check(
            "FINDING: a bounded decision's boundary is its rationale field",
            all([result["boundary_is_rationale"] is True,
                 result["shipped_boundary"] == "Stop after five sources or two minutes"]),
            f"compile_contract publishes the rationale as the boundary, so the same string "
            f"is the justification and the limit: {result['shipped_boundary']!r} is a "
            "limit, while a reason in the same field yields a boundary nobody can check",
        ),
        practice.Check(
            "FINDING: the delegated decisions ship with no explanation",
            all([result["delegated"] == 2, result["delegated_explained"] == 0]),
            f"{result['delegated']} of the {result['decisions']} decisions are delegated -- "
            "which exception to catch and how to name the helper, both cheap and reversible "
            f"-- and {result['delegated_explained']} carry the explanation the lesson says "
            "the agent owes",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
