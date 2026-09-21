"""Exercise 1 — forbidden is a path list, reversibility is a property.

    Add a sixth category if your product genuinely needs it. Defend why it
    does not collapse into one of the five.

Reading of the exercise: the defence has to be a demonstration. A sixth
category earns its place when there is a turn that passes every rule in the
five and it refuses, and when the obvious rewrite into one of the five is
shown to need a rule per path. The candidate is **reversibility**: every
destructive action carries a recorded undo.

**ANSWER: reversibility, and it refuses a turn the five accept 5 of 5.** A
turn that reads the state file, edits only allowed paths, passes its tests,
is confident, and adds no dependency -- while deleting
`migrations/0007_drop_users.sql` with no backup -- passes all **5** shipped
rules and fails a reversibility check. **1** refusal, **0** from the others.

**FINDING: it does not collapse into forbidden, because forbidden enumerates
paths.** `no_release_script_edits` names one literal path. Over **6**
destructive actions on paths nobody listed, the forbidden rule refuses **0**
and reversibility refuses the **4** that recorded no undo. Expressing the
same coverage as forbidden rules needs a new rule per file anyone ever adds.

**FINDING: it does not collapse into approval either, because approval is
about a person.** `new_dependency_approved` passes when `added_dependencies`
is empty, which it is on all **6** destructive turns. An approval rule
covering deletes would put a human in the loop for every one; the
reversibility rule asks only that an undo exists, and **4** of the **6**
turns could satisfy it without asking anybody.

**FINDING: nothing validates the category field, so a sixth is free and
unchecked.** `parse_rules` accepts any string after `category:` -- adding
`reversibility` needs **0** changes to the parser -- and the module contains
**0** references to the five category names. The taxonomy the lesson defends
is enforced by review, not by the checker, so "force the split" is advice the
code cannot give.

Structure: `DESTRUCTIVE` are the six turns; `reversible()` is the sixth
category's check, written against the shipped `TurnTrace`.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "33-instructions-as-executable-constraints"
CATEGORIES = ("startup", "forbidden", "definition_of_done", "uncertainty",
              "approval")
# (deleted path, whether an undo was recorded)
DESTRUCTIVE = (("migrations/0007_drop_users.sql", False),
               ("infra/terraform/state.tf", False),
               ("data/seed_customers.csv", False),
               ("docs/api.md", True),
               ("fixtures/orders.json", True),
               ("config/staging.yaml", False))


def trace_for(ref, deleted, undo_recorded, clean=True):
    """A turn that is impeccable by the five rules and deletes one file."""
    trace = ref.TurnTrace(
        read_state_file=True,
        edited_files=["app.py", "test_app.py", deleted] if clean else [deleted],
        confidence=0.9, asked_for_help=False, tests_exit_code=0,
        added_dependencies=[])
    trace.deleted_files = [deleted]
    trace.undo_records = [deleted] if undo_recorded else []
    return trace


def reversible(trace):
    """The sixth category: every destructive action carries a recorded undo."""
    deleted = getattr(trace, "deleted_files", [])
    undos = getattr(trace, "undo_records", [])
    return all(path in undos for path in deleted)


def five_rule_verdicts(ref, rules, trace):
    checker = ref.RuleChecker()
    return {row["slug"]: row["passed"] for row in ref.score(rules, checker, trace)}


def parse_with(ref, text, tmp_root):
    """parse_rules against a temporary file, so the reference tree is untouched."""
    path = tmp_root / "agent-rules.md"
    path.write_text(text)
    original = ref.RULES_PATH
    ref.RULES_PATH = path
    try:
        return ref.parse_rules()
    finally:
        ref.RULES_PATH = original


def sixth_rule_block():
    return ("\n## reversibility/undo-recorded\n"
            "- category: reversibility\n"
            "- check: undo_recorded\n"
            "Every deleted file must have a recorded undo.\n")


def solve():
    import pathlib
    import tempfile
    ref = parity.load_reference(PHASE, LESSON, "main")
    rules = parse_with(ref, ref.SEED_RULES, pathlib.Path(tempfile.mkdtemp()))
    headline = trace_for(ref, *DESTRUCTIVE[0])
    five = five_rule_verdicts(ref, rules, headline)
    caught = [path for path, undo in DESTRUCTIVE
              if not reversible(trace_for(ref, path, undo))]
    forbidden_caught = [path for path, undo in DESTRUCTIVE
                        if not ref.RuleChecker().no_release_script_edits(
                            trace_for(ref, path, undo))]
    with_sixth = parse_with(ref, ref.SEED_RULES + sixth_rule_block(),
                            pathlib.Path(tempfile.mkdtemp()))
    source = inspect.getsource(ref)
    return {
        "rules": len(rules), "five_pass": sum(five.values()),
        "five_verdicts": five, "sixth_verdict": reversible(headline),
        "destructive": len(DESTRUCTIVE),
        "reversibility_catches": len(caught),
        "forbidden_catches": len(forbidden_caught),
        "self_serviceable": sum(undo for _, undo in DESTRUCTIVE),
        "approval_passes": sum(
            ref.RuleChecker().new_dependency_approved(trace_for(ref, p, u))
            for p, u in DESTRUCTIVE),
        "parsed_with_sixth": len(with_sixth),
        "sixth_category": with_sixth[-1].category,
        "category_mentions": sum(source.count(f'"{name}"') for name in CATEGORIES),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: reversibility refuses a turn the five accept 5 of 5",
            all([result["rules"] == 5, result["five_pass"] == 5,
                 result["sixth_verdict"] is False,
                 all(result["five_verdicts"].values())]),
            f"a turn that reads state, edits allowed paths, passes tests, is confident "
            f"and adds no dependency -- while deleting a migration with no backup -- "
            f"passes {result['five_pass']}/{result['rules']} shipped rules and fails the "
            f"reversibility check ({result['sixth_verdict']})",
        ),
        practice.Check(
            "FINDING: it does not collapse into forbidden, which enumerates paths",
            all([result["forbidden_catches"] == 0,
                 result["reversibility_catches"] == 4,
                 result["destructive"] == 6]),
            f"no_release_script_edits names one literal path, so over "
            f"{result['destructive']} destructive actions on paths nobody listed it "
            f"catches {result['forbidden_catches']} where reversibility catches "
            f"{result['reversibility_catches']}. The same coverage as forbidden rules "
            "needs one rule per file anyone ever adds",
        ),
        practice.Check(
            "FINDING: it does not collapse into approval, which is about a person",
            all([result["approval_passes"] == 6,
                 result["self_serviceable"] == 2]),
            f"new_dependency_approved passes on all {result['approval_passes']} "
            f"destructive turns, because none adds a dependency. An approval rule "
            f"covering deletes puts a human in every one; reversibility asks only that an "
            f"undo exists, which {result['self_serviceable']} of the turns already do",
        ),
        practice.Check(
            "FINDING: nothing validates the category field",
            all([result["parsed_with_sixth"] == 6,
                 result["sixth_category"] == "reversibility",
                 result["category_mentions"] == 0]),
            f"parse_rules accepts any string after category:, so the sixth parses with no "
            f"parser change ({result['parsed_with_sixth']} rules, last category "
            f"{result['sixth_category']!r}), and the module mentions the five category "
            f"names {result['category_mentions']} times. The taxonomy is enforced by "
            "review, not by the checker",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
