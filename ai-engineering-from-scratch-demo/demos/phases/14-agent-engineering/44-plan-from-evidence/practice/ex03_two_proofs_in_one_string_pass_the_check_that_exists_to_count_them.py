"""Exercise 3 — two proofs in one string pass the check that exists to count them.

    Split one item that has two proof commands.

Reading of the exercise: `proof` is a single string, so "two proof commands"
is something the model permits rather than something it records. Splitting is
easy; the question is what the split buys, and what the shipped validator
would have said either way.

**ANSWER: splitting the integration item into its two proofs adds 1 item, 0
waves, and one resumable boundary.** `python3 -m unittest && python3
scripts/check_links.py` validates clean as one string -- **0** issues -- and
tells a resuming session nothing about which half ran. As two items in the
same wave, the plan has **5** items, still **3** waves, and a session can read
exactly one proof as done.

**FINDING: `validate` checks that `proof` is non-empty and nothing else.**
A proof of `&&`-joined commands passes, and so does the lesson's own
`"review contract"` -- **1** of the example's **4** proofs is prose rather
than a command, and it is the proof on the item every other item depends on.

**FINDING: the two commands fail differently, which is the actual argument for
splitting.** One is the acceptance suite and one is a link checker; a plan
that records them as one string cannot say "tests pass, docs link is broken".
Splitting gives **2** terminal nodes with **2** independent verdicts, which is
what the lesson means by a proof closing an item.

**FINDING: splitting widens wave 3 rather than lengthening the plan.** Both
halves depend on `implementation` and `docs`, so they run together: wave
widths go from `[1, 2, 1]` to `[1, 2, 2]`, with the last wave holding two
independent gates instead of one compound claim.

Structure: `compound()` is the single-string version; `split()` is the pair;
`shapes()` measures both.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "44-plan-from-evidence"
SUITE = "python3 -m unittest"
LINKS = "python3 scripts/check_links.py"


def compound(ref):
    """The lesson's plan with one integration item carrying two commands."""
    items = [row for row in ref.example() if row.id != "integration"]
    return items + [ref.WorkItem("integration", "Run the complete acceptance gate",
                                 ("pyproject.toml:31",), ("implementation", "docs"),
                                 f"{SUITE} && {LINKS}")]


def split(ref):
    """The same work as two terminal nodes with independent verdicts."""
    items = [row for row in ref.example() if row.id != "integration"]
    deps = ("implementation", "docs")
    return items + [
        ref.WorkItem("integration-tests", "Prove behaviour with the acceptance suite",
                     ("pyproject.toml:31",), deps, SUITE),
        ref.WorkItem("integration-links", "Prove the documented contract resolves",
                     ("docs/api.md:72",), deps, LINKS)]


def shapes(ref, items):
    waves = ref.execution_waves(items)
    return {"items": len(items), "waves": len(waves),
            "widths": [len(wave) for wave in waves], "last": waves[-1]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    one, two = compound(ref), split(ref)
    proofs = [row.proof for row in ref.example()]
    validator = inspect.getsource(ref.validate)
    return {
        "compound": shapes(ref, one), "split": shapes(ref, two),
        "compound_issues": ref.validate(one), "split_issues": ref.validate(two),
        "commands_in_one": len([part for part in
                                next(row.proof for row in one if row.id == "integration")
                                .split("&&") if part.strip()]),
        "proofs": proofs,
        "prose_proofs": [proof for proof in proofs if not proof.startswith("python3")],
        "prose_owner": [row.id for row in ref.example()
                        if not row.proof.startswith("python3")],
        "dependents": [row.id for row in ref.example() if "contract" in row.depends_on],
        "proof_checks": validator.count("item.proof"),
        "terminal_nodes": len(two[-1].depends_on),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the split adds 1 item, 0 waves, and one resumable boundary",
            all([result["compound_issues"] == [], result["commands_in_one"] == 2,
                 result["split"]["items"] == result["compound"]["items"] + 1,
                 result["split"]["waves"] == result["compound"]["waves"] == 3,
                 result["split"]["last"] == ["integration-links", "integration-tests"]]),
            f"the compound proof holds {result['commands_in_one']} commands and validates "
            f"with {len(result['compound_issues'])} issues; split, the plan has "
            f"{result['split']['items']} items, still {result['split']['waves']} waves, and "
            f"a last wave of {result['split']['last']}",
        ),
        practice.Check(
            "FINDING: validate checks that proof is non-empty and nothing else",
            all([result["proof_checks"] == 1, len(result["prose_proofs"]) == 1,
                 result["prose_owner"] == ["contract"],
                 result["prose_proofs"] == ["review contract"]]),
            f"the only proof rule is an emptiness test, so {result['prose_proofs']} passes "
            f"-- and it belongs to {result['prose_owner']}, the item that "
            f"{result['dependents']} both depend on",
        ),
        practice.Check(
            "FINDING: the two commands fail differently",
            all([result["split_issues"] == [], len(result["split"]["last"]) == 2,
                 result["terminal_nodes"] == 2]),
            f"one command is the acceptance suite and one is a link checker; as a single "
            f"string the plan cannot say 'tests pass, docs link is broken'. Split, wave 3 "
            f"holds {len(result['split']['last'])} terminal nodes with independent verdicts",
        ),
        practice.Check(
            "FINDING: splitting widens the last wave rather than lengthening the plan",
            all([result["compound"]["widths"] == [1, 2, 1],
                 result["split"]["widths"] == [1, 2, 2]]),
            f"wave widths go from {result['compound']['widths']} to "
            f"{result['split']['widths']}: both halves depend on implementation and docs, "
            "so they run together and the plan gains a verdict rather than a step",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
