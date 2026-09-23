"""Exercise 4 — the third question has no true answer for a public post.

    Design a challenge-and-response checklist for a specific action (e.g.,
    "post to a public Twitter account"). What three questions must the
    reviewer answer? Why those three?

Reading of the exercise: "why those three" is the part that can be checked,
because the shipped `checklist_approve` already takes exactly three booleans.
So the design names three questions, maps each to one of the shipped
parameters, and is then tested by asking whether a reviewer can answer them
honestly for this action.

**ANSWER: content, audience, withdrawal -- and for a public post the third
is No.** *Is the text exactly what will appear?* maps to `understood`. *Is
the account and its follower set what the proposal says?* maps to `verified`.
*Can this be withdrawn, and what does withdrawal cost?* maps to
`rollback_ready`, and for a public post the honest answer is no: deletion
removes the post and not the copies. Answered honestly, `checklist_approve`
**rejects**, the status stays `waiting`, `commit` refuses and **0** side
effects occur.

**FINDING: the three questions are the three irreducible ones.** They are
the only three that cannot be inferred from each other: content is what the
action does, audience is who it reaches, withdrawal is what happens if the
first two were wrong. `blast_radius` and `lineage` inform the second and the
first; `intent` informs none of them, which is why a checklist keyed on
intent would pass a proposal whose payload had changed.

**FINDING: the shipped run answers the third question yes for the one
proposal whose record says no.** `checklist_approve` is called with
`rollback_ready=True` for an email whose `rollback` field reads `no in-band
rollback`. Nothing compares the boolean to the record, so the checklist's
protection against rubber-stamping is itself rubber-stampable.

**FINDING: rejection is the only outcome the checklist can produce.** It
takes **3** booleans and **1** combination approves; the other **7** reject
with the same message, so a reviewer who cannot answer the third question
produces the same artifact as one who did not read the first. A checklist
that records *which* question failed is one field away and would be the
difference between "declined" and "declined because it cannot be undone".

Structure: `QUESTIONS` is the design; `ask()` runs the shipped checklist with
a given set of answers and reports what the store ends up holding.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import itertools
import os
import re
import tempfile

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "15-propose-then-commit"

TEMP = tempfile.mkdtemp(prefix="aiefs-checklist-")
# (question, the shipped parameter it maps to, honest answer for a public post)
QUESTIONS = (
    ("Is the text exactly what will appear?", "understood", True),
    ("Is the account and its follower set what the proposal says?", "verified", True),
    ("Can this be withdrawn, and what does withdrawal cost?", "rollback_ready", False),
)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def post_proposal(ref):
    return ref.Proposal(
        thread_id="t-tweet", action="social.post",
        payload={"account": "@acme", "text": "v1.2 is out"},
        intent="Announce the release publicly",
        lineage="release notes /releases/1.2",
        blast_radius="41k followers; public and archived",
        rollback="delete removes the post, not the copies")


def ask(ref, answers, name="ask.json"):
    """Run the shipped checklist with a set of answers; report what results."""
    path = os.path.join(TEMP, name)
    if os.path.exists(path):
        os.remove(path)
    store = ref.Store(path)
    key = quiet(ref.propose, store, post_proposal(ref))
    approved = quiet(ref.checklist_approve, store, key, *answers)
    before = len(ref.SIDE_EFFECTS)
    quiet(ref.commit, store, key)
    return approved, store.all()[key]["status"], len(ref.SIDE_EFFECTS) - before


def outcomes(ref):
    """How many of the eight answer combinations approve."""
    return sum(quiet(ask, ref, combo, f"combo-{index}.json")[0]
               for index, combo in enumerate(itertools.product([True, False], repeat=3)))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    honest = tuple(answer for _q, _p, answer in QUESTIONS)
    main_source = inspect.getsource(ref.main)
    return {
        "questions": len(QUESTIONS),
        "parameters": list(inspect.signature(ref.checklist_approve).parameters)[2:],
        "mapped": [param for _q, param, _a in QUESTIONS],
        "honest": list(honest),
        "result": list(ask(ref, honest)),
        "all_yes": list(ask(ref, (True, True, True), "yes.json")),
        "approving_combinations": outcomes(ref),
        "combinations": 8,
        "shipped_answers": re.findall(r"rollback_ready=(\w+)", main_source),
        "shipped_rollbacks": re.findall(r'rollback="([^"]+)"', main_source),
        "compares_record": 'rec["rollback"]' in inspect.getsource(ref.checklist_approve),
        "records_which": "reason" in inspect.getsource(ref.checklist_approve),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: content, audience, withdrawal -- and the third is No",
            all([result["questions"] == 3, result["mapped"] == result["parameters"],
                 result["honest"] == [True, True, False],
                 result["result"] == [False, "waiting", 0],
                 result["all_yes"] == [True, "committed", 1]]),
            f"the {result['questions']} questions map onto {result['parameters']}; "
            f"answered honestly {result['honest']} the checklist rejects, the status "
            f"stays {result['result'][1]!r} and {result['result'][2]} side effects "
            f"occur -- against {result['all_yes']} if the third is ticked anyway",
        ),
        practice.Check(
            "FINDING: the three questions are the three irreducible ones",
            all([result["mapped"] == ["understood", "verified", "rollback_ready"],
                 len(result["parameters"]) == 3]),
            "content is what the action does, audience is who it reaches, withdrawal is "
            "what happens if the first two were wrong -- none is inferable from the "
            "others, and intent informs none of them",
        ),
        practice.Check(
            "FINDING: the shipped run answers the third yes for the one no",
            all([result["shipped_answers"] == ["True", "False"],
                 any(text.startswith("no ") for text in result["shipped_rollbacks"]),
                 not result["compares_record"]]),
            f"checklist_approve is called with rollback_ready={result['shipped_answers'][0]} "
            "for a proposal whose rollback field begins 'no in-band rollback', and the "
            "function never reads the record -- so the anti-rubber-stamp check is "
            "itself rubber-stampable",
        ),
        practice.Check(
            "FINDING: rejection is the only outcome the checklist can produce",
            all([result["approving_combinations"] == 1, result["combinations"] == 8,
                 not result["records_which"]]),
            f"{result['approving_combinations']} of {result['combinations']} answer "
            "combinations approves and the other seven reject with one message, so a "
            "reviewer who cannot answer the third produces the same artifact as one who "
            "did not read the first",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
