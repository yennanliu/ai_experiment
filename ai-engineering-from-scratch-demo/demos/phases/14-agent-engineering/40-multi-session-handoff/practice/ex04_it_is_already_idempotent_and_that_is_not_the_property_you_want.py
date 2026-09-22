"""Exercise 4 — it is already idempotent, and that is not the property you want.

    Make the generator idempotent: running it twice produces the same packet.
    What needs to be stable for that to hold?

Reading of the exercise: run it twice before changing anything. The generator
has no clock, no uuid and no hash of a mutable, so on one snapshot it is
already byte-stable. The instability is upstream, in the snapshot, and the
question "what needs to be stable" is a question about the workbench.

**ANSWER: two runs on one snapshot are byte-identical, and three snapshot
inputs break it.** `generate_handoff` produces the same markdown and the same
payload both times. Change nothing but the *order* of `diff_summary["touched"]`
and the markdown differs while the file set is equal; append the generator's
own command to the feedback log, as Lesson 37's runner would, and
`commands_run` grows by **1** and the tail shifts; reorder the gate's findings
and `open_risks` follows. Stability is required of the diff listing, the
feedback log and the findings order -- none of which the generator owns.

**FINDING: nothing in the packet says which run produced it.**
`HandoffPayload` has **9** fields and **0** of them carry a commit, a branch or
a status. The lesson's own "one active handoff per branch and topic" pattern
asks for all three. Without them a re-run and a stale packet from last week are
byte-indistinguishable, which is the failure idempotency was supposed to
prevent.

**FINDING: the generator writes to one path and keeps no history.** `main`
writes `handoff.md` and `handoff.json` next to the script, so the second run
overwrites the first and "produces the same packet" is unobservable from disk.
Idempotence you cannot check is a claim, not a property.

**FINDING: a missing review total is silently worth 10.**
`derive_risks` reads `review.get("total", 10)` and falls back to 10 on a
missing or unparseable value, so a snapshot carrying no review at all and one
carrying a perfect 10 both come out at **3** risks, neither contributing
anything from the review branch. Two different worlds, one packet.

Structure: `packets()` renders a snapshot twice; `perturb()` runs the three
upstream changes.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "40-multi-session-handoff"
TOUCHED = ["app/signup.py", "tests/test_signup.py", "README.md"]
FEEDBACK = [{"command": "pytest", "exit_code": 0}, {"command": "ruff check .", "exit_code": 0}]
FINDINGS = [{"severity": "warn", "detail": "off-scope: README.md"},
            {"severity": "block", "detail": "rule failed: done/tests-pass"}]


def snapshot(ref, touched=None, feedback=None, findings=None, review=None):
    return ref.WorkbenchSnapshot(
        task_id="T-001",
        state={"active_task_id": None, "blockers": ["awaiting rate-limit decision"],
               "next_action": "open PR with current diff"},
        verdict={"passed": False, "findings": FINDINGS if findings is None else findings},
        review={"verdict": "pass", "total": 8} if review is None else review,
        feedback=FEEDBACK if feedback is None else feedback,
        diff_summary={"touched": TOUCHED if touched is None else touched})


def packets(ref, **kw):
    markdown, payload = ref.generate_handoff(snapshot(ref, **kw))
    return markdown, payload


def perturb(ref):
    """The three upstream changes the generator cannot defend against."""
    base_md, base = packets(ref)
    reordered_md, reordered = packets(ref, touched=list(reversed(TOUCHED)))
    grown_md, grown = packets(ref, feedback=FEEDBACK + [{"command": "generate_handoff.py",
                                                         "exit_code": 0}])
    _, swapped = packets(ref, findings=list(reversed(FINDINGS)))
    return {
        "order_changes_md": reordered_md != base_md,
        "order_same_set": sorted(reordered.changed_files) == sorted(base.changed_files),
        "log_growth": len(grown.commands_run) - len(base.commands_run),
        "tail_shifts": grown.feedback_tail != base.feedback_tail,
        "risks_follow": [r["detail"] for r in swapped.open_risks]
        != [r["detail"] for r in base.open_risks],
        "grown_differs": grown_md != base_md,
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    first_md, first = packets(ref)
    second_md, second = packets(ref)
    no_review = packets(ref, review={})[1]
    perfect = packets(ref, review={"verdict": "pass", "total": 10})[1]
    return {
        **perturb(ref),
        "md_stable": first_md == second_md, "payload_stable": first == second,
        "fields": list(ref.HandoffPayload.__dataclass_fields__),
        "identity": [f for f in ref.HandoffPayload.__dataclass_fields__
                     if f in ("branch", "head_commit", "status")],
        "writes": sorted(set(name for name in ("handoff.md", "handoff.json")
                             if name in inspect.getsource(ref.main))),
        "fallback": "total\", 10" in inspect.getsource(ref.derive_risks)
        or 'total", 10' in inspect.getsource(ref.derive_risks),
        "risk_counts": (len(no_review.open_risks), len(perfect.open_risks)),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: two runs are byte-identical, and three snapshot inputs break it",
            all([result["md_stable"] is True, result["payload_stable"] is True,
                 result["order_changes_md"] is True, result["order_same_set"] is True,
                 result["log_growth"] == 1, result["tail_shifts"] is True,
                 result["risks_follow"] is True]),
            f"the same snapshot renders identically twice ({result['md_stable']}), while "
            f"reordering the touched files changes the markdown on an equal file set, the "
            f"generator's own command adds {result['log_growth']} to commands_run and "
            "shifts the tail, and reordering the gate's findings reorders open_risks",
        ),
        practice.Check(
            "FINDING: nothing in the packet says which run produced it",
            all([len(result["fields"]) == 9, result["identity"] == []]),
            f"HandoffPayload carries {len(result['fields'])} fields and "
            f"{len(result['identity'])} of branch, head_commit or status, so a re-run and "
            "a packet from last week are byte-indistinguishable -- the failure the "
            "lesson's one-active-handoff-per-branch pattern exists to prevent",
        ),
        practice.Check(
            "FINDING: the generator writes to one path and keeps no history",
            result["writes"] == ["handoff.json", "handoff.md"],
            f"main writes {result['writes']} next to the script, so the second run "
            "overwrites the first and 'produces the same packet' is unobservable from "
            "disk. Idempotence you cannot check is a claim",
        ),
        practice.Check(
            "FINDING: a missing review total is silently worth 10",
            all([result["fallback"] is True, result["risk_counts"] == (3, 3)]),
            f"derive_risks defaults review['total'] to 10, so a snapshot with no review "
            f"at all and one with a perfect review both yield {result['risk_counts'][0]} "
            "risks. Two different worlds, one packet",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
