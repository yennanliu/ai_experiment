"""Exercise 2 — a reviewer with 8 false positives cuts the lesson's agent from 20/30 to 12/30, because it cannot revise.

    Add a `reviewer` sub-agent that reads the diff before PR posting and can request a revision loop. Measure whether false-positive reviews drop SWE-bench pass rate below the single-agent baseline (hint: usually yes).

Reading of the exercise: the reviewer reads a unified diff and applies
three rules a cautious code reviewer uses: no edits to tests, no removed
asserts, no swallowed exceptions. It runs on 30 labelled diffs, the size of
the lesson's SWE-bench Pro subset: 20 correct and 10 incorrect across 7
kinds. A flag starts a revision loop of up to 3 rounds; a diff still
flagged after that is not posted, so it fails. Two agents do the revising.
The first is the lesson's own `run_agent`, re-run with the review as its
task. The second is an ideal reviser: it complies with every flag, and when
the diff was wrong it writes the right fix. Baseline is posting every diff:
pass = correct.

**ANSWER: yes for the lesson's agent, no for an ideal one.** The reviewer
flags 8 of the 20 correct diffs (regression tests added, an obsolete assert
dropped, a `KeyError` handled) and 7 of the 10 wrong ones. The lesson's
agent returns the same run whatever the review says, so each flag holds
for all 3 rounds: pass falls from 20/30 to 12/30, at 4x the tokens on the
15 flagged tasks. The ideal reviser reaches 23/30. Complying breaks 4
correct diffs (the assert and `KeyError` ones) and fixes 7 wrong ones. The
review loses exactly when obeyed false positives outnumber fixed true
positives: 4 against 7 here.

**FINDING: the lesson's harness has no diff to review and no place to hook
a reviewer.** `TOOLS` holds 2 of the 6 tools the lesson lists (`read_file`,
`run_shell`), so there is no `edit_file` or `git`. A `Stop` hook that asks
for revision is ignored: the loop ends and the session closes anyway.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "01-terminal-native-coding-agent"
ROUNDS = 3
# kind: (count, correct, correct after the ideal reviser complies, diff body)
KINDS = {
    "plain_fix": (12, True, True, "--- a/src/m.py\n+++ b/src/m.py\n-    return a - b\n+    return b - a\n"),
    "adds_regression_test": (4, True, True, "--- a/tests/test_m.py\n+++ b/tests/test_m.py\n+def test_x():\n"
                             "+    assert f(2) == 3\n"),
    "drops_obsolete_assert": (2, True, False, "--- a/src/m.py\n+++ b/src/m.py\n-    assert len(xs) < 10\n"),
    "handles_keyerror": (2, True, False, "--- a/src/m.py\n+++ b/src/m.py\n+    except KeyError:\n+        pass\n"),
    "deletes_failing_assert": (4, False, True, "--- a/tests/test_m.py\n+++ b/tests/test_m.py\n"
                               "-    assert f(2) == 3\n"),
    "swallows_exception": (3, False, True, "--- a/src/m.py\n+++ b/src/m.py\n+    except Exception:\n"
                           "+        pass\n"),
    "wrong_logic": (3, False, False, "--- a/src/m.py\n+++ b/src/m.py\n-    return a - b\n+    return a + b\n"),
}


def fixture():
    return [{"kind": k, "correct": c, "after": a, "diff": d}
            for k, (n, c, a, d) in KINDS.items() for _ in range(n)]


RULES = [("edits tests", lambda rm, add, heads: any(h.startswith("+++ b/tests/") for h in heads)),
         ("removes an assert", lambda rm, add, heads: any("assert" in r for r in rm)),
         ("swallows an exception", lambda rm, add, heads: "pass" in add
          and any(a.startswith("except") for a in add))]


def reviewer(diff: str) -> list:
    """Read a unified diff; return the objections, empty to approve."""
    lines = diff.splitlines()
    heads = [ln for ln in lines if ln[:3] in ("---", "+++")]
    body = [ln for ln in lines if ln not in heads]
    rm = [ln[1:].strip() for ln in body if ln.startswith("-")]
    add = [ln[1:].strip() for ln in body if ln.startswith("+")]
    return [name for name, rule in RULES if rule(rm, add, heads)]


def fingerprint(run):
    """The run minus its GOAL line, which only echoes the task text back."""
    return run["plan"].splitlines()[1:], run["budget"]


def review_loop(ref, item, ideal, code):
    """Returns (passed, sessions run)."""
    diff, correct, sessions = item["diff"], item["correct"], 1
    first = fingerprint(ref.run_agent(item["kind"], code))
    for _ in range(ROUNDS):
        notes = reviewer(diff)
        if not notes:
            return correct, sessions
        again = ref.run_agent(f"revise {item['kind']}: {'; '.join(notes)}", code)
        sessions += 1
        if ideal:
            diff, correct = "", item["after"]
        elif fingerprint(again) != first:
            raise AssertionError("the lesson's agent changed its run after a review")
    return (not reviewer(diff)) and correct, sessions


def stop_veto_ignored(ref, code):
    class VetoBus(ref.HookBus):
        def __init__(self):
            super().__init__()
            self.on("Stop", lambda p: {**p, "revise": True})
    real, ref.HookBus = ref.HookBus, VetoBus
    try:
        run = ref.run_agent("veto", code)
        return run["budget"]["turns_used"], run["trace"][-1]["event"]
    finally:
        ref.HookBus = real


def with_review(ref, items, flags, ideal, code):
    runs = [review_loop(ref, i, ideal, code) for i in items]
    return sum(p for p, _ in runs), sorted({s for (_, s), f in zip(runs, flags) if f})


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    code = str(parity.lesson_dir(PHASE, LESSON) / "code")
    items = fixture()
    flags = [bool(reviewer(i["diff"])) for i in items]
    (lesson, sessions), ideal = with_review(ref, items, flags, False, code), with_review(ref, items, flags, True, code)
    return {"n": len(items), "baseline": sum(i["correct"] for i in items),
            "fp": sum(f and i["correct"] for f, i in zip(flags, items)),
            "tp": sum(f and not i["correct"] for f, i in zip(flags, items)),
            "tools": sorted(ref.TOOLS), "veto": stop_veto_ignored(ref, code),
            "lesson": lesson, "lesson_flagged_sessions": sessions, "ideal": ideal[0]}


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: false positives push the lesson's agent below baseline; an ideal reviser clears it",
            (r["n"], r["baseline"], r["fp"], r["tp"], r["lesson"], r["ideal"], r["lesson_flagged_sessions"])
            == (30, 20, 8, 7, 12, 23, [4]),
            f"baseline {r['baseline']}/{r['n']}; reviewer FP {r['fp']}/20, TP {r['tp']}/10; with review: "
            f"lesson agent {r['lesson']}/30 ({r['lesson_flagged_sessions']} sessions per flagged task), "
            f"ideal reviser {r['ideal']}/30",
        ),
        practice.Check(
            "FINDING: no edit or git tool to make a diff, and a Stop-hook revision request is ignored",
            (r["tools"], r["veto"]) == (["read_file", "run_shell"], (3, "end")),
            f"TOOLS = {r['tools']}; with a vetoing Stop hook the run still takes {r['veto'][0]} turns "
            f"and ends on {r['veto'][1]!r}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
