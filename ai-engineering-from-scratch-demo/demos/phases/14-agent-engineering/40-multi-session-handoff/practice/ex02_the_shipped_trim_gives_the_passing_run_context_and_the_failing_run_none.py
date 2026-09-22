"""Exercise 2 — the shipped trim gives the passing run context and the failing run none.

    Trim the feedback summary differently for failing runs versus passing
    ones. Defend the asymmetry.

Reading of the exercise: the defence has to say what each log is *evidence
of*. A passing run's log is evidence that the commands ran, so the last K
entries are the whole story. A failing run's log is evidence of a causal
sequence -- what ran before the failure, what was retried after -- and a tail
is the wrong window for that.

**ANSWER: the shipped trim hands the failing run 0 of its 3 neighbouring
commands and the passing run all 5 of its most recent.** On a 20-command log
whose only failure is at index 2, `trim_feedback` returns `['c15', 'c16',
'c17', 'c18', 'c19', 'c2']`: the failure arrives **last**, after five newer
records, with none of `c1`, `c3`, `c4` beside it. The asymmetry the exercise
asks for is positional, not sized -- a window around the failure, not a longer
tail.

**FINDING: `tail + nonzero` breaks chronological order.** The failure is at
index 2 of 20 and lands at position **6** of **6** in the packet. A next
session reading `feedback_tail` top to bottom sees the failure after the
commands that came fifteen steps later, which is the opposite of the order
that explains it.

**FINDING: the trim is unbounded in exactly the case it exists for.**
`failed_attempts` is built from every non-zero exit with no cap at all, so a
200-command session with **60** failures puts **60** lines in the markdown and
**64** records in `feedback_tail`. The trim shrinks the quiet packet and lets
the loud one through.

**FINDING: dedup by `id()` disagrees with the count beside it.** Two runs of
the same failing command passed as the same object collapse to **1** entry in
`feedback_tail` while `failed_attempts` reports **2**. One packet, two answers
to "how many times did this fail".

Structure: `windowed()` is the asymmetric replacement; `log()` builds the
20-command fixture.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "40-multi-session-handoff"
FAIL_AT = 2


def log(size=20, fail_at=FAIL_AT):
    return [{"command": f"c{i}", "exit_code": 1 if i == fail_at else 0} for i in range(size)]


def windowed(records, tail_k, radius=2):
    """Passing runs get the tail; failing runs get a window around each failure."""
    keep = set(range(max(0, len(records) - tail_k), len(records)))
    for index, record in enumerate(records):
        if record.get("exit_code") not in (0, None):
            keep |= set(range(max(0, index - radius), min(len(records), index + radius + 1)))
    return [records[i] for i in sorted(keep)]


def neighbours(records, trimmed, fail_at=FAIL_AT):
    """How many of the failure's immediate neighbours survived the trim."""
    wanted = {records[i]["command"] for i in (fail_at - 1, fail_at + 1, fail_at + 2)}
    return len(wanted & {r["command"] for r in trimmed}), len(wanted)


def loud(ref):
    """A 200-command session with 60 failures."""
    records = [{"command": f"c{i}", "exit_code": 1 if i % 3 == 0 else 0} for i in range(180)]
    snapshot = ref.WorkbenchSnapshot(task_id="T-002", state={}, verdict={}, review={},
                                     feedback=records, diff_summary={"touched": []})
    _, payload = ref.generate_handoff(snapshot)
    return {"failures": sum(r["exit_code"] == 1 for r in records),
            "attempts": len(payload.failed_attempts),
            "tail": len(payload.feedback_tail)}


def repeated(ref):
    """The same failing record object handed in twice."""
    record = {"command": "pytest", "exit_code": 1}
    snapshot = ref.WorkbenchSnapshot(task_id="T-003", state={}, verdict={}, review={},
                                     feedback=[record, record], diff_summary={"touched": []})
    _, payload = ref.generate_handoff(snapshot)
    return len(payload.feedback_tail), len(payload.failed_attempts)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    records = log()
    shipped = ref.trim_feedback(records)
    mine = windowed(records, ref.TAIL_K)
    passing = ref.trim_feedback([{"command": f"c{i}", "exit_code": 0} for i in range(20)])
    return {
        "shipped": [r["command"] for r in shipped],
        "shipped_neighbours": neighbours(records, shipped),
        "windowed_neighbours": neighbours(records, mine),
        "passing_tail": len(passing), "tail_k": ref.TAIL_K,
        "fail_position": [r["command"] for r in shipped].index(f"c{FAIL_AT}") + 1,
        "shipped_len": len(shipped),
        "ordered": [r["command"] for r in shipped] == sorted(
            (r["command"] for r in shipped), key=lambda c: int(c[1:])),
        "loud": loud(ref), "repeated": repeated(ref),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the failing run gets 0 of 3 neighbours, the passing run all 5",
            all([result["shipped_neighbours"] == (0, 3),
                 result["windowed_neighbours"] == (3, 3),
                 result["passing_tail"] == result["tail_k"] == 5,
                 result["shipped"] == ["c15", "c16", "c17", "c18", "c19", "c2"]]),
            f"trim_feedback returns {result['shipped']}: "
            f"{result['shipped_neighbours'][0]} of {result['shipped_neighbours'][1]} "
            f"commands around the failure survive, against "
            f"{result['windowed_neighbours'][0]} under a window, while the passing run "
            f"keeps {result['passing_tail']} of its last {result['tail_k']}",
        ),
        practice.Check(
            "FINDING: tail + nonzero breaks chronological order",
            all([result["fail_position"] == 6, result["shipped_len"] == 6,
                 result["ordered"] is False]),
            f"the failure sits at index {FAIL_AT} of 20 and lands at position "
            f"{result['fail_position']} of {result['shipped_len']} in the packet, after "
            "the commands that came fifteen steps later -- the opposite of the order that "
            "explains it",
        ),
        practice.Check(
            "FINDING: the trim is unbounded in exactly the case it exists for",
            all([result["loud"]["failures"] == 60, result["loud"]["attempts"] == 60,
                 result["loud"]["tail"] == 64]),
            f"a 180-command session with {result['loud']['failures']} failures writes "
            f"{result['loud']['attempts']} failed_attempts lines and "
            f"{result['loud']['tail']} feedback_tail records: the trim shrinks the quiet "
            "packet and lets the loud one through",
        ),
        practice.Check(
            "FINDING: dedup by id() disagrees with the count beside it",
            result["repeated"] == (1, 2),
            f"the same failing record handed in twice collapses to "
            f"{result['repeated'][0]} entry in feedback_tail while failed_attempts reports "
            f"{result['repeated'][1]} -- one packet with two answers to how many times it "
            "failed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
