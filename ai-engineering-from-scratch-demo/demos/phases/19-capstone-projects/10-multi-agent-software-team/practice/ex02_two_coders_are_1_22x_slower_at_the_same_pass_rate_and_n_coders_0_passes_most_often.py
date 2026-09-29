"""Exercise 2 — two coders are 1.22x slower at the same pass rate, and `run_team(n_coders=0)` passes most often.

    Reduce to two coders (architect + coder + reviewer + tester, coder runs two subtasks sequentially). Compare wall-clock and pass rate.

Reading of the exercise: two coders each run two of the four planned
subtasks back to back; architect, reviewer and tester are unchanged. The
lesson's code has no clock, so wall-clock is the critical path in generated
tokens: every serial message's tokens plus the slowest coder lane's
DIFF_READY tokens, measured from the board of 1,000 seeded runs. Pass rate is
`tested_passed` on the same seeds.

**ANSWER: 25,805 against 21,101 tokens on the critical path, 1.22x slower,
at the same 935/1000 pass rate.** One coder doing all four is 34,973 (1.66x).
The pass rate cannot move: `reviewer_check` and `tester_run` take only
`(diffs, rng)`, never the coder, so the same seed gives the same verdict
whichever lane did the work.

**FINDING: the lesson's own two-coder switch drops half the plan.**
`run_team(n_coders=2)` slices `plan[:2]`, so `api` and `migration` are never
dispatched, yet it passes 936/1000. `n_coders=0` codes nothing and passes
973/1000, the best score in the table, because a bug planted in an
undispatched subtask never reaches review.

**FINDING: four parallel coders only tie the single agent on wall-clock.**
`single_agent_baseline` spends 21,051 tokens serially (703/1000 pass), and
the four-coder critical path is 21,101, 1.002x of it. The serial handoffs
eat the whole parallel gain; the doc's "parallel speedup" criterion has
nothing to win in this scaffold.
"""

from __future__ import annotations

import contextlib
import inspect
import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "10-multi-agent-software-team"
RUNS = 1000


@contextlib.contextmanager
def recording(ref):
    boards, real = [], ref.Board

    class Recording(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            boards.append(self)

    ref.Board = Recording
    try:
        yield boards
    finally:
        ref.Board = real


def critical_path(ref, board, lanes):
    """Wall-clock in generated tokens: serial stages plus the slowest coder lane.

    `lanes` lists, per coder, the plan indices it runs back to back.
    """
    first = [m.tokens for m in board.messages if m.kind == ref.MsgKind.DIFF_READY][:4]
    coders = max(sum(first[i] for i in lane) for lane in lanes)
    serial = sum(m.tokens for m in board.messages) - sum(first)
    return serial + coders


LAYOUTS = {"4 coders": [[0], [1], [2], [3]], "2 coders, 2 subtasks each": [[0, 1], [2, 3]],
           "1 coder, 4 in a row": [[0, 1, 2, 3]]}


def layout_times(ref):
    """Mean critical path per layout, and the pass count, on the 4-coder runs."""
    times, passed = {k: 0 for k in LAYOUTS}, 0
    with recording(ref) as boards:
        for seed in range(RUNS):
            passed += ref.run_team(f"issue-{seed}", n_coders=4, rng=random.Random(seed))["tested_passed"]
            for name, lanes in LAYOUTS.items():
                times[name] += critical_path(ref, boards[-1], lanes)
    return {k: round(v / RUNS) for k, v in times.items()}, passed


def lesson_switch(ref, n):
    """What the lesson's own `n_coders` does: passes, and subtasks dispatched per run."""
    with recording(ref) as boards:
        runs = [ref.run_team(f"issue-{s}", n_coders=n, rng=random.Random(s)) for s in range(RUNS)]
    coded = {sum(m.kind == ref.MsgKind.SUBTASK for m in b.messages) for b in boards}
    return {"passed": sum(r["tested_passed"] for r in runs), "subtasks_coded": sorted(coded)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    times, passed = layout_times(ref)
    base = [ref.single_agent_baseline(f"issue-{s}", random.Random(s)) for s in range(RUNS)]
    return {
        "time": times, "passed_4": passed,
        "outcome_args": [list(inspect.signature(f).parameters) for f in (ref.reviewer_check, ref.tester_run)],
        "shipped": {n: lesson_switch(ref, n) for n in (0, 2, 4)},
        "base_pass": sum(b["passed"] for b in base),
        "base_time": round(sum(b["total_tokens"] for b in base) / RUNS),
    }


def verify(result):
    r, t, sh = result, result["time"], result["shipped"]
    two, four = t["2 coders, 2 subtasks each"], t["4 coders"]
    return [
        practice.Check(
            "ANSWER: two coders are 1.22x slower on wall-clock at the same 935/1000 pass rate",
            (four, two, t["1 coder, 4 in a row"], r["passed_4"], r["outcome_args"])
            == (21101, 25805, 34973, 935, [["diffs", "rng"], ["diffs", "rng"]]),
            f"mean critical path in generated tokens: 4 coders {four:,}, 2 coders {two:,} ({two / four:.2f}x), "
            f"1 coder {t['1 coder, 4 in a row']:,}; pass {r['passed_4']}/{RUNS} for any layout, since the "
            f"reviewer and tester take only {r['outcome_args'][0]}",
        ),
        practice.Check(
            "FINDING: run_team(n_coders=2) drops half the plan, and n_coders=0 passes most often",
            {n: (v["passed"], v["subtasks_coded"]) for n, v in sh.items()}
            == {0: (973, [0]), 2: (936, [2]), 4: (935, [4])},
            "; ".join(f"n_coders={n}: {v['subtasks_coded'][0]}/4 subtasks coded, {v['passed']}/{RUNS} pass"
                      for n, v in sh.items()),
        ),
        practice.Check(
            "FINDING: four parallel coders only tie the single agent on wall-clock",
            (r["base_time"], r["base_pass"]) == (21051, 703) and four > r["base_time"],
            f"single agent: {r['base_time']:,} tokens serial, {r['base_pass']}/{RUNS} pass; "
            f"4-coder team critical path {four:,} ({four / r['base_time']:.3f}x)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
