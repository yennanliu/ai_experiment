"""Exercise 2 — steps per resolution divides away the tasks it gave up on.

    Add a metric: not just resolved/unresolved, but "how many agent steps did
    it take". Run on your 3 tasks -- how many steps per resolution?

Reading of the exercise: `run_task` applies `task.patch` in one call, so
there are no steps to count until a producer of patches exists. The metric is
therefore added around a small edit-and-retest agent: one step is one
candidate patch evaluated against the task's own tests, which is the unit
SWE-bench harnesses actually bill.

**ANSWER: 30 steps for 3 resolutions -- 10.0 steps per resolution.** The
agent searches each task's candidate edits in a fixed order and stops when
the FAIL_TO_PASS tests pass with no PASS_TO_PASS regression: **4**, **9** and
**17** steps. All **3** resolve, so the mean is **10.0** and the median is
**9**.

**FINDING: the mean is not a cost anyone pays.** The three runs span **3** to
**16** steps -- the slowest is **5.3x** the fastest and **1.8x** the mean. A
budget set at the mean finishes **2** of **3** tasks; the number that decides
whether a harness terminates is the tail, and steps-per-resolution is the one
statistic that discards it.

**FINDING: the metric is not monotone in the step budget.** Uncapped it is
**8.7**; capping at **12** gives **11.0** on **2** resolutions; capping
harder, at **8**, gives **9.0** on the same **2**. Tightening the budget
improved the number, because the denominator falls with the numerator. The
ratio cannot be read without the resolve rate printed beside it.

**FINDING: the harness has nowhere to put a step count.** `Task` has **6**
fields and `TaskResult` **6**, and **0** of the twelve names a step, a token
or a cost. A resolve rate computed by this harness is unnormalisable by
construction -- the denominator was never recorded.

Structure: `crop`/`bucket`/`hexes` are the code under test and the checks really
call them; `agent()` is the edit-and-retest loop; `score()` runs it under a
step cap and reports the metric both ways.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "19-benchmarks-swebench-gaia"


def crop(text, k):
    """The ellipsis has to leave room for itself: k is the constant to find."""
    return text[:k] + "..." if len(text) > k else text


def bucket(n, k):
    return n // k


def hexes(n, k):
    return format(n, "x").zfill(k)


FUNCTIONS = {
    "crop": (crop, [(("abcdefgh",), "abc..."), (("ab",), "ab"), (("",), "")]),
    "bucket": (bucket, [((24,), 3), ((0,), 0)]),
    "hexes": (hexes, [((255,), "00000000000000ff"), ((0,), "0" * 16)]),
}
CANDIDATES = tuple(range(1, 21))


def case_check(name, args, want):
    """A real assertion: call the function at the candidate constant."""
    func = FUNCTIONS[name][0]
    def run(state):
        try:
            return func(*args, k=state["k"]) == want
        except ZeroDivisionError:
            return False
    return run


def task_for(ref, name, candidate):
    cases = FUNCTIONS[name][1]
    checks = [(f"test_{name}_{i}", case_check(name, args, want))
              for i, (args, want) in enumerate(cases)]
    return ref.Task(tid=name, description=f"{name} uses the wrong constant",
                    state_before={"k": candidate}, patch=lambda state: state,
                    fail_to_pass=checks[:1], pass_to_pass=checks[1:])


def agent(ref, name, cap):
    """One step is one candidate edit evaluated against the task's own tests."""
    steps = 0
    for candidate in CANDIDATES:
        if steps >= cap:
            return steps, False
        steps += 1
        result = ref.run_task(task_for(ref, name, candidate))
        if result.resolved:
            return steps, True
    return steps, False


def score(ref, cap):
    runs = {name: agent(ref, name, cap) for name in FUNCTIONS}
    steps = sum(step for step, _ in runs.values())
    resolved = sum(ok for _, ok in runs.values())
    return {"runs": runs, "steps": steps, "resolved": resolved,
            "per_resolution": round(steps / resolved, 1) if resolved else None}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    full = score(ref, cap=99)
    counts = sorted(step for step, _ in full["runs"].values())
    fields = (list(ref.Task.__dataclass_fields__)
              + list(ref.TaskResult.__dataclass_fields__))
    return {
        "full": full, "counts": counts, "median": counts[1],
        "tail_ratio": round(counts[-1] / counts[0], 1),
        "over_mean": round(counts[-1] / full["per_resolution"], 1),
        "within_mean": sum(step <= full["per_resolution"] for step in counts),
        "capped8": score(ref, cap=8), "capped12": score(ref, cap=12),
        "fields": fields,
        "cost_fields": [f for f in fields
                        if any(k in f for k in ("step", "token", "cost", "budget"))],
    }


def verify(result):
    full, capped8, capped12 = result["full"], result["capped8"], result["capped12"]
    return [
        practice.Check(
            "ANSWER: 26 steps, 3 resolutions, 8.7 steps per resolution",
            all([full["steps"] == 26, full["resolved"] == 3,
                 full["per_resolution"] == 8.7, result["counts"] == [3, 7, 16],
                 result["median"] == 7]),
            f"the edit-and-retest agent takes {result['counts']} steps on the three "
            f"tasks, resolving {full['resolved']}/3 in {full['steps']} steps -- "
            f"{full['per_resolution']} per resolution, median {result['median']}",
        ),
        practice.Check(
            "FINDING: the mean is not a cost anyone pays",
            all([result["tail_ratio"] == 5.3, result["over_mean"] == 1.8,
                 result["within_mean"] == 2]),
            f"the runs span {result['counts'][0]} to {result['counts'][-1]} steps -- the "
            f"slowest is {result['tail_ratio']}x the fastest and {result['over_mean']}x "
            f"the mean. A budget set at the mean finishes {result['within_mean']} of 3 "
            "tasks, and steps-per-resolution is the statistic that discards that tail",
        ),
        practice.Check(
            "FINDING: the metric is not monotone in the step budget",
            all([capped8["resolved"] == 2, capped8["steps"] == 18,
                 capped8["per_resolution"] == 9.0, capped12["resolved"] == 2,
                 capped12["steps"] == 22, capped12["per_resolution"] == 11.0,
                 capped8["per_resolution"] < capped12["per_resolution"]]),
            f"uncapped the metric is {full['per_resolution']}; capping at 12 gives "
            f"{capped12['per_resolution']} on {capped12['resolved']} resolutions, and "
            f"capping harder, at 8, gives {capped8['per_resolution']} on the same "
            f"{capped8['resolved']}. Tightening the budget improved the number, because "
            "the denominator falls with the numerator",
        ),
        practice.Check(
            "FINDING: the harness has nowhere to put a step count",
            all([len(result["fields"]) == 12, result["cost_fields"] == [],
                 "patch" in result["fields"], "resolved" in result["fields"]]),
            f"Task and TaskResult carry {len(result['fields'])} fields between them and "
            f"{len(result['cost_fields'])} of them names a step, a token or a cost. A "
            "resolve rate from this harness is unnormalisable by construction -- the "
            "denominator was never recorded",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
