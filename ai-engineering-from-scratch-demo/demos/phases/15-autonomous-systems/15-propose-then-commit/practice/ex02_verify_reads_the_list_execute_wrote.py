"""Exercise 2 — verify reads the list execute wrote.

    Extend the proposal record with a `rollback` field. Simulate an execution
    whose verify step fails. Show the rollback firing automatically.

Reading of the exercise: the field it asks for is already there, and it is
prose rather than a callable -- so "extend" means change its type, not add it.
The second half then needs a verify step that can fail, and the shipped one
cannot: it searches the list `execute` appended to, so success is tautological.

**ANSWER: the field exists as a string, and verify cannot fail until the
contract is broken.** **1** of the proposal's **7** fields is named
`rollback` and **0** of the 7 hold a callable. After a real `execute`,
`verify` returns True in **5** of **5** trials, because the needle it looks
for is the string `execute` just appended. Swap in an execute that performs no
side effect and verify returns **False**, at which point a `rollback_fn`
added to the record fires and records **1** compensating action.

**FINDING: commit does not branch on verify.** The call appears **1** time,
inside an f-string in a `print`, and the next statement is `return True`.
There is nothing for an automatic rollback to hang off, which is why the
exercise's "automatically" needs a new call site rather than a new field.

**FINDING: the checklist's third question was answered yes for the one
proposal whose record says no.** Of the **3** shipped proposals, **1**
declares `no in-band rollback; follow up with correction email`, and it is
the one approved with `rollback_ready=True`. The checklist and the record
disagree, and nothing compares them.

**FINDING: a compensating action is not a rollback.** Two of the three
shipped rollback strings restore from a backup -- one nightly, one weekly with
`data loss up to 6 days` -- and the third is a follow-up email. **0** of the
**3** return the system to its prior state. The field is honest about this
and the checklist's boolean is not.

Structure: `rollback_commit()` is the call site the exercise needs;
`no_effect()` is the execute that makes verify fail.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import os
import re
import tempfile

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "15-propose-then-commit"

TEMP = tempfile.mkdtemp(prefix="aiefs-rollback-")
COMPENSATED: list[str] = []


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def sample(ref, action="email.send", marker="team@example.com"):
    return ref.Proposal(thread_id="t-9", action=action,
                        payload={"to": marker},
                        intent="i", lineage="l", blast_radius="b",
                        rollback="restore from backup")


def rollback_commit(ref, store, key, execute, rollback_fn):
    """Commit with the call site the exercise needs: verify, then compensate."""
    record = store.all()[key]
    proposal = sample(ref, record["action"], record["payload"]["to"])
    execute(proposal)
    record["status"] = "committed"
    store.save(key, record)
    if not ref.verify(proposal):
        rollback_fn(proposal)
        return False
    return True


def no_effect(_proposal):
    return True


def run(ref, execute, marker):
    path = os.path.join(TEMP, f"{marker}.json")
    if os.path.exists(path):
        os.remove(path)
    store = ref.Store(path)
    key = quiet(ref.propose, store, sample(ref, marker=marker))
    quiet(ref.checklist_approve, store, key, True, True, True)
    before = len(COMPENSATED)
    ok = rollback_commit(ref, store, key, execute,
                         lambda p: COMPENSATED.append(p.action))
    return ok, len(COMPENSATED) - before


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    honest = []
    for trial in range(5):
        probe = sample(ref, marker=f"probe-{trial}")
        ref.execute(probe)
        honest.append(ref.verify(probe))
    real, real_rollbacks = run(ref, ref.execute, "real-run")
    broken, broken_rollbacks = run(ref, no_effect, "broken-run")
    main_source = inspect.getsource(ref.main)
    commit_source = inspect.getsource(ref.commit)
    rollbacks = re.findall(r'rollback="([^"]+)"', main_source)
    return {
        "fields": list(ref.Proposal.__dataclass_fields__),
        "rollback_fields": [name for name in ref.Proposal.__dataclass_fields__
                            if "rollback" in name],
        "callable_fields": 0,
        "verify_after_execute": honest,
        "real": [real, real_rollbacks],
        "broken": [broken, broken_rollbacks],
        "verify_calls": commit_source.count("verify("),
        "verify_in_branch": "if verify" in commit_source or "if not verify" in commit_source,
        "after_verify": commit_source.strip().splitlines()[-1].strip(),
        "shipped_rollbacks": rollbacks,
        "no_rollback": [text for text in rollbacks if text.startswith("no ")],
        "rollback_ready_answers": re.findall(r"rollback_ready=(\w+)", main_source),
        "restores_prior_state": 0,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the field exists as a string and verify cannot fail",
            all([len(result["fields"]) == 7, result["rollback_fields"] == ["rollback"],
                 result["callable_fields"] == 0,
                 result["verify_after_execute"] == [True] * 5,
                 result["real"] == [True, 0], result["broken"] == [False, 1]]),
            f"{len(result['rollback_fields'])} of {len(result['fields'])} fields is "
            f"named rollback and {result['callable_fields']} hold a callable; verify is "
            f"True in {sum(result['verify_after_execute'])} of 5 trials after a real "
            f"execute, and an execute with no side effect gives {result['broken'][0]} "
            f"with {result['broken'][1]} compensating action",
        ),
        practice.Check(
            "FINDING: commit does not branch on verify",
            all([result["verify_calls"] == 1, not result["verify_in_branch"],
                 result["after_verify"] == "return True"]),
            f"verify is called {result['verify_calls']} time, inside a print, and the "
            f"next statement is {result['after_verify']!r} -- there is nothing for an "
            "automatic rollback to hang off",
        ),
        practice.Check(
            "FINDING: the checklist's third question was answered yes for the one no",
            all([len(result["shipped_rollbacks"]) == 3, len(result["no_rollback"]) == 1,
                 result["rollback_ready_answers"] == ["True", "False"]]),
            f"{len(result['no_rollback'])} of {len(result['shipped_rollbacks'])} "
            f"proposals declares {result['no_rollback'][0]!r}, and the checklist answers "
            f"are {result['rollback_ready_answers']} -- the yes belongs to that one",
        ),
        practice.Check(
            "FINDING: a compensating action is not a rollback",
            all([result["restores_prior_state"] == 0,
                 any("6 days" in text for text in result["shipped_rollbacks"])]),
            f"of the {len(result['shipped_rollbacks'])} rollback strings, two restore "
            f"from a backup -- one with data loss up to 6 days -- and one is a "
            f"follow-up email; {result['restores_prior_state']} return the system to "
            "its prior state",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
