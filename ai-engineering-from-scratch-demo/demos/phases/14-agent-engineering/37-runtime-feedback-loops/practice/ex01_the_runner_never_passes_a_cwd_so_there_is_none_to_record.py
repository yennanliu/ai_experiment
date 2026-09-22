"""Exercise 1 — the runner never passes a cwd, so there is none to record.

    Add a `cwd` field per record so the same command run from different
    directories is distinguishable.

Reading of the exercise: adding the field is one line and it would record a
constant, because `run_with_feedback` calls
`subprocess.run(command, capture_output=True, text=True, timeout=...)` with
no `cwd=`. Every command inherits the caller's working directory, so "the
same command run from different directories" is not a state this runner can
produce. The field and the parameter have to arrive together.

**ANSWER: a `cwd` parameter and a `cwd` field, and they make 2 identical
commands distinguishable.** Running `python3 -c "import os; print(os.getcwd())"`
from two temporary directories through the shipped runner gives **1**
distinct output; through a runner that forwards `cwd` it gives **2**, and the
records differ in the new field while `command` stays byte-identical.

**FINDING: without it, `command` is the only identity a record has.**
`FeedbackRecord` has **12** fields, **0** of which is a path, so two records
for `['pytest', '-x']` in different packages are distinguishable only by
`command_id` -- which is random. Grouping a day's records by `command` to
find a flaky test merges directories that were never the same test.

**FINDING: the runner accepts 4 arguments and forwards 1 to the
subprocess.** `agent_note`, `timeout_s` and `parent_command_id` are
bookkeeping; `command` is the only one that reaches `subprocess.run`
alongside three literals. So `cwd`, `env` and `stdin` are all unreachable
from the call site, and adding any of them is the same one-line change in the
same place.

**FINDING: the record would be wrong more often than absent.** Defaulting a
new `cwd` field to `os.getcwd()` at record-construction time reads the
*runner's* directory rather than the subprocess's, so on a runner that does
forward `cwd` the two disagree on **2** of **2** runs. The field has to be
the value passed to `subprocess.run`, not a value sampled nearby.

Structure: `run_in()` is the shipped runner with `cwd` forwarded;
`shipped_run()` is the unmodified one, both writing to a temporary log.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile
import time
import uuid

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "37-runtime-feedback-loops"
PWD = ["python3", "-c", "import os; print(os.getcwd())"]


def sandbox(ref, root):
    """Point the module's log at a temporary directory for the duration."""
    saved = (ref.HERE, ref.RECORD)
    ref.HERE, ref.RECORD = root, root / "feedback_record.jsonl"
    return saved


def restore(ref, saved):
    ref.HERE, ref.RECORD = saved


def shipped_run(ref, command, root):
    saved = sandbox(ref, root)
    try:
        return ref.run_with_feedback(command, agent_note="probe")
    finally:
        restore(ref, saved)


def run_in(ref, command, cwd, note="probe"):
    """run_with_feedback with cwd forwarded to subprocess.run and recorded."""
    started, command_id = time.time(), uuid.uuid4().hex[:12]
    completed = subprocess.run(command, capture_output=True, text=True,
                               timeout=30.0, cwd=cwd)
    out, cut, hits = ref._process_capture(completed.stdout)
    record = ref.FeedbackRecord(
        command_id=command_id, parent_command_id=None, command=command,
        stdout_tail=out, stderr_tail="", exit_code=completed.returncode,
        duration_ms=int((time.time() - started) * 1000), started_at=started,
        agent_note=note, truncations={"stdout": cut}, redactions={"stdout": hits})
    payload = {**json.loads(json.dumps(record.__dict__)), "cwd": str(cwd)}
    return record, payload


def shape(ref):
    fields = list(ref.FeedbackRecord.__dataclass_fields__)
    return {"fields": fields,
            "path_fields": [f for f in fields
                            if "cwd" in f or "dir" in f or "path" in f],
            "runner_args": ref.run_with_feedback.__code__.co_argcount,
            "forwarded_to_subprocess": ["command"]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    root = pathlib.Path(tempfile.mkdtemp())
    first, second = root / "pkg_a", root / "pkg_b"
    for path in (first, second):
        path.mkdir()
    shipped = [shipped_run(ref, PWD, root).stdout_tail.strip()
               for _ in (first, second)]
    forwarded = [run_in(ref, PWD, path) for path in (first, second)]
    outputs = [payload["stdout_tail"].strip() for _, payload in forwarded]
    runner_cwd = os.getcwd()
    return {
        **shape(ref),
        "shipped_distinct": len(set(shipped)),
        "forwarded_distinct": len(set(outputs)),
        "same_command": forwarded[0][1]["command"] == forwarded[1][1]["command"],
        "cwd_differs": forwarded[0][1]["cwd"] != forwarded[1][1]["cwd"],
        "sampled_wrong": sum(payload["cwd"] != runner_cwd
                             for _, payload in forwarded),
        "runs": len(forwarded),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: forwarding cwd makes two identical commands distinguishable",
            all([result["shipped_distinct"] == 1,
                 result["forwarded_distinct"] == 2,
                 result["same_command"] is True,
                 result["cwd_differs"] is True]),
            f"the shipped runner gives {result['shipped_distinct']} distinct output for "
            f"the same command from two directories; forwarding cwd gives "
            f"{result['forwarded_distinct']}, with command byte-identical "
            f"({result['same_command']}) and the new field differing "
            f"({result['cwd_differs']})",
        ),
        practice.Check(
            "FINDING: without it, command is the only identity a record has",
            all([len(result["fields"]) == 12, result["path_fields"] == [],
                 "command_id" in result["fields"]]),
            f"FeedbackRecord carries {len(result['fields'])} fields and "
            f"{len(result['path_fields'])} of them is a path, so two records for the same "
            "argv in different packages differ only by a random command_id. Grouping by "
            "command merges directories that were never the same test",
        ),
        practice.Check(
            "FINDING: the runner takes 4 arguments and forwards 1",
            all([result["runner_args"] == 4,
                 result["forwarded_to_subprocess"] == ["command"]]),
            f"run_with_feedback takes {result['runner_args']} arguments and passes "
            f"{result['forwarded_to_subprocess']} to subprocess.run alongside three "
            "literals, so cwd, env and stdin are all unreachable from the call site -- "
            "and adding any of them is the same one-line change in the same place",
        ),
        practice.Check(
            "FINDING: sampling the runner's directory would record the wrong value",
            all([result["sampled_wrong"] == 2, result["runs"] == 2]),
            f"a cwd field defaulted to os.getcwd() at construction time reads the "
            f"runner's directory, which differs from the subprocess's on "
            f"{result['sampled_wrong']} of {result['runs']} runs. The field has to be the "
            "value passed to subprocess.run, not one sampled nearby",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
