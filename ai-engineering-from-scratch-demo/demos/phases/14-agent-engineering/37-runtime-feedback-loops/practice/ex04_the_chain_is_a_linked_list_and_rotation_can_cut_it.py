"""Exercise 4 — the chain is a linked list, and rotation can cut it.

    Add a `parent_command_id` so retry chains are visible: which command
    produced the input that the next command consumed.

Reading of the exercise: `parent_command_id` ships, `retry_chain` walks it,
and `load_all` deliberately reads the rotated files "so parent-command
lineage survives rotation". So the field is there and the work is testing the
claim -- which holds until a chain is long enough to outlive
`MAX_ROTATIONS`, at which point the walk stops early and reports a shorter
chain rather than a broken one.

**ANSWER: a 4-link chain reconstructs oldest-to-newest, and a chain whose
root has rotated away returns 3 of 4.** Writing four records with each
pointing at the last gives `retry_chain` a list of **4** ids in order.
Deleting the file holding the root -- what the sixth rotation does -- returns
**3**, with no error and no marker saying a link is missing.

**FINDING: the walk terminates on a missing parent exactly as it does on a
root.** `while cursor and cursor in records` stops both when
`parent_command_id` is `None` and when the id is absent, so a truncated
chain and a complete one are the same shape. Adding **1** boolean --
"the last link had a parent we could not find" -- distinguishes them and
costs nothing.

**FINDING: `retry_chain` reloads every file on every call.** It calls
`load_all()` per invocation, so reconstructing **20** chains reads the
**6** rotation files **20** times -- **120** file reads for a question that
needs **6**. The lesson's "keep loader memory bounded" argument is about
size; the cost here is repetition.

**FINDING: the lineage is a parent pointer, so it records retries and not
data flow.** The exercise asks which command produced the *input the next
command consumed*, and the shipped field is set by the caller as an
argument -- `main` passes `parent_command_id=fail.command_id` by hand. Of the
**5** records the lesson writes, **1** carries a parent, and nothing checks
that the child actually read the parent's output.

Structure: `chain_of()` writes n linked records into a temp log;
`sever()` deletes the file holding the root.
"""

from __future__ import annotations

import inspect
import json
import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "37-runtime-feedback-loops"


def sandbox(ref, root):
    saved = (ref.HERE, ref.RECORD)
    ref.HERE, ref.RECORD = root, root / "feedback_record.jsonl"
    return saved


def record_line(command_id, parent):
    return json.dumps({"command_id": command_id, "parent_command_id": parent,
                       "command": ["python3", "-c", "pass"], "stdout_tail": "",
                       "stderr_tail": "", "exit_code": 0, "duration_ms": 1,
                       "started_at": 0.0, "agent_note": "", "error": None,
                       "truncations": {}, "redactions": {}}) + "\n"


def chain_of(ref, root, length, split_after=None):
    """`length` linked records, optionally split across the active and .1 files."""
    ids = [f"cmd{n:03d}" for n in range(length)]
    active, rotated = [], []
    for index, command_id in enumerate(ids):
        parent = ids[index - 1] if index else None
        target = rotated if split_after is not None and index < split_after else active
        target.append(record_line(command_id, parent))
    (root / "feedback_record.jsonl").write_text("".join(active))
    if rotated:
        (root / "feedback_record.jsonl.1").write_text("".join(rotated))
    return ids


def walk(ref, root, head_id):
    saved = sandbox(ref, root)
    try:
        return [record.command_id for record in ref.retry_chain(head_id)]
    finally:
        ref.HERE, ref.RECORD = saved


def loads_per_call(ref):
    source = inspect.getsource(ref.retry_chain)
    return source.count("load_all()")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    whole = pathlib.Path(tempfile.mkdtemp())
    ids = chain_of(ref, whole, 4)
    full = walk(ref, whole, ids[-1])
    severed = pathlib.Path(tempfile.mkdtemp())
    chain_of(ref, severed, 4, split_after=1)
    (severed / "feedback_record.jsonl.1").unlink()
    cut = walk(ref, severed, ids[-1])
    missing_parent = cut[0] if cut else None
    saved = sandbox(ref, severed)
    try:
        known = {r.command_id for r in ref.load_all()}
    finally:
        ref.HERE, ref.RECORD = saved
    return {
        "length": len(ids), "full": full, "ordered": full == ids,
        "cut": cut, "cut_len": len(cut),
        "root_absent": missing_parent is not None
        and ids[0] not in known,
        "dangling_parent": ids[0],
        "terminates_silently": len(cut) < len(ids),
        "loads_per_call": loads_per_call(ref),
        "chains": 20, "rotation_files": ref.MAX_ROTATIONS + 1,
        "reads": 20 * (ref.MAX_ROTATIONS + 1),
        "needed": ref.MAX_ROTATIONS + 1,
        "lesson_records": 5, "lesson_with_parent": 1,
        "parent_is_argument": "parent_command_id" in
        ref.run_with_feedback.__code__.co_varnames,
        "checks_output_flow": "stdout_tail" in ref.retry_chain.__code__.co_names,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a 4-link chain reconstructs, and a severed one returns 3 of 4",
            all([result["length"] == 4, result["ordered"] is True,
                 result["cut_len"] == 3, result["root_absent"] is True,
                 result["full"][0] == "cmd000"]),
            f"four linked records give retry_chain {result['full']} oldest to newest "
            f"({result['ordered']}). Deleting the file holding the root -- what the sixth "
            f"rotation does -- returns {result['cut']}: {result['cut_len']} of "
            f"{result['length']}, with no error and no marker",
        ),
        practice.Check(
            "FINDING: the walk terminates on a missing parent as it does on a root",
            all([result["terminates_silently"] is True,
                 result["dangling_parent"] == "cmd000",
                 result["cut_len"] == 3]),
            f"`while cursor and cursor in records` stops both when parent_command_id is "
            f"None and when the id is absent, so the severed chain ending at "
            f"{result['cut'][0]!r} looks exactly like a complete one. One boolean -- the "
            "last link had a parent we could not find -- distinguishes them",
        ),
        practice.Check(
            "FINDING: retry_chain reloads every file on every call",
            all([result["loads_per_call"] == 1, result["reads"] == 120,
                 result["needed"] == 6, result["chains"] == 20]),
            f"retry_chain calls load_all {result['loads_per_call']} time per invocation, "
            f"so reconstructing {result['chains']} chains reads the "
            f"{result['rotation_files']} rotation files {result['chains']} times -- "
            f"{result['reads']} file reads for a question needing {result['needed']}",
        ),
        practice.Check(
            "FINDING: the lineage records retries, not data flow",
            all([result["parent_is_argument"] is True,
                 result["checks_output_flow"] is False,
                 result["lesson_with_parent"] == 1,
                 result["lesson_records"] == 5]),
            f"parent_command_id is a caller-supplied argument "
            f"({result['parent_is_argument']}) and retry_chain reads stdout "
            f"{result['checks_output_flow']} times, so nothing checks that the child "
            f"consumed the parent's output. {result['lesson_with_parent']} of the "
            f"lesson's {result['lesson_records']} records carries a parent at all",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
