"""Exercise 3 — the one-megabyte cap is really six, and checked before the write.

    Cap total `feedback_record.jsonl` size at 1 MB by rotating to `.1`, `.2`
    files. Defend the rotation policy.

Reading of the exercise: `maybe_rotate` ships with `ROTATE_BYTES` at 1 MB and
`MAX_ROTATIONS` at 5, so the mechanism is there and the *total* is not what
the exercise asks for. The policy is worth defending on its merits, and two
of its properties are accidents rather than choices: the cap applies to the
active file only, and it is evaluated before the append rather than after.

**ANSWER: the active file is capped at 1 MB and the directory holds up to 6.**
Appending until rotation cycles produces `feedback_record.jsonl` plus **5**
numbered siblings -- **5 MB** on this run and a ceiling of **6 MB**, where
the exercise says **1 MB**. Rotation is correct in ordering -- oldest dropped, `.1` newest --
and `load_all` reads every file, so lineage survives it.

**FINDING: the check runs before the write, so the active file exceeds the
cap by one record.** `maybe_rotate` is called before `RECORD.open("a")`, and
it returns early when the file is *under* `ROTATE_BYTES`. A record appended
to a 1048575-byte file leaves it at **1049319** bytes -- over `ROTATE_BYTES`
-- so the cap is a floor: the file is rotated once it is already too big,
never before.

**FINDING: a single record larger than the cap is written whole and never
split.** With `HEAD_LINES=5` and `TAIL_LINES=30` a capture is bounded at
**35** lines, so in practice a record cannot approach 1 MB -- the truncation
is what enforces the cap, and `ROTATE_BYTES` only decides how many records
fit. Remove the truncation and the rotation policy stops bounding anything.

**FINDING: rotation by size loses the property a reviewer wants, which is
time.** The **6** files carry no timestamps in their names and
`started_at` is inside the records, so "what happened on Tuesday" requires
opening every file. Rotating by size keeps memory bounded, which is the
stated goal; rotating by day would keep the *question* bounded, and the two
policies want different names.

Structure: `fill()` appends records until rotation fires; `sizes()` reads
the resulting directory.
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


def line(index, payload_bytes):
    return json.dumps({"command_id": f"c{index:06d}", "parent_command_id": None,
                       "command": ["x"], "stdout_tail": "y" * payload_bytes,
                       "stderr_tail": "", "exit_code": 0, "duration_ms": 1,
                       "started_at": 0.0, "agent_note": "", "error": None,
                       "truncations": {}, "redactions": {}}) + "\n"


def fill(ref, root, rotations=6, chunk=64 * 1024):
    """Append until the rotation policy has cycled past MAX_ROTATIONS."""
    saved = sandbox(ref, root)
    try:
        index = 0
        per_file = ref.ROTATE_BYTES // chunk + 1
        for _ in range(rotations * per_file):
            ref.maybe_rotate()
            with ref.RECORD.open("a") as handle:
                handle.write(line(index, chunk))
            index += 1
        return sorted(p.name for p in root.iterdir())
    finally:
        ref.HERE, ref.RECORD = saved


def boundary(ref, root):
    """One append onto a file that is one byte under the cap."""
    saved = sandbox(ref, root)
    try:
        ref.RECORD.write_bytes(b"x" * (ref.ROTATE_BYTES - 1))
        before = ref.RECORD.stat().st_size
        ref.maybe_rotate()
        rotated = ref.RECORD.stat().st_size if ref.RECORD.exists() else 0
        with ref.RECORD.open("a") as handle:
            handle.write(line(0, 512))
        return {"before": before, "after_rotate": rotated,
                "after_write": ref.RECORD.stat().st_size,
                "cap": ref.ROTATE_BYTES}
    finally:
        ref.HERE, ref.RECORD = saved


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    root = pathlib.Path(tempfile.mkdtemp())
    names = fill(ref, root)
    total = sum(p.stat().st_size for p in root.iterdir())
    edge = boundary(ref, pathlib.Path(tempfile.mkdtemp()))
    source = inspect.getsource(ref.run_with_feedback)
    return {
        "cap_mb": ref.ROTATE_BYTES // (1024 * 1024),
        "max_rotations": ref.MAX_ROTATIONS,
        "files": len(names), "names": names,
        "total_mb": round(total / (1024 * 1024)),
        "rotate_before_write": source.index("maybe_rotate") < source.index('open("a")'),
        "boundary": edge,
        "over_cap": edge["after_write"] > edge["cap"],
        "head": ref.HEAD_LINES, "tail": ref.TAIL_LINES,
        "max_capture_lines": ref.HEAD_LINES + ref.TAIL_LINES,
        "timestamped_names": sum(any(c.isdigit() for c in n.split(".")[-1])
                                 and len(n.split(".")[-1]) > 2 for n in names),
    }


def verify(result):
    edge = result["boundary"]
    return [
        practice.Check(
            "ANSWER: the active file is capped at 1 MB and the directory holds 6",
            all([result["cap_mb"] == 1, result["max_rotations"] == 5,
                 result["files"] == 6, result["total_mb"] == 5,
                 result["names"][0] == "feedback_record.jsonl"]),
            f"ROTATE_BYTES is {result['cap_mb']} MB and MAX_ROTATIONS is "
            f"{result['max_rotations']}, so appending until rotation cycles leaves "
            f"{result['files']} files totalling {result['total_mb']} MB on this run "
            f"with a ceiling of {result['max_rotations'] + 1}: {result['names']}",
        ),
        practice.Check(
            "FINDING: the check runs before the write, so the cap is a floor",
            all([result["rotate_before_write"] is True,
                 edge["before"] == edge["cap"] - 1,
                 edge["after_rotate"] == edge["before"],
                 result["over_cap"] is True,
                 edge["after_write"] > edge["cap"]]),
            f"maybe_rotate is called before the append ({result['rotate_before_write']}) "
            f"and returns early under the cap, so a file at {edge['before']} bytes is not "
            f"rotated and then grows to {edge['after_write']}. The file is rotated once "
            "it is already over, never before",
        ),
        practice.Check(
            "FINDING: truncation is what actually bounds a record",
            all([result["max_capture_lines"] == 35, result["head"] == 5,
                 result["tail"] == 30]),
            f"deterministic_tail keeps {result['head']} head and {result['tail']} tail "
            f"lines, so a capture is bounded at {result['max_capture_lines']} lines and a "
            "record cannot approach a megabyte. ROTATE_BYTES only decides how many "
            "records fit; remove the truncation and rotation stops bounding anything",
        ),
        practice.Check(
            "FINDING: rotation by size loses the property a reviewer wants",
            all([result["timestamped_names"] == 0, result["files"] == 6]),
            f"the {result['files']} files carry {result['timestamped_names']} timestamps "
            "in their names, and started_at lives inside the records -- so 'what happened "
            "on Tuesday' means opening every file. Size keeps memory bounded; a daily "
            "rotation would keep the question bounded",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
