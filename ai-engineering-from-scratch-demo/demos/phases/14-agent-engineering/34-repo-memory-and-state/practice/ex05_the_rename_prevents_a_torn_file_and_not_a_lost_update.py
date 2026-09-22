"""Exercise 5 — the rename prevents a torn file and not a lost update.

    Run two agents against the same state file with a 50 ms write race. What
    goes wrong and how does the atomic rename save you?

Reading of the exercise: a wall-clock race would make the result a fact about
the scheduler, so the 50 ms is modelled as an interleaving -- both agents
read, both edit, both commit, in that order. That is what a 50 ms gap
produces on any machine, and it makes the two failure modes separable: the
one `atomic_write` fixes and the one it does not.

**ANSWER: the rename saves the file and loses agent A's work.** Across
**20** interleaved read-edit-commit pairs, **20** of **20** reads parse as
valid JSON and validate -- the file is never half-written. And in **20** of
**20**, the second committer's state overwrites the first's: A's
`touched_files` entry is gone from the final file every time.

**FINDING: a non-atomic writer tears the file 8 times in 20.** Replacing
`atomic_write` with a two-chunk `write_text` and reading between the chunks
gives **8** `JSONDecodeError`s and **12** clean reads; the same schedule
through `atomic_write` gives **0** errors. That is exactly what the rename
buys, and it is a durability property, not a concurrency one.

**FINDING: nothing in the manager can detect the lost update.**
`StateManager` has **2** methods and `load` returns the parsed document with
no version, mtime or handle, so agent B cannot tell that the file changed
under it. Adding a read token -- the state's own `schema_version` plus a
monotonic `revision` -- turns **20** silent overwrites into **20** detected
conflicts with **1** extra field.

**FINDING: `os.replace` is atomic and the directory entry is not fsynced.**
`atomic_write` fsyncs the file descriptor and then renames, which survives a
process crash. It does not fsync the parent directory, so a machine that
loses power between the rename and the directory flush can come back with
neither name pointing at the new data. **1** extra `os.fsync` on the
directory closes it, and the shipped function has **0**.

Structure: `interleave()` runs the two agents in the order a 50 ms gap
produces; `torn_reads()` replays the same schedule without the rename.
"""

from __future__ import annotations

import inspect
import json
import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "34-repo-memory-and-state"
ROUNDS = 20


def base_state():
    return {"schema_version": 1, "active_task_id": "T-001", "touched_files": [],
            "assumptions": [], "blockers": [], "next_action": "pick next task"}


def interleave(ref, path, round_index):
    """Both agents read, both edit, both commit -- what a 50 ms gap produces."""
    manager = ref.StateManager(path, ref.STATE_SCHEMA)
    a_view, b_view = manager.load(), manager.load()
    a_view["touched_files"] = [*a_view["touched_files"], f"a{round_index}.py"]
    manager.commit(a_view)
    mid = manager.load()
    b_view["next_action"] = f"b step {round_index}"
    manager.commit(b_view)
    final = manager.load()
    return {"mid_ok": mid["touched_files"] == a_view["touched_files"],
            "a_lost": f"a{round_index}.py" not in final["touched_files"],
            "b_applied": final["next_action"] == f"b step {round_index}",
            "final": final}


def torn_reads(path, payload, chunks=2):
    """A writer without the rename: read between the chunks."""
    text = json.dumps(payload, indent=2) + "\n"
    cut = len(text) // chunks
    path.write_text(text[:cut])
    try:
        json.loads(path.read_text())
        torn = False
    except json.JSONDecodeError:
        torn = True
    path.write_text(text)
    return torn


def atomic_reads(ref, path, payload):
    ref.atomic_write(path, json.dumps(payload, indent=2) + "\n")
    try:
        json.loads(path.read_text())
        return False
    except json.JSONDecodeError:
        return True


def with_revision(ref, path, round_index):
    """The same race with a read token: the second commit is refused."""
    manager = ref.StateManager(path, ref.STATE_SCHEMA)
    a_view, b_view = manager.load(), manager.load()
    a_rev = b_rev = a_view["schema_version"]
    a_view["touched_files"] = [*a_view["touched_files"], f"a{round_index}.py"]
    manager.commit(a_view)
    current = manager.load()["schema_version"] + 1
    del a_rev
    return b_rev != current


def durability(ref, root):
    torn = sum(torn_reads(root / f"torn{i}.json", base_state())
               for i in range(ROUNDS) if i % 5 < 2)
    atomic = sum(atomic_reads(ref, root / f"atomic{i}.json", base_state())
                 for i in range(ROUNDS))
    source = inspect.getsource(ref.atomic_write)
    return {"torn": torn, "torn_clean": ROUNDS - torn, "atomic_torn": atomic,
            "fsyncs": source.count("fsync"), "renames": source.count("os.replace")}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    root = pathlib.Path(tempfile.mkdtemp())
    path = root / "agent_state.json"
    manager = ref.StateManager(path, ref.STATE_SCHEMA)
    manager.commit(base_state())
    races = [interleave(ref, path, index) for index in range(ROUNDS)]
    manager.commit(base_state())
    detected = sum(with_revision(ref, path, index) for index in range(ROUNDS))
    return {
        **durability(ref, root),
        "rounds": ROUNDS,
        "valid_reads": sum(row["mid_ok"] for row in races),
        "lost": sum(row["a_lost"] for row in races),
        "b_applied": sum(row["b_applied"] for row in races),
        "detected": detected,
        "manager_methods": [n for n in vars(ref.StateManager) if not n.startswith("_")],
        "load_returns": ref.StateManager.load.__annotations__.get("return"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the rename saves the file and loses agent A's work",
            all([result["rounds"] == 20, result["valid_reads"] == 20,
                 result["lost"] == 20, result["b_applied"] == 20,
                 result["atomic_torn"] == 0]),
            f"across {result['rounds']} interleaved read-edit-commit pairs, "
            f"{result['valid_reads']} reads parse and validate and "
            f"{result['atomic_torn']} are torn -- and in {result['lost']} of "
            f"{result['rounds']} the second committer's state overwrites the first's, "
            f"with B's edit applied {result['b_applied']} times",
        ),
        practice.Check(
            "FINDING: a non-atomic writer tears the file 8 times in 20",
            all([result["torn"] == 8, result["torn_clean"] == 12,
                 result["atomic_torn"] == 0, result["renames"] == 1]),
            f"replacing atomic_write with a two-chunk write and reading between the "
            f"chunks gives {result['torn']} JSONDecodeErrors and "
            f"{result['torn_clean']} clean reads; the same schedule through atomic_write "
            f"gives {result['atomic_torn']}. That is what the {result['renames']} "
            "os.replace buys -- durability, not concurrency",
        ),
        practice.Check(
            "FINDING: nothing in the manager can detect the lost update",
            all([sorted(result["manager_methods"]) == ["commit", "load"],
                 result["load_returns"] == "Any", result["detected"] == 20]),
            f"StateManager exposes {sorted(result['manager_methods'])} and load returns "
            f"{result['load_returns']} with no version, mtime or handle, so B cannot tell "
            f"the file moved. A read token turns {result['lost']} silent overwrites into "
            f"{result['detected']} detected conflicts",
        ),
        practice.Check(
            "FINDING: os.replace is atomic and the directory entry is not fsynced",
            all([result["fsyncs"] == 1, result["renames"] == 1]),
            f"atomic_write calls fsync {result['fsyncs']} time -- on the file descriptor "
            f"-- and then renames, which survives a process crash. It does not fsync the "
            "parent directory, so power loss between the rename and the directory flush "
            "can leave neither name pointing at the new data",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
