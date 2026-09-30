"""Exercise 2 — rotate-before-save needs keep_last=4, and five crashed saves rotate away all 5 good checkpoints.

    Add a `last_5_steps` rotation: keep the 5 most recent checkpoints, delete the oldest before saving a new one.

Reading of the exercise: `save_last_5` calls the lesson's own
`rotate_checkpoints` on the parent directory and then the lesson's
`save_sharded` into a fresh `step_NNNN` directory, over the lesson's 4 demo
ranks, for 8 steps. "Keep the 5 most recent" is read as: at most 5
checkpoint directories exist once a save finishes. A crash mid-save is
simulated by making the lesson's `_serialize_state` raise on the third rank,
which leaves exactly what a killed process leaves: two `.tmp` shards and no
manifest (the save has no cleanup).

**ANSWER: delete-before-save means `rotate_checkpoints(keep_last=4)`, not
5.** With keep_last=4 before each save, 8 saves leave step_0003..step_0007,
never more than 5 directories. Calling it with keep_last=5 before saving, as
the name suggests, leaves 6 after every save from the sixth on. (The lesson's
own `main` rotates once, after all 8 saves.)

**FINDING: the rotation counts crashed saves as checkpoints.** It keeps any
`step_*` directory, sorted by mtime, whether or not it has a manifest. Start
from 5 good checkpoints and let the next 5 saves crash (a node that keeps
failing): each rotation deletes one more good checkpoint to make room for a
broken one, and afterwards 0 of the 5 remaining directories load (each raises
"manifest missing"). A rotation that first deletes directories without a
`manifest.json` and then counts only complete ones still has 4 loadable
checkpoints, steps 1-4, after the same 5 crashes (the fifth directory is the
latest crashed save, removed by the next rotation).

Expected output: two PASS checks.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from harness import parity, practice

try:
    import torch  # noqa: F401 - the lesson's save path is torch.save
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "80-checkpoint-sharded-resume"


def complete_only_rotate(ref, parent, keep_last):
    """Drop directories with no manifest, then keep the newest complete ones by step."""
    dirs = sorted(Path(parent).glob("step_*"))
    for d in dirs:
        if not (d / ref.MANIFEST_NAME).exists():
            shutil.rmtree(d)
    good = sorted(Path(parent).glob("step_*"))
    for d in good[: max(len(good) - keep_last, 0)]:
        shutil.rmtree(d)


def save_last_5(ref, parent, states, step, rotate=None, keep_before=4):
    (rotate or ref.rotate_checkpoints)(parent, keep_last=keep_before)
    ref.save_sharded(states, f"{parent}/step_{step:04d}", step)
    return len(list(Path(parent).glob("step_*")))


def crash_on_third_rank(ref):
    original, calls = ref._serialize_state, [0]

    def crashing(state):
        calls[0] += 1
        if calls[0] == 3:
            raise OSError("node lost mid-save")
        return original(state)

    return original, crashing


def loadable(ref, parent):
    ok = []
    for d in sorted(Path(parent).glob("step_*")):
        try:
            ok.append(ref.load_sharded(str(d), 4)[0].step)
        except ref.CheckpointError:
            pass
    return ok, len(list(Path(parent).glob("step_*")))


def crash_run(ref, states, rotate):
    with tempfile.TemporaryDirectory(prefix="ex02-crash-") as tmp:
        for step in range(5):
            save_last_5(ref, tmp, states, step, rotate)
        for step in range(5, 10):
            original, ref._serialize_state = crash_on_third_rank(ref)
            try:
                save_last_5(ref, tmp, states, step, rotate)
            except OSError:
                pass
            finally:
                ref._serialize_state = original
        return loadable(ref, tmp)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    states = [ref.make_demo_state(r, 4) for r in range(4)]
    counts = {}
    for keep in (4, 5):
        with tempfile.TemporaryDirectory(prefix="ex02-") as tmp:
            counts[keep] = [save_last_5(ref, tmp, states, s, keep_before=keep) for s in range(8)]
            if keep == 4:
                kept = sorted(p.name for p in Path(tmp).iterdir())
    return {"counts": counts, "kept": kept, "lesson": crash_run(ref, states, None),
            "complete_only": crash_run(ref, states, lambda p, keep_last: complete_only_rotate(ref, p, keep_last))}


def verify(result):
    c = result["counts"]
    return [
        practice.Check(
            "ANSWER: delete-before-save needs rotate_checkpoints(keep_last=4), not 5",
            c[4] == [1, 2, 3, 4, 5, 5, 5, 5] and c[5] == [1, 2, 3, 4, 5, 6, 6, 6]
            and result["kept"] == [f"step_{s:04d}" for s in range(3, 8)],
            f"dirs after each save: keep_last=4 {c[4]}, keep_last=5 {c[5]}; kept {result['kept']}",
        ),
        practice.Check(
            "FINDING: the rotation counts crashed saves, and 5 crashes rotate away all 5 good checkpoints",
            result["lesson"] == ([], 5) and result["complete_only"] == ([1, 2, 3, 4], 5),
            f"after 5 crashed saves: lesson rotation loadable {result['lesson'][0]} of {result['lesson'][1]} dirs; "
            f"complete-only rotation loadable steps {result['complete_only'][0]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
