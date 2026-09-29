"""Exercise 3 — a 20 s wallclock trigger cuts the worst-case lost work from 124 s to 32 s, and scrambling the RNG on resume changes nothing.

    Add a `--ckpt-every-seconds` flag that triggers a save on a wallclock interval, not just step count.

Reading of the exercise: the lesson's CLI is extended, not edited -- a
parser with `--ckpt-every-steps` and `--ckpt-every-seconds` drives the
lesson's `train_until` one step at a time and calls its `save_checkpoint`
whenever either interval has elapsed, the doc's "whichever is shorter" rule.
The clock is injected, so the run is deterministic: 24 steps of 1 s each,
except steps 11-14, which stall for 30 s each (an eval or a slow disk).
Every checkpoint the wallclock trigger writes -- mid-epoch, at whatever
batch the clock fell on -- is then resumed and compared with the
uninterrupted run.

**ANSWER: `parser()` below adds the flag, and `due()` saves when either
interval has elapsed.** The lesson's own `parse_args` rejects
`--ckpt-every-seconds 20` with exit code 2. On the stalled run:

| trigger | saves at step | worst-case work lost to a kill |
|---|---|---:|
| every 8 steps | 8, 16, 24 | 124 s |
| every 20 s | 11, 12, 13, 14 | 40 s |
| both (whichever first) | 8, 11, 12, 13, 14, 22 | 32 s |

The step trigger cannot see the stall. The clock trigger is checked only at
step boundaries, so its bound is the interval plus one step, and during the
stall it saves on every 30 s step. Every checkpoint the combined trigger
wrote, including the mid-epoch ones at steps 11-14 and 22, resumes to the
uninterrupted losses with a max diff of 0.0. Stepping one step at a time
gives the same 24 losses as the lesson's single `train_until` call.

**FINDING: the lesson's resume test cannot see the RNG bucket.** The doc
says "without the RNG state the resumed loss curve is a different curve".
With `restore_rng_state` replaced by `torch.manual_seed(999)`, the lesson's
`run_resume_demo` still reports a max loss diff of exactly 0.0 at all 23
interrupt points (1-23 of 24). The data comes from a per-epoch generator
reseeded to `12345 + epoch`, and the model has no dropout, so nothing
downstream draws from the global RNG the checkpoint saves.

Expected output: two PASS checks.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
from pathlib import Path
from unittest import mock

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "47-checkpoint-save-resume"
DIMS = {"batches_per_epoch": 5, "batch_size": 4, "in_dim": 16, "out_dim": 4}
STEP_SECONDS = [30.0 if 10 <= i <= 13 else 1.0 for i in range(24)]
ARGV = ["--ckpt-every-steps", "8", "--ckpt-every-seconds", "20"]


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-every-steps", type=int, default=0, help="0 disables the step trigger")
    p.add_argument("--ckpt-every-seconds", type=float, default=0.0, help="0 disables the wallclock trigger")
    return p


def fresh(ref):
    ref.seed_everything(11)
    model = ref.make_model(16, 24, 4)
    return (model, *ref.make_optimizer_and_scheduler(model, lr=0.01, total_steps=24))


def due(args, steps, seconds):
    """Whichever interval elapses first; 0 turns a trigger off."""
    return bool(0 < args.ckpt_every_steps <= steps or 0 < args.ckpt_every_seconds <= seconds)


def train_with_saves(ref, args, folder):
    """Returns the steps saved at, the worst-case seconds of work a kill would lose, and the losses."""
    (model, opt, sched), state = fresh(ref), ref.TrainState(0, 0, 0)
    clock, last, saved, worst = 0.0, (0, 0.0), [], 0.0
    while state.step < 24:
        ref.train_until(model, opt, sched, torch.nn.CrossEntropyLoss(), state, stop_step=state.step + 1, **DIMS)
        clock += STEP_SECONDS[state.step - 1]
        worst = max(worst, clock - last[1])
        if due(args, state.step - last[0], clock - last[1]):
            ref.save_checkpoint(model, opt, sched, state, folder / f"ckpt-{state.step:04d}.pt")
            last, saved = (state.step, clock), [*saved, state.step]
    return saved, worst, state.losses


def resume_diff(ref, path, full):
    model, opt, sched = fresh(ref)
    start = (state := ref.load_checkpoint(path, model, opt, sched)).step
    ref.train_until(model, opt, sched, torch.nn.CrossEntropyLoss(), state, stop_step=24, **DIMS)
    return max(abs(a - b) for a, b in zip(full[start:], state.losses[start:], strict=True))


def lesson_cli_rejects(ref):
    try:
        with mock.patch.object(sys, "argv", ["main.py", "--ckpt-every-seconds", "20"]), \
                contextlib.redirect_stderr(io.StringIO()):
            ref.parse_args()
    except SystemExit as exc:
        return exc.code


def scrambled_rng_diffs(ref):
    """The lesson's own resume demo at every interrupt point, with RNG restore replaced by a wrong seed."""
    with mock.patch.object(ref, "restore_rng_state", lambda _state: torch.manual_seed(999)), \
            tempfile.TemporaryDirectory(prefix="ex03-rng-") as tmp:
        return [ref.run_resume_demo(ckpt_dir=Path(tmp) / str(k), total_steps=24, interrupt_at=k)
                ["max_loss_diff_after_resume"] for k in range(1, 24)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    policies = {"steps": parser().parse_args(ARGV[:2]), "seconds": parser().parse_args(ARGV[2:]),
                "both": parser().parse_args(ARGV)}
    out = {"cli_exit": lesson_cli_rejects(ref), "parsed": vars(policies["both"]), "rng": scrambled_rng_diffs(ref)}
    with tempfile.TemporaryDirectory(prefix="ex03-") as tmp:
        for name, pol in policies.items():
            (Path(tmp) / name).mkdir()
            saved, worst, losses = train_with_saves(ref, pol, Path(tmp) / name)
            out[name] = {"saved": saved, "worst": worst}
        single = ref.run_resume_demo(ckpt_dir=Path(tmp) / "demo", total_steps=24, interrupt_at=10)["full_losses"]
        out["same_losses"] = losses == single
        out["resume"] = {s: resume_diff(ref, Path(tmp) / "both" / f"ckpt-{s:04d}.pt", losses) for s in out["both"]["saved"]}
    return out


def verify(result):
    r = result
    table = {k: (r[k]["saved"], r[k]["worst"]) for k in ("steps", "seconds", "both")}
    return [
        practice.Check(
            "ANSWER: --ckpt-every-seconds 20 plus every 8 steps cuts the worst-case loss from 124 s to 32 s",
            r["cli_exit"] == 2 and r["parsed"] == {"ckpt_every_steps": 8, "ckpt_every_seconds": 20.0}
            and table == {"steps": ([8, 16, 24], 124.0), "seconds": ([11, 12, 13, 14], 40.0),
                          "both": ([8, 11, 12, 13, 14, 22], 32.0)}
            and r["same_losses"] and set(r["resume"].values()) == {0.0},
            f"lesson CLI exit {r['cli_exit']}; (saved at, worst s) {table}; resume diffs {r['resume']}",
        ),
        practice.Check(
            "FINDING: with the RNG restore replaced by a wrong seed the lesson's resume diff is still 0.0",
            len(r["rng"]) == 23 and set(r["rng"]) == {0.0},
            f"max loss diff at interrupt points 1-23: {sorted(set(r['rng']))}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
