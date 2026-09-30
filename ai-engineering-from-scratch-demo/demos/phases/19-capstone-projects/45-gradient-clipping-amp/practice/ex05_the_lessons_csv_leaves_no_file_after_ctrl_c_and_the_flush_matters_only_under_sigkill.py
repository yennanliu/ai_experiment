"""Exercise 5 — the lesson writes its CSV only after training, so a Ctrl-C on step 7 leaves no file; flushing each row keeps all 8.

    Wire the loop to write the canonical CSV (`step, lr, grad_l2_pre_clip, grad_l2_post_clip, loss, skipped, skip_reason, scaler_scale`) and confirm the file survives a Ctrl-C by flushing after every row.

Reading of the exercise: `train_streaming` runs the lesson's demo loop
(seed-7 toy model, `AmpTrainState` on CPU, Inf injected on step 5). It
writes the header and then each `StepLog.to_csv_row()` as the step
finishes, and flushes after every row. "Survives a Ctrl-C" is tested for
real, not simulated: this file re-runs itself as a child process that sends
itself SIGINT, the signal Ctrl-C delivers, right after step 7's row. The
parent then reads what is on disk. The same child is also run without the
flush, with SIGKILL (a death with no Python unwinding), and with the
lesson's own `write_step_log_csv` called after the loop.

**ANSWER: with a flush after every row, the file survives Ctrl-C with all 8
rows (steps 0-7).** The header is the exercise's 8 columns in order, and
row 5 reads `skipped=1, skip_reason=non_finite_grad`. It also survives
SIGKILL with the same 8 rows.

**FINDING: the lesson's CSV does not survive a Ctrl-C at all.**
`write_step_log_csv` takes the whole log after training. Interrupted on
step 7, the run leaves no file.

**FINDING: for Ctrl-C itself the flush is not what saves the rows.**
Without the flush, a SIGINT still leaves all 8 rows. The KeyboardInterrupt
unwinds the `with open(...)` block and closing the file writes the buffer.
Under SIGKILL there is no unwinding, and the unflushed file is 0 bytes;
the flushed one keeps 8 rows. A flush protects against the process dying,
not against the OS crashing: that would also need `os.fsync`, which is not
measured here.

Expected output: three PASS checks.
"""

from __future__ import annotations

import csv
import os
import pathlib
import signal
import subprocess
import sys
import tempfile

from harness import parity, practice

try:
    import torch  # noqa: F401  (the lesson's loop needs it)
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "45-gradient-clipping-amp"
COLUMNS = ["step", "lr", "grad_l2_pre_clip", "grad_l2_post_clip", "loss", "skipped", "skip_reason", "scaler_scale"]
KILL_AFTER = 7
REPO = pathlib.Path(practice.__file__).resolve().parent.parent


def train(ref, on_row, sig):
    model, inputs, targets = ref.build_toy_model()
    state = ref.AmpTrainState(model=model, lr=1e-2, max_norm=1.0, device_type="cpu")
    for i in range(20):
        on_row(state.step(inputs, targets, gradient_corruptor=ref.inject_inf_into_first_grad if i == 5 else None))
        if i == KILL_AFTER and sig:
            os.kill(os.getpid(), sig)
    return state


def train_streaming(ref, path, flush=True, sig=0):
    """The loop wired to the canonical CSV, one flushed row per step."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)

        def on_row(record):
            writer.writerow(record.to_csv_row())
            if flush:
                fh.flush()

        writer.writerow(COLUMNS)
        if flush:
            fh.flush()
        train(ref, on_row, sig)


def child(argv):
    mode, sig, path = argv[0], int(argv[1]), argv[2]
    ref = parity.load_reference(PHASE, LESSON, "main")
    if mode == "lesson":
        ref.write_step_log_csv(train(ref, lambda _row: None, sig).log, pathlib.Path(path))
    else:
        train_streaming(ref, path, flush=mode == "flush", sig=sig)
    return 0


def interrupted(mode, sig):
    """Run the loop in a child that signals itself after step 7; report what is on disk."""
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "steps.csv"
        env = {**os.environ, "PYTHONPATH": str(REPO)}
        proc = subprocess.run([sys.executable, __file__, "--child", mode, str(int(sig)), str(path)],
                              env=env, capture_output=True, timeout=120)
        if not path.exists():
            return {"code": proc.returncode, "exists": False, "rows": 0}
        rows = list(csv.reader(path.open(encoding="utf-8")))
        return {"code": proc.returncode, "exists": True, "bytes": path.stat().st_size,
                "header": rows[0] if rows else [], "rows": max(len(rows) - 1, 0),
                "row5": rows[6][5:7] if len(rows) > 6 else []}


def solve():
    cases = {"flush_int": ("flush", signal.SIGINT), "flush_kill": ("flush", signal.SIGKILL),
             "lesson_int": ("lesson", signal.SIGINT), "noflush_int": ("noflush", signal.SIGINT),
             "noflush_kill": ("noflush", signal.SIGKILL)}
    return {name: interrupted(*args) for name, args in cases.items()}


def verify(result):
    r = result
    ok = r["flush_int"]
    return [
        practice.Check(
            "ANSWER: flushed per row, the canonical CSV keeps all 8 rows through Ctrl-C (and SIGKILL)",
            (ok["code"], ok["header"], ok["rows"], ok["row5"]) == (-signal.SIGINT, COLUMNS, 8, ["1", "non_finite_grad"])
            and (r["flush_kill"]["code"], r["flush_kill"]["rows"]) == (-signal.SIGKILL, 8),
            f"SIGINT after step 7: exit {ok['code']}, {ok['rows']} rows, header {ok['header']}, row 5 {ok['row5']}; "
            f"SIGKILL: {r['flush_kill']['rows']} rows",
        ),
        practice.Check(
            "FINDING: the lesson's write_step_log_csv leaves no file when training is interrupted",
            (r["lesson_int"]["code"], r["lesson_int"]["exists"]) == (-signal.SIGINT, False),
            f"exit {r['lesson_int']['code']}, file exists: {r['lesson_int']['exists']}",
        ),
        practice.Check(
            "FINDING: Ctrl-C alone is survived by unwinding; only a hard kill shows what the flush buys",
            r["noflush_int"]["rows"] == 8 and (r["noflush_kill"]["exists"], r["noflush_kill"]["bytes"]) == (True, 0),
            f"no flush: SIGINT keeps {r['noflush_int']['rows']} rows, SIGKILL leaves {r['noflush_kill']['bytes']} bytes",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(child(sys.argv[2:]) if sys.argv[1:2] == ["--child"] else practice.selfcheck(globals()))
