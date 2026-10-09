"""Exercise 3 — bc drops the leading zero, so a number regex finds 1 loss of 100.

    Create a fake training log with `for i in $(seq 1 100); do echo "epoch
    $i loss: $(echo "scale=4; 1/$i" | bc)"; sleep 0.1; done >
    fake_train.log` and then use `grep`, `tail`, and `awk` to extract just
    the loss values.

Reading of the exercise: the log is the exercise's own loop. `bc` is not
guaranteed on a CI runner, so the log is written by `bc_line`, a model of what
`bc` prints for `scale=4; 1/i` (truncated to 4 places, no leading zero), and,
where `bc` exists, the loop itself is run without its `sleep` and must produce
the identical file. The extraction runs in bash with the real `grep`, `tail`
and `awk` in a temp dir.

**ANSWER: `grep 'loss:' fake_train.log | awk '{print $4}'` extracts all 100
values; `| tail -n 5` before the `awk` keeps the last five, `.0104 .0103 .0102
.0101 .0100`.**

**FINDING: 99 of the 100 values have no leading zero.** `bc` prints `.5000`,
not `0.5000`; only epoch 1's `1.0000` has a digit before the point, so the
usual number pattern `grep -oE 'loss: [0-9]+\\.[0-9]+'` finds 1 line of 100.
`awk` and Python's `float` both read `.5000` correctly; regexes do not.

**FINDING: the values are truncated, not rounded.** `scale=4` cuts digits, so
38 of the 100 differ from `round(1/i, 4)` -- `1/6` is logged as `.1666`.

**FINDING: the lesson's own log helper finds nothing here.** `taillog` (like
the `watchloss` alias, by its text) reads `logs/*.log`; run beside
`fake_train.log` it prints nothing and returns at once -- and on a matching
file its `tail -f` would never return, so it cannot extract a finished log.

**CONTROL: where `bc` is installed, the exercise's loop writes the same file
as the model, byte for byte.**

Structure: `bc_line` models bc; `bash_run` runs the pipelines in a temp dir.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "10-terminal-and-shell"
LOOP = 'for i in $(seq 1 100); do echo "epoch $i loss: $(echo "scale=4; 1/$i" | bc)"; done'
PIPES = {
    "all": "grep 'loss:' fake_train.log | awk '{print $4}'",
    "last5": "grep 'loss:' fake_train.log | tail -n 5 | awk '{print $4}'",
    "regex": "grep -oE 'loss: [0-9]+\\.[0-9]+' fake_train.log",
}


def bc_line(i):
    """What `echo "scale=4; 1/i" | bc` prints: truncated, and no leading zero below 1."""
    whole, frac = divmod(10_000 // i, 10_000)
    return f"{whole or ''}.{frac:04d}"


def bash_run(script, cwd):
    bash = shutil.which("bash")
    if bash is None:
        raise practice.Skip("bash is not on PATH")
    done = subprocess.run([bash, "--noprofile", "--norc", "-c", script], capture_output=True,
                          text=True, cwd=cwd, env={"PATH": "/usr/bin:/bin"}, timeout=30)
    return done.stdout.split()


def solve():
    aliases = parity.lesson_dir(PHASE, LESSON) / "code" / "shell_aliases.sh"
    log = "".join(f"epoch {i} loss: {bc_line(i)}\n" for i in range(1, 101))
    with tempfile.TemporaryDirectory() as tmp:
        (pathlib.Path(tmp) / "fake_train.log").write_text(log)
        out = {name: bash_run(cmd, tmp) for name, cmd in PIPES.items()}
        helper = bash_run(f"source '{aliases}'; taillog loss", tmp)
        real = None
        if shutil.which("bc", path="/usr/bin:/bin"):
            real = " ".join(bash_run(LOOP, tmp)) == " ".join(log.split())
    values = out["all"]
    return {
        "values": values,
        "last5": out["last5"],
        "regex_hits": len(out["regex"]) // 2,
        "no_zero": sum(v.startswith(".") for v in values),
        "floats_ok": [float(v) for v in values] == [int(10_000 / i) / 10_000 for i in
                                                     range(1, 101)],
        "truncated": sum(float(v) != round(1 / i, 4) for i, v in enumerate(values, 1)),
        "helper": helper,
        "bc_matches": real,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: grep | awk gives all 100 losses; tail -n 5 keeps the last five",
            len(result["values"]) == 100
            and result["last5"] == [".0104", ".0103", ".0102", ".0101", ".0100"],
            f"{len(result['values'])} values, first {result['values'][:3]}; last five "
            f"{result['last5']}",
        ),
        practice.Check(
            "FINDING: 99 of 100 values lack a leading zero; a number regex finds 1",
            result["no_zero"] == 99 and result["regex_hits"] == 1 and result["floats_ok"],
            f"{result['no_zero']} values start with '.'; grep -oE 'loss: [0-9]+\\.[0-9]+' "
            f"matches {result['regex_hits']} line; float() reads all of them",
        ),
        practice.Check(
            "FINDING: 38 of 100 values are truncated away from the rounded loss",
            result["truncated"] == 38,
            f"{result['truncated']} differ from round(1/i, 4); 1/6 is logged as "
            f"{result['values'][5]}",
        ),
        practice.Check(
            "FINDING: the lesson's taillog reads logs/*.log and finds nothing here",
            result["helper"] == [],
            f"taillog loss beside fake_train.log printed {result['helper']} and returned",
        ),
        practice.Check(
            "CONTROL: the exercise's own bc loop writes the modelled file",
            result["bc_matches"] in (True, None),
            "bc absent; the model alone stands" if result["bc_matches"] is None
            else f"real bc loop identical to the model: {result['bc_matches']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
