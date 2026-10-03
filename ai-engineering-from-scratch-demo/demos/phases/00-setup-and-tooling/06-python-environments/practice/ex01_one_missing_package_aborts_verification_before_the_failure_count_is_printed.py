"""Exercise 1 — one missing package aborts verification before the failure count is printed.

    Run `env_setup.sh` and verify all checks pass

Reading of the exercise: running the script for real downloads packages and
writes a `.venv` into the reference checkout, which no test may do. So the
script is read as text from the reference `phases/` tree and its own blocks are
executed in `bash`, with only the outside world faked: `python` and `python3`
become shell functions that report a chosen version or refuse a chosen import.
Every `if`, every `pass`/`fail` and every counter is the lesson's own code.

**ANSWER: with every import present, the verification tail ends "All checks
passed", exit 0.** It checks five packages (numpy, matplotlib, scikit-learn,
pandas, jupyter), a NumPy matmul, and warns, without failing, when torch is
absent.

**FINDING: one missing package stops the script at that package.** The script
runs under `set -euo pipefail`, and `verify_package` signals failure with
`return 1` from a bare call, so `set -e` exits right there. With matplotlib
missing the run ends after **2 of 5** checks: scikit-learn, pandas and jupyter
are never tested, `FAILURES` is never read, and the
"N package(s) failed verification" summary written for this case is
unreachable. Exit status is 1, so CI would notice; a person reading the output
sees one `[FAIL]` and no summary.

**FINDING: the version gate rejects Python 4.0.** It tests
`major -ge 3 && minor -ge 11` separately, so 3.11 and 3.13 pass, 3.10 fails --
and so does 4.0, whose minor 0 is below 11. The gate exits 1 with "Python 3.11+
not found" for a newer interpreter.

**CONTROL: the script's root really is the course root.** `REPO_ROOT` is
`dirname "$0"/../../../..`; from `phases/00-setup-and-tooling/06-python-environments/code`
that resolves to the directory holding `phases/`, so the `.venv` it creates
lands where the lesson says.

Structure: `block` cuts a span of the script; `run` executes it in bash with
the fakes; `solve` runs the tail, the gate and the root arithmetic.
"""

from __future__ import annotations

import re
import shutil
import subprocess

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "06-python-environments"
FAKE = """python() { for m in $MISSING torch; do case "$*" in *"import $m"*) return 1;; esac; done
case "$*" in *"import sys; print"*) echo "$FAKE_VERSION";; *) echo "ran: ${2%%;*}";; esac; }
python3() { python "$@"; }
REPO_ROOT=/course
"""
VERSIONS = ("3.10", "3.11", "3.13", "4.0")


def script():
    return (parity.lesson_dir(PHASE, LESSON) / "code" / "env_setup.sh").read_text()


def block(text, start, end=None):
    """The script from `start` up to (not including) `end`."""
    head = text.index(start)
    return text[head:text.index(end, head) if end else None]


def run(text, body, **env):
    """The script's settings and helpers, the fakes, then `body`, in bash."""
    bash = shutil.which("bash")
    if bash is None:
        raise practice.Skip("bash is not on PATH")
    prelude = block(text, "set -euo", "REPO_ROOT=")
    done = subprocess.run([bash, "-c", prelude + FAKE + body], capture_output=True,
                          text=True, env={"PATH": "/usr/bin:/bin", **env}, timeout=30)
    return done.returncode, re.sub(r"\x1b\[[0-9;]*m", "", done.stdout)


def solve():
    text = script()
    tail = block(text, "FAILURES=0")
    gate = block(text, 'PYTHON_CMD=""', 'echo ""\necho "--- Creating')
    ok = run(text, tail, MISSING="", FAKE_VERSION="3.12")
    missing = run(text, tail, MISSING="matplotlib", FAKE_VERSION="3.12")
    here = parity.lesson_dir(PHASE, LESSON) / "code"
    return {
        "ok": ok, "missing": missing,
        "packages": re.findall(r'^verify_package "([^"]+)"', text, re.M),
        "gate": {v: run(text, gate, MISSING="", FAKE_VERSION=v) for v in VERSIONS},
        "root": (here / "../../../..").resolve(),
        "phases": (here / "../../..").resolve(),
    }


def verify(result):
    (ok_code, ok_out), (miss_code, miss_out) = result["ok"], result["missing"]
    checked = re.findall(r"ran: import (\w+)", miss_out) + re.findall(r"\[FAIL\] (\S+)", miss_out)
    gate = {v: code for v, (code, _) in result["gate"].items()}
    refused = [v for v, (_, out) in result["gate"].items() if "3.11+ not found" in out]
    return [
        practice.Check(
            "ANSWER: with every import present the tail ends 'All checks passed'",
            ok_code == 0 and all(m in ok_out for m in ("All checks passed", "[WARN] PyTorch")),
            f"exit {ok_code}; packages verified {result['packages']}, then the matmul, then a "
            "torch WARN that does not fail the run",
        ),
        practice.Check(
            "FINDING: one missing package stops the script at that package",
            miss_code == 1 and len(checked) == 2 and "failed verification" not in miss_out,
            f"with matplotlib missing: exit {miss_code}, packages reached {checked} of "
            f"{len(result['packages'])}, and the 'package(s) failed verification' summary "
            "never prints -- set -e exits on verify_package's bare `return 1`",
        ),
        practice.Check(
            "FINDING: the version gate rejects Python 4.0",
            gate == {"3.10": 1, "3.11": 0, "3.13": 0, "4.0": 1} and refused == ["3.10", "4.0"],
            f"exit status by interpreter version {gate}, and 'Python 3.11+ not found' for "
            f"{refused}: major and minor are tested separately, so 4.0 fails 'minor -ge 11'",
        ),
        practice.Check(
            "CONTROL: REPO_ROOT is the directory that holds phases/",
            (result["root"] / "phases") == result["phases"],
            f"code/../../../.. resolves to {result['root'].name}/, the parent of "
            f"{result['phases'].name}/",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
