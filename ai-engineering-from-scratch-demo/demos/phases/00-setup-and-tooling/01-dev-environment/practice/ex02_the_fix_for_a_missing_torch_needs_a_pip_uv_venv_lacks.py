"""Exercise 2 — the fix for a missing PyTorch needs a pip that `uv venv` does not ship.

    Create a Python virtual environment for this course and install PyTorch

Reading of the exercise: create the environment for real, in a temp directory,
the way the lesson's Step 2 does -- `uv venv`, which seeds **no pip** -- and
then let the lesson's own `verify.py` judge it from inside. Installing PyTorch
itself needs the network and ~1 GB, so that step is replaced by running the
exact command the script prints as its fix, which is the step that matters
here: it is what a reader does next.

**ANSWER: the environment is real and isolated.** The venv's interpreter has
`sys.prefix != sys.base_prefix`, and `verify.py --show-later` run under it
reports PyTorch as **LATER** "not importable by" the venv's own python -- the
host's packages do not leak in.

**FINDING: the printed fix fails in the environment Step 2 builds.** The fix
line for PyTorch is `python3 -m pip install torch`, and **4** of the script's
fix lines say `python3 -m pip install`. In a pip-less venv (what `uv venv`
makes, and what Step 2 tells the reader to activate) that command exits
**1** with "No module named pip". The lesson installs packages with
`uv pip install`; its verifier tells the reader to use a tool that is not
there.

**FINDING: no route makes PyTorch a requirement.** PyTorch is required by
**0** of the **7** routes, so the exercise's second half can never turn
`verify.py` red: a reader who skips it still gets "Ready to start", and
`--show-later` is the only way to see it at all.

**CONTROL: pip's absence is the cause, not the command.** The same
`-m pip --version` in a venv created *with* pip exits **0**.

Structure: `make_venv` builds an environment with or without pip;
`run_in` runs one command with that environment's interpreter.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import venv

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "01-dev-environment"


def make_venv(root, name, with_pip):
    """A real virtual environment; returns its python."""
    home = pathlib.Path(root) / name
    venv.EnvBuilder(with_pip=with_pip, symlinks=sys.platform != "win32").create(home)
    bindir = "Scripts" if sys.platform == "win32" else "bin"
    return home / bindir / ("python.exe" if sys.platform == "win32" else "python")


def run_in(python, *args):
    """Run `python args...` with a clean environment; (exit code, combined output)."""
    done = subprocess.run([str(python), "-I", *args], capture_output=True, text=True,
                          timeout=120, env={"PATH": str(python.parent)})
    return done.returncode, done.stdout + done.stderr


def solve():
    ref = parity.load_reference(PHASE, LESSON, "verify")
    script = parity.lesson_dir(PHASE, LESSON) / "code" / "verify.py"
    fix = ref.PROBES["torch"].fix
    with tempfile.TemporaryDirectory() as root:
        bare = make_venv(root, "course", with_pip=False)
        _, isolated = run_in(bare, "-c", "import sys; print(sys.prefix != sys.base_prefix)")
        _, report = run_in(bare, str(script), "--route", "ml-foundations", "--show-later")
        pip_code, pip_out = run_in(bare, "-m", "pip", "--version")
        seeded = make_venv(root, "seeded", with_pip=True)
        seeded_code, _ = run_in(seeded, "-m", "pip", "--version")
    torch_block = report.partition("[LATER] PyTorch")[2].split("[")[0]
    return {
        "isolated": isolated.strip() == "True",
        "torch_later": "not importable by" in torch_block,
        "fix": fix,
        "pip_fixes": sum("python3 -m pip install" in p.fix for p in ref.PROBES.values()),
        "pip_code": pip_code, "no_pip": "No module named pip" in pip_out,
        "torch_required": sum("torch" in r.required for r in ref.ROUTES.values()),
        "routes": len(ref.ROUTES), "seeded_code": seeded_code,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a real, isolated environment, judged by the lesson's own script",
            result["isolated"] and result["torch_later"],
            "inside the new venv sys.prefix != sys.base_prefix, and verify.py --show-later run "
            "by that venv's python reports PyTorch LATER, 'not importable by' the venv's "
            "interpreter: host packages do not leak in",
        ),
        practice.Check(
            "FINDING: the printed fix needs pip, and Step 2's venv has none",
            result["pip_code"] != 0 and result["no_pip"] and result["pip_fixes"] == 4,
            f"the fix reads {result['fix']!r} and {result['pip_fixes']} fix lines say "
            f"python3 -m pip install; in a pip-less venv, which is what `uv venv` builds, "
            f"-m pip exits {result['pip_code']} with 'No module named pip'",
        ),
        practice.Check(
            "FINDING: PyTorch is required by no route",
            result["torch_required"] == 0 and result["routes"] == 7,
            f"torch is in route.required for {result['torch_required']} of {result['routes']} "
            "routes, so skipping the exercise's second half never turns verify.py red",
        ),
        practice.Check(
            "CONTROL: with pip seeded, the same command succeeds",
            result["seeded_code"] == 0,
            f"-m pip --version in a venv created with pip exits {result['seeded_code']}, so "
            "pip's absence, not the command, is what fails",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
