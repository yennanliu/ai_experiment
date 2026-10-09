"""Exercise 2 — isolation holds until one PYTHONPATH, and the script's venv check passes a non-venv.

    Create a second virtual environment, install a different version of numpy in it,
    and confirm the two environments are isolated

Reading of the exercise: two environments are created with the stdlib `venv`
module (no pip seeded, as `uv venv` does), each in a temp dir and each named
`.venv` like the lesson's. A real numpy download is off the table, so
"installing a different version" plants a stand-in `numpy` package declaring
`__version__` 1.26.4 in one and 2.1.0 in the other, in the site-packages each
interpreter reports for itself. Every probe runs that environment's own
`python` in a subprocess with a controlled, empty-by-default environment.

**ANSWER: the two environments are isolated.** Environment A imports 1.26.4
from A's site-packages and B imports 2.1.0 from B's, side by side. Neither sees
the numpy of the project venv that created them: a fresh venv is built from the
*base* interpreter, not from whichever venv was active.

**FINDING: isolation is a default, not a wall.** Run A's interpreter with one
environment variable, `PYTHONPATH` pointing at B's site-packages, and A imports
**2.1.0**. Nothing about A changed; `sys.path` order did. A stray `PYTHONPATH`
in a shell rc file defeats every venv on the machine the same way.

**FINDING: activation is only PATH.** A's interpreter, called by absolute path
with no `activate`, no `VIRTUAL_ENV` and an empty environment, reports
`sys.prefix` = A and `sys.prefix != sys.base_prefix`. "Forgetting to activate"
is only a problem for a bare `python` that PATH resolves elsewhere.

**FINDING: the lesson's own venv check cannot tell these apart, and passes a
non-venv.** `env_setup.sh` accepts the interpreter when `which python` merely
contains `.venv`. Run in bash on four paths, it accepts both A and B -- and
`/home/u/.venvs-old/bin/python`, which is no venv at all -- and rejects a conda
env. The test that means "inside a venv" is `sys.prefix != sys.base_prefix`.

**CONTROL:** with nothing planted, A cannot import numpy at all.

Structure: `make` builds an env and plants a version; `probe` asks an env's own
interpreter what it sees; `script_accepts` runs the script's check in bash.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import venv

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "06-python-environments"
ASK = ("import json, sys\ntry:\n    import numpy\n    v, f = numpy.__version__, numpy.__file__\n"
       "except ImportError:\n    v = f = None\n"
       "print(json.dumps([v, f, sys.prefix, sys.prefix != sys.base_prefix]))")
FAKE_PATHS = ("A", "B", "/home/u/.venvs-old/bin/python", "/opt/conda/envs/ml/bin/python")


def make(root, version):
    """A venv named .venv under `root`; plants numpy `version` unless it is None."""
    home = root / ".venv"
    venv.create(home, with_pip=False, symlinks=os.name != "nt")
    python = home / ("Scripts" if os.name == "nt" else "bin") / "python"
    site = run(python, "import sysconfig; print(sysconfig.get_paths()['purelib'])").strip()
    if version:
        (pathlib.Path(site) / "numpy").mkdir(parents=True)
        (pathlib.Path(site) / "numpy" / "__init__.py").write_text(f"__version__ = '{version}'\n")
    return python, site


def run(python, code, **env):
    done = subprocess.run([str(python), "-c", code], capture_output=True, text=True,
                          env=env, timeout=60, check=True)
    return done.stdout


def probe(python, **env):
    return json.loads(run(python, ASK, **env))


def script_accepts(paths):
    """The script's own `[[ $VENV_PYTHON != *$VENV_DIR* ]]` block, per fake `which`."""
    text = (parity.lesson_dir(PHASE, LESSON) / "code" / "env_setup.sh").read_text()
    check = text[text.index('VENV_PYTHON="$(which python)"'):text.index('pass "Python path')]
    head = 'VENV_DIR=".venv"\nfail() { echo "$1"; }\nwhich() { echo "$FAKE"; }\n'
    bash = shutil.which("bash")
    if bash is None:
        raise practice.Skip("bash is not on PATH")
    return {p: subprocess.run([bash, "-c", head + check], env={"FAKE": p}, timeout=30,
                              capture_output=True).returncode == 0 for p in paths}


def solve():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (a, a_site), (b, b_site) = make(tmp / "a", "1.26.4"), make(tmp / "b", "2.1.0")
        bare, _ = make(tmp / "c", None)
        found = {"A": probe(a), "B": probe(b), "A+PYTHONPATH": probe(a, PYTHONPATH=b_site),
                 "bare": probe(bare)}
        sites = {"A": a_site, "B": b_site, "A_python": str(a), "B_python": str(b)}
        paths = [str(a), str(b)] + list(FAKE_PATHS[2:])
        accepted = dict(zip(FAKE_PATHS, script_accepts(paths).values()))
    return {"found": found, "sites": sites, "accepted": accepted}


def verify(result):
    found, sites, accepted = result["found"], result["sites"], result["accepted"]
    a, b, leak = found["A"], found["B"], found["A+PYTHONPATH"]
    return [
        practice.Check(
            "ANSWER: A sees 1.26.4 and B sees 2.1.0, each from its own site-packages",
            a[0] == "1.26.4" and b[0] == "2.1.0"
            and a[1].startswith(sites["A"]) and b[1].startswith(sites["B"]),
            f"A imports numpy {a[0]} and B imports {b[0]}, each from the site-packages its own "
            "interpreter reports; neither sees the project venv that created them",
        ),
        practice.Check(
            "FINDING: one PYTHONPATH makes A import B's numpy",
            leak[0] == "2.1.0" and leak[1].startswith(sites["B"]),
            f"A's interpreter with PYTHONPATH set to B's site-packages imports {leak[0]} from "
            "B. Nothing in A changed; sys.path order did",
        ),
        practice.Check(
            "FINDING: activation is only PATH",
            a[3] and b[3] and pathlib.Path(sites["A_python"]).parent.parent.resolve()
            == pathlib.Path(a[2]).resolve(),
            f"called by absolute path with an empty environment, A reports sys.prefix inside its "
            f"own .venv and sys.prefix != sys.base_prefix is {a[3]}, with no activate script run",
        ),
        practice.Check(
            "FINDING: the script's venv check cannot tell A from B and passes a non-venv",
            accepted == {"A": True, "B": True, FAKE_PATHS[2]: True, FAKE_PATHS[3]: False},
            f"env_setup.sh's substring test, run in bash, accepts {accepted}: anything whose path "
            "contains '.venv' passes, a conda env fails",
        ),
        practice.Check(
            "CONTROL: an env with nothing planted cannot import numpy",
            found["bare"][0] is None and found["bare"][3],
            "a third venv, made the same way with no stand-in, raises ImportError for numpy",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
