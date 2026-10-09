"""Exercise 4 — inside an activated uv-style venv, `pip` still resolves to the global one.

    Deliberately install a package globally (without activating a venv), notice
    where it goes, then uninstall it

Reading of the exercise: a test must not write into the machine's real Python,
so nothing is installed or uninstalled anywhere outside a temp dir. What can be
measured without that is the exercise's middle clause, "notice where it goes":
which `site-packages` an interpreter installs into, and which `pip` and `python`
a shell actually finds. A venv is created with the stdlib `venv` module and no
pip seeded -- what the lesson's recommended `uv venv` produces -- and a stand-in
"global" bin directory holding `python` and `pip` sits on PATH behind it. The
venv's own `activate` script is then sourced in bash and the shell is asked.

**ANSWER: a global install lands in the base interpreter's site-packages.**
The base interpreter named in the venv's `pyvenv.cfg` reports `purelib` outside
any venv and `sys.prefix == sys.base_prefix`; the venv's interpreter reports a
`purelib` inside `.venv`. Uninstalling is removing from that first directory,
which is exactly why it is a step this test refuses to take.

**FINDING: activation does not protect `pip` in a uv-style venv.** After
`source .venv/bin/activate`, the shell resolves `python` to `.venv/bin/python`
but `pip` to the global bin -- the venv has no pip, so PATH falls through. The
lesson's "GOOD" line, `source .venv/bin/activate` then `pip install torch`, is
therefore a global install for anyone who followed Option 1.

**FINDING: the lesson's own check would catch it, and the safe form fails
loudly.** The lesson says `which pip` "should show .venv/bin/pip"; here it
shows the global one. `python -m pip` binds the installer to the interpreter
instead, and in this venv it exits non-zero with "No module named pip" rather
than writing somewhere else.

**CONTROL: the stand-in is what PATH finds without the venv.** With the same
PATH and no activation, both `python` and `pip` resolve to the global bin.

Structure: `shell` sources activate (or not) and asks `command -v`; `paths`
asks an interpreter where it installs; `solve` builds both bins.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import tempfile
import venv

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "06-python-environments"
WHERE = ("import json, sys, sysconfig\n"
         "print(json.dumps([sysconfig.get_paths()['purelib'], sys.prefix == sys.base_prefix]))")


def shell(home, global_bin, activate):
    """`command -v python pip` in bash, after sourcing the venv's activate or not."""
    bash = shutil.which("bash")
    if bash is None:
        raise practice.Skip("bash is not on PATH")
    body = (f'source "{home}/bin/activate"; ' if activate else "") + "command -v python pip"
    done = subprocess.run([bash, "-c", body], capture_output=True, text=True, timeout=30,
                          env={"PATH": f"{global_bin}:/usr/bin:/bin"})
    return done.stdout.split()


def paths(python):
    done = subprocess.run([str(python), "-c", WHERE], capture_output=True, text=True,
                          env={}, timeout=60, check=True)
    return json.loads(done.stdout)


def solve():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp).resolve()
        home, global_bin = tmp / "project" / ".venv", tmp / "global" / "bin"
        venv.create(home, with_pip=False, symlinks=True)
        global_bin.mkdir(parents=True)
        for tool in ("python", "pip"):
            (global_bin / tool).write_text("#!/bin/sh\necho global\n")
            (global_bin / tool).chmod(0o755)
        config = dict(line.split(" = ", 1) for line in
                      (home / "pyvenv.cfg").read_text().splitlines() if " = " in line)
        python = home / "bin" / "python"
        safe = subprocess.run([str(python), "-m", "pip", "--version"], capture_output=True,
                              text=True, env={}, timeout=60)
        return {
            "home": str(home), "global": str(global_bin),
            "bin": sorted(p.name for p in (home / "bin").iterdir()),
            "active": shell(home, global_bin, True), "inactive": shell(home, global_bin, False),
            "venv_paths": paths(python), "base_paths": paths(config["executable"]),
            "safe": (safe.returncode, safe.stderr.strip().splitlines()[-1].split(": ")[-1]),
            "doc": "should show .venv/bin/pip" in parity.doc_text(PHASE, LESSON, "en"),
        }


def verify(result):
    home, glob = result["home"], result["global"]
    (venv_site, venv_base), (base_site, base_base) = result["venv_paths"], result["base_paths"]
    active, inactive = result["active"], result["inactive"]
    return [
        practice.Check(
            "ANSWER: a global install lands in the base interpreter's site-packages",
            base_base and not venv_base and venv_site.startswith(home)
            and not base_site.startswith(home),
            f"the base interpreter from pyvenv.cfg installs into {base_site} "
            f"(sys.prefix == base_prefix: {base_base}); the venv's into "
            f"{venv_site.replace(home, '.venv')}",
        ),
        practice.Check(
            "FINDING: after activate, python is the venv's and pip is the global one",
            active[0] == f"{home}/bin/python" and active[1] == f"{glob}/pip",
            f"venv bin holds {result['bin']} -- no pip -- so the activated shell resolves "
            "python to .venv/bin/python and pip to the stand-in global bin",
        ),
        practice.Check(
            "FINDING: the lesson's which-pip check fails, and python -m pip fails loudly",
            result["doc"] and not [n for n in result["bin"] if n.startswith("pip")]
            and result["safe"] == (1, "No module named pip"),
            f"the lesson says which pip 'should show .venv/bin/pip'; here no pip exists in the "
            f"venv, and `python -m pip` exits {result['safe'][0]} with '{result['safe'][1]}' "
            "instead of installing elsewhere",
        ),
        practice.Check(
            "CONTROL: without activation both names resolve to the stand-in global bin",
            inactive == [f"{glob}/python", f"{glob}/pip"],
            "same PATH, activate not sourced: python and pip both come from the global bin",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
