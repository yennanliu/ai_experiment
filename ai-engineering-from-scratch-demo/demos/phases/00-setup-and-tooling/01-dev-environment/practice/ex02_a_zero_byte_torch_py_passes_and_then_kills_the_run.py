"""Exercise 2 — a zero-byte torch.py passes, and then kills the run.

    Create a Python virtual environment for this course and install PyTorch

Reading of the exercise: do it, then ask the shipped preflight whether it
worked -- because the exercise has two halves and the lesson's own script can
check neither. It never looks at whether a virtual environment is active, and
its PyTorch check is satisfied by an empty file.

**ANSWER: the virtual environment is the answer to both halves, and the
preflight reads neither.** `sys.prefix != sys.base_prefix` is the one-line
test for "a virtual environment is active", and **0** of the eleven probes
read `sys.prefix`, `VIRTUAL_ENV` or anything else that would distinguish a
course environment from the system interpreter. The PyTorch half is checked by
`module_result`, which is `importlib.util.find_spec` -- a question about
`sys.path`, not about an installation.

**FINDING: a zero-byte `torch.py` passes the PyTorch probe.** Writing an empty
file of **0** bytes named `torch.py` into a directory and putting it at the
front of `sys.path` flips the probe from `ok=False` to `ok=True`, and its
detail line then reads "importable by ..." naming this interpreter. Nothing
was installed. `find_spec` does not execute the module, so every module probe
answers "a file with this name is reachable", which is the question the reader
was not asking.

**FINDING: the stricter probe does not catch the fake -- it dies on it.**
`gpu_result` is the one probe that really imports. With the empty `torch.py`
in front it imports fine, and then `torch.cuda` raises **AttributeError**. The
`try`/`except Exception` wraps only the import statement, so the exception
escapes `gpu_result`, escapes `print_probe`, escapes `main`, and the preflight
exits on a traceback instead of printing a verdict. The one probe that could
have detected the fake is the reason the run produces no report at all.

**FINDING: the probe and its own fix name different interpreters.**
`module_result` reports against `sys.executable`. **4** of the fix strings
tell the reader to run `python3 -m pip install ...`, which installs into
whatever `python3` resolves to on `PATH`. **0** fix strings name
`sys.executable`. Following the fix is only guaranteed to fix the probe when
those two happen to be the same file.

Structure: `shadow()` installs an empty module in front of `sys.path` and puts
everything back; `probe_under()` asks the two PyTorch probes inside it.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import pathlib
import shutil
import sys
import tempfile

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "01-dev-environment"
VENV_TESTS = ("sys.prefix", "base_prefix", "VIRTUAL_ENV", "real_prefix")


@contextlib.contextmanager
def shadow(name):
    """Put an empty `name`.py at the front of sys.path, then undo all of it."""
    held = sys.modules.pop(name, None)
    directory = tempfile.mkdtemp()
    stub = pathlib.Path(directory, f"{name}.py")
    stub.write_bytes(b"")
    sys.path.insert(0, directory)
    importlib.invalidate_caches()
    try:
        yield stub.stat().st_size
    finally:
        sys.path.remove(directory)
        sys.modules.pop(name, None)
        if held is not None:
            sys.modules[name] = held
        importlib.invalidate_caches()
        shutil.rmtree(directory, ignore_errors=True)


def probe_under(ref, size):
    """What the two PyTorch-facing probes say while the empty module is in front."""
    module = ref.module_result("torch")
    try:
        gpu, raised = ref.gpu_result(), None
    except Exception as exc:
        gpu, raised = None, type(exc).__name__
    return {"stub_bytes": size, "module_ok": module.ok, "module_detail": module.detail,
            "gpu_raised": raised, "gpu_ok": None if gpu is None else gpu.ok}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "verify")
    with shadow("torch") as size:
        faked = probe_under(ref, size)
    fixes = [p.fix for p in ref.PROBES.values()]
    return {
        **faked,
        "in_venv": sys.prefix != sys.base_prefix,
        "venv_probes": sum(any(t in inspect.getsource(p.run) for t in VENV_TESTS)
                           for p in ref.PROBES.values() if p.run.__name__ != "<lambda>"),
        "probes": len(ref.PROBES),
        "find_spec": "find_spec" in inspect.getsource(ref.module_result),
        "pip_fixes": sum("python3 -m pip" in f for f in fixes),
        "executable_fixes": sum("sys.executable" in f for f in fixes),
        "guarded": inspect.getsource(ref.gpu_result).count("except Exception"),
        "path_python3": shutil.which("python3"),
        "running": sys.executable,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the virtual environment answers both halves and the preflight reads neither",
            all([result["venv_probes"] == 0, result["find_spec"], result["in_venv"]]),
            f"sys.prefix != sys.base_prefix settles the first half in one line and "
            f"{result['venv_probes']} of {result['probes']} probes consult it; the "
            "second half is find_spec, a question about sys.path rather than about an "
            "installation",
        ),
        practice.Check(
            "FINDING: a zero-byte torch.py passes the PyTorch probe",
            all([result["stub_bytes"] == 0, result["module_ok"],
                 "importable by" in result["module_detail"]]),
            f"an empty file of {result['stub_bytes']} bytes at the front of sys.path "
            f"flips the probe to ok=True with the detail {result['module_detail']!r} -- "
            "nothing was installed, and find_spec never executes the module",
        ),
        practice.Check(
            "FINDING: the stricter probe does not catch the fake -- it dies on it",
            all([result["gpu_raised"] == "AttributeError", result["gpu_ok"] is None,
                 result["guarded"] == 1]),
            f"gpu_result really imports, so the empty module reaches torch.cuda and "
            f"raises {result['gpu_raised']}; its {result['guarded']} except Exception "
            "wraps the import alone, so the exception escapes print_probe and main and "
            "the preflight exits on a traceback instead of a verdict",
        ),
        practice.Check(
            "FINDING: the probe and its own fix name different interpreters",
            all([result["pip_fixes"] == 4, result["executable_fixes"] == 0]),
            f"{result['pip_fixes']} fix strings install with python3 -m pip, which "
            f"targets {result['path_python3']} on PATH, while the probe reports on "
            f"{result['running']} -- {result['executable_fixes']} fix strings name the "
            f"interpreter the probe actually asks, and here the two "
            f"{'agree' if result['path_python3'] == result['running'] else 'differ'}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
