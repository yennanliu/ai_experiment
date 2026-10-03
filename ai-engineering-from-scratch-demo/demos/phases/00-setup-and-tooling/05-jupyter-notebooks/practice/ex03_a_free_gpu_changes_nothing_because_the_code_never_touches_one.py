"""Exercise 3 — a free GPU changes nothing, because the code never touches one.

    Take the code from `code/notebook_tips.py`, paste it into a Colab notebook, and
    run it with a free GPU

Reading of the exercise: Colab needs a browser, a Google account and a network,
none of which a test may assume, so this ships the scaled-down runnable
`DESIGN D11` asks for. Pasting a file into one cell means executing its source
in a kernel whose `__name__` is `"__main__"`; what a T4 could change is whatever
the source sends to a device. So the source is read with `ast` for device code,
and its numpy-only functions are run on this CPU through `parity`.

**ANSWER: the GPU is irrelevant to this file.** Its imports are `time`, `sys`,
`numpy`, `matplotlib` and `pandas`; it references none of `torch`, `cuda`,
`cupy`, `jax`, `tensorflow` or `device`. numpy runs on the CPU, so on Colab's T4
runtime every number it prints comes from the CPU, as it does here. Switching the
runtime type changes nothing but the queue you wait in.

**FINDING: pasted into a cell, the code never asks for an inline plot.** The module calls
`matplotlib.use("Agg")` at import -- the non-interactive file backend -- and
`inline_plotting` calls `savefig`, never `show`, then prints "In a notebook,
plt.show() displays this inline." The plot lands as `notebook_plot.png` in the
runtime's working directory, the one place a Colab user is least likely to look.

**FINDING: the "process memory" line measures the array, not the process.**
`memory_check` prints `~80.0 MB` from `sys.getsizeof(large)`, which for a numpy
array is the object header plus its buffer *if it owns one*: here 80,000,112
bytes against `nbytes` 80,000,000. A view of the same 80 MB reports **~112
bytes**, so this is the wrong tool for the memory-leak hunting the lesson
teaches.

**CONTROL: the guard runs, and the numpy functions run on CPU.** The
`if __name__ == "__main__"` test evaluates True in a kernel-like namespace, so
the pasted cell would execute `main`; `timing_comparison` and `memory_check`
complete here and print their figures with no GPU present.

Structure: `load_lesson` imports with plotting stubbed when absent; `scan`
reads the source tree; `solve` runs the numpy-only functions.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import inspect
import io
import re
import sys
import types

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "05-jupyter-notebooks"
DEVICE_WORDS = ("torch", "cuda", "cupy", "jax", "tensorflow", "device")


def load_lesson():
    """The lesson's module; matplotlib and pandas are stubbed only if absent."""
    added = []
    if importlib.util.find_spec("matplotlib") is None:
        mpl = types.ModuleType("matplotlib")
        mpl.use, mpl.pyplot = (lambda *a, **k: None), types.ModuleType("matplotlib.pyplot")
        sys.modules.update({"matplotlib": mpl, "matplotlib.pyplot": mpl.pyplot})
        added += ["matplotlib", "matplotlib.pyplot"]
    if importlib.util.find_spec("pandas") is None:
        sys.modules["pandas"] = types.ModuleType("pandas")
        added.append("pandas")
    try:
        return parity.load_reference(PHASE, LESSON, "notebook_tips")
    finally:
        for name in added:
            sys.modules.pop(name, None)


def called(node):
    """Names of attributes called anywhere under `node` (plt.savefig -> savefig)."""
    return {n.func.attr for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}


def in_kernel(tree):
    """The value of the module's `if __name__ == ...` test in a kernel's namespace."""
    guard = next(n.test for n in tree.body if isinstance(n, ast.If))
    return eval(compile(ast.Expression(guard), "<cell>", "eval"), {"__name__": "__main__"})


def scan(source):
    """Imports, device words, top-level calls, plotting calls, guard value."""
    tree = ast.parse(source)
    imports = [a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names]
    plot = next(n for n in tree.body if getattr(n, "name", "") == "inline_plotting")
    top = [ast.unparse(n.value) for n in tree.body if isinstance(n, ast.Expr)]
    names = {getattr(n, "id", None) or getattr(n, "attr", None) for n in ast.walk(tree)}
    return {"imports": imports, "device": [w for w in DEVICE_WORDS if w in names],
            "top": top, "plot_calls": called(plot), "guard": in_kernel(tree)}


def printed(function):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        function()
    return out.getvalue()


def solve():
    import numpy as np
    ref = load_lesson()
    memory = printed(ref.memory_check)
    large = np.zeros(10_000_000)
    return {
        **scan(inspect.getsource(ref)),
        "timing": printed(ref.timing_comparison),
        "printed_mb": float(re.search(r"memory: ~([\d.]+) MB", memory).group(1)),
        "owner": sys.getsizeof(large), "nbytes": large.nbytes, "view": sys.getsizeof(large[:]),
        "torch": importlib.util.find_spec("torch") is not None,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: nothing in the file can use a GPU",
            result["device"] == [] and "numpy" in result["imports"],
            f"imports {result['imports']}; of {list(DEVICE_WORDS)} the source references "
            f"{result['device']}. numpy runs on the CPU, so a T4 runtime prints the same numbers",
        ),
        practice.Check(
            "FINDING: pasted into a cell, the code never asks for an inline plot",
            "matplotlib.use('Agg')" in result["top"] and "savefig" in result["plot_calls"]
            and "show" not in result["plot_calls"],
            f"module-level calls {result['top']}; inline_plotting calls "
            f"{sorted(result['plot_calls'] & {'savefig', 'show', 'plot', 'hist'})} -- savefig, "
            "never show -- while printing that plt.show() displays the plot inline",
        ),
        practice.Check(
            "FINDING: the 'process memory' line measures the array, not the process",
            result["printed_mb"] == 80.0 and result["owner"] - result["nbytes"] < 200
            and result["view"] < 200,
            f"memory_check prints ~{result['printed_mb']} MB from sys.getsizeof: "
            f"{result['owner']:,} bytes for an owning array against nbytes {result['nbytes']:,}, "
            f"and {result['view']} bytes for a view of the same 80 MB",
        ),
        practice.Check(
            "CONTROL: the guard runs in a kernel, and the numpy functions run on CPU",
            result["guard"] is True and "Speedup:" in result["timing"],
            f"__name__ == '__main__' evaluates {result['guard']} in a kernel-like namespace; "
            f"timing_comparison printed {result['timing'].split('Speedup:')[1].strip()} "
            f"with torch installed: {result['torch']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
