"""Parity assertions: the from-scratch version vs the real library. (D5)

This is the repo's flagship artifact. A parity check imports the *actual code
the lesson ships*, runs the production equivalent on the same inputs, and
asserts the numbers agree. It turns "your toy attention is what torch does"
from a claim in prose into a green test.

Two rules keep these honest:

  * The reference implementation is **imported, never copied**. If the lesson
    changes, the parity check breaks -- which is the point.
  * A passing check prints the actual max deviation, not just "OK", so a result
    that passes only because the tolerance is loose is visible.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REFERENCE_DIRNAME = "ai-engineering-from-scratch"
REPO_ROOT = Path(__file__).resolve().parents[1]


class ParityError(AssertionError):
    """A from-scratch implementation and its library equivalent disagree."""


def _is_reference(path: Path) -> bool:
    return (path / "phases").is_dir() and (path / "ROADMAP.md").is_file()


def candidate_reference_paths() -> list[Path]:
    """Where the reference checkout might be, nearest first.

    `AIEFS_REF` wins. Otherwise every ancestor of this repo is checked for an
    `ai-engineering-from-scratch` sibling -- the demo repo may sit directly
    beside the reference, or nested a few levels down inside another repo (and
    a git worktree nests it deeper still), so a fixed `parents[2]` is wrong.
    """
    override = os.environ.get("AIEFS_REF")
    if override:
        return [Path(override).expanduser()]
    return [parent / REFERENCE_DIRNAME for parent in REPO_ROOT.parents]


def reference_repo() -> Path:
    """Path to the `ai-engineering-from-scratch` checkout this repo mirrors."""
    for candidate in candidate_reference_paths():
        if _is_reference(candidate):
            return candidate
    raise ParityError(
        f"reference repo not found. Looked for a `{REFERENCE_DIRNAME}` directory "
        f"beside any ancestor of {REPO_ROOT}.\n"
        "Clone it, or point AIEFS_REF at an existing checkout:\n"
        "  git clone https://github.com/yennanliu/ai-engineering-from-scratch"
    )


def load_reference(module_path: str):
    """Import a module from the reference repo by its repo-relative path.

    >>> load_reference("phases/07-transformers-deep-dive/03-multi-head-attention/code/main.py")

    Loaded under a private `aiefs_ref.` name so two lessons that both ship a
    `main.py` cannot collide in `sys.modules`.
    """
    file = reference_repo() / module_path
    if not file.exists():
        raise ParityError(f"reference module not found: {file}")

    name = "aiefs_ref." + module_path.removesuffix(".py").replace("/", ".").replace("-", "_")
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise ParityError(f"cannot import reference module: {file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


# --------------------------------------------------------------------------
# numeric comparison
# --------------------------------------------------------------------------


def flatten(value: Any) -> list[float]:
    """Flatten a tensor, array, nested sequence or scalar to a list of floats.

    Deliberately duck-typed: it has to accept torch tensors, numpy arrays,
    Python lists, and the bespoke `Matrix` classes the lessons hand-roll,
    without importing torch or numpy on a T0 machine.
    """
    if hasattr(value, "detach"):  # torch tensor
        value = value.detach().cpu()
    if hasattr(value, "tolist"):  # torch tensor / numpy array
        value = value.tolist()
    elif hasattr(value, "data") and not isinstance(value, (str, bytes)):
        # The lessons' hand-rolled `Matrix` classes keep their numbers in
        # `.data` -- nested lists in Phase 01, a flat row-major list in
        # Phase 07. Both flatten to the same row-major sequence, which is what
        # the library's `.tolist()` gives us too.
        value = value.data
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        out: list[float] = []
        for item in value:
            out.extend(flatten(item))
        return out
    raise ParityError(f"cannot flatten {type(value).__name__} into numbers")


@dataclass(frozen=True)
class ParityResult:
    """The outcome of one parity check, printable as a table row."""

    label: str
    count: int
    max_abs_diff: float
    max_rel_diff: float
    atol: float
    passed: bool

    def row(self) -> str:
        mark = "ok " if self.passed else "FAIL"
        return (
            f"  {mark}  {self.label:<34}  n={self.count:<6} "
            f"max|d|={self.max_abs_diff:.2e}  atol={self.atol:.0e}"
        )


def compare(mine: Any, theirs: Any, *, label: str, atol: float = 1e-5,
            rtol: float = 0.0) -> ParityResult:
    """Compare two numeric structures without raising.

    `rtol` defaults to 0 so `atol` means exactly what it says. A parity check
    that advertises 1e-12 must not be quietly passing on a relative slack.
    """
    a, b = flatten(mine), flatten(theirs)
    if len(a) != len(b):
        raise ParityError(
            f"{label}: shape mismatch -- from-scratch has {len(a)} values, "
            f"the library has {len(b)}"
        )
    if not a:
        raise ParityError(f"{label}: nothing to compare (both are empty)")

    max_abs = 0.0
    max_rel = 0.0
    passed = True
    for x, y in zip(a, b):
        diff = abs(x - y)
        scale = max(abs(x), abs(y))
        rel = diff / scale if scale else 0.0
        max_abs = max(max_abs, diff)
        max_rel = max(max_rel, rel)
        if diff > atol + rtol * abs(y):
            passed = False
    return ParityResult(label, len(a), max_abs, max_rel, atol, passed)


def assert_close(mine: Any, theirs: Any, *, label: str, atol: float = 1e-5,
                 rtol: float = 0.0) -> ParityResult:
    """Assert two numeric structures agree, and return the measured deviation.

    Raises `ParityError` with the worst offending pair, so a failure says *how
    far apart* the implementations drifted rather than just that they did.
    """
    result = compare(mine, theirs, label=label, atol=atol, rtol=rtol)
    if not result.passed:
        a, b = flatten(mine), flatten(theirs)
        worst = max(range(len(a)), key=lambda i: abs(a[i] - b[i]))
        raise ParityError(
            f"{label}: from-scratch and library implementations disagree.\n"
            f"  worst element [{worst}]: from-scratch={a[worst]!r} library={b[worst]!r}\n"
            f"  max abs diff {result.max_abs_diff:.3e} > atol {atol:.1e} "
            f"(+ rtol {rtol:.1e})"
        )
    return result


def report(results: list[ParityResult], *, title: str = "parity") -> None:
    """Print the small metric table D6 asks every demo to end with."""
    print(f"\n{title}: {len(results)} check(s)")
    for result in results:
        print(result.row())
    worst = max((r.max_abs_diff for r in results), default=0.0)
    print(f"  worst deviation across all checks: {worst:.3e}")
