"""The `PRACTICE_IMPL` grading shim (`DESIGN D13`).

Every solution doubles as its own auto-grader. A solution module exposes:

    PRACTICE_IMPL = {"solve": solve, "verify": verify}

`solve()` returns whatever the exercise asked for; `verify(result)` returns a
list of `Check`. The same pair backs three callers, so they can never disagree:
`uv run demo practice run`, `tests/test_practice.py`, and running the file
directly.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import traceback


@dataclasses.dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return f"    [{mark}] {self.name}" + (f" — {self.detail}" if self.detail else "")


@dataclasses.dataclass
class Result:
    exercise: str
    status: str                       # pass | fail | skip | error
    checks: tuple = ()
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("pass", "skip")


class Skip(Exception):
    """Raised by `solve` when a capability is missing; carries the remedy."""


def load_module(path: pathlib.Path):
    path = pathlib.Path(path)
    spec = importlib.util.spec_from_file_location(f"_aiefs_sol_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _impl(module, path):
    impl = getattr(module, "PRACTICE_IMPL", None)
    if impl is None:
        raise AttributeError(f"{path}: no PRACTICE_IMPL (DESIGN D13)")
    for key in ("solve", "verify"):
        if not callable(impl.get(key)):
            raise AttributeError(f"{path}: PRACTICE_IMPL['{key}'] missing or not callable")
    return impl


def grade_file(path: pathlib.Path, name: str | None = None) -> Result:
    """Run one solution's solve/verify pair and collect its checks."""
    name = name or pathlib.Path(path).stem
    try:
        module = load_module(path)
        impl = _impl(module, path)
    except Skip as exc:
        return Result(name, "skip", detail=str(exc))
    except Exception as exc:
        return Result(name, "error", detail=f"{type(exc).__name__}: {exc}")
    try:
        checks = tuple(impl["verify"](impl["solve"]()))
    except Skip as exc:
        return Result(name, "skip", detail=str(exc))
    except Exception as exc:
        return Result(name, "error",
                      detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")
    if not checks:
        return Result(name, "error", checks=(), detail="verify() returned no checks")
    status = "pass" if all(c.ok for c in checks) else "fail"
    return Result(name, status, checks=checks)


def report(result: Result) -> str:
    head = f"  {result.exercise}: {result.status.upper()}"
    lines = [head]
    lines.extend(str(c) for c in result.checks)
    if result.detail:
        lines.append(f"    {result.detail.strip()}")
    return "\n".join(lines)


def selfcheck(module_globals: dict) -> int:
    """`if __name__ == '__main__': raise SystemExit(selfcheck(globals()))`.

    Makes every solution independently runnable *and* independently graded,
    which is what `DESIGN D10` promises.
    """
    impl = module_globals.get("PRACTICE_IMPL")
    name = pathlib.Path(module_globals.get("__file__", "solution")).stem
    try:
        checks = tuple(impl["verify"](impl["solve"]()))
    except Skip as exc:
        print(f"  {name}: SKIP — {exc}")
        return 0
    for check in checks:
        print(check)
    failed = [c for c in checks if not c.ok]
    print(f"  {name}: {'PASS' if not failed else 'FAIL'} ({len(checks) - len(failed)}/{len(checks)})")
    return 1 if failed else 0
