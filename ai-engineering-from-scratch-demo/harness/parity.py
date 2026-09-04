"""Reference-checkout discovery and parity assertions (`DESIGN D5`).

`load_reference` **imports the lesson's own code**; it never copies it. A
solution that forked the lesson's implementation would assert against itself and
prove nothing.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import io
import os
import pathlib

REFERENCE_DIRNAME = "ai-engineering-from-scratch"


class ReferenceNotFound(RuntimeError):
    pass


def find_reference_root(start: pathlib.Path | None = None) -> pathlib.Path:
    """Search every ancestor for the reference checkout, not a fixed sibling.

    `DESIGN §8` Q1 keeps this repo inside `ai_experiment`, but the search is
    layout-independent on purpose so that answer can change without code changes.
    """
    override = os.environ.get("AIEFS_REFERENCE")
    if override:
        root = pathlib.Path(override).expanduser()
        if not (root / "phases").is_dir():
            raise ReferenceNotFound(f"AIEFS_REFERENCE={root} has no phases/ directory")
        return root
    here = (start or pathlib.Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / REFERENCE_DIRNAME
        if (candidate / "phases").is_dir():
            return candidate
        if parent.name == REFERENCE_DIRNAME and (parent / "phases").is_dir():
            return parent
    raise ReferenceNotFound(
        f"no {REFERENCE_DIRNAME}/ with a phases/ directory in any ancestor of {here}; "
        "clone it beside ai_experiment or set AIEFS_REFERENCE"
    )


def lesson_dir(phase: str, lesson: str) -> pathlib.Path:
    path = find_reference_root() / "phases" / phase / lesson
    if not path.is_dir():
        raise ReferenceNotFound(f"reference lesson not found: {path}")
    return path


def load_reference(phase: str, lesson: str, module: str):
    """Import `phases/<phase>/<lesson>/code/<module>.py` from the reference repo.

    Import-time stdout is swallowed. Not every lesson module guards its demos
    behind `if __name__ == "__main__"` — 02/02 and 02/03 do not — and importing
    one of those otherwise dumps its whole demo transcript into the runner's
    output (361 lines for one lesson, against ~30 of actual checks). We import
    their code as a library; they wrote it as a script, and that is the seam.
    stderr is left alone so a real import failure is still visible.
    """
    source = lesson_dir(phase, lesson) / "code" / f"{module}.py"
    if not source.is_file():
        raise ReferenceNotFound(f"reference module not found: {source}")
    name = f"_aiefs_ref_{phase}_{lesson}_{module}".replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, source)
    loaded = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(loaded)
    return loaded


def doc_text(phase: str, lesson: str, language: str = "en") -> str:
    return (lesson_dir(phase, lesson) / "docs" / f"{language}.md").read_text(encoding="utf-8")


@dataclasses.dataclass(frozen=True)
class Deviation:
    """A measured difference, reported rather than hidden (`DESIGN §9`)."""
    label: str
    worst: float
    tolerance: float

    @property
    def ok(self) -> bool:
        return self.worst <= self.tolerance

    def __str__(self) -> str:
        return f"{self.label}: worst {self.worst:.3g} (tol {self.tolerance:.3g})"


def _flatten(value):
    if hasattr(value, "components"):            # the lesson's Vector
        return [float(x) for x in value.components]
    if hasattr(value, "rows"):                  # the lesson's Matrix
        return [float(x) for row in value.rows for x in row]
    if hasattr(value, "tolist"):                # numpy
        flat = value.tolist()
        out = []
        stack = [flat]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
            else:
                out.append(float(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [float(value)]


def assert_close(mine, theirs, atol: float = 1e-9, label: str = "parity") -> Deviation:
    """Compare and **report the measured deviation**, not just pass/fail."""
    a, b = _flatten(mine), _flatten(theirs)
    if len(a) != len(b):
        raise AssertionError(f"{label}: shape mismatch, {len(a)} vs {len(b)} values")
    worst = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
    deviation = Deviation(label, worst, atol)
    if not deviation.ok:
        raise AssertionError(str(deviation))
    return deviation


def try_numpy():
    """numpy if installed, else None.

    `DESIGN D2` puts numpy inside T0, but the harness itself stays zero-dep, so a
    numpy cross-check is an *extra* assertion a solution adds when it can — never
    the one its correctness rests on.
    """
    try:
        import numpy
    except ImportError:
        return None
    return numpy
