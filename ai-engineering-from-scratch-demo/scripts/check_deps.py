#!/usr/bin/env python3
"""Assert every demo's imports are covered by its declared `deps_group`. (D8)

Per-phase dependency groups only stay useful if they stay honest. Without this
check a demo quietly imports `transformers` while declaring `deps_group: math`,
the `math` group grows to keep it working, and within a few phases the groups
have collapsed back into the single monolithic `requirements.txt` D8 exists to
avoid.

    python scripts/check_deps.py
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.manifest import Demo, discover  # noqa: E402
from harness.runner import DEMOS_ROOT, REPO_ROOT  # noqa: E402

# Import name -> distribution name, where they differ.
MODULE_TO_DIST = {
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
}
# Always available: the stdlib, the harness itself, the demo's own modules, and
# the test runner every group gets from `dev`.
ALWAYS_OK = {"harness", "run", "pytest", "_pytest"}


def declared_groups() -> dict[str, set[str]]:
    """Distribution names per extra, from pyproject."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"].get("optional-dependencies", {})
    return {
        name: {req.split(">=")[0].split("[")[0].split("==")[0].strip().lower()
               for req in reqs}
        for name, reqs in extras.items()
    }


def top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def third_party(names: set[str]) -> set[str]:
    return {
        n for n in names
        if n not in ALWAYS_OK and n not in sys.stdlib_module_names
    }


def check(demo: Demo, groups: dict[str, set[str]]) -> list[str]:
    available = groups.get(demo.deps_group, set()) | groups.get("dev", set())
    sources = [demo.entrypoint_path, *sorted((demo.path / "tests").glob("test_*.py"))]

    imported: set[str] = set()
    for source in sources:
        if source.exists():
            imported |= third_party(top_level_imports(source))

    missing = sorted(
        module for module in imported
        if MODULE_TO_DIST.get(module, module).lower() not in available
    )
    return [
        f"imports `{module}`, which `{demo.deps_group}` does not provide "
        f"(add it to the group, or move the demo to one that has it)"
        for module in missing
    ]


def main() -> int:
    groups = declared_groups()
    failures = 0
    for demo in discover(DEMOS_ROOT):
        problems = check(demo, groups)
        if problems:
            failures += 1
            print(f"FAIL  {demo.lesson}  [{demo.deps_group}]")
            for problem in problems:
                print(f"        {problem}")
        else:
            print(f"ok    {demo.lesson}  [{demo.deps_group}]")
    print(f"\n{failures} demo(s) import something their group does not declare")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
