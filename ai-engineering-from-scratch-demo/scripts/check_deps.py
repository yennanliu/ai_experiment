#!/usr/bin/env python3
"""Every artifact's imports must be covered by its declared `deps_group` (D8).

Otherwise the groups rot back into one monolithic requirements.txt.
"""

from __future__ import annotations

import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from harness import manifest, runner  # noqa: E402

GROUP_MODULES = {
    "none": set(),
    "math": {"numpy", "sklearn", "scipy"},
    # sklearn ships the sample images, pillow decodes them
    "vision": {"numpy", "PIL", "sklearn"},
    "audio": {"numpy"},
    "llm": {"openai", "torch", "transformers", "jax", "jaxlib", "optax"},
    "agents": set(),
    "infra": set(),
}
STDLIB = set(sys.stdlib_module_names)
LOCAL = {"harness"}


def _names(node) -> set:
    if isinstance(node, ast.Import):
        return {a.name.split(".")[0] for a in node.names}
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return {node.module.split(".")[0]}
    return set()


def imports(path: pathlib.Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        found |= _names(node)
    return found


def toplevel_imports(path: pathlib.Path) -> set:
    """Imports that run at import time — the ones that decide whether a module
    is importable on a bare Python.

    An import nested in a function, or guarded by `try: ... except ImportError`,
    does not, which is exactly how the harness offers optional numpy and torch
    probes while staying zero-dep itself.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found |= _names(node)
    return found


def main(argv=None) -> int:
    problems = []
    for man_path in runner._manifests(argv[0] if argv else None):
        pack = manifest.load_practice(man_path)
        for ex in pack.code_exercises:
            path = man_path.parent / ex.filename
            if not path.is_file():
                continue
            allowed = STDLIB | LOCAL | GROUP_MODULES[ex.deps_group]
            for module in sorted(imports(path) - allowed):
                problems.append(
                    f"{pack.lesson}/{ex.filename}: imports {module!r}, not covered by "
                    f"deps_group {ex.deps_group!r}")
    # the harness itself must stay importable with nothing installed
    for path in sorted((pathlib.Path(__file__).parent.parent / "harness").glob("*.py")):
        for module in sorted(toplevel_imports(path) - STDLIB - LOCAL):
            problems.append(f"harness/{path.name}: imports {module!r} at module level — the "
                            f"harness is zero-dep (DESIGN §4); guard it or move it into the "
                            f"function that needs it")
    for problem in problems:
        print("FAIL " + problem)
    print(f"check_deps: {'FAILED' if problems else 'ok'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
