"""Exercise 3 — the lesson's template installs neither torch nor anthropic, and drops pandas.

    Write a `pyproject.toml` for a project that needs both PyTorch and the Anthropic SDK

Reading of the exercise: "needs" is read as required, so both libraries belong
in `[project] dependencies`, not behind an extra. The file is written here and
parsed with the stdlib `tomllib`; the lesson's own `pyproject.toml` example is
parsed out of `docs/en.md` the same way and compared, along with the
`CORE_PACKAGES` line of `code/env_setup.sh`. Nothing is resolved or downloaded.

**ANSWER: a pyproject that parses, with both libraries required.** `[project]`
names the project, pins `requires-python = ">=3.11"`, and lists `torch>=2.3`
and `anthropic>=0.39` as dependencies; `[build-system]` names hatchling, so
`pip install -e .` has a backend to call (it finds the code in
`src/torch_claude_app/`, which this exercise does not create).

**FINDING: copied from the lesson's template, neither library would install.**
The template keeps torch in an extra called `torch` and anthropic in one called
`llm`. Its required `dependencies` are numpy, matplotlib, jupyter and
scikit-learn, so `pip install .` -- or `uv sync` with no flags -- gives a project
that "needs" PyTorch and the Anthropic SDK and has **0 of the 2**. Extras are for
what a project can run without.

**FINDING: the lesson's two dependency lists disagree.** `env_setup.sh`
installs `numpy matplotlib jupyter scikit-learn pandas`; the template's
`dependencies` omit **pandas**. The previous lesson's `notebook_tips.py` imports
pandas at module top, so an environment built from the template cannot import
the course's own Phase 0 code.

**FINDING: the template has no `[build-system]`,** while the lesson tells you to
run `uv pip install -e ".[torch]"` against it. An editable install then relies
on the PEP 517 fallback to setuptools' legacy backend rather than a backend the
file names.

**CONTROL: both lesson sources agree on Python.** The template says
`>=3.11` and the script's gate is `PYTHON_MIN_MAJOR=3`, `PYTHON_MIN_MINOR=11`.

Structure: `PYPROJECT` is the answer; `template` and `script_packages` read the
lesson; `names` strips requirement strings to distribution names.
"""

from __future__ import annotations

import ast
import re
import tomllib

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "06-python-environments"
PYPROJECT = """\
[project]
name = "torch-claude-app"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.3",
    "anthropic>=0.39",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""


def names(requirements):
    """Distribution names, lower-cased, from PEP 508 requirement strings."""
    return [re.match(r"[A-Za-z0-9._-]+", r).group(0).lower() for r in requirements]


def template():
    """The lesson's own pyproject.toml block, parsed."""
    doc = parity.doc_text(PHASE, LESSON, "en")
    return tomllib.loads(re.search(r"```toml\n(.*?)```", doc, re.S).group(1))


def lesson_text(lesson, *parts):
    return parity.lesson_dir(PHASE, lesson).joinpath(*parts).read_text(encoding="utf-8")


def solve():
    mine, theirs = tomllib.loads(PYPROJECT), template()
    script = lesson_text(LESSON, "code", "env_setup.sh")
    tips = ast.parse(lesson_text("05-jupyter-notebooks", "code", "notebook_tips.py"))
    return {
        "mine": names(mine["project"]["dependencies"]),
        "mine_python": mine["project"]["requires-python"],
        "mine_backend": mine.get("build-system", {}).get("build-backend"),
        "their_deps": names(theirs["project"]["dependencies"]),
        "their_extras": {k: names(v) for k, v in
                         theirs["project"]["optional-dependencies"].items()},
        "their_python": theirs["project"]["requires-python"],
        "their_backend": "build-system" in theirs,
        "script": re.search(r'CORE_PACKAGES="([^"]+)"', script).group(1).split(),
        "script_min": re.findall(r"PYTHON_MIN_(?:MAJOR|MINOR)=(\d+)", script),
        "tips_imports": [a.name for n in tips.body if isinstance(n, ast.Import)
                         for a in n.names],
    }


def verify(result):
    needed = {"torch", "anthropic"}
    plain = needed & set(result["their_deps"])
    missing = sorted(set(result["script"]) - set(result["their_deps"]))
    return [
        practice.Check(
            "ANSWER: a pyproject that parses, with both libraries required",
            needed <= set(result["mine"]) and result["mine_backend"] == "hatchling.build"
            and result["mine_python"] == ">=3.11",
            f"tomllib reads dependencies {result['mine']}, requires-python "
            f"{result['mine_python']}, build-backend {result['mine_backend']}",
        ),
        practice.Check(
            "FINDING: copied from the lesson's template, neither library would install",
            not plain and "torch" in result["their_extras"]["torch"]
            and "anthropic" in result["their_extras"]["llm"],
            f"the template's required dependencies are {result['their_deps']}, holding "
            f"{len(plain)} of the 2 needed; they sit in extras {result['their_extras']}",
        ),
        practice.Check(
            "FINDING: the lesson's two dependency lists disagree",
            missing == ["pandas"] and "pandas" in result["tips_imports"],
            f"env_setup.sh installs {result['script']}; the template's dependencies omit "
            f"{missing}, which lesson 05's notebook_tips.py imports at top level "
            f"({result['tips_imports']})",
        ),
        practice.Check(
            "FINDING: the template names no build backend",
            result["their_backend"] is False,
            "no [build-system] table, so `uv pip install -e \".[torch]\"` relies on the PEP 517 "
            "fallback to setuptools' legacy backend",
        ),
        practice.Check(
            "CONTROL: both lesson sources agree on Python 3.11",
            result["their_python"] == ">=3.11" and result["script_min"] == ["3", "11"],
            f"template requires-python {result['their_python']}; script gate "
            f"{'.'.join(result['script_min'])}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
