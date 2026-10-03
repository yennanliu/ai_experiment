"""Exercise 3 — adding flask to the list re-resolves 9 unpinned libraries; bind 0.0.0.0.

    Add `flask` to the Dockerfile, rebuild, and run a simple API server on port 5000.
    Map the port with `-p 5000:5000`

Reading of the exercise: with no daemon in CI, "add and rebuild" is run as a
build-cache simulation over the lesson's own `code/Dockerfile`: apply the edit
in memory, find the first instruction whose text changed (Docker reuses every
layer before it and re-runs every layer from it on), and report what that
re-run installs. Two edits are compared: appending `flask` to the existing
library RUN, and adding it as its own RUN after it. "Run a server on 5000" is
read for the one setting the lesson's own commands show matters, the bind host.

**ANSWER: add a separate `RUN python -m pip install --no-cache-dir flask` after
the library layer, then**
`docker run --rm -p 5000:5000 -v $(pwd):/workspace ai-dev`
`flask --app app run --host 0.0.0.0 --port 5000`.
Editing the list keeps the first 8 of 13 instructions -- the torch layer
included -- cached; appending keeps 9, the library layer too.

**FINDING: editing the existing list re-resolves 9 unpinned libraries.** The
edited RUN is a new layer, and pip re-installs everything on it: numpy,
transformers, datasets and the rest carry no pin, so "add flask" also silently
upgrades them to build-day versions. Appending a RUN installs only flask.

**FINDING: Flask's default bind is 127.0.0.1, which `-p 5000:5000` cannot
reach.** Published ports arrive on the container's network interface, not its
loopback. The lesson already pays this for Jupyter: all 3 Jupyter commands in
the doc and compose file pass `--ip=0.0.0.0`.

**CONTROL: the base swap is the worst case.** An edit to `FROM` invalidates all
13 instructions, and an edit to the torch RUN re-runs 6 of 13, re-downloading
torch.

Structure: `instructions` splits the Dockerfile; `rebuilt` diffs two of them.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "07-docker-for-ai"
FLASK_RUN = "RUN python -m pip install --no-cache-dir flask"


def instructions(text):
    """Dockerfile instruction lines, continuation lines joined and spaces collapsed."""
    lines = (" ".join(ln.split()) for ln in re.sub(r"\\\n", " ", text).splitlines())
    return [ln for ln in lines if ln and not ln.startswith("#")]


def rebuilt(old, new):
    """Index of the first changed instruction: every layer from it on is re-run."""
    return next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new)))


def library_run(steps):
    return max(i for i, s in enumerate(steps) if s.startswith("RUN") and "pip install" in s)


def unpinned(step):
    tokens = step.split("pip install", 1)[1].split()
    return [t for t in tokens if not t.startswith("-") and "==" not in t]


def solve():
    root = parity.lesson_dir(PHASE, LESSON)
    steps = instructions((root / "code" / "Dockerfile").read_text(encoding="utf-8"))
    lib = library_run(steps)
    torch = next(i for i, s in enumerate(steps) if "torch==" in s)
    edited = steps[:lib] + [steps[lib] + " flask"] + steps[lib + 1 :]
    appended = steps[: lib + 1] + [FLASK_RUN] + steps[lib + 1 :]
    swapped = [steps[0].replace("-devel-", "-runtime-")] + steps[1:]
    torch_bump = steps[:torch] + [steps[torch].replace("2.6.0", "2.5.1")] + steps[torch + 1 :]
    texts = [parity.doc_text(PHASE, LESSON), (root / "code" / "docker-compose.yml").read_text()]
    jupyter = [ln for t in texts for ln in t.splitlines() if "jupyter notebook" in ln]
    return {
        "steps": len(steps),
        "torch": torch,
        "edit": (rebuilt(steps, edited), unpinned(edited[lib])),
        "append": (rebuilt(steps, appended), unpinned(appended[lib + 1])),
        "swap": rebuilt(steps, swapped),
        "torch_bump": rebuilt(steps, torch_bump),
        "jupyter": (len(jupyter), sum("--ip=0.0.0.0" in ln for ln in jupyter)),
    }


def verify(result):
    n = result["steps"]
    (edit_at, edit_pkgs), (app_at, app_pkgs) = result["edit"], result["append"]
    runs, bound = result["jupyter"]
    return [
        practice.Check(
            "ANSWER: either edit keeps the torch layer; ship flask as its own RUN",
            (edit_at, app_at, n) == (8, 9, 13) and result["torch"] < edit_at,
            f"editing the list first differs at instruction {edit_at} (0-based) of {n}, "
            f"appending at {app_at}; the torch RUN is instruction {result['torch']}, so "
            f"{edit_at} and {app_at} layers, torch included, stay cached",
        ),
        practice.Check(
            "FINDING: editing the library list re-installs 9 unpinned libraries",
            len(edit_pkgs) == 10 and app_pkgs == ["flask"],
            f"the edited RUN installs {len(edit_pkgs)} unpinned packages ({', '.join(edit_pkgs)})"
            f", so all {len(edit_pkgs) - 1} besides flask re-resolve to build-day versions; "
            f"the appended RUN installs {app_pkgs}",
        ),
        practice.Check(
            "FINDING: the server must bind 0.0.0.0, as every Jupyter launch here does",
            runs == bound == 3,
            f"{bound} of {runs} Jupyter commands in docs/en.md and docker-compose.yml pass "
            "--ip=0.0.0.0; Flask's default 127.0.0.1 is unreachable through -p 5000:5000",
        ),
        practice.Check(
            "CONTROL: a base swap rebuilds every layer, a torch bump 6 of 13",
            result["swap"] == 0 and n - result["torch_bump"] == 6,
            f"devel -> runtime first differs at instruction {result['swap']} ({n} of {n} "
            f"re-run); torch 2.6.0 -> 2.5.1 at {result['torch_bump']} "
            f"({n - result['torch_bump']} of {n} re-run)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
