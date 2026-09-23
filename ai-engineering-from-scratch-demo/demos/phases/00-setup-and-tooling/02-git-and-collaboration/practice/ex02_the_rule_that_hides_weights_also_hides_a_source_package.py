"""Exercise 2 — the rule that hides weights also hides a source package.

    Create a `.gitignore` that excludes model checkpoint files (`.pt`, `.pth`,
    `.safetensors`)

Reading of the exercise: write it, check it with `git check-ignore` rather
than by eye, and then read the one this repository already ships -- because
that file answers the exercise on line 49 and breaks something else on line
45.

**ANSWER: three patterns, checked against a real index rather than read.**
`*.pt`, `*.pth` and `*.safetensors` in a `.gitignore` make `git check-ignore`
claim all **3** sample weight files and leave a `.py` beside them tracked.
Checking is the point: pattern syntax is matched by path, so the only honest
test is to put files in a repository and ask git.

**FINDING: the repository already ships the answer, and three more.** Lines
**49**, **50** and **52** of the root `.gitignore` are exactly the three
patterns the exercise asks for, and lines 51, 53 and 54 add `*.onnx`, `*.bin`
and `*.h5`. The exercise asks the reader to write a file that is already in
the checkout they were told to clone in the previous exercise.

**FINDING: the same file excludes source, not just weights.** Line **45** is
`models/` and line **44** is `data/`, and a bare `dir/` pattern matches at
every depth. `git check-ignore` confirms that
`phases/01-math-foundations/01-linear-algebra-intuition/code/models/encoder.py`
is excluded by line 45. In a repository teaching machine learning, `models/`
is the most likely name for a source package, and adding one would make it
invisible to `git add` with no error.

**FINDING: a `.gitignore` cannot untrack what is already committed.** Commit
`weights.pt`, then add `*.pt` to `.gitignore`, then change the file:
`git status --porcelain` still reports it modified. Ignoring is consulted for
untracked paths only, so a reader who does this exercise after committing a
checkpoint has changed nothing about the checkpoint.

Structure: `sandbox()` builds a throwaway repository; `ignored()` asks git,
not the reader, which paths a pattern claims.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "02-git-and-collaboration"
ASKED = ("*.pt", "*.pth", "*.safetensors")
SAMPLES = ("model.pt", "model.pth", "model.safetensors", "train.py")
SOURCE = "phases/01-math-foundations/01-linear-algebra-intuition/code/models/encoder.py"


def git(repo, *args):
    """One git command in `repo`; returns (exit code, stdout+stderr)."""
    done = subprocess.run(("git", "-C", str(repo), "-c", "user.name=P",
                           "-c", "user.email=p@example.invalid") + args,
                          capture_output=True, text=True, timeout=60)
    return done.returncode, (done.stdout + done.stderr).strip()


def sandbox():
    """A repository holding the three sample weights plus one source file."""
    repo = pathlib.Path(tempfile.mkdtemp())
    git(repo, "init", "-q", "-b", "main")
    for name in SAMPLES:
        (repo / name).write_bytes(b"")
    return repo


def ignored(repo, paths):
    """Which of `paths` git itself reports as excluded."""
    code, out = git(repo, "check-ignore", *paths)
    return sorted(out.split()) if code == 0 else []


def written():
    """The exercise, done: write the three patterns and ask git what they claim."""
    repo = sandbox()
    (repo / ".gitignore").write_text("\n".join(ASKED) + "\n", encoding="utf-8")
    claimed = ignored(repo, SAMPLES)
    shutil.rmtree(repo, ignore_errors=True)
    return claimed


def tracked_first():
    """Commit a checkpoint, ignore it, change it, and ask git."""
    repo = sandbox()
    git(repo, "add", "model.pt")
    git(repo, "commit", "-q", "-m", "Add weights")
    (repo / ".gitignore").write_text("*.pt\n", encoding="utf-8")
    (repo / "model.pt").write_bytes(b"changed")
    _, status = git(repo, "status", "--porcelain", "model.pt")
    claimed = ignored(repo, ["model.pt"])
    shutil.rmtree(repo, ignore_errors=True)
    return status, claimed


def shipped():
    """Line numbers of the checkpoint and directory rules in the root .gitignore."""
    root = parity.lesson_dir(PHASE, LESSON).parents[2]
    lines = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    numbered = {line.strip(): n for n, line in enumerate(lines, 1) if line.strip()}
    code, out = git(root, "check-ignore", "-v", "--no-index", SOURCE)
    return numbered, len(numbered), (out.split(":")[1] if code == 0 else "")


def solve():
    numbered, patterns, source_rule = shipped()
    status, still = tracked_first()
    return {
        "claimed": written(),
        "asked_lines": [numbered.get(p) for p in ASKED],
        "extra_lines": [numbered.get(p) for p in ("*.onnx", "*.bin", "*.h5")],
        "patterns": patterns,
        "models_line": numbered.get("models/"),
        "data_line": numbered.get("data/"),
        "source_rule": source_rule,
        "tracked_status": status,
        "tracked_still_ignored": still,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: three patterns, checked against a real index rather than read",
            result["claimed"] == ["model.pt", "model.pth", "model.safetensors"],
            f"{', '.join(ASKED)} make git check-ignore claim all "
            f"{len(result['claimed'])} weight files and leave train.py untouched",
        ),
        practice.Check(
            "FINDING: the repository already ships the answer, and three more",
            all([result["asked_lines"] == [49, 50, 52],
                 result["extra_lines"] == [51, 53, 54]]),
            f"lines {result['asked_lines']} of the root .gitignore are those three "
            f"patterns and {result['extra_lines']} add *.onnx, *.bin and *.h5 -- 6 of "
            f"{result['patterns']}, already in the checkout the reader was told to clone",
        ),
        practice.Check(
            "FINDING: the same file excludes source, not just weights",
            all([result["models_line"] == 45, result["data_line"] == 44,
                 result["source_rule"] == "45"]),
            f"line {result['models_line']} is models/ and {result['data_line']} is data/; "
            f"a bare dir/ pattern matches at every depth, and check-ignore blames line "
            f"{result['source_rule']} for {SOURCE} -- a source package in a machine "
            "learning repository, invisible to git add with no error",
        ),
        practice.Check(
            "FINDING: a .gitignore cannot untrack what is already committed",
            all([result["tracked_status"].endswith("model.pt"),
                 result["tracked_status"].lstrip().startswith("M"),
                 result["tracked_still_ignored"] == []]),
            f"after committing model.pt, ignoring *.pt and changing it, git status still "
            f"reports {result['tracked_status']!r} and check-ignore claims "
            f"{len(result['tracked_still_ignored'])} paths -- ignoring is consulted for "
            "untracked paths only",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
