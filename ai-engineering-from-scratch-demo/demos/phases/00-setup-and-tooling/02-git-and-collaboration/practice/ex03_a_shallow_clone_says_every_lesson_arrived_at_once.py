"""Exercise 3 — a shallow clone says every lesson arrived at once.

    Look at the commit history of this repo with `git log --oneline` and read
    how lessons were added

Reading of the exercise: this repository's own history is not something a
solution can depend on -- the checkout that grades it may be a one-commit
shallow clone, which is exactly the point. So the history is rebuilt
deterministically in a temp repository: the **12** real lessons of phase 00,
read from the reference `phases/` tree, are added one commit each with fixed
dates and identity. Then `git log --oneline` is read the way the exercise
asks, on a full clone and on a shallow one.

**ANSWER: `git log --oneline` shows one line per lesson added.** The full
history has **12** lines, and `git log --diff-filter=A -- <lesson>` names
**12** distinct commits, one per lesson: exactly how each was added.

**FINDING: on a depth-1 clone the same command says all 12 arrived in one
commit.** `git log --oneline` prints **1** line, and `--diff-filter=A` names
that **1** commit as the one that added every lesson. The only hint is a
`grafted` decoration, which `--decorate` shows and a piped `git log` omits.
CI checkouts (`fetch-depth: 1`) and `git clone --depth 1` both produce this,
so the exercise's question is only answerable from a full clone.

**FINDING: `--depth 1` on a plain local path is silently a full clone.**
`git clone --depth 1 <path>` still logs **12** lines and only warns
"--depth is ignored in local clones"; the shallow history needs a `file://`
URL. The same flag gives two different histories depending on how the source
is spelled.

**CONTROL: the shallow clone has the same files.** Its tip tree lists the
same **12** lesson directories as the full repository; only the history is
gone.

Structure: `git` runs one command in an isolated environment; `build` commits
one lesson per day; `history` reads a clone the way the exercise asks.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "02-git-and-collaboration"


def git(cwd, *args, day=1):
    """One git command with no global or system config: (exit code, output)."""
    home, stamp = pathlib.Path(cwd).parent, f"2026-01-{day:02d}T00:00:00Z"
    env = {"PATH": str(pathlib.Path(shutil.which("git")).parent), "HOME": str(home),
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": str(home / "gitconfig"),
           "GIT_AUTHOR_NAME": "L", "GIT_AUTHOR_EMAIL": "l@x", "GIT_COMMITTER_NAME": "L",
           "GIT_COMMITTER_EMAIL": "l@x", "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
    done = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=30)
    return done.returncode, (done.stdout + done.stderr).strip()


def build(repo, lessons):
    """One commit per real phase-00 lesson, a day apart."""
    git(repo, "init", "-q", "-b", "main")
    for day, lesson in enumerate(lessons, start=1):
        target = repo / "phases" / PHASE / lesson.name / "docs"
        target.mkdir(parents=True)
        shutil.copy(lesson / "docs" / "en.md", target / "en.md")
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", f"Add {lesson.name}", day=day)


def history(clone, lessons):
    """(oneline count, distinct adding commits, lessons in the tip tree)."""
    oneline = git(clone, "log", "--oneline")[1].splitlines()
    adders = {git(clone, "log", "--diff-filter=A", "--format=%h", "--",
                  f"phases/{PHASE}/{lesson.name}")[1] for lesson in lessons}
    tree = git(clone, "ls-tree", "--name-only", "HEAD", f"phases/{PHASE}/")[1].split()
    return len(oneline), len(adders), len(tree)


def solve():
    if shutil.which("git") is None:
        raise practice.Skip("git is not installed; install it to run this exercise")
    phase_dir = parity.lesson_dir(PHASE, LESSON).parent
    lessons = sorted(p for p in phase_dir.iterdir() if (p / "docs" / "en.md").is_file())
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "course").mkdir()
        build(root / "course", lessons)
        _, local_warning = git(root, "clone", "--depth", "1", "course", "local")
        git(root, "clone", "-q", "--depth", "1", (root / "course").as_uri(), "shallow")
        full = history(root / "course", lessons)
        local, shallow = history(root / "local", lessons), history(root / "shallow", lessons)
        decorated = git(root / "shallow", "log", "--oneline", "--decorate")[1]
    return {"grafted": "grafted" in decorated, "lessons": len(lessons), "full": full,
            "shallow": shallow, "local": local,
            "warned": "ignored in local clones" in local_warning}


def verify(result):
    full, shallow, local, n = result["full"], result["shallow"], result["local"], result["lessons"]
    return [
        practice.Check(
            "ANSWER: one oneline per lesson, one adding commit per lesson",
            n >= 2 and full[:2] == (n, n),
            f"{n} real phase-00 lessons committed a day apart: git log --oneline prints "
            f"{full[0]} lines and --diff-filter=A names {full[1]} distinct adding commits",
        ),
        practice.Check(
            "FINDING: a depth-1 clone attributes every lesson to one commit",
            shallow[:2] == (1, 1) and result["grafted"],
            f"on the file:// --depth 1 clone, git log --oneline prints {shallow[0]} line and "
            f"--diff-filter=A names {shallow[1]} commit as the adder of all {n} lessons; the "
            "only hint is a 'grafted' decoration under --decorate",
        ),
        practice.Check(
            "FINDING: --depth 1 on a plain local path is a full clone",
            local[0] == n and result["warned"],
            f"git clone --depth 1 <path> logs {local[0]} lines and only warns that --depth "
            "is ignored in local clones; shallowness needs a file:// URL",
        ),
        practice.Check(
            "CONTROL: the shallow clone has every lesson's files",
            shallow[2] == full[2] == n,
            f"the shallow tip lists {shallow[2]} lesson directories, the full one {full[2]}: "
            "only history is missing",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
