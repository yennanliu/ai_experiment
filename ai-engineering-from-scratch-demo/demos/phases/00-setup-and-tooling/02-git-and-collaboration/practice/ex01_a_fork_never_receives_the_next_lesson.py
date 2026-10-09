"""Exercise 1 — a fork never receives the next lesson.

    Fork this repo, clone your fork, create a branch called `my-progress`, make
    a file, commit it, push it

Reading of the exercise: GitHub's Fork button is `git clone --bare` on a
server, so the whole sequence runs locally, three repositories in a temp
directory: the course, a bare fork of it, and the reader's clone of the fork.
Every command after the fork is the lesson's Step 4 verbatim, including the
push, `git push origin my-progress`. Git runs with an isolated config so the
host's settings cannot change the outcome.

**ANSWER: the push lands.** The fork's `my-progress` points at the reader's
commit, hash for hash, and the commit's tree holds the new file.

**FINDING: the course moves on and the reader never sees it.** The course
then adds lesson 02. The reader's `git fetch` brings in **0** new commits,
because the clone has **1** remote, `origin`, which is the fork, and the fork
is a snapshot. The lesson text says `upstream` **0** times, `remote add` **0**
and `sync` **0** -- neither the git fix nor GitHub's Sync fork button -- so the
workflow it teaches can push progress but cannot pull new lessons.

**FINDING: the push as written leaves the branch without an upstream.**
`git push origin my-progress` (no `-u`) leaves `my-progress@{upstream}`
unresolved, so the next bare `git push` exits **128** with "has no upstream
branch", and `git status` cannot say whether the work is backed up. `push -u`
appears **0** times in the lesson.

**CONTROL: one `git remote add upstream` fixes the first, `-u` the second.**
After it, `git fetch upstream` brings in the lesson-02 commit; after
`git push -u`, the upstream resolves to `origin/my-progress`.

Structure: `git` runs one command in an isolated environment; `build` makes
the course, forks it and runs Step 4.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "02-git-and-collaboration"


def git(cwd, *args):
    """One git command with no global or system config: (exit code, output)."""
    home = pathlib.Path(cwd).parent
    env = {"PATH": str(pathlib.Path(shutil.which("git")).parent), "HOME": str(home),
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": str(home / "gitconfig"),
           "GIT_AUTHOR_NAME": "L", "GIT_AUTHOR_EMAIL": "l@x", "GIT_COMMITTER_NAME": "L",
           "GIT_COMMITTER_EMAIL": "l@x", "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
           "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}
    done = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=30)
    return done.returncode, (done.stdout + done.stderr).strip()


def commit_file(repo, name, message):
    (repo / name).write_text(message + "\n", encoding="utf-8")
    git(repo, "add", name)
    git(repo, "commit", "-qm", message)


def build(root):
    """Course, bare fork, reader's clone; then Step 4 verbatim."""
    course, me = root / "course", root / "me"
    course.mkdir()
    git(course, "init", "-q", "-b", "main")
    commit_file(course, "lesson01.md", "lesson 01")
    git(root, "clone", "-q", "--bare", "course", "fork.git")
    git(root, "clone", "-q", "fork.git", "me")
    git(me, "checkout", "-q", "-b", "my-progress")
    commit_file(me, "notes.md", "my notes")
    push_code, _ = git(me, "push", "-q", "origin", "my-progress")
    return course, me, push_code


def solve():
    if shutil.which("git") is None:
        raise practice.Skip("git is not installed; install it to run this exercise")
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        course, me, push_code = build(root)
        mine = git(me, "rev-parse", "HEAD")[1]
        forked = git(root / "fork.git", "rev-parse", "my-progress")[1]
        tree = git(me, "ls-tree", "--name-only", "HEAD")[1].split()
        upstream_code, _ = git(me, "rev-parse", "--abbrev-ref", "my-progress@{upstream}")
        bare_code, bare_out = git(me, "push")
        commit_file(course, "lesson02.md", "lesson 02")
        lesson02 = git(course, "rev-parse", "HEAD")[1]
        git(me, "fetch", "-q", "--all")
        remotes = git(me, "remote")[1].split()
        seen = git(me, "cat-file", "-t", lesson02)[0] == 0
        git(me, "remote", "add", "upstream", str(course))
        git(me, "fetch", "-q", "upstream")
        fixed_seen = git(me, "rev-parse", "upstream/main")[1] == lesson02
        git(me, "push", "-q", "-u", "origin", "my-progress")
        fixed_upstream = git(me, "rev-parse", "--abbrev-ref", "my-progress@{upstream}")[1]
    doc = parity.doc_text(PHASE, LESSON).lower()
    words = {w: doc.count(w) for w in ("upstream", "remote add", "sync", "push -u")}
    return {"words": words, "verbatim": "git push origin my-progress" in doc,
            "push_code": push_code, "landed": mine == forked, "tree": tree,
            "upstream_code": upstream_code, "bare_code": bare_code,
            "no_upstream": "has no upstream branch" in bare_out, "remotes": remotes,
            "seen": seen, "fixed_seen": fixed_seen, "fixed_upstream": fixed_upstream}


def verify(result):
    return [
        practice.Check(
            "ANSWER: fork, clone, branch, commit, push -- and the push lands",
            all((result["push_code"] == 0, result["landed"], "notes.md" in result["tree"],
                 result["verbatim"])),
            f"git push origin my-progress exits {result['push_code']}; the fork's my-progress "
            f"equals the reader's HEAD and its tree is {result['tree']}",
        ),
        practice.Check(
            "FINDING: the fork never receives the next lesson",
            result["remotes"] == ["origin"] and not result["seen"]
            and result["words"]["upstream"] == result["words"]["remote add"] == 0,
            f"after the course adds lesson 02, git fetch --all in the clone finds it: "
            f"{result['seen']}. The clone's remotes are {result['remotes']}, the fork, which "
            f"is a snapshot. The lesson mentions 'upstream' {result['words']['upstream']} times, "
            f"'remote add' {result['words']['remote add']} and 'sync' {result['words']['sync']}",
        ),
        practice.Check(
            "FINDING: Step 4's push leaves my-progress with no upstream",
            all((result["upstream_code"] != 0, result["bare_code"] == 128,
                 result["no_upstream"], result["words"]["push -u"] == 0)),
            f"my-progress@{{upstream}} does not resolve (exit {result['upstream_code']}), and "
            f"the next bare git push exits {result['bare_code']}: 'has no upstream branch'. "
            f"'push -u' appears {result['words']['push -u']} times in the lesson",
        ),
        practice.Check(
            "CONTROL: an upstream remote and -u fix both",
            result["fixed_seen"] and result["fixed_upstream"] == "origin/my-progress",
            f"after git remote add upstream, upstream/main is the lesson-02 commit: "
            f"{result['fixed_seen']}; after push -u the upstream is {result['fixed_upstream']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
