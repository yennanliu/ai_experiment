"""Exercise 3 — the checkout that grades this has one commit in it.

    Look at the commit history of this repo with `git log --oneline` and read
    how lessons were added

Reading of the exercise: try to obey it from inside the grader, find that the
history is not there, and answer the underlying question -- "how lessons were
added" -- from the thing that survives a shallow clone, which is the tree.

**ANSWER: the order lessons were added is written into the directory names,
not into the log.** Every phase directory and every lesson directory carries a
two-digit prefix, so the sequence is recoverable by sorting names with **0**
commits present. That is the answer that holds in any checkout, and it is the
answer the grader has to use.

**FINDING: `--depth=1` keeps every file and drops every commit but one.** A
repository with **3** commits, cloned with `--depth=1`, reports **1** commit
from `git log --oneline` while `git ls-files` returns the identical file list
and the identical tree hash. Nothing about the working copy says the history
is missing; only `git log` does, and it does so by simply being short.

**FINDING: that clone is the checkout grading this exercise.** The workflow
that runs these solutions checks the repository out with `actions/checkout`,
whose `fetch-depth` defaults to **1**, and pins the reference checkout to
`fetch-depth: 1` explicitly. So in CI the command this exercise names returns
one line for a repository with thousands of commits, and any solution that
read `git log` would assert a number that is a property of the fetch.

**FINDING: "this repo" names two repositories that disagree.** The lesson text
lives in the reference checkout and the solution lives in the practice
repository; they have different git directories and different HEADs. Asking
`git log --oneline` in one answers a different question than asking it in the
other, and the exercise does not say which.

Structure: `depth_demo()` builds three commits and clones one of them;
`ordering()` recovers the sequence from names alone.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import tempfile

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "02-git-and-collaboration"
PREFIX = re.compile(r"^\d\d-")


def git(repo, *args):
    """One git command in `repo`; returns stdout+stderr, stripped."""
    done = subprocess.run(("git", "-C", str(repo), "-c", "user.name=P",
                           "-c", "user.email=p@example.invalid") + args,
                          capture_output=True, text=True, timeout=60)
    return (done.stdout + done.stderr).strip()


def depth_demo():
    """Three commits, then a depth-1 clone of them: what survives and what does not."""
    work = pathlib.Path(tempfile.mkdtemp())
    source, clone = work / "source", work / "clone"
    source.mkdir()
    git(source, "init", "-q", "-b", "main")
    for n in range(3):
        (source / f"lesson{n}.md").write_text(f"lesson {n}\n", encoding="utf-8")
        git(source, "add", "-A")
        git(source, "commit", "-q", "-m", f"Add lesson {n}")
    subprocess.run(["git", "clone", "-q", "--depth=1", source.as_uri(), str(clone)],
                   capture_output=True, text=True, timeout=120)
    measured = {
        "source_commits": len(git(source, "log", "--oneline").splitlines()),
        "clone_commits": len(git(clone, "log", "--oneline").splitlines()),
        "same_files": git(source, "ls-files") == git(clone, "ls-files"),
        "same_tree": git(source, "rev-parse", "HEAD^{tree}") == git(clone, "rev-parse",
                                                                       "HEAD^{tree}"),
    }
    shutil.rmtree(work, ignore_errors=True)
    return measured


def ordering(root):
    """Phase and lesson directories, and how many carry an ordering prefix."""
    phases = sorted(p.name for p in root.iterdir() if p.is_dir())
    lessons = [d.name for p in root.iterdir() if p.is_dir()
               for d in p.iterdir() if d.is_dir()]
    return {
        "phases": len(phases),
        "lessons": len(lessons),
        "unprefixed": sorted(n for n in phases + lessons if not PREFIX.match(n)),
        "sorted_is_order": phases == sorted(phases, key=lambda n: int(n[:2])),
    }


def workflow(root):
    """How the CI workflow fetches: explicit depths, and how many checkouts there are."""
    text = (root / ".github" / "workflows" / "aiefs-demo-t0.yml").read_text(encoding="utf-8")
    return text.count("actions/checkout@"), text.count("fetch-depth: 1")


def solve():
    reference = parity.lesson_dir(PHASE, LESSON).parents[2]
    here = pathlib.Path(__file__).resolve()
    demo = next(p for p in here.parents if (p / "harness").is_dir()).parent
    checkouts, pinned = workflow(demo)
    return {
        **depth_demo(),
        **ordering(reference / "phases"),
        "checkouts": checkouts,
        "pinned_depth": pinned,
        "default_depth": 1,
        "separate_repos": git(reference, "rev-parse", "--git-dir") != git(demo, "rev-parse", "--git-dir"),
        "different_heads": git(reference, "rev-parse", "HEAD") != git(demo, "rev-parse", "HEAD"),
        "visible_here": len(git(demo, "log", "--oneline").splitlines()),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the order lessons were added is written into the directory names",
            all([result["unprefixed"] == [], result["sorted_is_order"],
                 result["phases"] >= 15, result["lessons"] >= 100]),
            f"{result['phases']} phase and {result['lessons']} lesson directories, "
            f"{len(result['unprefixed'])} without a two-digit prefix -- the sequence "
            "sorts out of the names with no commits present, the only form that "
            "survives the checkout doing the grading",
        ),
        practice.Check(
            "FINDING: --depth=1 keeps every file and drops every commit but one",
            all([result["source_commits"] == 3, result["clone_commits"] == 1,
                 result["same_files"], result["same_tree"]]),
            f"a {result['source_commits']}-commit repository cloned with --depth=1 "
            f"reports {result['clone_commits']} commit, while ls-files and the tree hash "
            "are identical -- nothing in the working copy says the history is gone",
        ),
        practice.Check(
            "FINDING: that clone is the checkout grading this exercise",
            all([result["checkouts"] == 2, result["pinned_depth"] == 1,
                 result["default_depth"] == 1]),
            f"the workflow runs {result['checkouts']} actions/checkout steps, pins "
            f"{result['pinned_depth']} to fetch-depth: 1 and lets the other take the "
            f"default of {result['default_depth']} -- in CI this command returns one "
            f"line where it returns {result['visible_here']} here",
        ),
        practice.Check(
            "FINDING: 'this repo' names two repositories that disagree",
            all([result["separate_repos"], result["different_heads"]]),
            "the lesson text lives in the reference checkout and the solution in the "
            "practice repository -- different git directories, different HEADs -- so "
            "the command answers a different question in each",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
