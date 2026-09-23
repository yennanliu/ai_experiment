"""Exercise 1 — the workflow pushes to the repo it later says you cannot.

    Fork this repo, clone your fork, create a branch called `my-progress`,
    make a file, commit it, push it

Reading of the exercise: do every step that does not need someone else's
server, then ask what the remaining step needs -- because the lesson spends
four Build It steps on this sequence and gets the one irreversible instruction
in the wrong order.

**ANSWER: five of the six steps run locally and the sixth is the only one that
leaves the machine.** `git init`, `git checkout -b my-progress`, writing a
file, `git add` and `git commit` produce a branch named `my-progress` holding
exactly **1** commit whose tree contains the file. `git push origin
my-progress` into that same repository fails immediately with "does not appear
to be a git repository" -- not because the branch is wrong but because a push
is the first step that needs a second machine, an account on it, and write
access to a repository there.

**FINDING: the daily workflow pushes to the course repo, and the lesson says
you cannot, 16 lines later.** Line **62** is `git push origin main` in the
"daily workflow" block. Line **78** is "You can't push to the course repo
itself -- only maintainers have write access." A reader working top to bottom
runs the failing command before reaching the sentence that explains why.

**FINDING: the lesson never says how the push is authenticated.** The clone in
Step 4 is anonymous HTTPS. The push two lines later needs a credential. The
words token, ssh, credential, authenticate, PAT and password appear **0** times
in the whole lesson, so the one step that can fail for a reason the reader
cannot guess is the one step with no instructions.

**FINDING: the "exactly these commands" table is wrong in both directions.**
Use It says "you need exactly these commands" and lists **6**. The Build It
blocks demonstrate **5** that the table omits -- config, fetch, merge, pull,
status -- and the table lists `git log --oneline`, which **0** Build It blocks
use. The list is neither a superset nor a subset of what the lesson teaches.

Structure: `run()` is one git invocation; `build()` performs the local half of
the exercise in a throwaway repository.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import tempfile

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "02-git-and-collaboration"
IDENT = ("-c", "user.name=Practice", "-c", "user.email=practice@example.invalid")
SECRETS = ("token", "ssh", "credential", "authenticate", "PAT", "password")


def run(repo, *args):
    """One git command in `repo`; returns (exit code, stdout+stderr)."""
    done = subprocess.run(("git", "-C", str(repo)) + IDENT + args,
                          capture_output=True, text=True, timeout=60)
    return done.returncode, (done.stdout + done.stderr).strip()


def build():
    """Fork-and-clone minus the network: branch, file, commit, then try to push."""
    repo = pathlib.Path(tempfile.mkdtemp())
    run(repo, "init", "-q", "-b", "main")
    run(repo, "commit", "-q", "--allow-empty", "-m", "base")
    run(repo, "checkout", "-q", "-b", "my-progress")
    (repo / "notes.md").write_text("# my progress\n", encoding="utf-8")
    run(repo, "add", "notes.md")
    run(repo, "commit", "-q", "-m", "Add notes")
    _, branch = run(repo, "rev-parse", "--abbrev-ref", "HEAD")
    _, ahead = run(repo, "rev-list", "--count", "main..my-progress")
    _, tree = run(repo, "ls-tree", "-r", "--name-only", "HEAD")
    code, pushed = run(repo, "push", "origin", "my-progress")
    shutil.rmtree(repo, ignore_errors=True)
    return {"branch": branch, "ahead": int(ahead), "tracked": tree.split(),
            "push_code": code, "push_says": pushed.splitlines()[0] if pushed else ""}


def fences(doc):
    """git verbs demonstrated inside fenced blocks, and those in the Use It table."""
    shown, inside = [], False
    for line in doc.splitlines():
        if line.startswith("```"):
            inside = not inside
        elif inside:
            shown += re.findall(r"\bgit ([a-z-]+)", line)
    listed = re.findall(r"`git ([a-z-]+)[^`]*`", doc[doc.index("## Use It"):])
    return set(shown), set(listed)


def line_of(doc, needle):
    """1-based line number of the first line containing `needle`."""
    return next(i for i, line in enumerate(doc.splitlines(), 1) if needle in line)


def solve():
    doc = parity.doc_text(PHASE, LESSON)
    shown, listed = fences(doc)
    return {
        **build(),
        "push_line": line_of(doc, "git push origin main"),
        "cannot_line": line_of(doc, "only maintainers"),
        "secret_words": sum(len(re.findall(word, doc, re.I)) for word in SECRETS),
        "listed": sorted(listed),
        "taught_not_listed": sorted(shown - listed),
        "listed_not_taught": sorted(listed - shown),
    }


def verify(result):
    gap = result["cannot_line"] - result["push_line"]
    return [
        practice.Check(
            "ANSWER: five of the six steps run locally and the sixth leaves the machine",
            all([result["branch"] == "my-progress", result["ahead"] == 1,
                 "notes.md" in result["tracked"], result["push_code"] != 0,
                 "origin" in result["push_says"]]),
            f"the branch is {result['branch']} with {result['ahead']} commit ahead of "
            f"main and {result['tracked']} in its tree; the push exits "
            f"{result['push_code']} with {result['push_says']!r} -- the first step that "
            "needs a second machine and an account on it",
        ),
        practice.Check(
            "FINDING: the daily workflow pushes to the course repo, and the lesson says you cannot",
            all([result["push_line"] == 62, result["cannot_line"] == 78, gap == 16]),
            f"line {result['push_line']} is `git push origin main`; line "
            f"{result['cannot_line']}, {gap} lines later, is \"only maintainers have "
            "write access\" -- a reader going top to bottom runs the failing command "
            "before reaching the sentence that explains it",
        ),
        practice.Check(
            "FINDING: the lesson never says how the push is authenticated",
            result["secret_words"] == 0,
            f"the clone is anonymous HTTPS and the push two lines later needs a "
            f"credential; {', '.join(SECRETS)} appear {result['secret_words']} times in "
            "the lesson, so the one step that fails for an unguessable reason has no "
            "instructions",
        ),
        practice.Check(
            "FINDING: the 'exactly these commands' table is wrong in both directions",
            all([len(result["listed"]) == 6, len(result["taught_not_listed"]) == 5,
                 result["listed_not_taught"] == ["log"]]),
            f"Use It lists {len(result['listed'])} commands; Build It demonstrates "
            f"{len(result['taught_not_listed'])} the table omits "
            f"({', '.join(result['taught_not_listed'])}) and the table lists "
            f"git {result['listed_not_taught'][0]}, which no Build It block uses",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
