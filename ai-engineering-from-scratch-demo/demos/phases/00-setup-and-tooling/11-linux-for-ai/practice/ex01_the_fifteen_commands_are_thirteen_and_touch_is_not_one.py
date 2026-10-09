"""Exercise 1 — the "15 commands" are 13, and `touch` is not one of them.

    SSH into any Linux machine (or open WSL2) and navigate to your home directory.
    Create a project folder, create three empty files inside it with `touch`, then
    list them with `ls -la`.

Reading of the exercise: there is no remote machine in CI, so the session runs in
a temporary directory standing in for `~`, with the umask pinned to 022 (Ubuntu's
default) so the result does not depend on the host. `mkdir`, `touch` and
`ls -la` are done with the system calls those commands make -- `mkdir(2)`,
`open(O_CREAT)` with mode 0666 under the umask, and `lstat(2)` per entry, `.` and
`..` included -- and the rows are printed in `ls -l`'s mode/links/size/name
layout. The lesson's own command list is read from `docs/en.md`.

**ANSWER: 5 rows for 3 files.** `ls -la` lists `.`, `..` and the three files;
each file is 0 bytes with mode `-rw-r--r--` (0666 masked by umask 022), and the
directory is `drwxr-xr-x`. Under a umask of 077 the same `touch` gives
`-rw-------`: the permissions come from the shell, not from `touch`.

**FINDING: the lesson's "15 commands" are 13.** The "Essential Commands"
section promises "the 15 commands that cover 95% of what you'll do"; its code
blocks use 13 distinct commands: pwd, ls, cd, mkdir, cp, mv, rm, cat, head,
tail, less, grep, find.

**FINDING: `touch`, which this exercise needs, appears nowhere in the lesson
body.** It occurs 0 times in the lesson's code blocks and inline code before the
Exercises section -- not in Essential Commands and not in the Quick Reference
Card. (The one prose "touch" is the verb: "the directories you'll actually touch".)

**CONTROL:** the lesson's permission example is right. `-rwxr-xr--` is mode
754, and `chmod 755` / `chmod 644` applied to a real file read back as
`-rwxr-xr-x` and `-rw-r--r--`, as the comments say.

Structure: `session` runs the exercise in a temp dir; `ls_la` renders its rows;
`essential_commands` reads the lesson's command list.
"""

from __future__ import annotations

import os
import pathlib
import re
import stat
import tempfile

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "11-linux-for-ai"
FILES = ("data.py", "model.py", "train.py")


def ls_la(folder):
    """(mode, links, size, name) for `.`, `..` and every entry, as `ls -la` prints them."""
    names = [".", "..", *sorted(os.listdir(folder))]
    rows = []
    for name in names:
        st = os.lstat(os.path.join(folder, name))
        rows.append((stat.filemode(st.st_mode), st.st_nlink, st.st_size, name))
    return rows


def session(umask):
    old = os.umask(umask)
    try:
        with tempfile.TemporaryDirectory() as home:
            project = pathlib.Path(home) / "my-project"
            project.mkdir(mode=0o777)                 # mkdir my-project
            for name in FILES:
                (project / name).touch(mode=0o666)    # touch, which opens with 0666
            return ls_la(project)
    finally:
        os.umask(old)


def chmod_modes():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "deploy.sh"
        path.touch()
        out = {}
        for mode in (0o755, 0o644):
            path.chmod(mode)
            out[oct(mode)[2:]] = stat.filemode(path.stat().st_mode)
        return out


def essential_commands(doc):
    """Distinct first words of the command lines in the 'Essential Commands' section."""
    section = doc.split("## Essential Commands")[1].split("\n## ")[0]
    blocks = re.findall(r"```bash\n(.*?)```", section, re.S)
    lines = [ln.strip() for block in blocks for ln in block.splitlines()]
    return sorted({ln.split()[0] for ln in lines if ln and not ln.startswith("#")})


def solve():
    doc = parity.doc_text(PHASE, LESSON)
    body = doc.split("## Exercises")[0]
    code = re.findall(r"```.*?```|`[^`\n]+`", body, re.S)
    return {
        "rows": session(0o022),
        "strict": session(0o077),
        "promise": re.search(r"These are the (\d+) commands", doc).group(1),
        "commands": essential_commands(doc),
        "touch_in_body": sum(len(re.findall(r"\btouch\b", span)) for span in code),
        "code_spans": len(code),
        "example": stat.filemode(stat.S_IFREG | 0o754),
        "doc_example": "-rwxr-xr--" in doc,
        "chmod": chmod_modes(),
    }


def verify(result):
    rows, strict = result["rows"], result["strict"]
    files = [r for r in rows if r[3] in FILES]
    cmds = result["commands"]
    return [
        practice.Check(
            "ANSWER: ls -la shows 5 rows for 3 empty files, -rw-r--r-- under umask 022",
            len(rows) == 5 and {(m, s) for m, _, s, _ in files} == {("-rw-r--r--", 0)},
            f"{len(rows)} rows ({', '.join(r[3] for r in rows)}); "
            + "; ".join(f"{m} {s} {name}" for m, _, s, name in files)
            + f"; directory {rows[0][0]} -- under umask 077 the files are {strict[2][0]}",
        ),
        practice.Check(
            "FINDING: the '15 commands' of Essential Commands are 13",
            result["promise"] == "15" and len(cmds) == 13,
            f"the doc says 'These are the {result['promise']} commands'; its code blocks "
            f"use {len(cmds)}: {', '.join(cmds)}",
        ),
        practice.Check(
            "FINDING: touch is never taught in the lesson body",
            (result["touch_in_body"], "touch" in cmds) == (0, False),
            f"'touch' occurs {result['touch_in_body']} times in the {result['code_spans']} code "
            "blocks and inline code spans before '## Exercises'",
        ),
        practice.Check(
            "CONTROL: the lesson's permission strings match real modes",
            (result["doc_example"], result["example"], result["chmod"])
            == (True, "-rwxr-xr--", {"755": "-rwxr-xr-x", "644": "-rw-r--r--"}),
            f"0754 renders as {result['example']}; chmod gives {result['chmod']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
