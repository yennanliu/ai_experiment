"""Exercise 2 — memhogs' macOS fallback never fires, and killtraining misses torchrun.

    Add the aliases from `code/shell_aliases.sh` to your shell config and
    reload with `source ~/.zshrc` (or `~/.bashrc`).

Reading of the exercise: editing a real rc file is host state, so the file is
sourced the way the rc line would source it, into `bash --noprofile --norc`
with HOME pointed at an empty temp dir, and what it defines is listed and then
exercised. External commands the definitions call (`ps`, `pkill`) are shadowed
by shell functions; the `pkill -f` pattern is matched with Python's `re`, which
agrees with POSIX ERE on `python.*train`.

**ANSWER: sourcing defines 22 aliases and 10 functions, the same counts as the
file's `alias` lines and `name() {` definitions; a second `source` (the
"reload") leaves both counts unchanged.**

**FINDING: `memhogs` prints nothing on macOS.** It is
`ps aux --sort=-%mem | head -11 || ps aux -m | head -11`, but a pipeline's
status is its last command's, and `head` succeeds; with a BSD-style `ps` that
rejects `--sort`, the output is empty and the exit status is 0, so the `-m`
fallback written for macOS never runs.

**FINDING: `killtraining` kills the wrong processes and misses common
launchers.** `pkill -f "python.*train"` matches 3 of 3 bystanders -- a server
started with `--config configs/train.yaml`, `pytest tests/test_trainer.py`,
a Jupyter kernel under `~/training-notes` -- and 0 of 3 jobs started as
`torchrun`, `accelerate launch` or `deepspeed`, which are all training.

**FINDING: the aliases only exist in interactive shells.** In the same
non-interactive bash, the function `lastexp` runs (exit 0) but the alias
`diskuse` is "command not found" (exit 127): a script that sources this file
gets the 10 functions and none of the 22 aliases.

**FINDING: the doc's rc line only works when the terminal opens in the repo.**
Step 7 says to add `source phases/00-setup-and-tooling/.../shell_aliases.sh`, a
relative path; a new shell starts in HOME, where that line exits 1 and defines
0 aliases.

**CONTROL: the plain `python train.py` job is matched.**

Structure: `bash_run` is the clean shell; `KILL_CASES` is the labelled fixture.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "10-terminal-and-shell"
BSD_PS = ('ps() { case "$*" in *--sort*) echo "ps: illegal option" >&2; return 1;; '
          '*) echo "USER PID %MEM"; echo "me 42 9.9";; esac; }\n')
KILL_CASES = {   # command line -> is it a training job?
    "python train.py --epochs 10": True,
    "torchrun --nproc_per_node=8 train.py": True,
    "accelerate launch train.py": True,
    "deepspeed train.py --deepspeed_config ds.json": True,
    "python serve.py --config configs/train.yaml": False,
    "python -m pytest tests/test_trainer.py": False,
    "python3 -m ipykernel_launcher --notebook-dir=/home/me/training-notes": False,
}


def bash_run(script, home):
    bash = shutil.which("bash")
    if bash is None:
        raise practice.Skip("bash is not on PATH")
    done = subprocess.run([bash, "--noprofile", "--norc", "-c", script], capture_output=True,
                          text=True, cwd=home, env={"PATH": "/usr/bin:/bin", "HOME": home},
                          timeout=30)
    return done.returncode, done.stdout


def solve():
    path = parity.lesson_dir(PHASE, LESSON) / "code" / "shell_aliases.sh"
    text = path.read_text()
    count = "echo $(alias -p | wc -l) $(declare -F | wc -l)"
    with tempfile.TemporaryDirectory() as home:
        src = f"source '{path}'\n"
        _, once = bash_run(src + count, home)
        _, twice = bash_run(src + src + count, home)
        hog_rc, hogs = bash_run(src + BSD_PS + "memhogs", home)
        alias_rc, _ = bash_run(src + "diskuse", home)
        func_rc, _ = bash_run(src + "lastexp", home)
        rel = re.search(r"^source (phases/\S+)$", parity.doc_text(PHASE, LESSON), re.M)
        _, doc_line = bash_run(f"source {rel.group(1)}; echo $? $(alias -p | wc -l)", home)
    pattern = re.search(r"alias killtraining='pkill -f \"(.+?)\"'", text).group(1)
    hits = {cmd: bool(re.search(pattern, cmd)) for cmd in KILL_CASES}
    return {
        "counts": (once.split(), twice.split()),
        "declared": (len(re.findall(r"^alias ", text, re.M)),
                     len(re.findall(r"^\w+\(\) \{", text, re.M))),
        "memhogs": (hog_rc, hogs.strip()),
        "rcs": (alias_rc, func_rc),
        "doc_line": tuple(doc_line.split()),
        "bystanders": sum(hits[c] for c, job in KILL_CASES.items() if not job),
        "launchers": sum(hits[c] for c, job in KILL_CASES.items() if job and "python" not in c),
        "plain": hits["python train.py --epochs 10"],
    }


def verify(result):
    once, twice = result["counts"]
    rc, out = result["memhogs"]
    alias_rc, func_rc = result["rcs"]
    return [
        practice.Check(
            "ANSWER: 22 aliases and 10 functions, unchanged by a second source",
            once == twice == ["22", "10"] and result["declared"] == (22, 10),
            f"after one source: {once}, after two: {twice}; the file declares "
            f"{result['declared'][0]} aliases and {result['declared'][1]} functions",
        ),
        practice.Check(
            "FINDING: memhogs prints nothing with a BSD ps, and exits 0",
            rc == 0 and out == "",
            f"exit {rc}, output {out!r}: the || tests head's status, so 'ps aux -m' never runs",
        ),
        practice.Check(
            "FINDING: killtraining hits 3 of 3 bystanders and 0 of 3 launcher jobs",
            result["bystanders"] == 3 and result["launchers"] == 0,
            f"'python.*train' matches {result['bystanders']} non-training commands and "
            f"{result['launchers']} of the torchrun / accelerate / deepspeed jobs",
        ),
        practice.Check(
            "FINDING: in a non-interactive shell the aliases do not exist",
            alias_rc == 127 and func_rc == 0,
            f"alias diskuse exits {alias_rc} (not found); function lastexp exits {func_rc}",
        ),
        practice.Check(
            "FINDING: the doc's relative 'source phases/...' line fails from HOME",
            result["doc_line"] == ("1", "0"),
            f"run from HOME it exits {result['doc_line'][0]} with "
            f"{result['doc_line'][1]} aliases defined",
        ),
        practice.Check(
            "CONTROL: 'python train.py' is matched",
            result["plain"],
            "the one job launched as python <...>train is the one the pattern was written for",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
