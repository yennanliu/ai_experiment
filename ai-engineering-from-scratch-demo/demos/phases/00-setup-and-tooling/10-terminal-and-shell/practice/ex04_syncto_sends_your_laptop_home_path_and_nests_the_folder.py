"""Exercise 4 — syncto sends your laptop's home path to the server and nests the folder.

    Set up an SSH config entry for a server you have access to (or use
    `localhost` to practice the syntax).

Reading of the exercise: the entry is the doc's own `Host gpu` block pointed at
`localhost`, written to a temp file and resolved by `ssh -G -F <file>`, which
prints the effective settings without connecting. The entry exists to be used
by the lesson's `syncto`/`syncfrom` (`syncto gpu ~/data ./data`), so `syncto` is
run for real in `bash --norc`, once with `rsync` shadowed by a recorder and once
with the `gpu:` prefix rewritten to a local "remote" dir and the real `rsync`.

**ANSWER: `Host gpu / HostName localhost / User learner / Port 2222 /
IdentityFile ~/.ssh/gpu_key` resolves to hostname localhost, user learner,
port 2222; `ssh gpu` and `syncto gpu ...` then need no address.**

**FINDING: the usage example sends your laptop's home to the server.** In
`syncto gpu ~/data ./data` the `~` is a word of its own, so the local shell
expands it: with HOME=/Users/learner the recorded call is `rsync -avz
--progress ./data gpu:/Users/learner/data`, a path a Linux GPU box (home
`/home/ubuntu`) does not have. The doc's own `rsync ... user@gpu-box-ip:~/data/`
keeps the `~` inside the word, unexpanded, for the remote side.

**FINDING: `syncto` nests the folder.** It passes `./data` without the
trailing slash the doc's rsync line has, so rsync copies the directory itself:
the file lands at `data/data/a.txt` on the "remote", not `data/a.txt`.

**FINDING: appending the entry below a `Host *` block loses its settings.**
ssh takes the first value it sees, so with `Host *` / `Port 22` above it the
same entry resolves to port 22.

**CONTROL: the entry is scoped.** `other` resolves to hostname other, port 22.

Structure: `ssh_settings` runs `ssh -G`; `bash_run` runs the lesson's syncto.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "10-terminal-and-shell"
ENTRY = ("Host gpu\n    HostName localhost\n    User learner\n    Port 2222\n"
         "    IdentityFile ~/.ssh/gpu_key\n")
STAR = "Host *\n    Port 22\n\n"
RECORD = 'rsync() { echo "$*"; }\n'
LOCAL = 'rsync() { command rsync "${@/#gpu:/$REMOTE/}"; }\n'


def need(tool):
    found = shutil.which(tool, path="/usr/bin:/bin")
    if found is None:
        raise practice.Skip(f"{tool} is not on PATH")
    return found


def ssh_settings(config, host, tmp):
    path = pathlib.Path(tmp) / "config"
    path.write_text(config)
    done = subprocess.run([need("ssh"), "-G", "-F", str(path), host], capture_output=True,
                          text=True, timeout=30, stdin=subprocess.DEVNULL)
    pairs = (line.split(" ", 1) for line in done.stdout.splitlines() if " " in line)
    return {k: v for k, v in pairs if k in ("hostname", "user", "port")}


def bash_run(script, cwd, **env):
    done = subprocess.run([need("bash"), "--noprofile", "--norc", "-c", script], cwd=cwd,
                          capture_output=True, text=True, timeout=30,
                          env={"PATH": "/usr/bin:/bin", **env})
    return done.stdout.strip()


def solve():
    src = f"source '{parity.lesson_dir(PHASE, LESSON) / 'code' / 'shell_aliases.sh'}'\n"
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        (base / "data").mkdir()
        (base / "data" / "a.txt").write_text("x\n")
        (base / "remote").mkdir()
        recorded = bash_run(src + RECORD + "syncto gpu ~/data ./data", tmp, HOME="/Users/learner")
        nested = None
        if shutil.which("rsync", path="/usr/bin:/bin"):
            bash_run(src + LOCAL + "syncto gpu data ./data", tmp, REMOTE=str(base / "remote"))
            nested = sorted(str(p.relative_to(base / "remote")) for p in
                            (base / "remote").rglob("*.txt"))
        return {
            "entry": ssh_settings(ENTRY, "gpu", tmp),
            "after_star": ssh_settings(STAR + ENTRY, "gpu", tmp),
            "other": ssh_settings(ENTRY, "other", tmp),
            "recorded": recorded,
            "nested": nested,
            "doc_keeps_tilde": "user@gpu-box-ip:~/data/" in parity.doc_text(PHASE, LESSON),
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the gpu entry resolves to localhost, learner, port 2222",
            result["entry"] == {"hostname": "localhost", "user": "learner", "port": "2222"},
            f"ssh -G gpu: {result['entry']}",
        ),
        practice.Check(
            "FINDING: syncto's usage example sends the local home path to the server",
            result["recorded"] == "-avz --progress ./data gpu:/Users/learner/data"
            and result["doc_keeps_tilde"],
            f"recorded: rsync {result['recorded']}; the doc's own rsync keeps ':~/data/'",
        ),
        practice.Check(
            "FINDING: syncto drops the trailing slash, so the folder nests",
            result["nested"] in (["data/data/a.txt"], None),
            "rsync absent; not run" if result["nested"] is None
            else f"after syncto gpu data ./data the remote holds {result['nested']}",
        ),
        practice.Check(
            "FINDING: under a Host * block above it, the entry's port is lost",
            result["after_star"]["port"] == "22",
            f"same entry after 'Host * / Port 22': {result['after_star']}",
        ),
        practice.Check(
            "CONTROL: other hosts are untouched by the entry",
            result["other"] == {"hostname": "other", "user": result["other"].get("user"),
                                "port": "22"},
            f"ssh -G other: {result['other']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
