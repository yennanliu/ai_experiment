"""Exercise 1 — trainenv fills two of its three panes, and running it twice makes five.

    Install tmux, create a session with three panes, and run `htop` in one,
    `watch -n1 date` in another, and a Python script in the third. Detach
    and reattach.

Reading of the exercise: installing tmux is the machine step DESIGN D11 scales
down. The lesson ships the three-pane session as its own `trainenv` function in
`code/shell_aliases.sh`, so that function is run for real in `bash --norc`
with `tmux` shadowed by a shell function that records every call. The recorded
calls are replayed through a small pane model (a split inserts the new pane
after the target and makes it active; `new-session` on an existing name fails,
as tmux's "duplicate session" does) to see what each pane runs.

**ANSWER: three panes, from `new-session`, `split-window -h` and
`split-window -v`; pane 1 runs `watch -n1 nvidia-smi`, pane 2 `htop`, pane 0
is left at a shell.** For the exercise, send `watch -n1 date` to pane 1 and
`python script.py` to pane 0; detach with `C-b d` and reattach with the lesson's
`ta train` (`alias ta='tmux attach -t'`).

**FINDING: the lesson's three-pane session fills two panes.** Of three panes
one runs nothing, and one runs `nvidia-smi`, which a laptop without an NVIDIA
GPU does not have -- so on most learners' machines one pane of three is useful.

**FINDING: a second `trainenv train` makes five panes.** The function checks
nothing: `new-session` fails on the existing name, and both `split-window`
calls still run against the old session's active pane (0, where the first run
left it), adding two panes; the shifted indices then send `watch -n1
nvidia-smi` and `htop` into the two new panes, so each now runs twice.

**FINDING: the pane targets hardcode window 0.** Three of the calls target
`train:0.N` while the splits target the bare session name, so under the common
`set -g base-index 1` the `send-keys` and `select-pane` calls name a window
that does not exist.

**CONTROL: nothing ran tmux.** The shim recorded exactly the function's seven
calls, ending in `attach -t train`, and `bash -n` accepts the file.

Structure: `record` runs trainenv under the shim; `replay` is the pane model.
"""

from __future__ import annotations

import shutil
import subprocess

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "10-terminal-and-shell"
SHIM = "tmux() { local IFS='|'; echo \"$*\"; }\n"


def bash_run(script):
    bash = shutil.which("bash")
    if bash is None:
        raise practice.Skip("bash is not on PATH")
    done = subprocess.run([bash, "--noprofile", "--norc", "-c", script], capture_output=True,
                          text=True, env={"PATH": "/usr/bin:/bin"}, timeout=30)
    return done.returncode, done.stdout


def record(path):
    """Every tmux call one `trainenv train` makes, as argument lists."""
    _, out = bash_run(f"source '{path}'\n{SHIM}trainenv train")
    return [line.split("|") for line in out.splitlines()]


def target(call):
    """(session, pane or None) named by a call's -t, or by new-session's -s."""
    name = call[call.index("-t") + 1] if "-t" in call else call[-1]
    session, _, pane = name.partition(":")
    return session, int(pane.rsplit(".", 1)[1]) if pane else None


def replay(calls, sessions):
    """Apply recorded calls to {session: [pane commands]}; a split inserts after active."""
    active = {name: 0 for name in sessions}      # the first run ends on select-pane 0.0
    for call in calls:
        verb, (name, pane) = call[0], target(call)
        if verb == "new-session" and name not in sessions:
            sessions[name], active[name] = ["shell"], 0
        elif verb == "split-window":
            active[name] = active.get(name, 0) + 1
            sessions[name].insert(active[name], "shell")
        elif verb == "select-pane":
            active[name] = pane
        elif verb == "send-keys":
            sessions[name][pane] = call[3]
    return sessions


def solve():
    path = parity.lesson_dir(PHASE, LESSON) / "code" / "shell_aliases.sh"
    calls = record(path)
    once = replay(calls, {})["train"][:]
    twice = replay(calls, replay(calls, {}))["train"]
    return {
        "calls": calls,
        "once": once,
        "twice": twice,
        "window0": sum(any(arg.startswith("train:0.") for arg in c) for c in calls),
        "syntax": bash_run(f"bash -n '{path}'")[0],
    }


def verify(result):
    once = result["once"]
    return [
        practice.Check(
            "ANSWER: three panes; pane 1 watch nvidia-smi, pane 2 htop, pane 0 a shell",
            once == ["shell", "watch -n1 nvidia-smi", "htop"],
            f"replayed trainenv gives panes {once}; send 'watch -n1 date' to pane 1 and a "
            "python script to pane 0, detach with C-b d, reattach with `ta train`",
        ),
        practice.Check(
            "FINDING: two of the three panes get a command, one of them nvidia-smi",
            sum(p != "shell" for p in once) == 2 and any("nvidia-smi" in p for p in once),
            f"{sum(p == 'shell' for p in once)} pane runs nothing; one runs nvidia-smi",
        ),
        practice.Check(
            "FINDING: running trainenv twice leaves five panes",
            len(result["twice"]) == 5 and result["twice"].count("htop") == 2,
            f"new-session fails on the existing name and both splits still run: "
            f"{result['twice']}",
        ),
        practice.Check(
            "FINDING: three targets hardcode window 0",
            result["window0"] == 3,
            f"{result['window0']} calls target 'train:0.N' while the splits use the bare "
            "session name; under base-index 1 those name a window that does not exist",
        ),
        practice.Check(
            "CONTROL: the shim saw the function's seven calls; bash -n accepts the file",
            len(result["calls"]) == 7 and result["calls"][-1] == ["attach", "-t", "train"]
            and result["syntax"] == 0,
            f"{len(result['calls'])} calls, last {result['calls'][-1]}; bash -n exit "
            f"{result['syntax']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
