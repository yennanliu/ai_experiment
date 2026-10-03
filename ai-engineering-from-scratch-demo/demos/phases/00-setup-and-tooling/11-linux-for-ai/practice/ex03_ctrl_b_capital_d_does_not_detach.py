"""Exercise 3 — a hang-up kills `sleep 300` unless it left the session; Ctrl+B D does not detach.

    Start a tmux session, run `sleep 300` inside it, detach, list sessions, and
    reattach.

Reading of the exercise: tmux is not installed in CI, so what tmux is *for* is
measured instead, with the kernel's own machinery: `sleep 300` is started on a
real pseudo-terminal (`pty.fork`, as sshd does), the terminal is hung up by
closing its master side (what a dropped SSH connection does), and the process's
fate is read back. The tmux-like case starts the same `sleep 300` with
`setsid()` (new session, no controlling terminal), which is what the tmux
server does. "List sessions and reattach" becomes: the detached pid is still
there and can be signalled. The lesson's tmux key list is checked against
tmux's default bindings.

**ANSWER: hang-up kills the plain `sleep`; the detached one survives.** Run
directly on the terminal, `sleep 300` dies of signal 1 (SIGHUP) the moment the
terminal closes. A `sleep 300` in the terminal's process group, under a shell
stand-in, dies too. A `sleep 300` started with `setsid` is alive after the
hang-up and is killed only by the solution's own cleanup.

**FINDING: the lesson's detach key is wrong.** It says "Ctrl+B, then D" to
detach. tmux's default binding for `d` is `detach-client`; `D` (shifted) is
`choose-client`, which opens a client picker and detaches nothing.

**FINDING: the split labels are swapped relative to tmux.** The lesson calls `%`
"split pane vertically" and `"` "split pane horizontally". tmux binds `%` to
`split-window -h` and documents it as "Split window horizontally", and `"` as
"Split window vertically". The lesson names the divider line, tmux names the
direction the panes are laid out; someone using the lesson's words to look up
`split-window -v` gets the other split.

**CONTROL:** before the hang-up all three sleeps are alive.

Structure: `on_terminal` runs argv on a fresh pty and hangs it up; `alive` probes
a pid; `TMUX_DEFAULTS` is tmux's key-bindings.c.
"""

from __future__ import annotations

import os
import pty
import re
import select
import signal
import sys
import time

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "11-linux-for-ai"
# tmux key-bindings.c: key -> (command, the note tmux shows for it)
TMUX_DEFAULTS = {
    "d": ("detach-client", "Detach the current client"),
    "D": ("choose-client -Z", "Choose a client from a list"),
    "%": ("split-window -h", "Split window horizontally"),
    '"': ("split-window", "Split window vertically"),
}
SHELL = """import subprocess, sys
kw = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
detached = subprocess.Popen(["sleep", "300"], start_new_session=True, **kw)
attached = subprocess.Popen(["sleep", "300"])  # a shell job: the terminal is its stdio
print(detached.pid, attached.pid, flush=True)
attached.wait()"""


def alive(pid, timeout=0.0):
    """True if `pid` still exists after up to `timeout` seconds of waiting for it to go."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        if time.monotonic() >= deadline:
            return True
        time.sleep(0.02)


def on_terminal(argv):
    """Run argv as the session leader of a new pty, hang the pty up; (signal, first line)."""
    pid, master = pty.fork()
    if pid == 0:
        signal.signal(signal.SIGHUP, signal.SIG_DFL)  # even if the runner ignores it (nohup)
        os.execvp(argv[0], argv)
    line = b""
    while argv[0] == sys.executable and not line.endswith(b"\n"):
        if not select.select([master], [], [], 10)[0]:
            break
        line += os.read(master, 64)
    if argv[0] != sys.executable:
        time.sleep(0.2)
    before = alive(pid)
    os.close(master)                                  # the SSH connection drops
    _, status = os.waitpid(pid, 0)
    sig = os.WTERMSIG(status) if os.WIFSIGNALED(status) else None
    return sig, line.decode(), before


def solve():
    plain_sig, _, plain_before = on_terminal(["sleep", "300"])
    _, pids, shell_before = on_terminal([sys.executable, "-c", SHELL])
    detached, attached = map(int, pids.split())
    fate = {"detached": alive(detached, 0.5), "attached": alive(attached, 5.0)}
    for pid in (detached, attached):
        if alive(pid):
            os.kill(pid, signal.SIGKILL)  # tmux kill-session
    doc = parity.doc_text(PHASE, LESSON)
    keys = dict(re.findall(r"Ctrl\+B, then (\S+)\s+# (.+)", doc))
    return {"plain": plain_sig, "fate": fate, "keys": keys,
            "before": plain_before and shell_before}


def verify(result):
    keys, fate = result["keys"], result["fate"]
    return [
        practice.Check(
            "ANSWER: hang-up kills the attached sleep; the setsid one survives",
            result["plain"] == signal.SIGHUP and fate == {"detached": True, "attached": False},
            f"sleep 300 on the terminal ended by signal {result['plain']} (SIGHUP); in the "
            f"terminal's group under a shell: alive={fate['attached']}; after setsid: "
            f"alive={fate['detached']}",
        ),
        practice.Check(
            "FINDING: 'Ctrl+B, then D' is choose-client; detach is lowercase d",
            keys.get("D", "").startswith("Detach") and TMUX_DEFAULTS["D"][0] != "detach-client",
            f"lesson: D -> {keys.get('D')!r}; tmux: D -> {TMUX_DEFAULTS['D'][0]}, "
            f"d -> {TMUX_DEFAULTS['d'][0]}",
        ),
        practice.Check(
            "FINDING: the lesson's split labels are tmux's swapped",
            "vertically" in keys.get("%", "") and "horizontally" in keys.get('"', ""),
            f"lesson: % -> {keys.get('%')!r}, \" -> {keys.get(chr(34))!r}; tmux: % is "
            f"{TMUX_DEFAULTS['%'][0]} ({TMUX_DEFAULTS['%'][1]}), \" is "
            f"{TMUX_DEFAULTS[chr(34)][0]} ({TMUX_DEFAULTS[chr(34)][1]})",
        ),
        practice.Check(
            "CONTROL: every sleep was alive before the hang-up",
            result["before"],
            "the pty's session leader answered kill(pid, 0) before its master was closed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
