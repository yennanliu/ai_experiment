"""Exercise 2 — the process with the biggest VIRT is the one using the least memory.

    Install `htop` with apt, run it, and identify which process is using the most
    memory.

Reading of the exercise: `apt` needs root and a network, and htop is interactive,
so neither runs in CI. What the exercise tests is reading htop's memory columns,
and those are `ps`'s VSZ (htop's VIRT) and RSS (htop's RES), both in KiB on Linux
and macOS alike. So two real processes are started whose answers differ by
column: a *reserver* that maps 1 GiB and never touches it -- what a CUDA or JAX
process does with its address space -- and a *worker* that fills 128 MiB.
`ps -o pid=,vsz=,rss=` is read for both.

**ANSWER: the worker, by RES.** Its RSS is ~130 MiB above the reserver's (143
against 13 MiB), and the reserver's stays at an idle interpreter's.

**FINDING: by VIRT the answer is the other process.** The reserver's VSZ is
896 MiB above the worker's (its 1 GiB less the worker's own 128), so sorting on
VIRT -- the column htop prints first -- names the process using the least
memory. GPU frameworks reserve tens of GiB
of address space at start-up, so on a GPU box VIRT is where they rank first.

**FINDING: the lesson never says which column to read.** `docs/en.md` lists
`htop` as an "interactive process viewer" and gives no column, sort key or key
binding: RES, VIRT, RSS and VSZ appear 0 times. htop sorts by CPU% until told
otherwise (`M` or F6), so its top row is not the answer either.

**CONTROL:** both processes are the same interpreter, so their baselines match;
the reserver's RSS is within 32 MiB of a third, idle process.

Structure: `start` launches a child and waits for its ready line; `ps_memory`
reads VSZ/RSS in KiB.
"""

from __future__ import annotations

import re
import subprocess
import sys

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "11-linux-for-ai"
MIB = 1024  # ps reports KiB
CHILDREN = {
    "idle": "pass",
    "reserver": "import mmap; held = mmap.mmap(-1, 1 << 30)",
    "worker": "import os; held = bytearray(os.urandom(1 << 20)) * 128",
}


def start(code):
    script = f"{code}\nimport sys, time\nprint('ready', flush=True)\nsys.stdin.read()"
    child = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True)
    child.stdout.readline()
    return child


def ps_memory(pids):
    """{pid: (vsz_kib, rss_kib)} from `ps`, which prints KiB on Linux and macOS."""
    out = subprocess.run(["ps", "-o", "pid=,vsz=,rss=", "-p", ",".join(map(str, pids))],
                         capture_output=True, text=True, check=True).stdout
    rows = [line.split() for line in out.splitlines() if line.strip()]
    return {int(pid): (int(vsz), int(rss)) for pid, vsz, rss in rows}


def solve():
    children = {name: start(code) for name, code in CHILDREN.items()}
    try:
        mem = ps_memory([c.pid for c in children.values()])
    finally:
        for child in children.values():
            child.stdin.close()
            child.wait(timeout=10)
    doc = parity.doc_text(PHASE, LESSON)
    return {
        "mem": {name: mem[c.pid] for name, c in children.items()},
        "columns": len(re.findall(r"\b(?:RES|VIRT|RSS|VSZ)\b|%MEM", doc)),
        "htop_lines": [ln.split("#")[1].strip() for ln in doc.splitlines()
                       if ln.startswith("htop ") and "#" in ln],
    }


def verify(result):
    (idle_v, idle_r), (res_v, res_r), (work_v, work_r) = (
        result["mem"][k] for k in ("idle", "reserver", "worker"))
    by_rss = max(("reserver", res_r), ("worker", work_r), key=lambda t: t[1])[0]
    by_vsz = max(("reserver", res_v), ("worker", work_v), key=lambda t: t[1])[0]
    return [
        practice.Check(
            "ANSWER: by RES the worker uses the most memory",
            by_rss == "worker" and work_r - res_r > 100 * MIB,
            f"RSS: worker {work_r / MIB:.0f} MiB, reserver {res_r / MIB:.0f} MiB "
            f"({(work_r - res_r) / MIB:.0f} MiB apart)",
        ),
        practice.Check(
            "FINDING: by VIRT the answer is the reserver, which uses the least",
            by_vsz == "reserver" and res_v - work_v > 500 * MIB,
            f"VSZ: reserver is {(res_v - work_v) / MIB:.0f} MiB above the worker, "
            "for 1 GiB mapped and never touched",
        ),
        practice.Check(
            "FINDING: the lesson names no memory column and no sort key",
            result["columns"] == 0 and result["htop_lines"],
            f"RES/VIRT/RSS/VSZ/%MEM in docs/en.md: {result['columns']}; what it says of htop: "
            f"{result['htop_lines']}",
        ),
        practice.Check(
            "CONTROL: the reserver's RSS stays at an idle interpreter's",
            abs(res_r - idle_r) < 32 * MIB,
            f"reserver {res_r / MIB:.1f} MiB vs idle {idle_r / MIB:.1f} MiB RSS",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
