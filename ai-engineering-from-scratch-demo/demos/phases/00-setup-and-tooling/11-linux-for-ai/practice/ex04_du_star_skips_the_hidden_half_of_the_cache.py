"""Exercise 4 — `du -sh ~/.cache/*` skips the hidden entries, and size on disk is not file size.

    Use `df -h` to check available disk space, then use `du -sh ~/.cache/*` to find
    what's taking up space in your cache.

Reading of the exercise: the real `~/.cache` differs on every machine, so a
labelled stand-in is built in a temp dir: `pip/` with 500 files of 100 bytes,
`huggingface/` with a 4 MiB blob and a 2 GiB sparse `.safetensors` (created by
`truncate`, the way a pre-allocated download starts), and a hidden `.hidden/`
holding 8 MiB. The real `du` and `df` run on it, through `sh` so that the `*` is
the shell's own glob. `-k` replaces `-h` so the numbers are exact.

**ANSWER: `huggingface` 4 MiB and `pip` ~2 MiB,** totalling ~6 MiB -- while
`du -sk ~/.cache` on the directory itself says ~14 MiB.

**FINDING: the `*` drops every dotfile.** The shell glob does not match names
starting with `.`, so `.hidden/` -- 8 MiB, more than half the cache -- is
missing from the exercise's listing. `du -sh ~/.cache/* ~/.cache/.[!.]*` or
`du -sh ~/.cache` sees it.

**FINDING: du measures blocks, not bytes, in both directions.** The 500 pip
files hold 50,000 bytes and occupy 2,000 KiB: one 4 KiB block each, 41x. The
2 GiB sparse file occupies 0 KiB, so `ls -l` and `du` disagree by 2 GiB on the
same file. Neither number is wrong; they answer "how big" and "how much disk".

**FINDING: the lesson's own "biggest space hogs" line fails silently on a Mac.**
`du -h --max-depth=1 / 2>/dev/null` uses GNU's `--max-depth`; BSD `du` (macOS)
spells it `-d`, and the `2>/dev/null` discards the error, so the pipeline prints
nothing. The lesson's macOS-to-Linux table does not list `du`.

**CONTROL:** `df -Pk` and `os.statvfs` agree on the free space of the volume
holding the temp dir (within 1%); summing `st_blocks` by hand reproduces `du`.

Structure: `make_cache` builds the labelled stand-in; `du` runs the real tool;
`blocks_kib` is the hand-rolled check.
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import tempfile

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "11-linux-for-ai"
PIP_FILES, PIP_BYTES, BLOB, SPARSE, HIDDEN = 500, 100, 4 << 20, 2 << 30, 8 << 20


def make_cache(root):
    cache = pathlib.Path(root) / ".cache"
    for sub in ("pip", "huggingface", ".hidden"):
        (cache / sub).mkdir(parents=True)
    for i in range(PIP_FILES):
        (cache / "pip" / f"wheel-{i:03d}.json").write_bytes(b"x" * PIP_BYTES)
    (cache / "huggingface" / "blob").write_bytes(os.urandom(BLOB))
    with open(cache / "huggingface" / "model.safetensors", "wb") as f:
        f.truncate(SPARSE)  # truncate -s 2G: a size with no blocks behind it
    (cache / ".hidden" / "kernels.bin").write_bytes(os.urandom(HIDDEN))
    return cache


def du(command, cwd):
    """{name: KiB} from a real `du -sk` line, run by the real `sh`."""
    out = subprocess.run(["sh", "-c", command], cwd=cwd, capture_output=True, text=True,
                         check=True).stdout
    return {pathlib.Path(p).name: int(k) for k, p in (ln.split("\t") for ln in out.splitlines())}


def blocks_kib(path):
    """What du adds up: st_blocks (512-byte units) of the path and everything under it."""
    entries = [pathlib.Path(path), *pathlib.Path(path).rglob("*")]
    return sum(p.lstat().st_blocks for p in entries) * 512 // 1024


def df_free_kib(path):
    line = subprocess.run(["df", "-Pk", str(path)], capture_output=True, text=True,
                          check=True).stdout.splitlines()[1]
    return int(line.split()[3])


def solve():
    with tempfile.TemporaryDirectory() as home:
        cache = make_cache(home)
        star = du("du -sk .cache/*", home)
        whole = du("du -sk .cache", home)[".cache"]
        sparse = cache / "huggingface" / "model.safetensors"
        st = os.statvfs(home)
        doc = parity.doc_text(PHASE, LESSON)
        table = doc.split("## Gotchas")[1]
        return {
            "star": star, "whole": whole, "hidden": du("du -sk .cache/.hidden", home),
            "pip_disk": star["pip"], "sparse": (sparse.stat().st_size, blocks_kib(sparse.parent)),
            "by_hand": blocks_kib(cache), "df": df_free_kib(home),
            "statvfs": st.f_bavail * st.f_frsize // 1024, "disk": shutil.disk_usage(home).total,
            "max_depth": "du -h --max-depth=1" in doc,
            "du_in_table": bool(re.search(r"\|\s*`du", table)),
        }


def verify(result):
    star, whole, hidden = result["star"], result["whole"], result["hidden"][".hidden"]
    size, hf_disk = result["sparse"]
    pip_ratio = result["pip_disk"] * 1024 / (PIP_FILES * PIP_BYTES)
    return [
        practice.Check(
            "ANSWER: huggingface 4 MiB and pip ~2 MiB, against ~14 MiB for the directory",
            sorted(star) == ["huggingface", "pip"] and sum(star.values()) < whole / 2,
            f"du -sk .cache/*: {star} (sum {sum(star.values())} KiB); du -sk .cache: {whole} KiB",
        ),
        practice.Check(
            "FINDING: the * glob drops the hidden directory, over half the cache",
            ".hidden" not in star and hidden > whole / 2,
            f".hidden is {hidden} KiB of {whole} KiB and absent from the * listing",
        ),
        practice.Check(
            "FINDING: du counts blocks -- 500 tiny files cost 41x, a 2 GiB sparse file 0",
            pip_ratio > 10 and size == SPARSE and hf_disk < 2 * BLOB // 1024,
            f"pip: {PIP_FILES * PIP_BYTES} bytes in {result['pip_disk']} KiB ({pip_ratio:.0f}x); "
            f"model.safetensors: st_size {size} bytes, huggingface/ on disk {hf_disk} KiB",
        ),
        practice.Check(
            "FINDING: the lesson's --max-depth is GNU-only and du is not in its macOS table",
            result["max_depth"] and not result["du_in_table"],
            "docs/en.md runs `du -h --max-depth=1 / 2>/dev/null`; BSD du spells it -d and "
            f"2>/dev/null hides the error; du in the Gotchas table: {result['du_in_table']}",
        ),
        practice.Check(
            "CONTROL: df agrees with statvfs, and st_blocks reproduces du",
            abs(result["df"] - result["statvfs"]) <= 0.01 * result["statvfs"] + 1024
            and abs(result["by_hand"] - whole) <= 4,
            f"df -Pk free {result['df']} KiB, statvfs {result['statvfs']} KiB; st_blocks sum "
            f"{result['by_hand']} KiB vs du {whole} KiB",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
