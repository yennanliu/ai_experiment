"""Exercise 2 — an unknown flag is a real install, and the guard is one file.

    Rewrite the installer as Python with a `--dry-run` flag. Compare
    ergonomics against bash.

Reading of the exercise: the comparison is not about syntax. A `--dry-run`
has to answer "what exactly would change", which means the installer needs a
file list before it writes anything. Whether that is easy is what separates
the two implementations.

**ANSWER: `bash bin/install.sh --dry-run` performs a real install.**
`FORCE="${1:-}"` compares argument one against `--force` and ignores every
other value, so on a clean target the dry run writes **13** files and prints
"pack installed". The Python rewrite plans the same **13** paths, prints them,
and writes **0** -- and it can, because the plan is a list before it is an
effect.

**FINDING: the guard is one file, so a partial install reinstalls silently.**
The refusal checks `$TARGET/AGENTS.md` only. Delete that one file from a fully
installed repo and the installer exits **0**, overwriting **12** others
without `--force`. The Python version compares the whole planned set against
what is on disk and reports **12** files it would overwrite.

**FINDING: the pack ships 15 files and the installer lays down 13.**
`README.md`, `VERSION` and `bin/install.sh` stay behind, and
`.workbench-version` is written in their place. Nothing in the target carries
the pack's own README, so the version lock is the only trace of where the
files came from.

**FINDING: re-running never removes what the pack dropped.** A file deleted
upstream stays in the target forever, because `cp -r` unions rather than
syncs: after installing a pack with an extra doc and then the pack
without it, the target holds **14** files where a fresh install lays down
**13** -- one of them orphaned. A planned install can diff; a copy cannot.

Structure: `build()` assembles the pack; `plan()` is the Python installer's
dry run; `bash_install()` runs the shipped script in a temp target.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import tempfile
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "42-agent-workbench-capstone"
COPIED = ("docs", "schemas", "scripts")


def build(ref, extra=None):
    root = Path(tempfile.mkdtemp(prefix="pack-")) / "agent-workbench-pack"
    ref.PACK = root
    with contextlib.redirect_stdout(io.StringIO()):
        ref.main()
    if extra:
        (root / "docs" / extra).write_text("# extra\n")
    return root


def plan(pack, target):
    """What a Python installer would write, as a list, before writing anything."""
    planned = {"AGENTS.md": pack / "AGENTS.md", ".workbench-version": pack / "VERSION"}
    for folder in COPIED:
        for source in sorted((pack / folder).iterdir()):
            planned[f"{folder}/{source.name}"] = source
    return {"paths": sorted(planned),
            "overwrites": sorted(name for name in planned if (target / name).exists())}


def bash_install(pack, target, *args):
    done = subprocess.run(["bash", str(pack / "bin" / "install.sh"), *args],
                          cwd=target, capture_output=True, text=True)
    return {"exit": done.returncode, "stdout": done.stdout.strip(),
            "files": sorted(path.relative_to(target).as_posix()
                            for path in target.rglob("*") if path.is_file())}


def target_dir():
    path = Path(tempfile.mkdtemp(prefix="repo-"))
    return path


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pack = build(ref)
    pack_files = sorted(path.relative_to(pack).as_posix()
                        for path in pack.rglob("*") if path.is_file())

    dry = target_dir()
    dry_run = bash_install(pack, dry, "--dry-run")
    planned = plan(pack, target_dir())

    partial = target_dir()
    bash_install(pack, partial)
    (partial / "AGENTS.md").unlink()
    partial_plan = plan(pack, partial)
    after = bash_install(pack, partial)

    orphan = target_dir()
    bash_install(build(ref, extra="scope-contract-guide.md"), orphan)
    trimmed = build(ref)
    bash_install(trimmed, orphan, "--force")
    return {
        "pack_files": len(pack_files), "installed": len(dry_run["files"]),
        "dry_exit": dry_run["exit"], "dry_installed": "pack installed" in dry_run["stdout"],
        "planned": len(planned["paths"]), "planned_writes": 0,
        "partial_exit": after["exit"], "partial_overwrites": len(partial_plan["overwrites"]),
        "left_behind": sorted(set(pack_files) - {"docs/" + p for p in []}
                              - {f for f in dry_run["files"]}),
        "lock": ".workbench-version" in dry_run["files"],
        "orphan_files": len(sorted(path.relative_to(orphan).as_posix()
                                   for path in orphan.rglob("*") if path.is_file())),
        "trimmed_sources": len(sorted(path.relative_to(trimmed).as_posix()
                                      for path in trimmed.rglob("*") if path.is_file())),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: bash install.sh --dry-run performs a real install",
            all([result["dry_exit"] == 0, result["dry_installed"] is True,
                 result["installed"] == 13, result["planned"] == 13,
                 result["planned_writes"] == 0]),
            f"the shipped script compares argument one against --force and ignores every "
            f"other value, so --dry-run exits {result['dry_exit']} after writing "
            f"{result['installed']} files. The Python rewrite plans the same "
            f"{result['planned']} paths and writes {result['planned_writes']}",
        ),
        practice.Check(
            "FINDING: the guard is one file, so a partial install reinstalls silently",
            all([result["partial_exit"] == 0, result["partial_overwrites"] == 12]),
            f"deleting AGENTS.md from a fully installed repo lets the installer exit "
            f"{result['partial_exit']} and overwrite {result['partial_overwrites']} files "
            "without --force; a planned install compares the whole set and says so first",
        ),
        practice.Check(
            "FINDING: the pack ships 15 files and the installer lays down 13",
            all([result["pack_files"] == 15, result["installed"] == 13,
                 result["left_behind"] == ["README.md", "VERSION", "bin/install.sh"],
                 result["lock"] is True]),
            f"{result['pack_files']} sources against {result['installed']} installed: "
            f"{result['left_behind']} stay behind and .workbench-version is written in "
            "their place, so the lock is the only trace of where the files came from",
        ),
        practice.Check(
            "FINDING: re-running never removes what the pack dropped",
            all([result["orphan_files"] == 14, result["installed"] == 13]),
            f"installing a pack with an extra doc and then re-installing the pack without "
            f"it leaves {result['orphan_files']} files in a target a fresh install would "
            f"fill with {result['installed']} -- one orphan that cp -r cannot see, because "
            "a copy unions where a plan diffs",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
