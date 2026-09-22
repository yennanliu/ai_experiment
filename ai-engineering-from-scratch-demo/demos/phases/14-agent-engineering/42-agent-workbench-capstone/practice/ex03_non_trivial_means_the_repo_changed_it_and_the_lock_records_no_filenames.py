"""Exercise 3 — non-trivial means the repo changed it, and the lock records no filenames.

    Add a `bin/uninstall.sh` that safely removes the pack and refuses if
    state files have non-trivial history. What counts as non-trivial?

Reading of the exercise: "non-trivial" has to be decidable from the
repository, not from intent. Three signals are available in a git checkout:
the file is dirty in the working tree, its history has commits beyond the one
that installed it, or its bytes no longer match what the pack laid down.

**ANSWER: non-trivial is "the repo changed it", which on a real checkout
refuses 1 of the 12 installed files and never asks about state at all.**
Installed into a temp git repo and committed, an uninstall removes **11** of
the **12** files the pack wrote and refuses on the locally customised
`docs/agent-rules.md`. `agent_state.json` is never a candidate, because the
pack never wrote it -- the state question is answered by the manifest, not by
inspecting history.

**FINDING: `.workbench-version` records a version and no filenames.** The
installer writes `1.0.0` and **0** paths, so an uninstaller has to re-derive
the file list from a pack source it may no longer have -- and if the pack has
moved on, the derived list is the *new* pack's, not the installed one's. A
manifest is **12** lines and removes the guesswork.

**FINDING: the user's artifacts are the ones the installer never created.**
`agent_state.json`, `task_board.json` and `outputs/` are absent after an
install and present after a session, so "do not delete what you did not
write" is mechanical: **3** user artifacts survive an uninstall that removes
12 pack files.

**FINDING: without hashes, a customised pack file is indistinguishable from an
untouched one.** Removing by name deletes an edited `docs/agent-rules.md`
silently; comparing bytes against the pack catches it as **1** refusal. The
rules doc is the file a team is most likely to edit, which makes it the worst
one to delete by name.

Structure: `repo()` builds a real git checkout with the pack installed;
`uninstall()` classifies every installed path before removing anything.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import subprocess
import tempfile
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "42-agent-workbench-capstone"
USER_ARTIFACTS = ("agent_state.json", "task_board.json", "outputs/verification/T-001.json")


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout


def build(ref):
    root = Path(tempfile.mkdtemp(prefix="pack-")) / "agent-workbench-pack"
    ref.PACK = root
    with contextlib.redirect_stdout(io.StringIO()):
        ref.main()
    return root


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo(pack):
    """A git checkout with the pack installed, committed, then used for a session."""
    root = Path(tempfile.mkdtemp(prefix="repo-"))
    git(root, "init", "-q")
    git(root, "config", "user.email", "a@b.co")
    git(root, "config", "user.name", "tester")
    subprocess.run(["bash", str(pack / "bin" / "install.sh")], cwd=root,
                   capture_output=True, text=True)
    git(root, "add", "-A")
    git(root, "commit", "-qm", "install pack")
    for name in USER_ARTIFACTS:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n")
    (root / "docs" / "agent-rules.md").write_text("# Agent Rules\n\nlocal house style.\n")
    return root


def installed_paths(pack):
    """What the installer wrote, re-derived from the pack source."""
    paths = ["AGENTS.md"]
    for folder in ("docs", "schemas", "scripts"):
        paths += [f"{folder}/{item.name}" for item in sorted((pack / folder).iterdir())]
    return paths


def uninstall(pack, root, keep_agents_md=False):
    """Remove what the pack wrote and only where the repo has not changed it."""
    removed, refused = [], []
    for name in installed_paths(pack):
        if keep_agents_md and name == "AGENTS.md":
            continue
        target, source = root / name, pack / name
        dirty = bool(git(root, "status", "--porcelain", "--", name).strip())
        if not target.exists() or dirty or digest(target) != digest(source):
            refused.append(name)
            continue
        removed.append(name)
    for name in removed:
        (root / name).unlink()
    return removed, refused


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pack = build(ref)
    root = repo(pack)
    state = root / "agent_state.json"
    state.write_text('{"next_action": "keep going"}\n')
    removed, refused = uninstall(pack, root)
    survivors = [name for name in USER_ARTIFACTS if (root / name).exists()]
    lock = (root / ".workbench-version").read_text().strip()
    return {
        "installed": len(installed_paths(pack)),
        "removed": len(removed), "refused": sorted(refused),
        "survivors": survivors, "lock": lock,
        "lock_names": sum(name in lock for name in installed_paths(pack)),
        "manifest_lines": len(installed_paths(pack)),
        "state_absent_after_install": "agent_state.json" not in installed_paths(pack),
        "by_name_would_delete": "docs/agent-rules.md" in installed_paths(pack),
        "lock_survives": (root / ".workbench-version").exists(),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: non-trivial is 'the repo changed it', refusing 1 of 12 and never state",
            all([result["installed"] == 12, result["removed"] == 11,
                 result["refused"] == ["docs/agent-rules.md"],
                 result["state_absent_after_install"] is True]),
            f"of the {result['installed']} files the pack wrote, an uninstall removes "
            f"{result['removed']} and refuses on {result['refused']}; agent_state.json is "
            "never a candidate because the pack never wrote it, so the state question is "
            "answered by the manifest rather than by history",
        ),
        practice.Check(
            "FINDING: .workbench-version records a version and no filenames",
            all([result["lock"] == "1.0.0", result["lock_names"] == 0,
                 result["manifest_lines"] == 12, result["lock_survives"] is True]),
            f"the lock holds {result['lock']!r} and {result['lock_names']} paths, so an "
            f"uninstaller re-derives the list from a pack that may have moved on; a "
            f"manifest is {result['manifest_lines']} lines and removes the guesswork",
        ),
        practice.Check(
            "FINDING: the user's artifacts are the ones the installer never created",
            all([len(result["survivors"]) == 3,
                 result["state_absent_after_install"] is True]),
            f"{result['survivors']} are absent after an install and present after a "
            "session, so 'do not delete what you did not write' is mechanical rather than a "
            "judgment call",
        ),
        practice.Check(
            "FINDING: without hashes a customised pack file looks untouched",
            all([result["by_name_would_delete"] is True,
                 "docs/agent-rules.md" in result["refused"]]),
            "removing by name would delete an edited docs/agent-rules.md silently; "
            f"comparing bytes against the pack catches it among {result['refused']}. The "
            "rules doc is the file a team is most likely to edit",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
