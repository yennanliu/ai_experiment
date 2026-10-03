"""Exercise 2 — the curriculum writes checkpoints the three rules miss.

    Create a `.gitignore` that excludes model checkpoint files (`.pt`, `.pth`,
    `.safetensors`)

Reading of the exercise: write the three-line `.gitignore` and let git, not
the eye, judge it -- in a throwaway repository with an isolated config, so
nothing on the host is read. The files it is judged against are the
checkpoint names the curriculum's own code writes, each confirmed by finding
its literal in that lesson's source under `phases/`.

**ANSWER: `*.pt`, `*.pth`, `*.safetensors` -- and git confirms them.** `git
check-ignore` claims a new `fresh.pt`, `weights.pth` and `model.safetensors`,
and leaves `train.py` beside them tracked.

**FINDING: 3 of the 6 checkpoint names the curriculum writes get through.**
Of `mnist_mlp.pt` (03/11), `ckpt.pt` and its atomic-save temp
`ckpt.pt.<random>.tmp` (19/47), `gpt2-stub.safetensors` (19/37), and the
sharded `rank0.bin` and `rank0.bin.tmp` (19/80, torch-serialized), the rules
ignore **3** and pass **3**. A training run killed mid-save in 19/47 leaves
a full checkpoint named `*.tmp`, which `*.pt` does not match.

**FINDING: the file does nothing for a checkpoint that is already committed.**
Commit `weights.pt` first, then add the `.gitignore` and retrain: `git status`
still reports ` M weights.pt`. Ignore rules apply to untracked paths only.

**CONTROL: `git rm --cached` is what makes the rule bite.** After it, the
checkpoint shows as staged for deletion and is ignored, with no `??` line.

Structure: `git` runs one command in an isolated environment; `CURRICULUM`
ties each checkpoint name to the literal that produces it.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile

from harness import parity, practice

RULES = "*.pt\n*.pth\n*.safetensors\n"
SAMPLES = ["fresh.pt", "weights.pth", "model.safetensors", "train.py"]
# (written name, lesson under phases/, literal in its code/main.py that produces it)
CURRICULUM = [
    ("mnist_mlp.pt", "03-deep-learning-core/11-intro-to-pytorch", "mnist_mlp.pt"),
    ("ckpt.pt", "19-capstone-projects/47-checkpoint-save-resume", '"ckpt.pt"'),
    ("ckpt.pt.k3x9a1.tmp", "19-capstone-projects/47-checkpoint-save-resume", 'suffix=".tmp"'),
    ("gpt2-stub.safetensors", "19-capstone-projects/37-loading-pretrained-weights",
     "gpt2-stub.safetensors"),
    ("rank0.bin", "19-capstone-projects/80-checkpoint-sharded-resume", 'f"rank{rank}.bin"'),
    ("rank0.bin.tmp", "19-capstone-projects/80-checkpoint-sharded-resume",
     'f"rank{rank}.bin.tmp"'),
]


def git(repo, *args):
    """One git command with no global or system config; returns stdout."""
    home = pathlib.Path(repo).parent
    env = {"PATH": str(pathlib.Path(shutil.which("git")).parent), "HOME": str(home),
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": str(home / "gitconfig"),
           "GIT_AUTHOR_NAME": "L", "GIT_AUTHOR_EMAIL": "l@x", "GIT_COMMITTER_NAME": "L",
           "GIT_COMMITTER_EMAIL": "l@x", "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
           "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}
    return subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True,
                          text=True, timeout=30).stdout


def ignored(repo, names):
    for name in names:
        (repo / name).write_bytes(b"\0")
    return sorted(git(repo, "check-ignore", *names).split())


def evidence():
    """How many CURRICULUM literals really appear in their lesson's code."""
    phases = parity.lesson_dir("00-setup-and-tooling", "02-git-and-collaboration").parent.parent
    return sum(any(literal in f.read_text(encoding="utf-8")
                   for f in (phases / lesson / "code").glob("*.py"))
               for _, lesson, literal in CURRICULUM)


def solve():
    if shutil.which("git") is None:
        raise practice.Skip("git is not installed; install it to run this exercise")
    with tempfile.TemporaryDirectory() as root:
        repo = pathlib.Path(root) / "repo"
        repo.mkdir()
        git(repo, "init", "-q")
        (repo / "weights.pt").write_bytes(b"v1")
        git(repo, "add", "weights.pt")
        git(repo, "commit", "-qm", "checkpoint before the rules")
        (repo / ".gitignore").write_text(RULES, encoding="utf-8")
        samples = ignored(repo, SAMPLES)
        curriculum = ignored(repo, [name for name, _, _ in CURRICULUM])
        (repo / "weights.pt").write_bytes(b"v2")
        tracked = git(repo, "status", "--porcelain", "weights.pt")
        git(repo, "rm", "-q", "--cached", "weights.pt")
        after = git(repo, "status", "--porcelain", "--untracked-files=all", "weights.pt")
    return {"samples": samples, "curriculum": curriculum, "evidence": evidence(),
            "tracked": tracked.rstrip("\n"), "after": after.rstrip("\n")}


def verify(result):
    missed = [name for name, _, _ in CURRICULUM if name not in result["curriculum"]]
    return [
        practice.Check(
            "ANSWER: three rules, confirmed by git check-ignore",
            result["samples"] == ["fresh.pt", "model.safetensors", "weights.pth"],
            f"check-ignore claims {result['samples']} and leaves train.py tracked",
        ),
        practice.Check(
            "FINDING: 3 of the 6 checkpoint names the curriculum writes get through",
            result["evidence"] == len(CURRICULUM) and len(missed) == 3,
            f"all {result['evidence']} names are confirmed in their lessons' code; the rules "
            f"ignore {result['curriculum']} and pass {missed} -- an interrupted atomic save "
            "in 19/47 and every torch-serialized shard in 19/80",
        ),
        practice.Check(
            "FINDING: a checkpoint committed before the rules stays tracked",
            result["tracked"] == " M weights.pt",
            f"after adding the .gitignore and retraining, git status reports "
            f"{result['tracked']!r}: ignore rules apply to untracked paths only",
        ),
        practice.Check(
            "CONTROL: git rm --cached is what makes the rule apply",
            result["after"] == "D  weights.pt",
            f"after git rm --cached the status is {result['after']!r}: staged for deletion, "
            "and no '??' line, because the rule now ignores it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
