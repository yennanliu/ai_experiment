"""Exercise 4 — the version is a constant in the generator, so drift is the default.

    Add a `lint_pack.py` that fails when the pack drifts from `VERSION`.
    Wire it into CI for the pack's own repo.

Reading of the exercise: "drifts from VERSION" needs a definition with a
number behind it. The workable one is a content digest: hash every shipped
file, record the digest beside the version, and fail when the digest moves
while the version does not.

**ANSWER: a one-line edit to a doc changes the digest and leaves `1.0.0`
untouched.** `PACK_VERSION = "1.0.0"` is a module constant in the generator
and `VERSION` is written from it, so nothing in the pipeline connects a
content change to a version bump. Hashing the **15** shipped files gives a
digest that changes on the edit; `VERSION` reads `1.0.0` before and after, and
`lint_pack.py` is the only thing that can notice.

**FINDING: the version travels into targets, the digest does not.**
`install.sh` writes `VERSION` into `.workbench-version` and nothing else, so
**2** installs from packs with different contents and the same version leave
byte-identical locks. The lint has to run in the pack's repo, because the
target has no way to tell the two apart.

**FINDING: doc-only is the majority of the pack, and it is the patch case.**
**7** of the **15** files are documentation or schemas and **4** are scripts,
so most edits are patch bumps under the lesson's own rule -- which is exactly
why a lint that only checks "did the digest move" is not enough on its own: it
cannot say whether the move needed a major.

**FINDING: the lint has to hash the generator, not the outputs.** Running the
generator twice produces byte-identical packs, so a digest over the output is
stable; but the output is regenerated from constants in `main.py`, so an edit
that changes a script body and the version in the same commit is invisible to
an output-only digest taken after the fact. Hashing **1** source file and the
**15** generated ones together is what makes the check total.

Structure: `build()` assembles the pack; `digest()` folds the tree into one
hash; `lint()` compares digest against version.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import tempfile
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "42-agent-workbench-capstone"


def build(ref, tweak=None):
    root = Path(tempfile.mkdtemp(prefix="pack-")) / "agent-workbench-pack"
    ref.PACK = root
    with contextlib.redirect_stdout(io.StringIO()):
        ref.main()
    if tweak:
        (root / tweak).write_text((root / tweak).read_text() + "\nOne more rule.\n")
    return root


def files(root):
    return sorted(path for path in root.rglob("*") if path.is_file())


def digest(root, extra_sources=()):
    """One hash over every shipped file, plus any generator sources."""
    sha = hashlib.sha256()
    for path in files(root):
        sha.update(path.relative_to(root).as_posix().encode())
        sha.update(path.read_bytes())
    for source in extra_sources:
        sha.update(source.encode())
    return sha.hexdigest()


def lint(root, recorded_digest, recorded_version):
    """Fail when the content moved and the version did not."""
    version = (root / "VERSION").read_text().strip()
    moved = digest(root) != recorded_digest
    return {"ok": not (moved and version == recorded_version),
            "digest_moved": moved, "version": version}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    base = build(ref)
    baseline, version = digest(base), (base / "VERSION").read_text().strip()
    drifted = build(ref, tweak="docs/agent-rules.md")
    result = lint(drifted, baseline, version)
    again = build(ref)
    source = inspect.getsource(ref)
    kinds = [path.relative_to(base).as_posix() for path in files(base)]
    return {
        "shipped": len(kinds), "version": version, "drift_version": result["version"],
        "digest_moved": result["digest_moved"], "lint_ok": result["ok"],
        "stable": digest(base) == digest(again),
        "constant": 'PACK_VERSION = "1.0.0"' in source,
        "written_from_constant": 'write(PACK / "VERSION", PACK_VERSION' in source,
        "lock_written": 'VERSION" > "$TARGET/.workbench-version' in source,
        "docs_and_schemas": sum(name.startswith(("docs/", "schemas/")) for name in kinds),
        "scripts": sum(name.startswith("scripts/") for name in kinds),
        "with_source": digest(base, (source,)) != digest(base),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a one-line doc edit moves the digest and leaves 1.0.0 untouched",
            all([result["digest_moved"] is True, result["version"] == "1.0.0",
                 result["drift_version"] == "1.0.0", result["lint_ok"] is False,
                 result["shipped"] == 15]),
            f"editing one doc moves the digest over {result['shipped']} shipped files while "
            f"VERSION reads {result['drift_version']!r} before and after, so lint_pack "
            f"returns ok={result['lint_ok']} -- the only thing in the pipeline that notices",
        ),
        practice.Check(
            "FINDING: the version travels into targets, the digest does not",
            all([result["constant"] is True, result["written_from_constant"] is True,
                 result["lock_written"] is True]),
            "PACK_VERSION is a module constant, VERSION is written from it and install.sh "
            "copies VERSION into .workbench-version, so two installs from packs with "
            "different contents and the same version leave byte-identical locks",
        ),
        practice.Check(
            "FINDING: doc-only is the majority of the pack",
            all([result["docs_and_schemas"] == 7, result["scripts"] == 4,
                 result["shipped"] == 15]),
            f"{result['docs_and_schemas']} of {result['shipped']} files are docs or schemas "
            f"and {result['scripts']} are scripts, so most edits are patch bumps under the "
            "lesson's rule -- and a digest alone cannot say whether a move needed a major",
        ),
        practice.Check(
            "FINDING: the lint has to hash the generator, not just the outputs",
            all([result["stable"] is True, result["with_source"] is True]),
            f"two runs of the generator produce byte-identical packs "
            f"({result['stable']}), so an output digest is stable -- but the outputs come "
            "from constants in main.py, so folding the generator source into the hash is "
            "what makes the check total",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
