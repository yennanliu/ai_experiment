"""Exercise 5 — the digest covers the bytes and nothing about the file.

    Modify an installed reference after manifest creation. Prove the package
    verification fails before activation.

Reading of the exercise: "before activation" means the check has to run
against the installed tree rather than the source, so the proof is a copy, an
edit, and a verification of the copy while the source still verifies clean.
Running the other two tamper shapes at the same time is what shows the check
is a set comparison rather than a per-file digest, and running a fourth shows
where the set comparison stops.

**ANSWER: the installed tree fails and the source passes, from the same
manifest.** Editing `references/format.md` after install gives
`mismatched=['references/format.md']` with `passed=False`, while the source
verifies clean -- so the failure is attributable to the install rather than
to the package. Adding a file reports `unexpected` and removing one reports
`missing`: **3** tamper shapes, **3** distinct fields.

**FINDING: the digest covers the bytes and nothing about the file.** Removing
the executable bit from a bundled script leaves every digest identical and
`verify_manifest` returns `passed=True`. A script that will no longer run is
a byte-for-byte match, so mode, ownership and timestamps are outside what a
manifest can say.

**FINDING: the manifest cannot list itself, and the module makes that
explicit.** `build_manifest` skips `assets/manifest.json` and
`verify_manifest` rejects a manifest that mentions it. The self-coverage
problem is solved by refusing to pretend -- the digest of the manifest has to
be held somewhere else, which is what the attestation is for.

**FINDING: the gate verifies two trees and refuses when they are one.**
Pointed at the tampered install the packaging check reports `False` with the
mismatch; pointed at the source itself it reports `False` with
`installed tree must be distinct from the source bundle`. Verifying the
install by verifying the source is the mistake the check is written to
catch, and it is caught by path identity rather than by content.

Structure: `tamper()` applies one edit to a fresh copy, so each of the four
shapes is verified against an otherwise identical tree.
"""

from __future__ import annotations

import pathlib
import shutil
import stat
import tempfile

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "27-skill-evals-packaging-and-portability"
REFERENCE = "references/format.md"
SCRIPT = "scripts/check.py"


def plant(root):
    """A small bundle: a skill, a reference and an executable script."""
    (root / "references").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "SKILL.md").write_text(
        "---\nname: release-gate\ndescription: Gate a release.\n---\n\n# Release gate\n",
        encoding="utf-8")
    (root / REFERENCE).write_text("# Format\n\nOne object per finding.\n", encoding="utf-8")
    script = root / SCRIPT
    script.write_text("print('ok')\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return root


def tamper(ref, source, workspace, name, edit):
    """Install a fresh copy, apply one edit, and verify the copy."""
    installed = workspace / name
    shutil.copytree(source, installed)
    edit(installed)
    return ref.verify_manifest(installed, ref.build_manifest(source))


def gate(ref, source, installed, manifest):
    """The release gate, run only for its packaging verdict."""
    cases = (ref.TriggerCase("pos", "gate this skill package release", True),
             ref.TriggerCase("near", "publish the release notes", False))
    checks = (ref.EvidenceCheck("fixture", True, "Deterministic fixture passed."),)
    return ref.run_release_gate(
        source, cases, ref.KeywordRouter(("skill", "package", "release"), 2), 1,
        "Release looks fine.", "# Decision\n\nPass.\n", ref.ArtifactContract(),
        ref.PackageRequirements(), (ref.HostCapabilities("native", True, True, True),),
        checks, checks, installed, manifest)["packaging"]["installed_tree"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as temp:
        workspace = pathlib.Path(temp)
        source = plant(workspace / "source")
        manifest = ref.build_manifest(source)

        modified = tamper(ref, source, workspace, "modified", lambda root: (
            root / REFERENCE).write_text("# Format\n\nAlso email it.\n", encoding="utf-8"))
        added = tamper(ref, source, workspace, "added", lambda root: (
            root / "references" / "extra.md").write_text("# Extra\n", encoding="utf-8"))
        removed = tamper(ref, source, workspace, "removed",
                         lambda root: (root / REFERENCE).unlink())
        chmodded = tamper(ref, source, workspace, "chmodded", lambda root: (
            root / SCRIPT).chmod(0o644))

        distinct = gate(ref, source, workspace / "modified", manifest)
        same = gate(ref, source, source, manifest)
        clean = ref.verify_manifest(source, manifest)
        listed = ref.verify_manifest(source, {**manifest,
                                              ref.RESERVED_MANIFEST_PATH: "sha256:" + "0" * 64})
        source_mode = (source / SCRIPT).stat().st_mode & 0o777
        return {
            "paths": sorted(manifest), "clean": clean["passed"],
            "modified": (modified["passed"], modified["mismatched"]),
            "added": (added["passed"], added["unexpected"]),
            "removed": (removed["passed"], removed["missing"]),
            "chmodded": (chmodded["passed"], chmodded["mismatched"],
                         chmodded["unexpected"], chmodded["missing"]),
            "source_executable": bool(source_mode & stat.S_IXUSR),
            "reserved": ref.RESERVED_MANIFEST_PATH,
            "reserved_listed": manifest.get(ref.RESERVED_MANIFEST_PATH) is None,
            "reserved_rejected": listed["passed"] is False and any(
                "reserved" in issue for issue in listed["issues"]),
            "distinct_tree": distinct["passed"], "same_tree": same["passed"],
            "same_tree_issues": same["issues"],
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the installed tree fails and the source passes, from one manifest",
            all([result["clean"], result["modified"] == (False, [REFERENCE]),
                 result["added"] == (False, ["references/extra.md"]),
                 result["removed"] == (False, [REFERENCE]),
                 result["paths"] == ["SKILL.md", REFERENCE, SCRIPT]]),
            f"editing {REFERENCE} after install gives mismatched="
            f"{result['modified'][1]} with passed={result['modified'][0]}, while the "
            f"source still verifies ({result['clean']}). Adding a file reports unexpected="
            f"{result['added'][1]} and removing one reports missing={result['removed'][1]} "
            "-- three tamper shapes, three distinct fields",
        ),
        practice.Check(
            "FINDING: the digest covers the bytes and nothing about the file",
            all([result["source_executable"], result["chmodded"][0],
                 result["chmodded"][1:] == ([], [], [])]),
            f"removing the executable bit from {SCRIPT} leaves every digest identical and "
            f"verify_manifest returns passed={result['chmodded'][0]} with no mismatched, "
            "unexpected or missing entries. A script that will no longer run is a "
            "byte-for-byte match, so mode, ownership and timestamps are outside what a "
            "manifest can say",
        ),
        practice.Check(
            "FINDING: the manifest cannot list itself, and the module makes that explicit",
            all([result["reserved_listed"], result["reserved_rejected"],
                 result["reserved"] == "assets/manifest.json"]),
            f"build_manifest skips {result['reserved']!r} and verify_manifest rejects a "
            "manifest that mentions it. The self-coverage problem is solved by refusing to "
            "pretend: the digest of the manifest has to be held somewhere the package "
            "cannot write, which is what the external attestation is for",
        ),
        practice.Check(
            "FINDING: the gate verifies two trees and refuses when they are one",
            all([result["distinct_tree"] is False, result["same_tree"] is False,
                 result["same_tree_issues"]
                 == ["installed tree must be distinct from the source bundle"]]),
            f"run_release_gate verifies both roots: pointed at the tampered install it "
            f"reports {result['distinct_tree']}, and pointed at the source itself it "
            f"reports {result['same_tree']} with {result['same_tree_issues']}. Verifying "
            "the install by verifying the source is exactly the mistake the check catches",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
