"""Exercise 7 — nothing in the module knows what a version is.

    Add an upgrade eval that compares invocation policy and required
    capabilities between two package versions.

Reading of the exercise: an upgrade eval is a diff with a direction, so each
delta is classified as widening, narrowing or neutral rather than merely
listed -- widening is what needs review, and narrowing is what breaks hosts
that were relying on the old answer. Staging the comparison at all turns out
to be the work, because the module has no version anywhere and every input to
the gate has to be held twice.

**ANSWER: five deltas, three widening, one narrowing, one neutral.**
`script_execution` false to true and `+user-invocable` widen what the host
must provide; `allowed-tools` read to read,write widens the authority the
skill asks for; `-network-access` narrows it. Widening is the review gate:
**3** of **5** deltas need a human, and **1** needs a migration note.

**FINDING: nothing in the module knows what a version is.** **0** of the
**8** dataclasses carries a version field, so "between two versions" means
holding two `PackageRequirements`, two frontmatter dicts and two manifests
side by side. The upgrade eval is not an extension of the gate; it is a
second harness that runs the gate twice.

**FINDING: narrowing is a break that the portability matrix reports as an
improvement.** Dropping `network-access` moves `no-network-host` from
`adapter-required` to `native`, so the matrix looks strictly better while a
skill that used the extension has silently stopped asking for it. A
capability diff and a compatibility diff point in opposite directions here.

**FINDING: the manifest cannot say what changed.** Between the two versions
**3** paths differ by digest and **1** appears only in v2, which is exactly
what a rename plus an edit looks like. The manifest answers "is this the
bytes I signed" and never "what is different about this release", so the
upgrade eval has to read fields rather than digests.

Structure: `classify_deltas()` is the eval, and `matrix_for()` runs the
shipped portability matrix on each version so the two answers can disagree.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "27-skill-evals-packaging-and-portability"
V1_TOOLS, V2_TOOLS = "read", "read, write"
V1_EXTENSIONS = ("allowed-tools", "network-access")
V2_EXTENSIONS = ("allowed-tools", "user-invocable")


def requirements_for(ref, version):
    if version == 1:
        return ref.PackageRequirements(companion_files=True, script_execution=False,
                                       runtime_extensions=V1_EXTENSIONS)
    return ref.PackageRequirements(companion_files=True, script_execution=True,
                                   runtime_extensions=V2_EXTENSIONS)


def hosts_for(ref):
    return (ref.HostCapabilities("modern-host", True, True, True,
                                 ("allowed-tools", "user-invocable", "network-access")),
            ref.HostCapabilities("no-network-host", True, True, True,
                                 ("allowed-tools", "user-invocable")),
            ref.HostCapabilities("prompt-only-host", False, False, False))


def tool_delta(before, after):
    gained = sorted(set(after.split(", ")) - set(before.split(", ")))
    lost = sorted(set(before.split(", ")) - set(after.split(", ")))
    if gained:
        return "widening", f"allowed-tools gains {gained}"
    if lost:
        return "narrowing", f"allowed-tools loses {lost}"
    return "neutral", "allowed-tools unchanged"


def classify_deltas(before, after):
    """Every difference with a direction: widening needs review, narrowing needs a note."""
    deltas = {}
    for field in ("companion_files", "script_execution"):
        was, now = getattr(before, field), getattr(after, field)
        deltas[field] = ("widening" if now and not was
                         else "narrowing" if was and not now else "neutral")
    for name in sorted(set(after.runtime_extensions) - set(before.runtime_extensions)):
        deltas[f"+{name}"] = "widening"
    for name in sorted(set(before.runtime_extensions) - set(after.runtime_extensions)):
        deltas[f"-{name}"] = "narrowing"
    direction, _ = tool_delta(V1_TOOLS, V2_TOOLS)
    deltas["allowed-tools"] = direction
    return deltas


def matrix_for(ref, version):
    return {row["host"]: row["status"]
            for row in ref.portability_matrix(requirements_for(ref, version), hosts_for(ref))}


def dataclass_names(ref):
    return [name for name in dir(ref)
            if hasattr(getattr(ref, name), "__dataclass_fields__")
            and not name.startswith("_")]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    before, after = requirements_for(ref, 1), requirements_for(ref, 2)
    deltas = classify_deltas(before, after)
    v1, v2 = matrix_for(ref, 1), matrix_for(ref, 2)
    versioned = [name for name in dataclass_names(ref)
                 if any("version" in field for field
                        in vars(getattr(ref, name))["__dataclass_fields__"])]
    manifest_v1 = {"SKILL.md": "sha256:" + "a" * 64, "references/format.md": "sha256:" + "b" * 64,
                   "scripts/run.py": "sha256:" + "c" * 64}
    manifest_v2 = {"SKILL.md": "sha256:" + "d" * 64, "references/format.md": "sha256:" + "e" * 64,
                   "scripts/run.py": "sha256:" + "f" * 64,
                   "references/tools.md": "sha256:" + "0" * 64}
    return {
        "deltas": deltas,
        "widening": sorted(name for name, kind in deltas.items() if kind == "widening"),
        "narrowing": sorted(name for name, kind in deltas.items() if kind == "narrowing"),
        "neutral": sorted(name for name, kind in deltas.items() if kind == "neutral"),
        "tool_reason": tool_delta(V1_TOOLS, V2_TOOLS)[1],
        "v1_matrix": v1, "v2_matrix": v2,
        "quiet_before": v1["no-network-host"], "quiet_after": v2["no-network-host"],
        "modern_before": v1["modern-host"], "modern_after": v2["modern-host"],
        "dataclasses": len(dataclass_names(ref)), "versioned": versioned,
        "digest_changes": sorted(path for path in manifest_v1
                                 if manifest_v1[path] != manifest_v2.get(path)),
        "added_paths": sorted(set(manifest_v2) - set(manifest_v1)),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: five deltas, three widening, one narrowing, one neutral",
            all([len(result["deltas"]) == 5,
                 result["widening"] == ["+user-invocable", "allowed-tools",
                                        "script_execution"],
                 result["narrowing"] == ["-network-access"],
                 result["neutral"] == ["companion_files"]]),
            f"the eval reports {result['deltas']}. {len(result['widening'])} widen what "
            f"the host must provide or what the skill may do -- "
            f"{result['tool_reason']} -- and {len(result['narrowing'])} narrows it. "
            "Widening is the review gate; narrowing is the migration note",
        ),
        practice.Check(
            "FINDING: nothing in the module knows what a version is",
            all([result["versioned"] == [], result["dataclasses"] >= 8]),
            f"{len(result['versioned'])} of the {result['dataclasses']} dataclasses "
            "carries a version field, so 'between two versions' means holding two "
            "PackageRequirements, two frontmatter dicts and two manifests side by side. "
            "The upgrade eval is not an extension of the gate; it is a second harness that "
            "runs the gate twice",
        ),
        practice.Check(
            "FINDING: narrowing is a break the portability matrix reports as an improvement",
            all([result["quiet_before"] == "adapter-required",
                 result["quiet_after"] == "native",
                 result["modern_before"] == "native",
                 result["modern_after"] == "native",
                 result["narrowing"] == ["-network-access"]]),
            f"dropping network-access moves no-network-host from {result['quiet_before']!r} to "
            f"{result['quiet_after']!r}, so the matrix reads strictly better while a skill "
            "that used the extension has silently stopped asking for it. The capability "
            "diff and the compatibility diff point in opposite directions",
        ),
        practice.Check(
            "FINDING: the manifest cannot say what changed",
            all([len(result["digest_changes"]) == 3, result["added_paths"]
                 == ["references/tools.md"]]),
            f"{len(result['digest_changes'])} paths differ by digest "
            f"({result['digest_changes']}) and {result['added_paths']} appears only in v2 "
            "-- exactly what a rename plus an edit looks like. The manifest answers 'is "
            "this the bytes I signed' and never 'what is different about this release'",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
