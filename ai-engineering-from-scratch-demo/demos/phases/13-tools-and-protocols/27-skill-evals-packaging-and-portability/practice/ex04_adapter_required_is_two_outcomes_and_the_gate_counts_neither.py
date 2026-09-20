"""Exercise 4 — adapter-required is two outcomes, and the gate counts neither.

    Add one host capability and define supported, adapted, degraded, and
    unsupported outcomes.

Reading of the exercise: four outcome names against a matrix that returns
three is the whole exercise, so the new capability is chosen to land in the
gap -- tool enforcement is something a host either does or does not do, and
no adapter can supply it. That separates "works differently here" from "works
less here", which is the distinction `adapter-required` currently hides.

**ANSWER: one capability, four outcomes, and the rule is whether the gap can
be emulated.** `enforces_allowed_tools` is added to `HostCapabilities` and
`requires_tool_enforcement` to `PackageRequirements`. Five hosts come out
`supported`, `adapted`, `degraded`, `degraded`, `unsupported` -- **4**
outcomes against the shipped matrix's **3**.

**FINDING: `adapter-required` is two outcomes, and the new gap reads as
`native`.** A host that cannot preserve companion files can be served by
inlining them; a host that cannot run scripts cannot. Both answer
`adapter-required`. And the host that does not enforce `allowed-tools`
answers `native`, because a capability the matrix does not ask about cannot
be missing -- adding the field is what makes the gap exist.

**FINDING: capabilities are hard-coded fields and extensions are data.**
`HostCapabilities` has **5** fields, **4** of them fixed booleans and **1** a
tuple of extension names. Adding a runtime extension is a string; adding a
capability edits two dataclasses and the matrix function. The extensibility
the design has is on the axis that needed it least.

**FINDING: the release gate counts native hosts, so a degraded host is
invisible.** `min_native_hosts` counts `status == "native"` and nothing
counts the rest, so shipping to a host that silently drops tool enforcement
passes a gate that never mentions it. Degraded is the status that most needs
a threshold and is the one with no field to hold it.

Structure: `classify()` is the four-way rule, written over the shipped
`missing` list so it adds a distinction rather than a second matrix.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "27-skill-evals-packaging-and-portability"
EMULABLE = {"companion-files"}
UNEMULABLE = {"script-execution", "tool-enforcement"}


def build(ref):
    class Capabilities(ref.HostCapabilities):
        """The shipped record plus the capability no adapter can supply."""

        def __init__(self, *args, enforces_allowed_tools=False, **kwargs):
            super().__init__(*args, **kwargs)
            object.__setattr__(self, "enforces_allowed_tools", enforces_allowed_tools)

    return Capabilities


def missing_for(ref, requirements, host, tool_enforcement):
    row = ref.portability_matrix(requirements, (host,))[0]
    gaps = list(row["missing"])
    if tool_enforcement and row["status"] != "unsupported" \
            and not getattr(host, "enforces_allowed_tools", False):
        gaps.append("tool-enforcement")
    return row, gaps


def classify(row, gaps):
    """Four outcomes: the question is whether every gap can be emulated."""
    if row["status"] == "unsupported":
        return "unsupported"
    if not gaps:
        return "supported"
    if set(gaps) <= EMULABLE:
        return "adapted"
    return "degraded"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    Capabilities = build(ref)
    hosts = (
        Capabilities("full-host", True, True, True, enforces_allowed_tools=True),
        Capabilities("inlining-host", True, False, True, enforces_allowed_tools=True),
        Capabilities("no-scripts-host", True, True, False, enforces_allowed_tools=True),
        Capabilities("unenforced-host", True, True, True, enforces_allowed_tools=False),
        Capabilities("prompt-only-host", False, False, False),
    )
    requirements = ref.PackageRequirements(companion_files=True, script_execution=True)
    rows = {}
    for host in hosts:
        row, gaps = missing_for(ref, requirements, host, tool_enforcement=True)
        rows[host.name] = {"shipped": row["status"], "gaps": gaps,
                           "status": classify(row, gaps)}
    shipped_statuses = [row["shipped"] for row in rows.values()]
    ours = [row["status"] for row in rows.values()]
    native = sum(status == "native" for status in shipped_statuses)
    return {
        "hosts": list(rows), "shipped": shipped_statuses, "ours": ours,
        "shipped_distinct": len(set(shipped_statuses)), "distinct": len(set(ours)),
        "adapted_gaps": rows["inlining-host"]["gaps"],
        "degraded_gaps": rows["no-scripts-host"]["gaps"],
        "unseen_gaps": rows["unenforced-host"]["gaps"],
        "unseen_shipped": rows["unenforced-host"]["shipped"],
        "same_shipped": rows["inlining-host"]["shipped"] == rows["no-scripts-host"]["shipped"],
        "host_fields": list(vars(ref.HostCapabilities)["__dataclass_fields__"]),
        "requirement_fields": list(vars(ref.PackageRequirements)["__dataclass_fields__"]),
        "extension_field": "supported_extensions",
        "native": native, "min_native": ref.ReleaseThresholds().min_native_hosts,
        "gate_passes": native >= ref.ReleaseThresholds().min_native_hosts,
        "degraded_count": sum(status == "degraded" for status in ours),
        "threshold_fields": list(vars(ref.ReleaseThresholds)["__dataclass_fields__"]),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one capability, four outcomes, and the rule is whether the gap emulates",
            all([result["ours"] == ["supported", "adapted", "degraded", "degraded",
                                    "unsupported"],
                 result["distinct"] == 4, result["shipped_distinct"] == 3,
                 result["hosts"] == ["full-host", "inlining-host", "no-scripts-host",
                                     "unenforced-host", "prompt-only-host"]]),
            f"the four hosts come out {result['ours']} where the shipped matrix gives "
            f"{result['shipped']} -- {result['distinct']} outcomes against "
            f"{result['shipped_distinct']}. The rule is whether every gap can be emulated: "
            "companion files can be inlined, tool enforcement cannot be supplied",
        ),
        practice.Check(
            "FINDING: adapter-required is two outcomes, and the new gap reads as native",
            all([result["same_shipped"], result["adapted_gaps"] == ["companion-files"],
                 result["degraded_gaps"] == ["script-execution"],
                 result["ours"][1] != result["ours"][2],
                 result["unseen_shipped"] == "native",
                 result["unseen_gaps"] == ["tool-enforcement"]]),
            f"the inlining host is missing {result['adapted_gaps']} and the no-scripts host "
            f"{result['degraded_gaps']}; both answer {result['shipped'][1]!r} though one "
            f"still does everything the skill claims and the other does not. The host "
            f"missing {result['unseen_gaps']} answers {result['unseen_shipped']!r}, because "
            "a capability the matrix does not ask about cannot be missing",
        ),
        practice.Check(
            "FINDING: capabilities are hard-coded fields and extensions are data",
            all([len(result["host_fields"]) == 5,
                 result["extension_field"] in result["host_fields"],
                 len(result["requirement_fields"]) == 3]),
            f"HostCapabilities carries {result['host_fields']} -- four fixed booleans and "
            f"one tuple -- against PackageRequirements' {result['requirement_fields']}. "
            "Adding a runtime extension is a string; adding a capability edits two "
            "dataclasses and the matrix function, so the extensible axis is the one that "
            "needed it least",
        ),
        practice.Check(
            "FINDING: the release gate counts native hosts, so a degraded host is invisible",
            all([result["native"] == 2, result["gate_passes"],
                 result["degraded_count"] == 2,
                 not any("degraded" in name for name in result["threshold_fields"])]),
            f"min_native_hosts is {result['min_native']} and {result['native']} hosts are "
            f"native, so the gate passes with {result['degraded_count']} degraded hosts "
            f"-- one of which silently drops tool enforcement. ReleaseThresholds carries "
            f"{result['threshold_fields']} and none of them counts a degraded host -- the "
            "status that most needs a threshold has no field to hold one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
