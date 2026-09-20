"""Exercise 5 — the substitute is tempting because it is a superset.

    Design a failure response for a skill that references an unavailable MCP
    tool. Do not silently substitute a tool with broader permissions.

Reading of the exercise: "do not silently substitute" names the wrong answer,
so the right one is built beside it and the two are compared on what the
caller learns rather than on whether the work completes. The substitution is
attractive precisely because the broader tool *can* do the job -- it is a
superset -- and that is what makes the refusal a policy rather than a
limitation.

**ANSWER: refuse with the tool named, the reason, and no fallback.** A skill
declaring `allowed-tools: notes_search notes_export` on a host offering only
`notes_search` answers `unavailable-tool` naming `notes_export`, the
**1** tool that is missing, and performs **0** work. The response says which
skill, which tool and which host, so the caller can fix exactly one thing.

**FINDING: the substitute is a strict superset, which is the whole problem.**
`files_write` covers everything `notes_export` does and **3** capabilities
more. Substituting completes the task, so a test that checks the output
passes -- and the skill has been granted write access to the filesystem by a
resolver, not by its `allowed-tools`. The declaration becomes advisory at the
moment the fallback is written.

**FINDING: the validator checks the field's shape and never its contents.**
`allowed-tools` is required only to be a non-empty string, so a skill naming
a tool that has never existed reports `valid=True`. Availability is a
runtime fact about a host and validity is a static fact about a file -- the
report cannot carry the first, which is why the failure has to be a response
and not an issue code.

**FINDING: refusing needs the same parse the substitution would have used.**
Both paths split the declaration into **2** names and diff it against the
host's **1**; they differ only in what they do with the difference. So the
cost of refusing is zero and the choice is entirely a policy one -- which is
why it has to be stated somewhere a reviewer can see.

Structure: `resolve` takes a policy -- refuse or substitute -- so both paths
run the same lookup, and `Capability` records what each tool can reach.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "22-skills-and-agent-sdks"
NAME = "incident-report"
DECLARED = "notes_search notes_export"
CAPABILITIES = {
    "notes_search": {"notes:read"},
    "notes_export": {"notes:read", "notes:write"},
    "files_write": {"notes:read", "notes:write", "fs:write", "fs:delete", "net:post"},
}
HOST = {"notes_search", "files_write"}
SUBSTITUTES = {"notes_export": "files_write"}


def skill_text(allowed=DECLARED):
    return "\n".join(["---", f"name: {NAME}",
                      "description: Use when writing an incident report.",
                      f"allowed-tools: {allowed}", "---", "",
                      "Search the notes, then export the report.\n"])


def resolve(declared, host, *, substitute=False):
    """Bind declared tools to host tools, refusing or substituting the gaps."""
    wanted = declared.split()
    missing = [tool for tool in wanted if tool not in host]
    if not missing:
        return {"status": "ok", "bound": {tool: tool for tool in wanted},
                "performed": True}
    if not substitute:
        return {"status": "unavailable-tool", "skill": NAME, "missing": missing,
                "host": sorted(host), "performed": False,
                "detail": f"{NAME} requires {missing[0]!r}, which this host does not offer"}
    bound = {tool: SUBSTITUTES.get(tool, tool) if tool in missing else tool
             for tool in wanted}
    return {"status": "ok", "bound": bound, "performed": True}


def granted(bound):
    return set().union(*(CAPABILITIES[tool] for tool in bound.values()))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    report = ref.validate_skill_text(skill_text(), NAME)
    ghost = ref.validate_skill_text(skill_text("a_tool_that_never_existed"), NAME)
    refused = resolve(DECLARED, HOST)
    substituted = resolve(DECLARED, HOST, substitute=True)
    complete_host = resolve(DECLARED, {"notes_search", "notes_export"})
    return {
        "valid": report.valid, "issues": [i.code for i in report.issues],
        "ghost_valid": ghost.valid, "ghost_issues": [i.code for i in ghost.issues],
        "status": refused["status"], "missing": refused["missing"],
        "performed": refused["performed"], "detail": refused["detail"],
        "names_host": refused["host"] == sorted(HOST),
        "substituted_status": substituted["status"],
        "substituted_performed": substituted["performed"],
        "substituted_bound": substituted["bound"],
        "declared_grant": sorted(granted(complete_host["bound"])),
        "substituted_grant": sorted(granted(substituted["bound"])),
        "superset": granted(substituted["bound"]) > granted(complete_host["bound"]),
        "extra": sorted(granted(substituted["bound"]) - granted(complete_host["bound"])),
        "declared_count": len(DECLARED.split()), "host_count": len(HOST),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: refuse with the tool named, the reason, and no fallback",
            all([result["status"] == "unavailable-tool",
                 result["missing"] == ["notes_export"], not result["performed"],
                 result["names_host"], NAME in result["detail"]]),
            f"the skill answers {result['status']!r} naming {result['missing']} -- the "
            f"{len(result['missing'])} tool the host lacks -- and performs no work "
            f"({result['performed']}). The detail is {result['detail']!r}, so the caller can "
            "fix exactly one thing",
        ),
        practice.Check(
            "FINDING: the substitute is a strict superset, which is the whole problem",
            all([result["substituted_status"] == "ok",
                 result["substituted_performed"], result["superset"],
                 result["extra"] == ["fs:delete", "fs:write", "net:post"]]),
            f"substituting binds {result['substituted_bound']} and completes the task, so an "
            f"output test passes -- while the grant goes from {result['declared_grant']} to "
            f"{result['substituted_grant']}, adding {result['extra']}. The skill has been "
            "given filesystem write by a resolver rather than by its allowed-tools",
        ),
        practice.Check(
            "FINDING: the validator checks the field's shape and never its contents",
            all([result["valid"], result["issues"] == [],
                 result["ghost_valid"], result["ghost_issues"] == []]),
            f"a skill naming a tool that has never existed reports "
            f"valid={result['ghost_valid']} with {result['ghost_issues']} issues, because "
            "allowed-tools is required only to be a non-empty string. Availability is a "
            "runtime fact about a host and validity a static fact about a file, so the "
            "failure has to be a response and not an issue code",
        ),
        practice.Check(
            "FINDING: refusing needs the same parse the substitution would have used",
            all([result["declared_count"] == 2, result["host_count"] == 2,
                 result["missing"] == ["notes_export"]]),
            f"both paths split the declaration into {result['declared_count']} names and "
            f"diff it against the host's {result['host_count']}, differing only in what they "
            "do with the difference. The cost of refusing is zero, so the choice is entirely "
            "a policy one -- which is why it has to be stated where a reviewer can see it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
