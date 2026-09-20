"""Exercise 1 — deleting the workspace root is inside the workspace jail.

    Add separate read, create, overwrite, and delete path permissions. Test
    the same path under every operation.

Reading of the exercise: "the same path under every operation" is the test
that makes the split worth doing, so one target is run through all four and
the four verdicts are read as a row. Splitting `write` then forces the
question the shipped review never asks -- whether the file is already there
-- and asking it turns up the jail check that lets the root itself through.

**ANSWER: four permissions, one path, and the answer moves with both the
operation and the file.** With `reports/summary.json` on disk the row is
`allow, require-approval, require-approval, deny`; with it absent,
`allow, allow, allow, deny`. The shipped policy answers
`allow, require-approval, require-approval, require-approval` either way --
**2** distinct answers to **4** questions, and the same **2** whatever is on
disk.

**FINDING: `write` cannot distinguish create from overwrite, because the
review never looks.** `normalize_workspace_path` resolves with
`strict=False`, so the same request is byte-identical whether the file
exists or not -- **1** verdict for both. Splitting them requires a `stat`,
which makes the verdict a claim about a file rather than about a path, and
therefore stale the moment it is returned.

**FINDING: deleting the workspace root is inside the workspace jail.**
`normalize_workspace_path(root, ".")` returns the root, and the containment
test is `resolved != root and root not in resolved.parents` -- an equal path
passes both halves. The shipped review answers `require-approval` for
deleting the entire workspace, which is the one delete a policy should never
put in front of a human.

**FINDING: the four operations need four path sets, not one root.** Read
over `**`, create under `reports/`, overwrite nowhere, delete nowhere is a
policy `SandboxPolicy` cannot hold: it has **1** path field and **1** kind
allowlist, so "may read anything, may create in one directory" has to be
written as code.

Structure: `PathPolicy` holds one glob set per operation and `review()`
consults the filesystem, which is the difference the exercise is asking for.
"""

from __future__ import annotations

import fnmatch
import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "26-skill-permissions-sandboxes-and-trust"
OPERATIONS = ("read", "create", "overwrite", "delete")
TARGET = "reports/summary.json"
GRANTS = {"read": ("**",), "create": ("reports/*",),
          "overwrite": ("reports/*",), "delete": ()}
ASK = {"overwrite"}


def matches(relative, patterns):
    return any(fnmatch.fnmatch(relative, pattern) or pattern == "**"
               for pattern in patterns)


def review(ref, policy, operation, target):
    """One verdict per operation, decided against the file rather than the path."""
    resolved = ref.normalize_workspace_path(policy.workspace_root, target)
    root = policy.workspace_root.resolve()
    if resolved == root:
        return "deny", "the workspace root is not an operand"
    relative = str(resolved.relative_to(root))
    exists = resolved.exists()
    if operation == "create" and exists:
        operation = "overwrite"
    if operation == "overwrite" and not exists:
        operation = "create"
    if not matches(relative, GRANTS[operation]):
        return "deny", f"{operation} is not granted for {relative}"
    return ("require-approval" if operation in ASK else "allow"), f"{operation} granted"


def shipped(ref, policy, operation, target):
    kind = {"read": "read", "create": "write", "overwrite": "write",
            "delete": "delete"}[operation]
    return ref.review_action(policy, ref.ActionRequest(kind, target=target)).verdict.value


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as temp:
        workspace = pathlib.Path(temp) / "workspace"
        (workspace / "reports").mkdir(parents=True)
        policy = ref.SandboxPolicy(workspace_root=workspace,
                                   allowed_kinds=("read", "write", "delete"))
        absent = [review(ref, policy, operation, TARGET)[0] for operation in OPERATIONS]
        shipped_absent = [shipped(ref, policy, op, TARGET) for op in OPERATIONS]
        (workspace / TARGET).write_text("{}", encoding="utf-8")
        present = [review(ref, policy, operation, TARGET)[0] for operation in OPERATIONS]
        shipped_present = [shipped(ref, policy, op, TARGET) for op in OPERATIONS]

        root_delete = ref.review_action(policy, ref.ActionRequest("delete", target="."))
        root_ours = review(ref, policy, "delete", ".")
        jailed = ref.normalize_workspace_path(workspace, ".") == workspace.resolve()
        fields = list(vars(ref.SandboxPolicy)["__dataclass_fields__"])
        return {
            "operations": list(OPERATIONS), "present": present, "absent": absent,
            "shipped_present": shipped_present, "shipped_absent": shipped_absent,
            "shipped_distinct": len(set(shipped_present)), "distinct": len(set(present)),
            "shipped_stable": shipped_present == shipped_absent,
            "root_verdict": root_delete.verdict.value, "root_rule": root_delete.rule,
            "root_ours": root_ours, "root_jailed": jailed,
            "root_normalized": root_delete.normalized_target == str(workspace.resolve()),
            "policy_fields": fields,
            "path_fields": [name for name in fields if "path" in name or "root" in name],
            "grants": {name: len(patterns) for name, patterns in GRANTS.items()},
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: four permissions, one path, and the answer moves with the file too",
            all([result["present"] == ["allow", "require-approval", "require-approval",
                                       "deny"],
                 result["absent"] == ["allow", "allow", "allow", "deny"],
                 result["distinct"] == 3, result["shipped_distinct"] == 2]),
            f"the same target answers {result['present']} for "
            f"{result['operations']} once the file exists and {result['absent']} before it "
            f"does, while the shipped policy gives {result['shipped_present']} -- "
            f"{result['shipped_distinct']} distinct answers to 4 questions",
        ),
        practice.Check(
            "FINDING: write cannot distinguish create from overwrite, because it never looks",
            all([result["shipped_stable"], result["present"] != result["absent"],
                 result["shipped_present"][1] == result["shipped_present"][2]]),
            f"normalize_workspace_path resolves with strict=False, so the shipped verdicts "
            f"are {result['shipped_present']} whether the file exists or not. Splitting the "
            f"two needs a stat, and our verdicts do move ({result['absent']} to "
            f"{result['present']}) -- which makes the answer a claim about a file rather "
            "than about a path, and stale the moment it is returned",
        ),
        practice.Check(
            "FINDING: deleting the workspace root is inside the workspace jail",
            all([result["root_jailed"], result["root_verdict"] == "require-approval",
                 result["root_rule"] == "approval-gate", result["root_normalized"],
                 result["root_ours"][0] == "deny"]),
            f"normalize_workspace_path(root, '.') returns the root and the containment test "
            f"is 'resolved != root and root not in resolved.parents', which an equal path "
            f"passes. The shipped review answers {result['root_verdict']!r} under "
            f"{result['root_rule']!r} for deleting the entire workspace; ours answers "
            f"{result['root_ours'][0]!r} because {result['root_ours'][1]}",
        ),
        practice.Check(
            "FINDING: the four operations need four path sets, not one root",
            all([result["path_fields"] == ["workspace_root"],
                 len(result["policy_fields"]) == 6,
                 result["grants"] == {"read": 1, "create": 1, "overwrite": 1, "delete": 0}]),
            f"SandboxPolicy carries {result['policy_fields']}, of which "
            f"{result['path_fields']} is the only path. 'Read anything, create and overwrite "
            f"under reports/, never delete' is {result['grants']} -- four sets, "
            "and the policy object has one root and one kind allowlist to hold them",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
