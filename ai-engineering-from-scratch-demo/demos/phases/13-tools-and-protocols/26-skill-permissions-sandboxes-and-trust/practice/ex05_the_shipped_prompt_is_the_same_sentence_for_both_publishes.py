"""Exercise 5 — the shipped prompt is the same sentence for both publishes.

    Write an approval message for a staging publish, then for a production
    publish. Make the target, artifact, and rollback consequence explicit.

Reading of the exercise: "make X explicit" is a requirement on a renderer,
not on two strings, so the message is generated from named facts and the
renderer refuses to produce anything when one is missing. That turns
"explicit" into something a test can fail. Comparing the two outputs then
shows that the only interesting difference is reversibility, which is the
one fact the module has nowhere to put.

**ANSWER: one renderer, four required facts, and two messages that differ
only where the deployment differs.** Both name the target host, the artifact
and its digest, and what rollback costs; the production message adds that
the version is immutable once published and that consumers may already have
fetched it. Omitting any required fact raises rather than rendering -- **4**
of **4** omissions refused.

**FINDING: the shipped prompt is the same sentence for both publishes.**
Two requests differing only in their URL both return `host policy gates
'network' behind approval` under rule `approval-gate`. The reason names the
policy that fired; an approver needs the consequence, and the review has no
way to know one publish is reversible and the other is not.

**FINDING: `ReviewDecision` has no field for the prompt.** Its **7** fields
carry a verdict, a rule and a reason, so the approval text is generated
somewhere else from something else. Nothing structurally guarantees that
what the approver reads and what the reviewer decided describe the same
action.

**FINDING: reversibility is the axis and it is not in the model.**
`approval_kinds` is keyed by kind, so a staging publish and a production
publish are both `network` and get the identical gate; neither
`ActionRequest` nor `SandboxPolicy` has an environment or a reversibility
field. "Approval should follow consequence" needs a fact the request does
not carry.

Structure: `render()` is the renderer with its required-fact check, so
"explicit" is enforced by a raise rather than by proofreading.
"""

from __future__ import annotations

import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "26-skill-permissions-sandboxes-and-trust"
REQUIRED = ("target", "artifact", "digest", "rollback")
STAGING = {"target": "https://registry.example.test:8443/staging",
           "artifact": "report-tool 1.4.0", "digest": "sha256:9f2c…41ab",
           "rollback": "redeploy 1.3.9 over it; nothing outside staging has seen 1.4.0",
           "irreversible": ""}
PRODUCTION = {"target": "https://registry.example.test/production",
              "artifact": "report-tool 1.4.0", "digest": "sha256:9f2c…41ab",
              "rollback": "publish 1.4.1; 1.4.0 stays fetchable forever",
              "irreversible": " This version is immutable once published and consumers "
                              "may have fetched it before you notice."}


def render(facts):
    """Refuse to render rather than render something vague."""
    missing = [name for name in REQUIRED if not facts.get(name)]
    if missing:
        raise ValueError(f"approval message needs {missing}")
    return (f"Publish {facts['artifact']} ({facts['digest']}) to {facts['target']}. "
            f"To undo: {facts['rollback']}.{facts['irreversible']}")


def omissions(facts):
    """Every single-fact omission, and whether the renderer refused it."""
    refused = []
    for name in REQUIRED:
        try:
            render({**facts, name: ""})
        except ValueError:
            refused.append(name)
    return refused


def shipped_reason(ref, policy, url):
    decision = ref.review_action(policy, ref.ActionRequest(
        "network", url=url, payload='{"digest": "sha256:9f2c"}'))
    return decision.verdict.value, decision.rule, decision.reason


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    staging, production = render(STAGING), render(PRODUCTION)
    with tempfile.TemporaryDirectory() as temp:
        workspace = pathlib.Path(temp) / "workspace"
        workspace.mkdir(parents=True)
        policy = ref.SandboxPolicy(
            workspace_root=workspace, allowed_kinds=("network",),
            approval_kinds=("network",),
            network_allowlist=("https://registry.example.test",
                               "https://registry.example.test:8443"))
        staging_review = shipped_reason(ref, policy, STAGING["target"])
        production_review = shipped_reason(ref, policy, PRODUCTION["target"])
    differing = sorted(name for name in STAGING if STAGING[name] != PRODUCTION[name])
    return {
        "staging": staging, "production": production,
        "names_target": [STAGING["target"] in staging, PRODUCTION["target"] in production],
        "names_digest": [STAGING["digest"] in staging, PRODUCTION["digest"] in production],
        "names_rollback": [STAGING["rollback"] in staging,
                           PRODUCTION["rollback"] in production],
        "refused": omissions(STAGING), "required": list(REQUIRED),
        "differing_facts": differing,
        "immutable": "immutable" in production and "immutable" not in staging,
        "shipped_same": staging_review[2] == production_review[2],
        "shipped_reason": staging_review[2], "shipped_rule": staging_review[1],
        "shipped_verdict": staging_review[0],
        "decision_fields": list(vars(ref.ReviewDecision)["__dataclass_fields__"]),
        "prompt_fields": [name for name in vars(ref.ReviewDecision)["__dataclass_fields__"]
                          if "prompt" in name or "consequence" in name],
        "env_fields": [name for name in
                       list(vars(ref.ActionRequest)["__dataclass_fields__"])
                       + list(vars(ref.SandboxPolicy)["__dataclass_fields__"])
                       if "environ" in name or "revers" in name],
        "approval_kinds": list(policy.approval_kinds),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one renderer, four required facts, two messages differing where it matters",
            all([result["names_target"] == [True, True],
                 result["names_digest"] == [True, True],
                 result["names_rollback"] == [True, True],
                 result["refused"] == list(REQUIRED), result["immutable"],
                 result["differing_facts"] == ["irreversible", "rollback", "target"]]),
            f"both messages name the target, the artifact digest and what rollback costs, "
            f"and the two differ in {result['differing_facts']}: production adds that the "
            f"version is immutable and may already have been fetched. Omitting any of "
            f"{result['required']} raises -- {len(result['refused'])} of "
            f"{len(result['required'])} refused rather than rendered",
        ),
        practice.Check(
            "FINDING: the shipped prompt is the same sentence for both publishes",
            all([result["shipped_same"], result["shipped_verdict"] == "require-approval",
                 result["shipped_rule"] == "approval-gate",
                 "network" in result["shipped_reason"]]),
            f"two requests differing only in their URL both return "
            f"{result['shipped_reason']!r} under {result['shipped_rule']!r}. The reason "
            "names the policy that fired; an approver needs the consequence, and the "
            "review has no way to know one publish is reversible and the other is not",
        ),
        practice.Check(
            "FINDING: ReviewDecision has no field for the prompt",
            all([len(result["decision_fields"]) == 7, result["prompt_fields"] == [],
                 "reason" in result["decision_fields"]]),
            f"its {len(result['decision_fields'])} fields {result['decision_fields']} carry "
            f"a verdict, a rule and a reason, and {len(result['prompt_fields'])} of them is "
            "a prompt or a consequence. The approval text is generated elsewhere from "
            "something else, so nothing guarantees the approver and the reviewer are "
            "describing the same action",
        ),
        practice.Check(
            "FINDING: reversibility is the axis and it is not in the model",
            all([result["env_fields"] == [], result["approval_kinds"] == ["network"],
                 result["shipped_same"]]),
            f"approval_kinds is {result['approval_kinds']}, keyed by kind, so a staging "
            f"publish and a production publish are both network and get the identical "
            f"gate; {len(result['env_fields'])} fields across ActionRequest and "
            "SandboxPolicy mention an environment or reversibility. Approval following "
            "consequence needs a fact the request does not carry",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
