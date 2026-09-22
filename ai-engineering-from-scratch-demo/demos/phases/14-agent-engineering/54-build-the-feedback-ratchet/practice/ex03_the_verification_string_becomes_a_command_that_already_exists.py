"""Exercise 3 — the verification string becomes a command that already exists.

    Add verification commands or observations to the lab output.

Reading of the exercise: the lab already emits a `verification_evidence`
field, so the work is replacing five canned phrases with something a machine
can run -- and checking that the things they name are actually here.

**ANSWER: five phrases become four commands and one observation, and every
path they name exists.** "Record a passing regression evaluation" becomes
`uv run pytest demos/phases/14-agent-engineering`; the policy, context and
runtime phrases become the audit, the coverage check and the dependency
check; the backlog phrase stays an observation, because reviewing a shaped
item against an outcome frame is not a command. **4** of the **5** name a
script or directory in this repository and **4** of **4** of those paths
exist.

**FINDING: the canned phrases describe an artifact class, not this
repository.** All **5** begin with "Record", which is an instruction to a
person; **0** contain a path, a flag or an executable. The field is named
`verification_evidence` and holds neither -- it holds the name of the
evidence somebody should go and produce.

**FINDING: the replacement is destination-shaped, which is why one of them
cannot be a command.** Evaluation, policy, context and runtime all have a
runnable check here; `backlog` is a decision about whether work should exist,
and no command settles that. A ratchet that demands a command for every
destination would push backlog items into whichever other bucket has one.

**FINDING: a command in a string is still a string.** Replacing the phrase
changes what a human would paste, not what the system runs: `promote` writes
the value and `backlog` sorts the actions -- **0** lines execute anything.
The lesson's own loop ends at "verify that recurrence becomes less likely",
and the artifact stops one step earlier.

Structure: `COMMANDS` is the replacement table; `resolves()` checks each
named path against the tree.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "54-build-the-feedback-ratchet"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "demos").is_dir())
PHASE_DIR = "demos/phases/14-agent-engineering"

# destination -> (replacement, the path it names)
COMMANDS = {
    "evaluation": (f"uv run pytest {PHASE_DIR}", PHASE_DIR),
    "policy": (f"uv run python scripts/audit_practice.py 14-agent-engineering",
               "scripts/audit_practice.py"),
    "context": ("uv run python scripts/coverage.py --check", "scripts/coverage.py"),
    "runtime": ("uv run python scripts/check_deps.py", "scripts/check_deps.py"),
    "backlog": ("observation: the shaped item is reviewed against the outcome frame", ""),
}
TRIGGERS = {"evaluation": "regression", "policy": "unsafe", "context": "missing context",
            "runtime": "timeout", "backlog": "new product need"}


def shipped(ref):
    """The lab's own verification strings, one per destination."""
    rows = {}
    for destination, text in TRIGGERS.items():
        action = ref.promote(ref.Signal("s", text, 3, 1, "owner", 30))
        rows[action.destination] = action.verification_evidence
    return rows


def resolves():
    named = {path for _, path in COMMANDS.values() if path}
    return len(named), sum((ROOT / path).exists() for path in named)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    phrases = shipped(ref)
    named, existing = resolves()
    module = inspect.getsource(ref)
    return {
        "phrases": len(phrases), "replacements": len(COMMANDS),
        "record_prefixed": sum(text.startswith("Record") for text in phrases.values()),
        "with_path": sum(any(mark in text for mark in ("/", ".py"))
                         for text in phrases.values()),
        "commands": sum(text.startswith("uv run") for text, _ in COMMANDS.values()),
        "observations": sum(text.startswith("observation") for text, _ in COMMANDS.values()),
        "named": named, "existing": existing,
        "backlog_replacement": COMMANDS["backlog"][0].startswith("observation"),
        "executes": sum(word in module for word in ("subprocess", "os.system", "run(")),
        "loop_step": "verify" in parity.doc_text(PHASE, LESSON).lower(),
        "fields": list(ref.RatchetAction.__dataclass_fields__),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: five phrases become four commands and one observation",
            all([result["phrases"] == 5, result["replacements"] == 5,
                 result["commands"] == 4, result["observations"] == 1,
                 result["named"] == 4, result["existing"] == 4]),
            f"{result['commands']} of the {result['replacements']} replacements are "
            f"runnable commands and {result['observations']} stays an observation; they "
            f"name {result['named']} paths in this repository and {result['existing']} of "
            "them exist",
        ),
        practice.Check(
            "FINDING: the canned phrases describe an artifact class, not this repository",
            all([result["record_prefixed"] == 5, result["with_path"] == 0]),
            f"all {result['record_prefixed']} shipped phrases begin with 'Record' and "
            f"{result['with_path']} contain a path or an executable: the field holds the "
            "name of the evidence somebody should go and produce",
        ),
        practice.Check(
            "FINDING: the replacement is destination-shaped",
            all([result["backlog_replacement"] is True, result["commands"] == 4]),
            "evaluation, policy, context and runtime each have a runnable check here, and "
            "backlog is a decision about whether work should exist -- a ratchet demanding "
            "a command for every destination would push backlog items into another bucket",
        ),
        practice.Check(
            "FINDING: a command in a string is still a string",
            all([result["executes"] == 0, len(result["fields"]) == 7,
                 result["loop_step"] is True]),
            f"the module executes nothing ({result['executes']} calls) and the action's "
            f"{len(result['fields'])} fields only carry text, so replacing the phrase "
            "changes what a human would paste rather than what the system runs",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
