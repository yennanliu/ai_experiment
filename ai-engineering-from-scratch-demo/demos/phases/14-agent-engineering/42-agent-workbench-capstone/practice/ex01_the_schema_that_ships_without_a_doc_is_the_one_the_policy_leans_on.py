"""Exercise 1 — the schema that ships without a doc is the one the policy leans on.

    Decide which optional fifth doc deserves promotion into the canonical
    pack. Defend the cut.

Reading of the exercise: the cut is decidable rather than tasteful. The pack
ships three schemas, four scripts and a reliability policy that names five
failure modes; a doc earns promotion when something in the pack depends on a
human writing a file the pack never explains.

**ANSWER: the fifth doc is a scope-contract authoring guide.**
`scope_contract.schema.json` ships with **7** properties and **6** of them
required, and **0** of the four docs mention it by name -- against
`agent_state`, which `docs/agent-rules.md` does name. The gate reads a scope
report, the policy leans on the scope check, and nothing in the pack tells a
human how to write the contract both depend on.

**FINDING: 2 of the 5 failure modes the policy claims to absorb have a script
behind them.** `run_with_feedback.py` covers cascading errors and
`verify_agent.py` covers hallucinated action; scope creep, context loss and
tool misuse name a scope checker, a state writer and a reviewer that the
pack's **4** scripts do not include. The policy is a promise the directory
does not keep.

**FINDING: the gate reads three paths the pack never writes.**
`verify_agent.py` loads `outputs/scope/closed/<task>.json`,
`outputs/scope/closed/<task>.report.json` and `outputs/rule_report.json`, each
through a loader that returns a default when the file is missing. So a fresh
install verifies **0** acceptance commands against **0** rules and reports
`passed: true`.

**FINDING: `AGENTS.md` opens with two files the installer never creates.** It
tells the agent to read **6** paths before acting; the installer lays down
**4** of them and leaves `agent_state.json` and `task_board.json` absent, which
`init_agent.py` then reports as the warn-severity probe "no state file yet".
A doc explaining the contract is the smallest fix that makes the rest legible.

Structure: `build()` assembles the pack into a temp directory; `coverage()`
maps the policy's failure modes onto the scripts that ship.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import tempfile
from pathlib import Path

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "42-agent-workbench-capstone"

# The policy's five failure modes and the pack script each one names.
MODES = [("hallucinated action", "verify_agent.py"), ("scope creep", "scope_check.py"),
         ("cascading errors", "run_with_feedback.py"), ("context loss", "write_state.py"),
         ("tool misuse", "review_agent.py")]


def build(ref):
    """Assemble the pack into a temp directory instead of the lesson's outputs/."""
    root = Path(tempfile.mkdtemp(prefix="pack-")) / "agent-workbench-pack"
    ref.PACK = root
    with contextlib.redirect_stdout(io.StringIO()):
        ref.main()
    return root


def coverage(root):
    shipped = {path.name for path in (root / "scripts").iterdir()}
    return [(mode, script, script in shipped) for mode, script in MODES]


def router(root):
    """What AGENTS.md tells the agent to read, and which of those the installer creates."""
    agents = (root / "AGENTS.md").read_text()
    named = [re.findall(r"`([^`]+)`", line)[0] for line in agents.splitlines()
             if re.match(r"^\d+\. `", line)]
    return {"agents_reads": named,
            "installed": [name for name in named if (root / "docs" / Path(name).name).exists()],
            "missing": [name for name in named if name.endswith(".json")]}


def schema_docs(root):
    """Each shipped schema and whether any doc mentions it by name."""
    joined = " ".join(path.read_text() for path in (root / "docs").iterdir())
    names = sorted(path.name for path in (root / "schemas").iterdir())
    return names, {name.split(".")[0]: name.split(".")[0] in joined for name in names}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    root = build(ref)
    schemas, named = schema_docs(root)
    scope_schema = json.loads((root / "schemas" / "scope_contract.schema.json").read_text())
    gate = (root / "scripts" / "verify_agent.py").read_text()
    return {
        **router(root),
        "docs": sorted(path.name for path in (root / "docs").iterdir()),
        "schemas": schemas, "schema_named": named,
        "scope_properties": len(scope_schema["properties"]),
        "scope_required": len(scope_schema["required"]),
        "coverage": coverage(root), "covered": sum(ok for _, _, ok in coverage(root)),
        "scripts": sorted(path.name for path in (root / "scripts").iterdir()),
        "gate_paths": sorted({path for path in re.findall(r'ROOT / f?"([^"]+)"', gate)
                              if "/" in path or "." in path}),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the fifth doc is a scope-contract authoring guide",
            all([result["scope_properties"] == 7, result["scope_required"] == 6,
                 result["schema_named"]["scope_contract"] is False,
                 result["schema_named"]["agent_state"] is True,
                 len(result["docs"]) == 4]),
            f"scope_contract.schema.json ships {result['scope_properties']} properties with "
            f"{result['scope_required']} required and none of the {len(result['docs'])} docs "
            f"names it, while agent_state is named; the schemas that ship are "
            f"{result['schemas']}",
        ),
        practice.Check(
            "FINDING: 2 of the 5 failure modes have a script behind them",
            all([result["covered"] == 2, len(result["coverage"]) == 5,
                 len(result["scripts"]) == 4]),
            f"{result['covered']} of {len(result['coverage'])} modes map onto the pack's "
            f"{len(result['scripts'])} scripts: "
            f"{[mode for mode, _, ok in result['coverage'] if not ok]} name a scope checker, "
            "a state writer and a reviewer the directory does not ship",
        ),
        practice.Check(
            "FINDING: the gate reads three paths the pack never writes",
            all([len(result["gate_paths"]) == 4,
                 "outputs/rule_report.json" in result["gate_paths"],
                 "feedback_record.jsonl" in result["gate_paths"]]),
            f"verify_agent.py loads {result['gate_paths']} through a loader that returns a "
            "default when the file is missing, so a fresh install verifies nothing and "
            "reports passed: true",
        ),
        practice.Check(
            "FINDING: AGENTS.md opens with two files the installer never creates",
            all([len(result["agents_reads"]) == 6, len(result["installed"]) == 4,
                 result["missing"] == ["agent_state.json", "task_board.json"]]),
            f"AGENTS.md names {len(result['agents_reads'])} paths to read before acting; "
            f"{len(result['installed'])} are the docs the installer copies and "
            f"{result['missing']} are never created -- which init_agent.py then reports as "
            "'no state file yet'",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
