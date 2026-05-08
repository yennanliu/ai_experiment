"""
Generator agent — implements a single plan component using the tool loop.

Harness pattern: reads AGENTS.md for standing instructions, receives the full
plan as context, and writes its output via write_artifact so other agents
(Constraint Checker, Evaluator) can read it without coupling to this module.
"""

from pathlib import Path

from . import config, memory, provider, tools

_AGENTS_MD = Path(__file__).parent.parent / "AGENTS.md"


def run(task: str, plan: dict, component: str) -> str:
    """
    Execute the generator for one plan component.

    Returns the agent's final text reply (summary of what was written).
    Side-effects: artifacts written to disk, memory updated.
    """
    mem = memory.load()

    system = _AGENTS_MD.read_text()

    constraints_block = "\n".join(f"  - {c}" for c in plan.get("constraints", []))
    criteria_block = "\n".join(f"  - {c}" for c in plan.get("success_criteria", []))

    prompt = (
        f"Task: {task}\n\n"
        f"Full plan:\n"
        f"  Components : {', '.join(plan['components'])}\n"
        f"  Artifacts  : {', '.join(plan['artifacts'])}\n"
        f"  Constraints:\n{constraints_block}\n"
        f"  Success criteria:\n{criteria_block}\n\n"
        f"Your assignment: implement the '{component}' component.\n"
        f"Write a thorough artifact using write_artifact. "
        f"The artifact slug should be derived from the component name."
    )

    conv = provider.make_conversation(system=system, tools=tools.SCHEMAS)
    response = conv.send(prompt)

    while not response.done:
        for tc in response.tool_calls:
            short_args = ", ".join(
                f"{k}={str(v)[:40]!r}" if len(str(v)) > 40 else f"{k}={v!r}"
                for k, v in tc.input.items()
            )
            print(f"    Tool : {tc.name}({short_args})")
        results = [tools.run(tc.name, tc.input, mem) for tc in response.tool_calls]
        response = conv.add_tool_results(response.tool_calls, results)

    memory.save(mem)
    return response.text
