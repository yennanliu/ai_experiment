"""
Generator agent — tool loop for one plan component.

Key difference from app2: memory is passed in (not loaded from disk here).
This lets fan_out give each concurrent agent an isolated memory snapshot
and merge the deltas back after all threads complete.
"""

from pathlib import Path

from . import provider, tools

_AGENTS_MD = Path(__file__).parent.parent / "AGENTS.md"


def run(task: str, plan: dict, component: str, mem: dict, verbose: bool = True) -> str:
    """
    Execute the generator for one plan component.

    Args:
        mem:     Caller-supplied memory dict (fan_out passes an isolated copy).
        verbose: When False, tool calls are not printed (avoids interleaved output
                 from concurrent threads).

    Returns the agent's final text reply. Side-effects: artifact written to disk.
    """
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
        if verbose:
            for tc in response.tool_calls:
                short_args = ", ".join(
                    f"{k}={str(v)[:40]!r}" if len(str(v)) > 40 else f"{k}={v!r}"
                    for k, v in tc.input.items()
                )
                print(f"    Tool : {tc.name}({short_args})")
        results = [tools.run(tc.name, tc.input, mem) for tc in response.tool_calls]
        response = conv.add_tool_results(response.tool_calls, results)

    return response.text
