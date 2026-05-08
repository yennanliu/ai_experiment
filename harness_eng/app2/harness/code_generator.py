"""
Software Engineer agent — writes actual Python source files.

Harness pattern: uses SOFTWARE_ENG.md for standing instructions and
CODE_SCHEMAS (memory + workspace tools) instead of design-artifact tools,
keeping each agent's tool surface scoped to its role.
"""

from pathlib import Path

from . import config, memory, provider, tools

_SOFTWARE_ENG_MD = Path(__file__).parent.parent / "SOFTWARE_ENG.md"


def run(task: str, plan: dict, component: str) -> str:
    """
    Run the software engineer for one plan component.

    Writes .py files to workspace/ via the write_code tool.
    Returns the agent's final text summary.
    """
    mem = memory.load()

    system = _SOFTWARE_ENG_MD.read_text()

    components_block = ", ".join(plan["components"])
    prompt = (
        f"Task: {task}\n\n"
        f"All components in this project: {components_block}\n\n"
        f"Your assignment: implement the '{component}' component as Python source.\n"
        f"Check list_workspace first, then write the file with write_code."
    )

    conv = provider.make_conversation(
        system=system,
        tools=tools.CODE_SCHEMAS,
        max_tokens=config.CODE_MAX_TOKENS,
    )
    response = conv.send(prompt)

    while not response.done:
        for tc in response.tool_calls:
            short_args = ", ".join(
                f"{k}={str(v)[:50]!r}" if len(str(v)) > 50 else f"{k}={v!r}"
                for k, v in tc.input.items()
            )
            print(f"    Tool : {tc.name}({short_args})")
        results = [tools.run(tc.name, tc.input, mem) for tc in response.tool_calls]
        response = conv.add_tool_results(response.tool_calls, results)

    memory.save(mem)
    return response.text
