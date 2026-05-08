"""Core agent loop — provider-agnostic."""

from pathlib import Path

from . import config, memory, provider, tools

_AGENTS_MD = Path(__file__).parent.parent / "AGENTS.md"


def run(question: str) -> str:
    mem = memory.load()

    print(f"  Agent  : {_AGENTS_MD.name}")
    print(f"  Model  : {config.PROVIDER} / {config.active_model()}")

    conv = provider.make_conversation(system=_AGENTS_MD.read_text(), tools=tools.SCHEMAS)
    response = conv.send(question)

    while not response.done:
        for tc in response.tool_calls:
            args = ", ".join(f"{k}={v!r}" for k, v in tc.input.items())
            print(f"  Tool   : {tc.name}({args})")
        results = [tools.run(tc.name, tc.input, mem) for tc in response.tool_calls]
        response = conv.add_tool_results(response.tool_calls, results)

    memory.save(mem)
    return response.text
