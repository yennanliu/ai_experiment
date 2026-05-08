"""Core agent loop — the engine inside the harness."""

from pathlib import Path

import anthropic

from . import memory, tools

_AGENTS_MD = Path(__file__).parent.parent / "AGENTS.md"
_MODEL = "claude-haiku-4-5-20251001"


def run(question: str) -> str:
    client = anthropic.Anthropic()
    mem = memory.load()
    system = _AGENTS_MD.read_text()
    messages = [{"role": "user", "content": question}]

    while True:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            system=system,
            tools=tools.SCHEMAS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            memory.save(mem)
            return next(b.text for b in response.content if hasattr(b, "text"))

        # Agentic tool-call loop
        messages.append({"role": "assistant", "content": response.content})
        results = [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": tools.run(block.name, block.input, mem),
            }
            for block in response.content
            if block.type == "tool_use"
        ]
        messages.append({"role": "user", "content": results})
