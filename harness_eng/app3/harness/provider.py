"""
Unified LLM interface — identical to app2, supports Anthropic and OpenAI.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from . import config


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.text is not None


# ── Anthropic ──────────────────────────────────────────────────────────────────

class AnthropicConversation:
    def __init__(self, system: str, tools: list[dict], max_tokens: int = config.MAX_TOKENS):
        import anthropic
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._system = system
        self._tools = tools
        self._max_tokens = max_tokens
        self._messages: list[dict] = []

    def send(self, content: str) -> LLMResponse:
        self._messages.append({"role": "user", "content": content})
        return self._call()

    def add_tool_results(self, tool_calls: list[ToolCall], results: list[str]) -> LLMResponse:
        self._messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tc.id, "content": r}
                for tc, r in zip(tool_calls, results)
            ],
        })
        return self._call()

    def _call(self) -> LLMResponse:
        resp = self._client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=self._max_tokens,
            temperature=config.TEMPERATURE,
            system=self._system,
            tools=self._tools,
            messages=self._messages,
        )
        self._messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason == "end_turn":
            text = next((b.text for b in resp.content if hasattr(b, "text")), "")
            return LLMResponse(text=text)

        return LLMResponse(
            text=None,
            tool_calls=[
                ToolCall(id=b.id, name=b.name, input=b.input)
                for b in resp.content
                if b.type == "tool_use"
            ],
        )


# ── OpenAI ─────────────────────────────────────────────────────────────────────

def _to_openai_tools(schemas: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in schemas
    ]


class OpenAIConversation:
    def __init__(self, system: str, tools: list[dict], max_tokens: int = config.MAX_TOKENS):
        import openai
        self._client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self._tools = _to_openai_tools(tools)
        self._max_tokens = max_tokens
        self._messages: list[Any] = [{"role": "system", "content": system}]

    def send(self, content: str) -> LLMResponse:
        self._messages.append({"role": "user", "content": content})
        return self._call()

    def add_tool_results(self, tool_calls: list[ToolCall], results: list[str]) -> LLMResponse:
        for tc, r in zip(tool_calls, results):
            self._messages.append({"role": "tool", "tool_call_id": tc.id, "content": r})
        return self._call()

    def _call(self) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=config.OPENAI_MODEL,
            max_tokens=self._max_tokens,
            temperature=config.TEMPERATURE,
            tools=self._tools,
            messages=self._messages,
        )
        msg = resp.choices[0].message
        self._messages.append(msg)

        if not msg.tool_calls:
            return LLMResponse(text=msg.content or "")

        tool_calls = []
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))

        return LLMResponse(text=None, tool_calls=tool_calls)


# ── Factory ────────────────────────────────────────────────────────────────────

def make_conversation(
    system: str,
    tools: list[dict],
    max_tokens: int = config.MAX_TOKENS,
) -> AnthropicConversation | OpenAIConversation:
    if config.PROVIDER == "openai":
        return OpenAIConversation(system, tools, max_tokens=max_tokens)
    return AnthropicConversation(system, tools, max_tokens=max_tokens)


def simple_complete(system: str, user: str, max_tokens: int = config.MAX_TOKENS) -> str:
    """One-shot completion without tools."""
    if config.PROVIDER == "openai":
        import openai
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            max_tokens=max_tokens,
            temperature=config.TEMPERATURE,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            temperature=config.TEMPERATURE,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text
