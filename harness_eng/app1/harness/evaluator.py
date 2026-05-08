"""Separate evaluator agent — never let the generator grade its own work."""

import json

import anthropic

_MODEL = "claude-haiku-4-5-20251001"
_SYSTEM = (
    "You are a strict quality evaluator. "
    "Given a question and an answer, rate quality 1–5 and explain briefly. "
    'Respond with valid JSON only: {"score": <int>, "reason": "<str>"}'
)


def evaluate(question: str, answer: str) -> dict:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=_MODEL,
        max_tokens=256,
        system=_SYSTEM,
        messages=[{"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"}],
    )
    try:
        return json.loads(response.content[0].text)
    except (json.JSONDecodeError, IndexError):
        return {"score": 0, "reason": "parse error"}
