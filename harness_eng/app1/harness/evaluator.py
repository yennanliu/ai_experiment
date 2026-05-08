"""Separate evaluator agent — never let the generator grade its own work."""

import json

from . import provider

_SYSTEM = (
    "You are a strict quality evaluator. "
    "Given a question and an answer, rate quality 1–5 and explain briefly. "
    'Respond with valid JSON only: {"score": <int>, "reason": "<str>"}'
)


def evaluate(question: str, answer: str) -> dict:
    text = provider.simple_complete(
        system=_SYSTEM,
        user=f"Question: {question}\n\nAnswer: {answer}",
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"score": 0, "reason": "parse error"}
